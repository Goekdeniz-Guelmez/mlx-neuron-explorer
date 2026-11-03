"""MLX backend for the explorer's live model (Apple Silicon): the same Runtime, with every forward pass run by mlx-lm.

The forward-pass parts (steering, reads, activations, head captures) are rewritten here on MLX. The weight-only
analysis in runtime.py (connections, head links, R/J++ lenses, attention-row maths on captures) runs unchanged on CPU
torch tensors: self.lm and self.model are thin views over the MLX weights with the attribute names runtime.py reads,
and captures are handed over as CPU torch tensors. mlx-lm stores RMSNorm weights as (1 + w); the views give back w.

Selected with EXPLORER_BACKEND=mlx (the default on Apple Silicon when mlx-lm is installed). EXPLORER_MODEL may name
an mlx-community conversion, quantized ones included (weights are dequantized when read).
"""
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import ArraysCache, BatchKVCache
from transformers import AutoTokenizer

import runtime as base
from runtime import Runtime, _locked


def _np(a):
    return np.array(a.astype(mx.float32))


def _t(a, half=False):
    t = torch.from_numpy(_np(a))
    return t.half() if half else t


def _weight(lin):
    if isinstance(lin, nn.QuantizedLinear):
        return mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
    return lin.weight


# ---------- torch views of the MLX weights, with the names runtime.py reads ----------
class _Lin:
    def __init__(self, m):
        self._m = m

    @property
    def weight(self):
        return _t(_weight(self._m))

    @property
    def in_features(self):
        return _weight(self._m).shape[1]  # lazy: the shape is known without dequantizing

    @property
    def out_features(self):
        return _weight(self._m).shape[0]

    def __call__(self, x):
        return x.float() @ self.weight.T


class _Norm:
    """An RMSNorm whose MLX weight is (1 + w): .weight gives w, calling it applies x / rms * (1 + w)."""
    def __init__(self, m, offset=1.0):
        self._m, self._off = m, offset
        self.eps = self.variance_epsilon = m.eps

    @property
    def weight(self):
        return _t(self._m.weight) - self._off

    def __call__(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * _t(self._m.weight)


class _View:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _attn_view(a):
    return _View(q_proj=_Lin(a.q_proj), k_proj=_Lin(a.k_proj), v_proj=_Lin(a.v_proj), o_proj=_Lin(a.o_proj.inner),
                 head_dim=a.head_dim, scaling=a.scale)


def _linear_view(a):
    return _View(in_proj_qkv=_Lin(a.in_proj_qkv), in_proj_a=_Lin(a.in_proj_a), in_proj_b=_Lin(a.in_proj_b),
                 out_proj=_Lin(a.out_proj.inner), norm=_Norm(a.norm.inner, 0.0), num_v_heads=a.num_v_heads,
                 num_k_heads=a.num_k_heads, head_v_dim=a.head_v_dim, head_k_dim=a.head_k_dim, key_dim=a.key_dim,
                 value_dim=a.value_dim, conv_dim=a.conv_dim)


def _layer_view(L):
    v = _View(input_layernorm=_Norm(L.input_layernorm), post_attention_layernorm=_Norm(L.post_attention_layernorm),
              mlp=_View(gate_proj=_Lin(L.mlp.inner.gate_proj), up_proj=_Lin(L.mlp.inner.up_proj),
                        down_proj=_Lin(L.mlp.inner.down_proj)))
    if L.is_linear:
        v.linear_attn = _linear_view(L.linear_attn)
    else:
        v.self_attn = _attn_view(L.self_attn)
    return v


class _Head:
    def __init__(self, lm):
        self._lm = lm
        self.tied = lm.args.tie_word_embeddings

    @property
    def weight(self):
        return _t(_weight(self._lm.model.embed_tokens) if self.tied else _weight(self._lm.lm_head))

    @property
    def out_features(self):
        return self._lm.args.vocab_size

    def __call__(self, x):
        return x.float() @ self.weight.T


# ---------- hooks: MLX modules wrapped in place ----------
class _MLP(nn.Module):
    """The MLP with the activations silu(gate) * up exposed: read, captured, added to or multiplied per batch row."""
    def __init__(self, inner, rt, i):
        super().__init__()
        self.inner, self._rt, self._i = inner, rt, i

    def __call__(self, x):
        m = self.inner
        h = nn.silu(m.gate_proj(x)) * m.up_proj(x)
        return m.down_proj(self._rt._on_act(self._i, h))


class _OutProj(nn.Module):
    """An attention output projection, whose input is the heads side by side: captured and scaled per head."""
    def __init__(self, inner, rt, i, n_heads):
        super().__init__()
        self.inner, self._rt, self._i, self._h = inner, rt, i, n_heads

    def __call__(self, x):
        return self.inner(self._rt._on_heads(self._i, x, self._h, self.inner))


class _GNorm(nn.Module):
    """Linear attention's gated norm: while capturing heads, each head's output scale."""
    def __init__(self, inner, rt, i):
        super().__init__()
        self.inner, self._rt, self._i = inner, rt, i

    def __call__(self, o, z=None):
        c = self._rt._capture_heads
        if c is not None:
            of, zf = o[0].astype(mx.float32), z[0].astype(mx.float32)
            sc = self.inner.weight.astype(mx.float32) * nn.silu(zf) * mx.rsqrt((of * of).mean(-1, keepdims=True) + self.inner.eps)
            c["s"][self._i] = sc.transpose(1, 0, 2)  # [H, T, D]
        return self.inner(o, z)


class _Layer(nn.Module):
    """A decoder layer with the residual after it, the MLP input and the attention query/key side exposed."""
    def __init__(self, inner, rt, i):
        super().__init__()
        self.inner, self._rt, self._i = inner, rt, i
        self.is_linear = inner.is_linear

    def __call__(self, x, mask=None, cache=None):
        L, rt, i = self.inner, self._rt, self._i
        xn = L.input_layernorm(x)
        a = L.linear_attn if L.is_linear else L.self_attn
        if rt._capture_heads is not None:
            rt._capture_qk(i, a, xn)
        h = x + a(xn, mask, cache)
        m_in = L.post_attention_layernorm(h)
        if rt._capture_heads is not None:
            hf = h[0].astype(mx.float32)
            rt._capture_heads["x"][i] = m_in[0]
            rt._capture_heads["rms"][i] = mx.sqrt((hf * hf).mean(-1) + L.post_attention_layernorm.eps)
        out = h + L.mlp(m_in)
        if rt._seen.get("want_resid") == i:
            rt._seen["resid"] = mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1)
        return out


def _l2norm(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)


class MLXRuntime(Runtime):
    backend = "mlx"

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.model is not None:
            return
        if base.NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            m, _ = mlx_load(base.MODEL)
            self.tok = AutoTokenizer.from_pretrained(base.MODEL)
            lm = m.language_model if hasattr(m, "language_model") else m
            inner = lm.model
            if not hasattr(inner.layers[0].mlp, "down_proj"):
                raise RuntimeError(f"{base.MODEL} is a mixture-of-experts model: the explorer needs dense MLP neurons")
            for i, L in enumerate(inner.layers):
                L.mlp = _MLP(L.mlp, self, i)
                if L.is_linear:
                    a = L.linear_attn
                    a.out_proj = _OutProj(a.out_proj, self, i, a.num_v_heads)
                    a.norm = _GNorm(a.norm, self, i)
                else:
                    a = L.self_attn
                    a.o_proj = _OutProj(a.o_proj, self, i, a.num_attention_heads)
            inner.layers = [_Layer(L, self, i) for i, L in enumerate(inner.layers)]
            self.mx_model, self._mlm, self._inner = m, lm, inner
            self.dev = "cpu"  # where the torch views and captures live
            self.lm = _View(layers=[_layer_view(L.inner) for L in inner.layers], norm=_Norm(inner.norm))
            self.model = _View(lm_head=_Head(lm), dtype=torch.float32)
            self.status = "ready"

    def _layers(self):
        return self._inner.layers

    # ---------- hook bodies ----------
    def _on_act(self, i, h):
        if self._capture_full is not None:
            self._capture_full[i] = h[0]
        if self._capture_all is not None:
            a = h[:, self._capture_from:].astype(mx.float32).reshape(-1, h.shape[-1])
            self._capture_all[i] = (a.max(0), a.argmax(0), a.min(0), a.argmin(0))
        if self._read is not None and self._read[0] == i:
            self._seen["act"] = h[0, :, self._read[1]]
        if "grab" in self._seen and self._seen["grab"][0] == i:
            self._seen["grabbed"] = h[:, :, self._seen["grab"][1]]
        for _, n, mode, vals in [p for p in self._push or [] if p[0] == i]:
            v = vals.astype(h.dtype)
            if n is None:  # the whole layer's MLP (multiply only)
                h = h * v[:, None, None]
            elif mode == "mul":
                h[:, :, n] = h[:, :, n] * v[:, None]
            else:
                h[:, :, n] = h[:, :, n] + v[:, None]
        return h

    def _on_heads(self, i, x, n_heads, proj):
        c = self._capture_heads
        if c is not None:  # each head's write into the residual: |W_O[:, head] . out_head|
            xv = x[0].reshape(x.shape[1], n_heads, -1)
            W = _weight(proj).reshape(-1, n_heads, xv.shape[-1])
            w = mx.einsum("thd,ohd->tho", xv.astype(mx.float32), W.astype(mx.float32))
            c["writes"][i] = mx.linalg.norm(w, axis=-1)  # [T, H]
            c["y"][i] = xv.transpose(1, 0, 2)  # [H, T, D]
            mx.eval(c["writes"][i])
        mine = [h for h in self._heads or [] if h[0] == i]
        if not mine:
            return x
        s = [1.0] * n_heads
        for _, head, scale in mine:
            s = [v * scale for v in s] if head is None else [v * scale if j == head else v for j, v in enumerate(s)]
        v = x.reshape(*x.shape[:-1], n_heads, -1) * mx.array(s, dtype=x.dtype)[:, None]
        return v.reshape(x.shape)

    def _capture_qk(self, i, a, x):
        """runtime._qk_hook on MLX: queries and keys after norm and rotary (full attention), or after the short conv
        and l2 norm with beta and the per-step log decay (linear attention), for the first batch row."""
        T = x.shape[1]
        c = self._capture_heads["q"]
        if not self._layers()[i].is_linear:
            hd = a.head_dim
            q, gate = mx.split(a.q_proj(x).reshape(1, T, a.num_attention_heads, -1), 2, axis=-1)
            q = a.rope(a.q_norm(q).transpose(0, 2, 1, 3))
            k = a.rope(a.k_norm(a.k_proj(x).reshape(1, T, a.num_key_value_heads, hd)).transpose(0, 2, 1, 3))
            v = a.v_proj(x).reshape(T, -1, hd).transpose(1, 0, 2)
            c[i] = {"kind": "full", "q": q[0], "k": k[0], "scale": a.scale, "v": v,
                    "s": mx.sigmoid(gate[0].astype(mx.float32)).transpose(1, 0, 2)}
        else:
            qkv = a.in_proj_qkv(x)
            qkv = mx.concatenate([mx.zeros((1, a.conv_kernel_size - 1, qkv.shape[-1]), qkv.dtype), qkv], 1)
            qkv = nn.silu(a.conv1d(qkv))[0]
            q, k, v = mx.split(qkv, [a.key_dim, 2 * a.key_dim], -1)
            q = _l2norm(q.astype(mx.float32).reshape(T, -1, a.head_k_dim)) / a.head_k_dim ** 0.5
            k = _l2norm(k.astype(mx.float32).reshape(T, -1, a.head_k_dim))
            beta = mx.sigmoid(a.in_proj_b(x)[0].astype(mx.float32))
            logd = -mx.exp(a.A_log.astype(mx.float32)) * nn.softplus(a.in_proj_a(x)[0].astype(mx.float32) + a.dt_bias)
            c[i] = {"kind": "linear", "q": q.transpose(1, 0, 2), "k": k.transpose(1, 0, 2), "beta": beta.T, "logd": logd.T,
                    "rep": a.num_v_heads // a.num_k_heads, "v": v.reshape(T, -1, a.head_v_dim).transpose(1, 0, 2)}

    def _heads_to_torch(self, c):
        """A finished MLX head capture as the CPU torch tensors runtime.py's attention maths reads."""
        half = lambda d: {l: _t(v, True) for l, v in d.items()}
        q = {}
        for l, d in c["q"].items():
            q[l] = {k: (_t(v, k not in ("beta", "logd")) if isinstance(v, mx.array) else v) for k, v in d.items()}
        return {"q": q, "writes": {l: _t(v) for l, v in c["writes"].items()}, "y": half(c["y"]), "s": half(c["s"]),
                "x": half(c["x"]), "rms": {l: _t(v) for l, v in c["rms"].items()}}

    # ---------- passes ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        l, n, mode, vals = super()._push_entry(layer, n, dose, mode, rows)
        return (l, n, mode, mx.array(vals.numpy()))

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        import os
        path = os.path.join(self.stats_dir, f"q35-{base.TAG}_L{layer}.npz")
        if os.path.exists(path):
            v = float(np.load(path)["resid_norm_q10_50_90"][1])
        else:
            ids = self.tok(base.REFERENCE_TEXT).input_ids
            self._seen["want_resid"] = layer
            try:
                self._inner(mx.array([ids]))  # lazy: only layers up to `layer` are computed
                v = float(np.median(_np(self._seen.pop("resid"))))
            finally:
                self._seen.pop("want_resid", None)
        self.resid_p50[layer] = v
        return v

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None, temperature=0.0, repetition_penalty=1.0, seed=0, min_p=0.0):
        """runtime.steer_multi on MLX: the three rows left-padded into one batch with mlx-lm's batch caches; sampling,
        repetition penalty and seeding as in the torch version, on CPU."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                lp = [T - len(q) for q in seqs]
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                ids = mx.array([[pad] * p + q for p, q in zip(lp, seqs)])
                cache = [ArraysCache(2, left_padding=lp) if L.is_linear else BatchKVCache(lp) for L in self._layers()]
                eos = self.tok.convert_tokens_to_ids("<|im_end|>")
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                gens = [torch.Generator().manual_seed(int(seed)) for _ in range(3)]
                seen = None
                if repetition_penalty != 1.0:
                    seen = torch.zeros(3, self._mlm.args.vocab_size, dtype=torch.bool)
                    for r, q in enumerate(seqs):
                        seen[r, torch.tensor(q)] = True
                self._push = push
                out = self._mlm(ids, cache=cache)[:, -1]
                for _ in range(max_new):
                    if cancel is not None and cancel.is_set():
                        break
                    logits = _t(out)
                    if seen is not None:
                        pen = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                        logits = torch.where(seen, pen, logits)
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, -1)
                        if min_p > 0:
                            probs = torch.where(probs >= min_p * probs.max(-1, keepdim=True).values, probs, 0.0)
                            probs = probs / probs.sum(-1, keepdim=True)
                        nxt = torch.cat([torch.multinomial(probs[r], 1, generator=gens[r]) for r in range(3)])
                    else:
                        nxt = logits.argmax(-1)
                    if seen is not None:
                        seen[torch.arange(3), nxt] = True
                    for r in range(3):
                        if done[r]:
                            continue
                        t = int(nxt[r])
                        if t == eos:
                            done[r] = True
                            continue
                        out_ids[r].append(t)
                        text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                        if not text.endswith("�"):
                            yield r, text[len(sent[r]):]
                            sent[r] = text
                    if all(done):
                        break
                    out = self._mlm(mx.array(nxt.numpy())[:, None], cache=cache)[:, -1]
            finally:
                self._push = None
        yield None, None

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        self.ensure_loaded()
        with self.lock:
            ids = self.chat_ids(text) if chat else self.tok(text).input_ids
            self._read = (layer, n)
            self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
            self._heads = list(heads) or None if heads else None
            try:
                self._inner(mx.array([ids]))
                act = _np(self._seen.pop("act")).tolist()
            finally:
                self._read = self._push = self._heads = None
        units = []
        for text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(push, hd):
            with self.lock:
                self._push, self._heads = push, hd
                try:
                    h = self._inner(mx.array([ids]))[:, -1:]
                    logits = self._inner.embed_tokens.as_linear(h) if self._mlm.args.tie_word_embeddings else self._mlm.lm_head(h)
                    return torch.log_softmax(_t(logits[0, -1]), -1)
                finally:
                    self._push = self._heads = None

        base_lp = run(None, None)
        if not pushes and not heads:
            return base_lp, None
        push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
        return base_lp, run(push, list(heads) if heads else None)

    @_locked
    def neuron_traces(self, layer, n, seqs):
        self.ensure_loaded()
        S = max(len(s) for s in seqs)
        ids = np.zeros((len(seqs), S), np.int32)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = s
        self._seen["grab"] = (layer, n)
        try:
            self._inner(mx.array(ids))  # lazy: only layers up to `layer` are computed
            a = _np(self._seen.pop("grabbed"))
        finally:
            self._seen.pop("grab", None)
        return [a[r, :len(s)] for r, s in enumerate(seqs)]

    def all_activations(self, ids, skip_first=False):
        self.ensure_loaded()
        with self.lock:
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = mx.array(ids)
                self._inner(x if x.ndim == 2 else x[None])
                cap = self._capture_all
                mx.eval([v for t in cap.values() for v in t])
                return {l: tuple(torch.from_numpy(np.array(v).astype(np.int64 if j % 2 else np.float32)) for j, v in enumerate(t))
                        for l, t in cap.items()}
            finally:
                self._capture_all, self._capture_from = None, 0

    def full_activations(self, ids, heads=False):
        self.ensure_loaded()
        with self.lock:
            self._capture_full = {}
            if heads:
                self._capture_heads = {"q": {}, "writes": {}, "y": {}, "s": {}, "x": {}, "rms": {}}
            try:
                self._inner(mx.array([ids]))
                if heads:
                    self.heads = {"T": len(ids), "ids": list(ids), **self._heads_to_torch(self._capture_heads)}
                return np.stack([_np(self._capture_full[l]) for l in sorted(self._capture_full)])
            finally:
                self._capture_full = None
                self._capture_heads = None

    def live_stream(self, ids, max_new=0, cancel=None):
        """runtime.live_stream on MLX: the prompt in one pass, then one cached greedy step per new token."""
        self.ensure_loaded()
        with self.lock:
            eos = self.tok.convert_tokens_to_ids("<|im_end|>")
            cache = self._mlm.make_cache()
            try:
                self._capture_full = {}
                logits = self._mlm(mx.array([ids]), cache=cache)[0, -1]
                acts = np.stack([_np(self._capture_full[l]) for l in sorted(self._capture_full)])  # [L, T, N]
                for p, t in enumerate(ids):
                    if cancel is not None and cancel.is_set():
                        return
                    yield p, t, acts[:, p], True
                pos = len(ids)
                for _ in range(max_new):
                    nxt = int(mx.argmax(logits).item())
                    if nxt == eos or (cancel is not None and cancel.is_set()):
                        return
                    self._capture_full = {}
                    logits = self._mlm(mx.array([[nxt]]), cache=cache)[0, -1]
                    yield pos, nxt, np.stack([_np(self._capture_full[l][-1]) for l in sorted(self._capture_full)]), False
                    pos += 1
            finally:
                self._capture_full = None

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024):
        self.ensure_loaded()
        layers = self._layers()
        L, N = len(layers), _weight(layers[0].inner.mlp.inner.down_proj).shape[1]
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        head = (lambda h: self._inner.embed_tokens.as_linear(h)) if self._mlm.args.tie_word_embeddings else self._mlm.lm_head
        for li, layer in enumerate(layers):
            self.status = f"indexing lens {li}/{L}"
            with self.lock:
                W = _weight(layer.inner.mlp.inner.down_proj)  # [hidden, inter]
                for s in range(0, N, chunk):
                    logits = head(self._inner.norm(W[:, s:s + chunk].T)).astype(mx.float32)
                    for out, lg in ((up, logits), (down, -logits)):
                        top = mx.argpartition(-lg, k, axis=-1)[:, :k]
                        order = mx.argsort(-mx.take_along_axis(lg, top, -1), -1)
                        out[li, s:s + chunk] = np.array(mx.take_along_axis(top, order, -1))
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path
