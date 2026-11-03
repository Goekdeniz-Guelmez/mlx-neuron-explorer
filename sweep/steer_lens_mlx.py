"""steer_lens.py on MLX (Apple Silicon): for each neuron and sign, the next-token tokens a push raises most and the KL
from the unsteered prediction, over the same short contexts and push size.

    uv run python sweep/steer_lens_mlx.py --model 2b --layers 0-23

Writes stats/steer_lens_<size>_L{l}.npz per layer (fields as steer_lens.py), and once every layer of the model has
one, the explorer's steer_index.npz for that size (up, down [L, N, k]; up_kl, down_kl [L, N]).
"""
import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache, BatchKVCache
from transformers import AutoTokenizer

from mlx_util import cache_dir, load_text_model, model_tag, resolve, weight
from steer_lens import CHAT, PLAIN
from steer_sweep_mlx import PushMLP


class Tap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.is_linear, self.out = inner, inner.is_linear, None

    def __call__(self, x, mask=None, cache=None):
        self.out = self.inner(x, mask, cache)
        return self.out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="repo, or a size: 0.8b, 2b, 4b, 9b, 27b")
    ap.add_argument("--layers", default="all", help='"all", or a range such as 0-7')
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default=None, help="output prefix (default stats/steer_lens_<size>)")
    a = ap.parse_args()
    model = resolve(a.model)
    out = a.out or os.path.join("stats", f"steer_lens_{model_tag(model)}")

    tok = AutoTokenizer.from_pretrained(model)
    lm, dec = load_text_model(model)
    L = len(dec.layers)
    lo, _, hi = a.layers.partition("-")
    layers = list(range(L)) if a.layers == "all" else list(range(int(lo), int(hi or lo) + 1))
    ctxs = [tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, enable_thinking=False,
                                    tokenize=True) for c in CHAT] + [tok(p).input_ids for p in PLAIN]
    ctxs = [list(c["input_ids"] if hasattr(c, "keys") else c) for c in ctxs]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    left = [T - len(c) for c in ctxs]
    ids = np.array([[pad] * p + c for p, c in zip(left, ctxs)], np.int32)

    taps = [Tap(x) for x in dec.layers]
    dec.layers = taps

    def last_logprobs(rows):
        """Last-position log-probabilities of the contexts, repeated `rows` times (left padding masked by the caches)."""
        lp = left * rows
        cache = [ArraysCache(2, left_padding=lp) if t.is_linear else BatchKVCache(lp) for t in taps]
        x = lm(mx.array(np.tile(ids, (rows, 1))), cache=cache)[:, -1].astype(mx.float32)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    base = last_logprobs(1)  # [C, V]
    real = np.arange(T)[None] >= np.array(left)[:, None]
    norms = [float(np.median(np.linalg.norm(np.array(t.out.astype(mx.float32)), axis=-1)[real])) for t in taps]
    mx.eval(base)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    B = a.neurons_per_batch
    for l in layers:
        W = weight(taps[l].inner.mlp.down_proj).astype(mx.float32)  # [hidden, inter]
        N = W.shape[1]
        unit = norms[l] / mx.linalg.norm(W, axis=0)
        push = taps[l].inner.mlp = PushMLP(taps[l].inner.mlp)
        res = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, N, B):
            n = mx.arange(s, min(s + B, N))
            b = n.shape[0]
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                cols = mx.repeat(n, C)  # rows: neuron-major, contexts repeated
                push.vec = (W[:, cols] * (sign * a.dose * unit[cols])[None]).T  # [b*C, hidden]
                lp = last_logprobs(b).reshape(b, C, -1)
                delta = (lp - base[None]).mean(1)
                kl = (mx.exp(lp) * (lp - base[None])).sum(-1).mean(1)
                top = mx.argpartition(-delta, a.k, axis=-1)[:, :a.k]
                top = mx.take_along_axis(top, mx.argsort(-mx.take_along_axis(delta, top, -1), -1), -1)
                mx.eval(top, kl)
                res[f"{key}_tok"].append(np.array(top).astype(np.int32)); res[f"{key}_kl"].append(np.array(kl))
            push.vec = None
        taps[l].inner.mlp = push.inner
        arrs = {k: np.concatenate(v) for k, v in res.items()}
        np.savez(f"{out}_L{l}.npz", **arrs, dose=a.dose, contexts=np.array(CHAT + PLAIN))
        print(f"L{l}: {N} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}", flush=True)

    paths = [f"{out}_L{l}.npz" for l in range(L)]
    if all(os.path.exists(p) for p in paths):
        zs = [np.load(p) for p in paths]
        idx = os.path.join(cache_dir(model), "steer_index.npz")
        os.makedirs(os.path.dirname(idx), exist_ok=True)
        np.savez(idx, up=np.stack([z["up_tok"] for z in zs]), down=np.stack([z["down_tok"] for z in zs]),
                 up_kl=np.stack([z["up_kl"] for z in zs]), down_kl=np.stack([z["down_kl"] for z in zs]), dose=a.dose)
        print(f"every layer done -> {idx}")


if __name__ == "__main__":
    main()
