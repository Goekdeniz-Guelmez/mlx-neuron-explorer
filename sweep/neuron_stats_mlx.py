"""neuron_stats.py on MLX (Apple Silicon): the same per-neuron activation ranges by sign regime, down columns and
residual norms, in the same npz, from the same web + chat corpus.

    uv run python sweep/neuron_stats_mlx.py --model 2b --layers 12 --tokens 200000

Writes stats/q35-<size>_L<layer>.npz (see neuron_stats.py for the fields). Smaller batches than the CUDA script by
default: a batch's activations for every target layer live in unified memory at once.
"""
import argparse
import itertools
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from mlx_util import load_text_model, resolve, stats_prefix, weight
from neuron_stats import text_stream


class StatMLP(nn.Module):
    """A layer's MLP that folds each batch's activations into the regime extremes as it runs."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        n = weight(inner.down_proj).shape[1]
        self.pp_max, self.nn_max, self.pn_min, self.np_min = (mx.zeros(n) for _ in range(4))

    def __call__(self, x):
        m = self.inner
        g, u = m.gate_proj(x), m.up_proj(x)
        a = nn.silu(g) * u
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        gf, uf, af = (t[:, 1:].astype(mx.float32).reshape(-1, t.shape[-1]) for t in (g, u, a))
        gp, up_ = gf > 0, uf > 0
        self.pp_max = mx.maximum(self.pp_max, mx.where(gp & up_, af, -mx.inf).max(0))
        self.nn_max = mx.maximum(self.nn_max, mx.where(~gp & ~up_, af, -mx.inf).max(0))
        self.pn_min = mx.minimum(self.pn_min, mx.where(gp & ~up_, af, mx.inf).min(0))
        self.np_min = mx.minimum(self.np_min, mx.where(~gp & up_, af, mx.inf).min(0))
        return m.down_proj(a)


class ResidTap(nn.Module):
    """A decoder layer that keeps the norms of the residual leaving it (positions >= 1)."""
    def __init__(self, inner):
        super().__init__()
        self.inner, self.is_linear, self.norms = inner, inner.is_linear, None

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        self.norms = mx.linalg.norm(out[:, 1:].astype(mx.float32), axis=-1).reshape(-1)
        return out


def windows(tok, seq_len, batch, seed):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    buf, eos = [], tok.eos_token_id
    for text in text_stream(tok, seed):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch:]
            yield np.array(chunk, np.int32).reshape(batch, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="repo, or a size: 0.8b, 2b, 4b, 9b, 27b")
    ap.add_argument("--layers", default="22", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output prefix (default stats/q35-<size>)")
    a = ap.parse_args()
    model = resolve(a.model)
    out = a.out or stats_prefix(model)
    layers = [int(x) for x in a.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(model)
    lm, dec = load_text_model(model)
    stats, taps = {}, {}
    for l in layers:
        dec.layers[l].mlp = stats[l] = StatMLP(dec.layers[l].mlp)
        dec.layers[l] = taps[l] = ResidTap(dec.layers[l])
    resid = {l: [] for l in layers}

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed), n_steps)):
        dec(mx.array(ids))
        # lazy: evaluating the stats runs the model only as deep as the deepest target layer
        mx.eval([[s.pp_max, s.nn_max, s.pn_min, s.np_min, taps[l].norms] for l, s in stats.items()])
        for l in layers:
            resid[l].append(np.array(taps[l].norms))
        if step % 10 == 0 or step == n_steps - 1:
            done = (step + 1) * a.seq_len * a.batch
            print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for l, s in stats.items():
        pp, nn_, pn, np_ = (np.array(x) for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn_, -pn, -np_])
        w_down = np.array(weight(s.inner.down_proj).astype(mx.float32)).T.astype(np.float16)
        resid_q = np.quantile(np.concatenate(resid[l]), [0.1, 0.5, 0.9])
        path = f"{out}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn_, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, model=model, layer=l, tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {(nn_ > pp).mean():.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
