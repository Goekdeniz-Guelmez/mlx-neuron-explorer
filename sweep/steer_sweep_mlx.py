"""steer_sweep.py on MLX (Apple Silicon): the same rollouts, designs, random-direction controls and jsonl records, run
with mlx-lm instead of the vLLM steering fork.

A push adds scale * vector to the steered layer's MLP output at every position (prompt and generated), the same
`mlp_out` hook the fork uses. Rollouts of one prompt run together in a batch, each row with its own push, decoding
greedily with a shared cache; every row keeps its chosen token's logprob and the top-k alternatives at each step.

    uv run python sweep/neuron_stats_mlx.py --model 2b --layers 12 --tokens 200000
    uv run python sweep/steer_sweep_mlx.py --model 2b --stats stats/q35-2b_L12.npz --design main_rel --dose-unit rel \\
        --neurons random:256 --random-controls 64 --run q35-2b-L12

Writes <runs>/<run>.jsonl and .meta.json (runs/ for the 4B, runs/<size>/ otherwise, as the explorer reads them).
Rerunning the same command resumes. No activation captures: sweep/capture_prefill.py replays prefill captures later.
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from mlx_util import load_text_model, resolve, runs_dir
from steer_sweep import DESIGNS, PROMPTS, parse_neurons


class PushMLP(nn.Module):
    """The steered layer's MLP, adding each batch row's own push to its output while `vec` is set ([B, hidden])."""
    def __init__(self, inner):
        super().__init__()
        self.inner, self.vec = inner, None

    def __call__(self, x):
        out = self.inner(x)
        return out if self.vec is None else out + self.vec[:, None, :].astype(out.dtype)


def run_batch(lm, push, prompt, vecs, max_tokens, top_k, eos):
    """Greedy rollouts of one prompt, one per row of vecs [B, hidden] (zeros = unsteered). Returns per row:
    (token ids, chosen logprobs, top-k ids, top-k logprobs, finish reason)."""
    B = vecs.shape[0]
    cache = lm.make_cache()
    push.vec = mx.array(vecs)
    ids = [[] for _ in range(B)]
    lps, tids, tlps = ([[] for _ in range(B)] for _ in range(3))
    done = np.zeros(B, bool)
    try:
        logits = lm(mx.array([prompt] * B), cache=cache)[:, -1]
        for _ in range(max_tokens):
            lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
            top = mx.argpartition(-lp, top_k, axis=-1)[:, :top_k]
            top_v = mx.take_along_axis(lp, top, -1)
            order = mx.argsort(-top_v, -1)
            top, top_v = mx.take_along_axis(top, order, -1), mx.take_along_axis(top_v, order, -1)
            nxt = top[:, 0]
            mx.eval(nxt, top, top_v)
            nxt_np, top_np, tv_np = np.array(nxt), np.array(top), np.array(top_v)
            for r in range(B):
                if done[r]:
                    continue
                t = int(nxt_np[r])
                ids[r].append(t); lps[r].append(round(float(tv_np[r, 0]), 4))
                tids[r].append(top_np[r].tolist()); tlps[r].append([round(float(v), 4) for v in tv_np[r]])
                done[r] = t == eos
            if done.all():
                break
            logits = lm(nxt[:, None], cache=cache)[:, -1]
    finally:
        push.vec = None
    return [(ids[r], lps[r], tids[r], tlps[r], "stop" if done[r] else "length") for r in range(B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="repo, or a size: 0.8b, 2b, 4b, 9b, 27b")
    ap.add_argument("--stats", required=True, help="npz from neuron_stats(_mlx).py for the layer to steer")
    ap.add_argument("--neurons", default="all", help='"all", "random:128", "2742", "0-99", or "1,5,9-12"')
    ap.add_argument("--design", default="main", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="absmax", choices=["absmax", "rel"], help="absmax: dose x the neuron's "
                    "abs_max; rel: dose x the median residual norm, the same push size for every neuron")
    ap.add_argument("--run", required=True, help="run name: output file")
    ap.add_argument("--out-dir", default=None, help="default runs/ (4B) or runs/<size>/")
    ap.add_argument("--max-tokens", type=int, default=256, help="steered rollouts often loop; this bounds them")
    ap.add_argument("--top-k", type=int, default=5, help="alternative tokens kept per generated step")
    ap.add_argument("--random-controls", type=int, default=256, help="random-direction controls spread through the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32, help="rollouts of one prompt decoded together")
    a = ap.parse_args()
    model = resolve(a.model)
    out_dir = a.out_dir or runs_dir(model)

    st = np.load(a.stats)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    neurons = parse_neurons(a.neurons, len(abs_max), a.seed)
    design = DESIGNS[a.design]
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    down_norm = np.linalg.norm(w_down.astype(np.float32), axis=1)
    assert not getattr(AutoConfig.from_pretrained(model).get_text_config(), "layer_scale", False), \
        "mlp_out sits after an FFN layer scale on this model, so adding W_down[:, n] there is no pure push on neuron n"

    tok = AutoTokenizer.from_pretrained(model)
    prompt_ids = {k: tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=True) for k, v in PROMPTS.items()}
    prompt_ids = {k: list(v["input_ids"] if hasattr(v, "keys") else v) for k, v in prompt_ids.items()}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.run}.jsonl")
    with open(os.path.join(out_dir, f"{a.run}.meta.json"), "w") as f:
        json.dump({"model": model, "layer": layer, "stats": os.path.abspath(a.stats), "prompts": PROMPTS,
                   "design_name": a.design, "design": design, "resid_norm_p50": resid_p50, "dose_unit": a.dose_unit,
                   "hook": "mlp_out", "backend": "mlx", "max_tokens": a.max_tokens, "top_k": a.top_k,
                   "capture_layers": [], "prompt_ids": prompt_ids}, f, indent=1)

    # random-direction controls, drawn exactly as steer_sweep.py draws them, so a seed gives the same controls
    rng = np.random.default_rng(a.seed)
    ctrl_ref = rng.integers(len(abs_max), size=a.random_controls)
    ctrl_vecs = rng.standard_normal((a.random_controls, w_down.shape[1])).astype(np.float32)
    ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref].astype(np.float32), axis=1)
                  / np.linalg.norm(ctrl_vecs, axis=1))[:, None]

    units = [("neuron", n) for n in neurons]
    for k in range(a.random_controls):
        units.insert(round((k + 1) * len(units) / (a.random_controls + 1)), ("random", k))

    key = lambda j: (j["kind"], j["neuron"], j["prompt"], j["dose"])
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {key(json.loads(line)) for line in f}
    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"L{layer}: {len(neurons)} neurons + {a.random_controls} random controls, {len(jobs)} rollouts to run "
          f"({len(done)} already done)", flush=True)
    if not jobs:
        return

    for j in jobs:
        kind, n, p, d = j["kind"], j["neuron"], j["prompt"], j["dose"]
        ref = n if kind == "neuron" else int(ctrl_ref[n]) if kind == "random" else None
        j["ref_neuron"] = ref
        if ref is None:
            j["add"] = 0.0
        elif a.dose_unit == "rel":
            j["add"] = float(d * resid_p50 / down_norm[ref])
        else:
            j["add"] = float(d * abs_max[ref])
        j["rel_push"] = abs(j["add"]) * float(down_norm[ref]) / resid_p50 if ref is not None else 0.0
        j["capture_id"] = f"L{layer}_{kind}{n}_{p}_{d:+g}".replace("+", "p").replace("-", "m")

    def vector(j):
        if j["ref_neuron"] is None:
            return np.zeros(w_down.shape[1], np.float32)
        v = w_down[j["neuron"]] if j["kind"] == "neuron" else ctrl_vecs[j["neuron"]]
        return v.astype(np.float32) * j["add"]

    lm, dec = load_text_model(model)
    push = dec.layers[layer].mlp = PushMLP(dec.layers[layer].mlp)
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    # batches of one prompt each (same length, so no padding), in job order within each prompt
    batches = []
    for p in PROMPTS:
        mine = [j for j in jobs if j["prompt"] == p]
        batches += [mine[i:i + a.batch] for i in range(0, len(mine), a.batch)]
    order = {id(j): i for i, j in enumerate(jobs)}
    batches.sort(key=lambda b: order[id(b[0])])  # roughly the run's order, so a stopped run has every prompt's share

    t_all, n_all = time.time(), 0
    with open(out_path, "a") as f:
        for bi, chunk in enumerate(batches):
            t0 = time.time()
            outs = run_batch(lm, push, prompt_ids[chunk[0]["prompt"]], np.stack([vector(j) for j in chunk]),
                             a.max_tokens, a.top_k, eos)
            n_tok = 0
            for j, (ids, lp, tid, tlp, finish) in zip(chunk, outs):
                text = tok.decode([t for t in ids if t != eos], skip_special_tokens=True)
                f.write(json.dumps({
                    "layer": layer, "kind": j["kind"], "neuron": j["neuron"], "ref_neuron": j["ref_neuron"],
                    "prompt": j["prompt"], "dose": j["dose"], "add": j["add"], "rel_push": j["rel_push"],
                    "text": text, "finish": finish, "n_gen": len(ids), "token_ids": ids,
                    "logprob": lp, "top_ids": tid, "top_lps": tlp, "capture_id": j["capture_id"]}) + "\n")
                n_tok += len(ids)
            f.flush()
            n_all += len(chunk)
            dt = time.time() - t0
            print(f"{n_all}/{len(jobs)} rollouts, {n_tok / dt:,.0f} gen tok/s, {len(chunk) / dt:.1f} rollouts/s, "
                  f"{(len(jobs) - n_all) * (time.time() - t_all) / n_all / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
