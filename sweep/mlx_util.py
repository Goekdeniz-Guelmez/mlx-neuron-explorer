"""Shared pieces of the MLX sweep scripts (Apple Silicon): loading a Qwen3.5 model with mlx-lm, reading weights
(dequantized when the model is quantized), and the default output names per model size, which match what the explorer
reads (stats/q35-<size>_L<layer>.npz; runs/ for the 4B, runs/<size>/ for any other size)."""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "explorer"))
from runtime import SIZES, model_tag  # noqa: E402


def resolve(name):
    """A short size name (2b) or a repo, as the explorer takes it."""
    return SIZES.get(name.lower(), name)


def stats_prefix(model):
    return os.path.join("stats", f"q35-{model_tag(model)}")


def runs_dir(model):
    tag = model_tag(model)
    return "runs" if tag == "4b" else os.path.join("runs", tag)


def load_text_model(name):
    """(the mlx-lm text model, which returns logits; its decoder, which returns the final normed hidden state)."""
    m, _ = load(resolve(name))
    lm = m.language_model if hasattr(m, "language_model") else m
    if not hasattr(lm.model.layers[0].mlp, "down_proj"):
        raise SystemExit(f"{name} is a mixture-of-experts model: the sweep needs dense MLP neurons")
    return lm, lm.model


def weight(lin):
    if isinstance(lin, nn.QuantizedLinear):
        return mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
    return lin.weight


def cache_dir(model):
    """The explorer cache this size reads: explorer/cache for the 4B, explorer/cache/<size> otherwise."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "explorer", "cache")
    tag = model_tag(model)
    return root if tag == "4b" else os.path.join(root, tag)
