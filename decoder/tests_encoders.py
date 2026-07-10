"""Task A1 — swappable frozen-encoder loader (decoder/encoders.py).

The RETFound-MAE path MUST reproduce training._encode exactly: the frozen ViT-L/16 forward
with NO MAE random patch shuffle and enc.norm applied, so a re-gridded decoder is byte-identical
to today's model. Other backbones (DINOv2 / DINOv3 / RETFound-DINOv2 / VisionFM) are wired behind
skipif guards (weights are downloaded on demand and may be absent on this box).

Repo convention (no decoder/__init__.py, no conftest): put decoder/ on sys.path and import bare.
Run:  python -m pytest decoder/tests_encoders.py -q
"""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoders as EN


def test_retfound_mae_default_shapes():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(2, 3, enc.input_size, enc.input_size)
    h = enc.encode_prefix(x)
    assert h.shape == (2, 1 + enc.grid[0] * enc.grid[1], enc.dim)
    assert enc.grid == (14, 14) and enc.dim == 1024      # ViT-L/16 @224


def test_encode_prefix_is_deterministic_and_nograd():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(1, 3, enc.input_size, enc.input_size)
    a = enc.encode_prefix(x); b = enc.encode_prefix(x)
    assert torch.allclose(a, b, atol=1e-5)               # eval mode, no MAE shuffle
    assert not a.requires_grad


def test_retfound_mae_matches_training_encode():
    """default-identity guard: encoders' retfound_mae == training.PerPointVFModel._encode (frozen).

    This is the load-bearing invariant for A3: swapping the backbone through encoders.py must leave
    the current model bit-for-bit unchanged when --encoder retfound_mae.
    """
    import training as T
    enc = EN.load_encoder("retfound_mae")
    model = T.PerPointVFModel(T.base_model, copy_encoder=True).eval()
    x = torch.randn(2, 3, enc.input_size, enc.input_size)
    with torch.no_grad():
        ref = model._encode(x)          # frozen forward + enc.norm, no shuffle
        got = enc.encode_prefix(x)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5)
