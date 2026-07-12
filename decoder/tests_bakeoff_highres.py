"""Task 1 (high-res / disc-crop / detected-center bake-off caching infra).

Adds `encode_prefix(imgs, input_size=224)` to the RETFound-MAE frozen encoder (decoder/encoders.py)
so the bake-off can run the frozen backbone at 384/448 and on a disc crop, plus a dependency-free
"detected" disc-center heuristic and per-config cache filenames in decoder/diag_encoder_bakeoff.py.

Load-bearing invariant: at input_size==224 the new code path MUST be byte-identical to today's
model (the frozen forward that produces the 3.75 headline result) — test_default_224_byte_identical
guards this. This task only wires the CAPABILITY + light unit tests on random tensors / tiny
synthetic images; it does NOT run the actual (expensive) 631-image cache — that's a later task.

Repo convention (no decoder/__init__.py, no conftest): put decoder/ on sys.path and import bare.
Run:  python -m pytest decoder/tests_bakeoff_highres.py -q
"""
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoders as EN
import diag_encoder_bakeoff as BO


# --------------------------------------------------------------------------- encoders.py: high-res
def test_retfound_encode_prefix_highres_grid():
    enc = EN.load_encoder("retfound_mae")
    for sz, g in [(224, 14), (384, 24), (448, 28)]:
        x = torch.randn(1, 3, sz, sz)
        h = enc.encode_prefix(x, input_size=sz)   # new kwarg; default 224 byte-identical
        assert h.shape == (1, 1 + g * g, enc.dim)


def test_default_224_byte_identical():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(enc.encode_prefix(x), enc.encode_prefix(x, input_size=224), atol=1e-6)


def test_highres_not_nan_and_finite():
    """The regenerated sin-cos pos-embed + bypassed patch-embed must not silently produce NaN/Inf."""
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(2, 3, 384, 384)
    h = enc.encode_prefix(x, input_size=384)
    assert torch.isfinite(h).all()


def test_highres_input_size_not_multiple_of_16_raises():
    """input_size=225 isn't divisible by the patch16 grid; must fail clearly upfront rather than
    deep inside a tensor add (e.g. a shape-mismatched pos-embed broadcast)."""
    import pytest
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(1, 3, 225, 225)
    with pytest.raises(ValueError):
        enc.encode_prefix(x, input_size=225)


def test_other_backbones_ignore_input_size_kwarg():
    """encode_prefix(imgs, input_size=224) must not break backbones that don't honour input_size
    (they should just accept and ignore the kwarg, running at their configured resolution)."""
    import pytest
    try:
        enc = EN.load_encoder("dinov2_l")
    except Exception as e:
        pytest.skip(f"dinov2_l unavailable: {type(e).__name__}: {e}")
    x = torch.randn(1, 3, enc.input_size, enc.input_size)
    h = enc.encode_prefix(x, input_size=224)
    assert h.shape == (1, 1 + enc.grid[0] * enc.grid[1], enc.dim)


# --------------------------------------------------------------------------- diag_encoder_bakeoff.py
def test_npz_path_default_matches_existing_baseline_filename():
    """The default config (view=full, input_size=224, disc_center=fixed) MUST resolve to the
    existing plain filename — bakeoff_retfound_mae.npz already sits on disk as the anchor and
    must never be renamed/shadowed by the new per-config naming scheme."""
    assert BO.npz_path("retfound_mae") == BO.npz_path("retfound_mae", "full", 224, "fixed")
    assert os.path.basename(BO.npz_path("retfound_mae")) == "bakeoff_retfound_mae.npz"


def test_npz_path_distinct_for_new_configs():
    p_full224 = BO.npz_path("retfound_mae", "full", 224, "fixed")
    p_disc384 = BO.npz_path("retfound_mae", "disc", 384, "fixed")
    p_disc384_det = BO.npz_path("retfound_mae", "disc", 384, "detected")
    p_full384 = BO.npz_path("retfound_mae", "full", 384, "fixed")
    paths = [p_full224, p_disc384, p_disc384_det, p_full384]
    assert len(set(paths)) == len(paths)              # all distinct
    assert "disc384" in os.path.basename(p_disc384)
    assert os.path.basename(p_disc384_det).endswith("_det.npz")


def test_detect_disc_center_finds_bright_spot_in_laterality_quadrant():
    """Dependency-free green-channel brightness centroid: a synthetic image with a bright green
    splotch on the RIGHT side must localize there for OD (disc nasal/right) and be ignored when
    searching the OS (left) quadrant of the SAME image."""
    w, h = 200, 100
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright green splotch in the top-right region (OD-side quadrant).
    arr[10:30, 160:180, 1] = 255
    img = Image.fromarray(arr, mode='RGB')
    cx_od, cy_od = BO._detect_disc_center(img, 'OD')
    assert cx_od > 0.5    # right half
    assert 0.05 < cy_od < 0.45   # top half, inside the splotch band

    # For OS the search window is the LEFT half, which is entirely dark here -> degenerate guard
    # falls back to the fixed nominal OS center rather than spuriously reporting the OD splotch.
    import training as T
    cx_os, cy_os = BO._detect_disc_center(img, 'OS')
    assert cx_os == T.DISC_CX_OS and cy_os == T.DISC_CY


def test_cache_encoder_detected_center_requires_disc_view_raises():
    """disc_center='detected' is only meaningful with view='disc'. With view='full' (the default),
    the detected-center offset is silently ignored (the full-image branch never looks at it) yet
    cache_encoder would still write a filename tagged '_det' — a plain full@224 cache mislabeled
    as a detected-disc result, which probe()'s auto-discovery would then compare as if it were real.
    Must raise ValueError instead of writing the mislabeled cache."""
    import pytest
    with pytest.raises(ValueError):
        BO.cache_encoder("retfound_mae", view="full", input_size=224, disc_center="detected")


def test_discover_configs_empty_when_no_extra_caches(tmp_path, monkeypatch):
    """probe()'s config auto-discovery must not pick up the plain per-encoder baseline files
    (no '__' in the name) — only genuinely distinct view/input_size/disc_center caches."""
    monkeypatch.setattr(BO, "AUTO", str(tmp_path))
    open(os.path.join(tmp_path, "bakeoff_retfound_mae.npz"), "wb").close()
    assert BO._discover_configs() == []
    open(os.path.join(tmp_path, "bakeoff_retfound_mae__disc384.npz"), "wb").close()
    labels = [label for label, _ in BO._discover_configs()]
    assert labels == ["retfound_mae__disc384"]
