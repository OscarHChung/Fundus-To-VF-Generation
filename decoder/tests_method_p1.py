"""P1 — disc-ROI as the SOLE encoder input (not an extra averaged view).

Design ref: decoder/specs/fundus_only_ceiling_design.md §4 P1. The existing --disc-crop adds the
disc as a second view fused by prediction-averaging (the iter-11 approach that failed). P1 instead
REPLACES the input with a laterality-aware disc/ROI crop, so the frozen encoder sees high-res
peripapillary detail and the anatomical machinery is derived once, for one view.

These tests are ENCODER-FREE (dataset + crop geometry only) so they run on the memory-tight box.
Default OFF (disc_only=False) must be byte-identical to the current behavior.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T

CV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")
VAL_JSON = os.path.join(CV, "fold0_val.json")


def _ds(disc_only=False, disc_crop=False, mode='train'):
    return T.MultiImageDataset(VAL_JSON, T.FUNDUS_DIR, T.val_transform, mode=mode,
                               disc_crop=disc_crop, disc_only=disc_only)


def test_off_is_full_view_unchanged():
    """disc_only=False (default) → every train sample is the 'full' view (current behavior)."""
    ds = _ds(disc_only=False)
    assert len(ds.samples) > 0
    assert all(s['view'] == 'full' for s in ds.samples), \
        "default must be full-view only (byte-identical to baseline)"


def test_disc_only_makes_disc_the_sole_view():
    """disc_only=True → every train sample is the 'disc' view; NO full view remains."""
    ds = _ds(disc_only=True)
    assert len(ds.samples) > 0
    views = {s['view'] for s in ds.samples}
    assert views == {'disc'}, f"disc_only must yield ONLY disc views, got {views}"
    # same number of samples as full (one view per image), NOT doubled like --disc-crop
    assert len(ds.samples) == len(_ds(disc_only=False).samples)


def test_disc_only_overrides_disc_crop():
    """disc_only=True wins even if disc_crop=True (sole input, not the averaged 2-view mode)."""
    ds = _ds(disc_only=True, disc_crop=True)
    assert {s['view'] for s in ds.samples} == {'disc'}


def test_disc_crop_tensor_differs_from_full():
    """The cropped input actually differs from the full image (crop is applied)."""
    full = _ds(disc_only=False)[0][0]
    disc = _ds(disc_only=True)[0][0]
    assert full.shape == disc.shape == (3, 224, 224)
    assert not torch.allclose(full, disc), "disc crop must change the pixels vs the full image"


def test_disc_half_controls_crop_size():
    """A larger DISC_HALF captures more of the image (disc+macula ROI vs tight disc)."""
    from PIL import Image
    rec = _ds()[0]  # trigger dataset build; grab a real image path
    img_path = _ds().samples[0]['image']
    lat = _ds().samples[0]['laterality']
    img = Image.open(os.path.join(T.FUNDUS_DIR, img_path)).convert('RGB')
    old = T.DISC_HALF
    try:
        T.DISC_HALF = 0.27
        tight = T.disc_crop_pil(img, lat)
        T.DISC_HALF = 0.45
        wide = T.disc_crop_pil(img, lat)
    finally:
        T.DISC_HALF = old
    assert wide.size[0] > tight.size[0] and wide.size[1] > tight.size[1], \
        "larger DISC_HALF must yield a larger crop box"


def test_val_mode_disc_only_single_view():
    """In val mode, disc_only yields exactly one (disc) view per image — no full-view fusion."""
    ds = _ds(disc_only=True, mode='val')
    s = ds.samples[0]
    assert all(v == 'disc' for _, v in s['image_views']), "val disc_only must use disc views only"
    n_imgs = len(s['images'])
    assert len(s['image_views']) == n_imgs, "one disc view per image (no full+disc doubling)"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
