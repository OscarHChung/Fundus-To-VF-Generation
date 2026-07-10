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


def test_crop_safe_tta_deg0_is_plain_crop():
    """Dir-1: deg=0 crop-safe TTA == a plain disc crop → the no-TTA disc path is byte-identical."""
    from PIL import Image
    img = Image.new('RGB', (300, 300), (70, 90, 110))
    a = np.asarray(T.rotate_then_disc_crop(img, 'OD', 0))
    b = np.asarray(T.disc_crop_pil(img, 'OD'))
    assert a.shape == b.shape and np.array_equal(a, b), "deg=0 must equal a plain disc crop"


def test_crop_safe_tta_avoids_black_border():
    """Dir-1: rotating the FULL image THEN disc-cropping injects far less black border than the old
    crop-then-rotate path (the +1.4 dB TTA bias). Background is mid-gray so black pixels can only
    come from the rotation fill, not the image content."""
    from PIL import Image, ImageDraw
    import torchvision.transforms.functional as TF
    W = 400
    img = Image.new('RGB', (W, W), (128, 128, 128))            # gray bg (never black)
    dr = ImageDraw.Draw(img)
    cx, cy, r = int(T.DISC_CX_OD * W), int(T.DISC_CY * W), 40
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))   # bright disc blob

    def black_frac(pil):
        a = np.asarray(pil).reshape(-1, 3)
        return float((a.sum(1) == 0).mean())

    old = T.DISC_HALF
    try:
        T.DISC_HALF = 0.27
        crop_then_rot = TF.rotate(T.disc_crop_pil(img, 'OD'), 5)     # OLD broken path (black corners)
        rot_then_crop = T.rotate_then_disc_crop(img, 'OD', 5)        # NEW crop-safe path
    finally:
        T.DISC_HALF = old

    bf_old, bf_new = black_frac(crop_then_rot), black_frac(rot_then_crop)
    assert bf_new < bf_old, f"crop-safe TTA should reduce black border ({bf_new:.3f} !< {bf_old:.3f})"
    assert bf_new < 0.06, f"crop-safe TTA should inject <6% black border, got {bf_new:.3f}"


def test_disc_jitter_default_is_byte_identical():
    """B2: disc_crop_pil defaults (cx_off=cy_off=0, half=None) reproduce the fixed champion crop."""
    from PIL import Image
    img = Image.new('RGB', (321, 289), (60, 80, 100))
    base = np.asarray(T.disc_crop_pil(img, 'OD'))
    same = np.asarray(T.disc_crop_pil(img, 'OD', 0.0, 0.0, None))
    assert base.shape == same.shape and np.array_equal(base, same)
    # matches the hand-computed box (same edge clamping as disc_crop_pil: OD cx+half=1.05 -> clamp to w)
    w, h = img.size
    exp = img.crop((int(max(0, (T.DISC_CX_OD - T.DISC_HALF) * w)), int(max(0, (T.DISC_CY - T.DISC_HALF) * h)),
                    int(min(w, (T.DISC_CX_OD + T.DISC_HALF) * w)), int(min(h, (T.DISC_CY + T.DISC_HALF) * h))))
    assert np.array_equal(base, np.asarray(exp))


def test_disc_jitter_is_train_only():
    """B2: the jitter magnitude is stored only in train mode; val/eval never jitter (0.0)."""
    tr = T.MultiImageDataset(VAL_JSON, T.FUNDUS_DIR, T.val_transform, mode='train',
                             disc_only=True, disc_jitter=0.15)
    va = T.MultiImageDataset(VAL_JSON, T.FUNDUS_DIR, T.val_transform, mode='val',
                             disc_only=True, disc_jitter=0.15)
    assert tr.disc_jitter == 0.15 and va.disc_jitter == 0.0


def test_disc_jitter_crops_vary_and_stay_valid():
    """B2: jittered crops (scale ±j, shift ±0.2j) are all non-degenerate + interior, and vary."""
    from PIL import Image
    img = Image.new('RGB', (400, 360), (90, 110, 130))
    j = 0.15
    np.random.seed(0)
    boxes = []
    for _ in range(30):
        cx_off = np.random.uniform(-0.2 * j, 0.2 * j)
        cy_off = np.random.uniform(-0.2 * j, 0.2 * j)
        half   = T.DISC_HALF * (1.0 + np.random.uniform(-j, j))
        c = T.disc_crop_pil(img, 'OD', cx_off, cy_off, half)
        assert c.size[0] >= 8 and c.size[1] >= 8              # never degenerate
        assert c.size[0] <= img.size[0] and c.size[1] <= img.size[1]   # interior
        boxes.append(c.size)
    assert len(set(boxes)) > 1, "jitter must actually vary the crop box"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
