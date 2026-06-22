"""
Garway–Heath sector-weighted loss for the fundus → VF model.
=============================================================

Opt-in spatial (sector) loss weighting that penalises poorly-predicted
peripheral / arcuate VF regions more than the well-behaved central field, to
attack the known severe/peripheral *underestimation* problem in fundus→VF
models. This is a NEW weight, distinct from the existing value-based (severity)
weight in `training.compute_loss`; the two combine multiplicatively and the
interaction is configurable (`both` / `sector_only` / `value_only`).

Design notes (full write-up: decoder/specs/garway_heath_weighting.md):

  * 52 valid points live in an 8×9 = 72 grid masked by `mask_OD`. The model
    outputs a length-52 vector in `valid_indices_od` (query) order; OS targets
    are gathered with `valid_indices_os` derived from `fliplr(mask_OD)` — the
    SAME convention as decoder/training.py (NOT the `reversed()` convention
    used by predict_vf_from_fundus.py / visualize_MAE_heatmap.py).
  * `SECTOR_GRID` is the single source of truth, defined ONCE on the 8×9 OD
    grid. OS sectors are derived by mirroring (`fliplr`) so a query index maps
    to the anatomically-correct sector for each eye. Because the blind-spot
    column is asymmetric, the per-query sector vectors differ between eyes even
    though the *displayed* sector maps mirror exactly (see selftest).
  * Target is raw threshold sensitivity in dB (clamped 0–35), not total
    deviation — consistent with the rest of the codebase.

This module deliberately keeps the lightweight sector logic (constants,
`sector_weight_vector`, `sector_id_vector`, `per_sector_mae`, `selftest`) free
of any `import training`, so training.py can import those without a circular
import. The heavy evaluation / figure helpers lazy-import training.py,
generate_scatterplot.py and visualize_MAE_heatmap.py *inside* the functions and
reuse their model class, metric helpers and plotting code.
"""

import os
import sys
import csv
import json
import math
import argparse
from collections import OrderedDict

import numpy as np
import torch

# ==============================================================
# PATHS
# ==============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR  = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
VAL_JSON    = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
TRAIN_JSON  = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")

RESULTS_DIR    = os.path.join(CURRENT_DIR, "results", "garway_heath")
BASELINE_MODEL = os.path.join(CURRENT_DIR, "inference_model.pth")           # baseline (value-only) checkpoint
GH_MODEL       = os.path.join(RESULTS_DIR, "inference_model_gh.pth")        # sector-weighted checkpoint

MASKED_VALUE_THRESHOLD = 99.0

# ==============================================================
# VALID-POINT GEOMETRY  (mirror of decoder/training.py)
# ==============================================================
mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]
], dtype=bool)

valid_indices_od = [i for i, v in enumerate(mask_OD.flatten()) if v]
mask_OS          = np.fliplr(mask_OD)
valid_indices_os = [i for i, v in enumerate(mask_OS.flatten()) if v]   # training.py convention (fliplr)
NUM_VALID_POINTS = len(valid_indices_od)                              # 52

# ==============================================================
# SECTOR MAP  —  EDITABLE SOURCE OF TRUTH (8 sectors, draft)
# --------------------------------------------------------------
# Defined ONCE on the 8×9 OD (right-eye) grid. Orientation:
#   rows 0-3 = SUPERIOR field, rows 4-7 = INFERIOR field
#   low col  = NASAL,  col 4 = centre,  high col = TEMPORAL
#   (blind spot is temporal: masked at col 7, rows 3 & 4)
#
# Sector ids → names (see SECTOR_NAMES). -1 marks the 20 masked grid cells.
# This is a Garway–Heath-*style* 8-section variant adapted to this grid; it is a
# DRAFT meant to be hand-edited. Just change the ids below to re-sector — the
# rest of the module (weights, OS mirroring, metrics, figures) follows
# automatically. `selftest()` re-verifies coverage / laterality after edits.
# ==============================================================
_C = -1
SECTOR_GRID = np.array([
    [_C, _C, _C,  2,  2,  3,  3, _C, _C],
    [_C, _C,  2,  2,  3,  3,  3,  7, _C],
    [_C,  6,  2,  2,  0,  3,  3,  7,  7],
    [ 6,  6,  2,  0,  0,  0,  3, _C,  7],
    [ 6,  6,  4,  1,  1,  1,  5, _C,  7],
    [_C,  6,  4,  4,  1,  5,  5,  7,  7],
    [_C, _C,  4,  4,  5,  5,  5,  7, _C],
    [_C, _C, _C,  4,  4,  5,  5, _C, _C],
], dtype=int)

N_SECTORS = 8
SECTOR_NAMES = OrderedDict([
    (0, "Central-Superior"),
    (1, "Central-Inferior"),
    (2, "Superonasal"),
    (3, "Superior-arcuate"),
    (4, "Inferonasal"),
    (5, "Inferior-arcuate"),
    (6, "Nasal-periphery"),
    (7, "Temporal-wedge"),
])

# ── Per-sector RAW weights (EDITABLE) ──────────────────────────
# Upweight the peripheral / arcuate / nasal-step zones (where severe loss and
# model underestimation concentrate); downweight the well-predicted centre.
# These are normalised to mean 1.0 over the 52 valid points before use, so the
# loss and the weighted-MAE stay on the dB scale (see sector_weight_vector).
SECTOR_WEIGHTS = {
    0: 0.75,   # Central-Superior   (spared until late; downweight)
    1: 0.75,   # Central-Inferior
    2: 1.10,   # Superonasal
    3: 1.55,   # Superior-arcuate   (Bjerrum zone; lagged most in run-1 → biggest bump)
    4: 1.15,   # Inferonasal
    5: 1.50,   # Inferior-arcuate
    6: 1.35,   # Nasal-periphery    (nasal step)
    7: 1.25,   # Temporal-wedge     (far periphery)
}

# ── Deep-floor loss shaping (EDITABLE) ─────────────────────────
# Goal-1 rescue, driven by the first GH run: severe *eyes* and 7/8 sectors
# improved, BUT the very deepest *points* (GT 0-10 dB) got slightly WORSE and the
# global bias drifted positive (+0.73 dB = the model over-predicts = it
# UNDER-deepens scotomata). Sector weights are spatial and cannot target a
# value band, so this adds a *value-axis* rescue, active ONLY under
# --weighting garway_heath:
#   • FLOOR_BOOST  — extra additive loss weight on GT points below FLOOR_DB,
#                    peaking at GT=0 and tapering linearly to 0 at FLOOR_DB. Adds
#                    on top of the existing severity weight (training.WEIGHT_SCALE),
#                    so the deepest points get ~5x weight vs healthy (was ~3x).
#   • OVERPRED_PENALTY — multiplies the per-point Huber by (1+penalty) when the
#                    model OVER-predicts a deep point (pred>gt AND gt<FLOOR_DB).
#                    Asymmetric → directly pushes deep predictions DOWN, attacking
#                    the positive bias / depth underestimation.
# Set FLOOR_BOOST=0 and OVERPRED_PENALTY=0 (or pass --floor-boost 0 etc.) to
# recover the pure sector-weighted run from before.
FLOOR_DB         = 12.0
FLOOR_BOOST      = 2.0
OVERPRED_PENALTY = 0.6


def deep_loss_config(floor_db=None, floor_boost=None, overpred_penalty=None):
    """Resolved deep-floor shaping config consumed by training.compute_loss.

    Returns a plain dict so it crosses the module boundary without importing
    training (keeps the no-circular-import guarantee). Any arg left None falls
    back to the module-level default, so CLI flags can override individual
    fields while leaving the others at their tuned defaults."""
    return {
        "floor_db":         FLOOR_DB         if floor_db         is None else float(floor_db),
        "floor_boost":      FLOOR_BOOST      if floor_boost      is None else float(floor_boost),
        "overpred_penalty": OVERPRED_PENALTY if overpred_penalty is None else float(overpred_penalty),
    }

# Sanity on the constants themselves (cheap; runs at import).
assert SECTOR_GRID.shape == (8, 9), "SECTOR_GRID must be 8×9"
assert np.array_equal(SECTOR_GRID == _C, ~mask_OD), \
    "SECTOR_GRID masked cells must match mask_OD exactly"
assert set(np.unique(SECTOR_GRID[mask_OD]).tolist()) == set(range(N_SECTORS)), \
    "every sector id 0..N_SECTORS-1 must be used, and no others"
assert set(SECTOR_WEIGHTS.keys()) == set(range(N_SECTORS)), \
    "SECTOR_WEIGHTS must define exactly sectors 0..N_SECTORS-1"

# Derived: query-index → sector id, per eye.
#   OD: query k → OD grid cell valid_indices_od[k]      → SECTOR_GRID
#   OS: query k → OS grid cell valid_indices_os[k]      → fliplr(SECTOR_GRID)
# Using fliplr(SECTOR_GRID) with the fliplr-derived valid_indices_os makes the
# sector anatomically consistent (the displayed OD/OS maps mirror — see selftest).
_SECTOR_GRID_OS = np.fliplr(SECTOR_GRID)
SECTOR_ID_OD = np.array([SECTOR_GRID.flat[fi]      for fi in valid_indices_od], dtype=int)
SECTOR_ID_OS = np.array([_SECTOR_GRID_OS.flat[fi]  for fi in valid_indices_os], dtype=int)

# SECTOR_MAP keyed by query index (OD reference order) for convenience / docs.
SECTOR_MAP = OrderedDict((k, int(SECTOR_ID_OD[k])) for k in range(NUM_VALID_POINTS))


# ==============================================================
# CORE HELPERS  (lightweight — no training.py dependency)
# ==============================================================
def sector_id_vector(eye="OD"):
    """Length-52 int array of sector ids in QUERY order for the given eye.

    Aligns element-for-element with the model's pred_52 / target_52 (which use
    valid_indices_od for OD and valid_indices_os (fliplr) for OS)."""
    eye = eye.upper()
    if eye.startswith("OD"):
        return SECTOR_ID_OD.copy()
    if eye.startswith("OS"):
        return SECTOR_ID_OS.copy()
    raise ValueError(f"Unknown laterality: {eye!r}")


def _raw_weight_vector(eye="OD"):
    sid = sector_id_vector(eye)
    return np.array([SECTOR_WEIGHTS[int(s)] for s in sid], dtype=np.float64)


def sector_weight_vector(eye="OD", normalize=True):
    """Length-52 float weight vector in QUERY order for the given eye.

    With `normalize=True` the weights average to 1.0 over the 52 valid points,
    which keeps a sector-weighted MAE on the same dB scale as the plain MAE.
    (Both eyes share the same multiset of sector ids over the 52 points, so the
    normaliser is identical for OD and OS.)"""
    w = _raw_weight_vector(eye)
    if normalize:
        w = w / w.mean()
    return w


def sector_weight_tensors(device=None, normalize=True, dtype=torch.float32):
    """Dict {'OD': tensor[52], 'OS': tensor[52]} of query-order weights.

    This is what training.compute_loss consumes behind the --weighting flag."""
    out = {}
    for eye in ("OD", "OS"):
        t = torch.tensor(sector_weight_vector(eye, normalize=normalize), dtype=dtype)
        if device is not None:
            t = t.to(device)
        out[eye] = t
    return out


def per_sector_mae(pred_pts, true_pts, sector_pts):
    """Per-sector MAE + point count from flat, already-masked valid points.

    pred_pts / true_pts / sector_pts are 1-D arrays of equal length holding,
    for every valid (non-masked) VF point across all samples, the prediction,
    ground truth and sector id respectively. Masking (gt < 99) is expected to
    have been applied by the caller (same rule as training/eval)."""
    pred_pts   = np.asarray(pred_pts, dtype=np.float64)
    true_pts   = np.asarray(true_pts, dtype=np.float64)
    sector_pts = np.asarray(sector_pts, dtype=int)
    abs_err = np.abs(pred_pts - true_pts)
    rows = OrderedDict()
    for sid in range(N_SECTORS):
        m = sector_pts == sid
        n = int(m.sum())
        mae = float(abs_err[m].mean()) if n > 0 else float("nan")
        rows[sid] = {"sector_id": sid, "sector": SECTOR_NAMES[sid],
                     "mae": mae, "n_points": n}
    return rows


def weighted_mae(abs_err, weights):
    """Σ w·|e| / Σ w. Equals the plain MAE when all weights are equal."""
    abs_err = np.asarray(abs_err, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float((abs_err * weights).sum() / weights.sum())


# ── Severity stratification (goal 1: are severe cases better?) ────────────────
# Per-point GT sensitivity bins. Lower sensitivity = deeper (more severe) loss.
# Edges in dB; the lowest bin (<10 dB) is the "severe defect" region the sector
# weighting is meant to rescue. NOTE: TD-vs-sensitivity is a per-point shift, so
# pointwise MAE here is identical whether expressed in sensitivity or TD.
GT_BIN_EDGES  = [0, 10, 20, 30, 36]
DEEP_DB       = 16.0   # points with GT < 16 dB are treated as "severe/deep"
# Per-eye severity by mean GT sensitivity (mirrors categorize_severity in
# predict_vf_from_fundus.py: severe < 16, moderate 16–26, early > 26 dB).
EYE_SEVERE_DB   = 16.0
EYE_MODERATE_DB = 26.0


def _bin_label(lo, hi):
    return f"{lo}-{hi}dB" if hi < 99 else f"{lo}+dB"


def severity_pointwise_mae(pt_true, pt_pred):
    """Pointwise MAE stratified by ground-truth sensitivity bin, plus a
    deep (<DEEP_DB) vs. preserved (>=DEEP_DB) split."""
    pt_true = np.asarray(pt_true, float); pt_pred = np.asarray(pt_pred, float)
    abs_err = np.abs(pt_pred - pt_true)
    rows = OrderedDict()
    for lo, hi in zip(GT_BIN_EDGES[:-1], GT_BIN_EDGES[1:]):
        m = (pt_true >= lo) & (pt_true < hi)
        n = int(m.sum())
        rows[_bin_label(lo, hi)] = {"gt_lo": lo, "gt_hi": hi, "n_points": n,
                                    "mae": float(abs_err[m].mean()) if n else float("nan")}
    deep = pt_true < DEEP_DB
    rows[f"deep(<{int(DEEP_DB)})"]      = {"n_points": int(deep.sum()),
        "mae": float(abs_err[deep].mean()) if deep.any() else float("nan")}
    rows[f"preserved(>={int(DEEP_DB)})"] = {"n_points": int((~deep).sum()),
        "mae": float(abs_err[~deep].mean()) if (~deep).any() else float("nan")}
    return rows


def per_eye_severity_mae(eye_records):
    """Eye-level MAE grouped by per-eye severity (mean GT sensitivity).

    eye_records: list of {'mae':float, 'mean_gt':float, 'n':int}."""
    groups = {"severe(<16)": [], "moderate(16-26)": [], "early(>26)": []}
    for r in eye_records:
        mg = r["mean_gt"]
        if np.isnan(mg):
            continue
        if mg < EYE_SEVERE_DB:
            groups["severe(<16)"].append(r["mae"])
        elif mg <= EYE_MODERATE_DB:
            groups["moderate(16-26)"].append(r["mae"])
        else:
            groups["early(>26)"].append(r["mae"])
    out = OrderedDict()
    for g, vals in groups.items():
        out[g] = {"n_eyes": len(vals),
                  "mae": float(np.mean(vals)) if vals else float("nan")}
    return out


def vector_to_grid_query(vec_52, eye="OD"):
    """Place a length-52 QUERY-ORDER vector into an 8×9 display grid.

    Uses the training.py OS convention (valid_indices_os from fliplr(mask_OD)),
    then flips OS back to left-eye display orientation. This FIXES the
    inconsistency in visualize_MAE_heatmap.py / predict_vf_from_fundus.py, which
    place OS values with reversed(valid_indices_od); see the spec doc."""
    grid = np.full(72, np.nan)
    idx  = valid_indices_od if eye.upper().startswith("OD") else valid_indices_os
    for k, flat_idx in enumerate(idx):
        grid[flat_idx] = vec_52[k]
    grid = grid.reshape(8, 9)
    if eye.upper().startswith("OS"):
        grid = np.fliplr(grid)
    return grid


def resolved_config(extra=None):
    """The full, reproducible config (sector map + weights + meta)."""
    cfg = {
        "sector_grid": SECTOR_GRID.tolist(),
        "n_sectors": N_SECTORS,
        "sector_names": {int(k): v for k, v in SECTOR_NAMES.items()},
        "sector_weights_raw": {int(k): v for k, v in SECTOR_WEIGHTS.items()},
        "sector_weights_normalized_od": sector_weight_vector("OD").tolist(),
        "sector_id_query_od": SECTOR_ID_OD.tolist(),
        "sector_id_query_os": SECTOR_ID_OS.tolist(),
        "deep_loss": deep_loss_config(),
        "os_convention": "fliplr(mask_OD)  (matches training.py)",
        "target": "raw threshold sensitivity (dB), clamped 0-35",
        "masked_value_threshold": MASKED_VALUE_THRESHOLD,
    }
    if extra:
        cfg.update(extra)
    return cfg


def save_resolved_config(path=None, extra=None):
    path = path or os.path.join(RESULTS_DIR, "config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(resolved_config(extra), f, indent=2)
    return path


# ==============================================================
# §5 SANITY CHECKS  (runnable: `python garway_heath_weighting.py --selftest`)
# ==============================================================
def selftest(verbose=True):
    def log(*a):
        if verbose:
            print(*a)

    log("=" * 64)
    log("Garway–Heath weighting — sanity checks (§5)")
    log("=" * 64)

    # 1. Coverage: every valid point → exactly one sector; masked excluded.
    covered = SECTOR_GRID[mask_OD]
    assert covered.size == NUM_VALID_POINTS == 52
    assert np.all(covered >= 0) and np.all(covered < N_SECTORS)
    assert int((SECTOR_GRID[~mask_OD] != _C).sum()) == 0
    sizes = {int(s): int((covered == s).sum()) for s in range(N_SECTORS)}
    assert sum(sizes.values()) == 52
    log(f"[1] coverage OK — 52 points, no unmapped/double-mapped; sizes={sizes}")

    # 2. Uniform-weight invariant: equal sector weights ⇒ weighted MAE == plain.
    rng = np.random.default_rng(0)
    err = rng.uniform(0, 12, size=523)
    uni = np.ones_like(err) * 3.3            # any constant
    assert abs(weighted_mae(err, uni) - err.mean()) < 1e-9
    log(f"[2] uniform-weight invariant OK — weighted MAE == plain MAE "
        f"({weighted_mae(err, uni):.6f} == {err.mean():.6f})")

    # 3. Normalization: normalized weights average to 1 over 52 valid points.
    for eye in ("OD", "OS"):
        w = sector_weight_vector(eye, normalize=True)
        assert w.shape == (52,)
        assert abs(w.mean() - 1.0) < 1e-9, f"{eye} mean={w.mean()}"
    log("[3] normalization OK — OD & OS weight vectors average to 1.000")

    # 4. Laterality round-trip: OD and OS sector maps, placed back via the
    #    query-order vector_to_grid, must land in anatomically mirrored
    #    positions (identical display grid). Guards the OS-convention bug.
    disp_od = vector_to_grid_query(sector_id_vector("OD").astype(float), "OD")
    disp_os = vector_to_grid_query(sector_id_vector("OS").astype(float), "OS")
    assert np.allclose(disp_od, disp_os, equal_nan=True), \
        "OD/OS sector displays are not mirror-consistent"
    # weights mirror too
    wdisp_od = vector_to_grid_query(sector_weight_vector("OD"), "OD")
    wdisp_os = vector_to_grid_query(sector_weight_vector("OS"), "OS")
    assert np.allclose(wdisp_od, wdisp_os, equal_nan=True)
    log("[4] laterality round-trip OK — OD/OS sector & weight maps mirror exactly")

    # 4b. Per-query vectors are EXPECTED to differ between eyes (blind-spot
    #     asymmetry reorders indices); document it so it isn't mistaken for a bug.
    differ = not np.array_equal(sector_id_vector("OD"), sector_id_vector("OS"))
    log(f"[4b] per-query OD vs OS sector vectors differ = {differ} "
        f"(expected True; only the DISPLAY mirrors)")

    # 5. Tensor consistency: torch weights match numpy weights.
    tens = sector_weight_tensors(device=None, normalize=True)
    for eye in ("OD", "OS"):
        assert np.allclose(tens[eye].numpy(), sector_weight_vector(eye)), eye
    log("[5] tensor/numpy weight parity OK")

    # 6. Deep-floor weight shaping: the additive floor weight must be
    #    non-negative, monotonically non-increasing in GT, and exactly 0 at/above
    #    FLOOR_DB. (Mirrors the formula used in training.compute_loss.)
    cfg = deep_loss_config()
    gts = np.array([0.0, 3.0, 6.0, cfg["floor_db"] - 1e-6, cfg["floor_db"], 20.0])
    fw  = cfg["floor_boost"] * np.clip(cfg["floor_db"] - gts, 0, None) / cfg["floor_db"]
    assert abs(fw[0] - cfg["floor_boost"]) < 1e-9          # peak at GT=0
    assert np.all(np.diff(fw) <= 1e-12)                    # non-increasing
    assert fw[-1] == 0.0 and fw[-2] == 0.0                 # zero at/above floor
    log(f"[6] deep-floor weight OK — peak={fw[0]:.2f} at GT=0, "
        f"0 at GT≥{cfg['floor_db']:.0f}; overpred_penalty={cfg['overpred_penalty']:.2f}")

    log("-" * 64)
    log("ALL SANITY CHECKS PASSED ✓")
    log("=" * 64)
    return True


# ==============================================================
# HEAVY: INFERENCE + METRICS + FIGURES  (lazy imports below)
# ==============================================================
def _device():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_perpoint_model(model_path, device):
    """Load the LIVE PerPointVFModel (v10.2) with the given checkpoint.

    Reuses training.py's encoder load + model class (single 1.2 GB encoder load)."""
    import training as T  # noqa: heavy import; loads RETFound encoder once
    model = T.PerPointVFModel(T.base_model)
    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    meta = {"val_mae": ckpt.get("val_mae", ckpt.get("mae")),
            "val_corr": ckpt.get("val_corr", ckpt.get("corr"))}
    return model, meta


def _run_inference(model, json_path, device, use_tta=True):
    """TTA inference over a dataset using training.py's val dataset/transforms.

    Returns stacked tensors (for the existing metric helpers) plus flat,
    masked, per-point arrays (for scatter / per-sector / per-location)."""
    import training as T
    from torch.utils.data import DataLoader

    ds = T.MultiImageDataset(json_path, FUNDUS_DIR, T.val_transform,
                             mode="val", use_tta=use_tta)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=T.val_collate_fn)

    preds, targets, lats = [], [], []
    # flat per-point (valid only)
    pt_pred, pt_true, pt_sid, pt_w = [], [], [], []
    # per-eye records (for per-eye severity stratification)
    eye_records = []
    # per-location accumulators (query order) per eye
    acc = {e: {"sum_ae": np.zeros(52), "sum_gt": np.zeros(52),
               "count": np.zeros(52, int)} for e in ("OD", "OS")}

    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(device)
            pred = model(imgs, laterality=lat, average_multi=True)  # [1,52]
            preds.append(pred.cpu())
            targets.append(hvf.unsqueeze(0) if hvf.dim() == 1 else hvf)
            lats.append(lat)

            eye       = "OD" if str(lat).startswith("OD") else "OS"
            valid_idx = valid_indices_od if eye == "OD" else valid_indices_os
            true52    = np.asarray(hvf).flatten()[valid_idx].astype(float)
            p52       = pred.squeeze(0).cpu().numpy().astype(float)
            sid       = sector_id_vector(eye)
            w         = sector_weight_vector(eye)
            m         = true52 < MASKED_VALUE_THRESHOLD

            pt_pred.extend(p52[m].tolist())
            pt_true.extend(true52[m].tolist())
            pt_sid.extend(sid[m].tolist())
            pt_w.extend(w[m].tolist())

            if m.any():
                eye_records.append({
                    "eye": eye,
                    "mae": float(np.abs(p52[m] - true52[m]).mean()),
                    "mean_gt": float(true52[m].mean()),
                    "n": int(m.sum()),
                })

            for loc in range(52):
                if m[loc]:
                    acc[eye]["sum_ae"][loc] += abs(p52[loc] - true52[loc])
                    acc[eye]["sum_gt"][loc] += true52[loc]
                    acc[eye]["count"][loc]  += 1

    return {
        "preds": preds, "targets": targets, "lats": lats,
        "pt_pred": np.array(pt_pred), "pt_true": np.array(pt_true),
        "pt_sid": np.array(pt_sid, int), "pt_w": np.array(pt_w),
        "eye_records": eye_records, "acc": acc, "n_samples": len(preds),
    }


def _metrics_from_inference(inf):
    """Overall + per-sector metrics, reusing training.py's helper functions."""
    import training as T
    stacked_pred = torch.cat(inf["preds"], dim=0)
    stacked_tgt  = torch.cat(inf["targets"], dim=0)

    corr      = T.pearson_correlation(stacked_pred, stacked_tgt, inf["lats"])
    r2, slope = T.compute_r2_slope(stacked_pred, stacked_tgt, inf["lats"])
    eye_corr  = T.compute_per_eye_metrics(stacked_pred, stacked_tgt, inf["lats"])

    err       = inf["pt_pred"] - inf["pt_true"]
    plain_mae = float(np.abs(err).mean())
    w_mae     = weighted_mae(np.abs(err), inf["pt_w"])
    bias      = float(err.mean())
    rmse      = float(np.sqrt((err ** 2).mean()))

    sectors = per_sector_mae(inf["pt_pred"], inf["pt_true"], inf["pt_sid"])
    sev_point = severity_pointwise_mae(inf["pt_true"], inf["pt_pred"])
    sev_eye   = per_eye_severity_mae(inf.get("eye_records", []))
    return {
        "n_samples": inf["n_samples"], "n_points": int(inf["pt_pred"].size),
        "plain_mae": plain_mae, "weighted_mae": w_mae,
        "bias": bias, "rmse": rmse,
        "pearson_r": float(corr), "r2": float(r2), "slope": float(slope),
        "per_eye_corr": float(eye_corr),
        "per_sector": sectors,
        "severity_pointwise": sev_point,   # goal 1: per-point GT-bin MAE
        "severity_per_eye": sev_eye,       # goal 1: per-eye severity-group MAE
    }


def _save_metrics(metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ("n_samples", "n_points", "plain_mae", "weighted_mae",
                  "bias", "rmse", "pearson_r", "r2", "slope", "per_eye_corr"):
            w.writerow([k, metrics[k]])
        w.writerow([])
        w.writerow(["sector_id", "sector", "mae", "n_points"])
        for sid, row in metrics["per_sector"].items():
            w.writerow([row["sector_id"], row["sector"], row["mae"], row["n_points"]])
        w.writerow([])
        w.writerow(["gt_sensitivity_bin", "mae", "n_points"])
        for label, row in metrics["severity_pointwise"].items():
            w.writerow([label, row["mae"], row["n_points"]])
        w.writerow([])
        w.writerow(["per_eye_severity_group", "mae", "n_eyes"])
        for label, row in metrics["severity_per_eye"].items():
            w.writerow([label, row["mae"], row["n_eyes"]])


def _save_scatter(inf, out_path, n_samples):
    """Figure 4 — identical style to generate_scatterplot.py, fed new preds."""
    import generate_scatterplot as GS
    GS.plot_vf_scatter(inf["pt_true"], inf["pt_pred"], out_path,
                       n_samples=n_samples, title_tag="Garway–Heath weighted")


def _save_heatmap(inf, out_path, split="val", overlay_sectors=True):
    """Figure 3 — bilateral 4-panel, driven by PerPointVFModel (model-class fix).

    Reuses visualize_MAE_heatmap.plot_single for styling; uses the query-order
    convention-A vector_to_grid (the OS-convention fix)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import visualize_MAE_heatmap as VH

    res = {}
    for eye in ("OD", "OS"):
        a = inf["acc"][eye]
        with np.errstate(invalid="ignore"):
            mean_ae = np.where(a["count"] > 0, a["sum_ae"] / a["count"], np.nan)
            mean_gt = np.where(a["count"] > 0, a["sum_gt"] / a["count"], np.nan)
        res[eye] = {"mean_ae": mean_ae, "mean_gt": mean_gt, "count": a["count"]}

    od_mae = vector_to_grid_query(res["OD"]["mean_ae"], "OD")
    os_mae = vector_to_grid_query(res["OS"]["mean_ae"], "OS")
    od_gt  = vector_to_grid_query(res["OD"]["mean_gt"], "OD")
    os_gt  = vector_to_grid_query(res["OS"]["mean_gt"], "OS")

    od_v = np.nanmean(res["OD"]["mean_ae"])
    os_v = np.nanmean(res["OS"]["mean_ae"])
    comb = np.nanmean(np.concatenate([
        res["OD"]["mean_ae"][~np.isnan(res["OD"]["mean_ae"])],
        res["OS"]["mean_ae"][~np.isnan(res["OS"]["mean_ae"])]]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"VF Prediction (Garway–Heath) — MAE & GT Sensitivity  |  {split} set\n"
        f"OD MAE: {od_v:.2f} dB   |   OS MAE: {os_v:.2f} dB   |   Combined: {comb:.2f} dB",
        fontsize=13, fontweight="bold", y=0.98)

    VH.plot_single(axes[0, 0], od_mae, "OD — Avg MAE per Location (Right Eye)",
                   "inferno", 0, 10, "MAE (dB)")
    VH.plot_single(axes[0, 1], os_mae, "OS — Avg MAE per Location (Left Eye)",
                   "inferno", 0, 10, "MAE (dB)")
    VH.plot_single(axes[1, 0], od_gt, "OD — Avg GT Sensitivity (Right Eye)",
                   "inferno", 0, 30, "Sensitivity (dB)")
    VH.plot_single(axes[1, 1], os_gt, "OS — Avg GT Sensitivity (Left Eye)",
                   "inferno", 0, 30, "Sensitivity (dB)")

    if overlay_sectors:
        _overlay_sector_boundaries(axes[0, 0], "OD")
        _overlay_sector_boundaries(axes[0, 1], "OS")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _overlay_sector_boundaries(ax, eye):
    """Draw thin black borders between differing sectors on an 8×9 imshow grid."""
    sid_grid = vector_to_grid_query(sector_id_vector(eye).astype(float), eye)
    for r in range(8):
        for c in range(9):
            if np.isnan(sid_grid[r, c]):
                continue
            s = sid_grid[r, c]
            # right edge
            if c + 1 >= 9 or np.isnan(sid_grid[r, c + 1]) or sid_grid[r, c + 1] != s:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color="k", lw=1.0)
            # bottom edge
            if r + 1 >= 8 or np.isnan(sid_grid[r + 1, c]) or sid_grid[r + 1, c] != s:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color="k", lw=1.0)


def _save_comparison(baseline_m, gh_m, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "garway_heath", "delta(new-base)"])
        for k in ("plain_mae", "weighted_mae", "pearson_r", "r2", "slope",
                  "bias", "rmse", "per_eye_corr"):
            b, g = baseline_m[k], gh_m[k]
            w.writerow([k, f"{b:.4f}", f"{g:.4f}", f"{g - b:+.4f}"])
        w.writerow([])
        w.writerow(["sector_id", "sector", "baseline_mae", "garway_heath_mae",
                    "delta(new-base)", "n_points"])
        for sid in range(N_SECTORS):
            b = baseline_m["per_sector"][sid]["mae"]
            g = gh_m["per_sector"][sid]["mae"]
            w.writerow([sid, SECTOR_NAMES[sid], f"{b:.4f}", f"{g:.4f}",
                        f"{g - b:+.4f}", gh_m["per_sector"][sid]["n_points"]])
        # Severity (goal 1): per-point GT-bin MAE, baseline vs new.
        w.writerow([])
        w.writerow(["gt_sensitivity_bin", "baseline_mae", "garway_heath_mae",
                    "delta(new-base)", "n_points"])
        for label in gh_m["severity_pointwise"]:
            b = baseline_m["severity_pointwise"][label]["mae"]
            g = gh_m["severity_pointwise"][label]["mae"]
            w.writerow([label, f"{b:.4f}", f"{g:.4f}", f"{g - b:+.4f}",
                        gh_m["severity_pointwise"][label]["n_points"]])
        # Severity (goal 1): per-eye severity-group MAE, baseline vs new.
        w.writerow([])
        w.writerow(["per_eye_severity_group", "baseline_mae", "garway_heath_mae",
                    "delta(new-base)", "n_eyes"])
        for label in gh_m["severity_per_eye"]:
            b = baseline_m["severity_per_eye"][label]["mae"]
            g = gh_m["severity_per_eye"][label]["mae"]
            w.writerow([label, f"{b:.4f}", f"{g:.4f}", f"{g - b:+.4f}",
                        gh_m["severity_per_eye"][label]["n_eyes"]])


def evaluate(model_path=GH_MODEL, baseline_path=BASELINE_MODEL,
             json_path=VAL_JSON, out_dir=RESULTS_DIR, use_tta=True,
             split="val"):
    """Full results pipeline: evaluate the (sector-weighted) model + baseline on
    the held-out set, save metrics / scatter / heatmap / comparison / config.

    Regenerate everything with: `python decoder/garway_heath_weighting.py --evaluate`
    """
    os.makedirs(out_dir, exist_ok=True)
    device = _device()
    print(f"Device: {device}  |  TTA: {use_tta}  |  split: {split}")

    # --- The (new) sector-weighted model -------------------------------------
    if not os.path.isfile(model_path):
        print(f"⚠  GH checkpoint not found: {model_path}")
        if os.path.isfile(baseline_path):
            print(f"   Falling back to baseline checkpoint for a PARITY run: {baseline_path}")
            model_path = baseline_path
        else:
            raise FileNotFoundError("No GH or baseline checkpoint available to evaluate.")

    print(f"\n[new] loading model: {model_path}")
    gh_model, gh_meta = _load_perpoint_model(model_path, device)
    gh_inf = _run_inference(gh_model, json_path, device, use_tta=use_tta)
    gh_m   = _metrics_from_inference(gh_inf)
    print(f"[new] plain MAE={gh_m['plain_mae']:.3f} dB | weighted MAE="
          f"{gh_m['weighted_mae']:.3f} dB | r={gh_m['pearson_r']:.3f} | "
          f"R²={gh_m['r2']:.3f} | slope={gh_m['slope']:.3f} | bias={gh_m['bias']:+.3f}")

    _save_metrics(gh_m, out_dir)
    _save_scatter(gh_inf, os.path.join(out_dir, "vf_scatterplot_garway_heath.png"),
                  gh_m["n_samples"])
    _save_heatmap(gh_inf, os.path.join(out_dir, "bilateral_MAE_garway_heath.png"),
                  split=split)
    save_resolved_config(os.path.join(out_dir, "config.json"),
                         extra={"evaluated_model": os.path.abspath(model_path),
                                "checkpoint_meta": gh_meta, "use_tta": use_tta})

    # --- Baseline (for the side-by-side comparison) --------------------------
    baseline_m = None
    if os.path.isfile(baseline_path) and os.path.abspath(baseline_path) != os.path.abspath(model_path):
        print(f"\n[baseline] loading model: {baseline_path}")
        b_model, _ = _load_perpoint_model(baseline_path, device)
        b_inf = _run_inference(b_model, json_path, device, use_tta=use_tta)
        baseline_m = _metrics_from_inference(b_inf)
        print(f"[baseline] plain MAE={baseline_m['plain_mae']:.3f} dB | "
              f"weighted MAE={baseline_m['weighted_mae']:.3f} dB")
        _save_comparison(baseline_m, gh_m,
                         os.path.join(out_dir, "comparison_baseline_vs_garway_heath.csv"))
    else:
        print("\n[baseline] skipped (same as evaluated model or not found) — "
              "no comparison CSV written.")

    _print_summary(baseline_m, gh_m)
    print(f"\nAll outputs → {out_dir}")
    return {"garway_heath": gh_m, "baseline": baseline_m}


def _print_summary(baseline_m, gh_m):
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    if baseline_m is not None:
        print(f"  baseline plain MAE : {baseline_m['plain_mae']:.3f} dB")
    print(f"  new      plain MAE : {gh_m['plain_mae']:.3f} dB")
    print(f"  new   weighted MAE : {gh_m['weighted_mae']:.3f} dB")
    print(f"\n  Per-sector MAE (baseline → new):")
    print(f"  {'sector':<18}{'base':>8}{'new':>8}{'Δ':>9}{'N':>6}")
    print(f"  {'-'*48}")
    for sid in range(N_SECTORS):
        g = gh_m["per_sector"][sid]
        if baseline_m is not None:
            b = baseline_m["per_sector"][sid]["mae"]
            print(f"  {SECTOR_NAMES[sid]:<18}{b:>8.3f}{g['mae']:>8.3f}"
                  f"{g['mae'] - b:>+9.3f}{g['n_points']:>6d}")
        else:
            print(f"  {SECTOR_NAMES[sid]:<18}{'—':>8}{g['mae']:>8.3f}"
                  f"{'—':>9}{g['n_points']:>6d}")

    # Goal 1: severe-case MAE (per-point GT bins + per-eye severity groups).
    print(f"\n  Severity — per-point GT-sensitivity bin MAE (baseline → new):")
    print(f"  {'GT bin':<18}{'base':>8}{'new':>8}{'Δ':>9}{'N':>7}")
    print(f"  {'-'*50}")
    for label, g in gh_m["severity_pointwise"].items():
        if baseline_m is not None:
            b = baseline_m["severity_pointwise"][label]["mae"]
            print(f"  {label:<18}{b:>8.3f}{g['mae']:>8.3f}"
                  f"{g['mae'] - b:>+9.3f}{g['n_points']:>7d}")
        else:
            print(f"  {label:<18}{'—':>8}{g['mae']:>8.3f}{'—':>9}{g['n_points']:>7d}")

    print(f"\n  Severity — per-eye group MAE (baseline → new):")
    print(f"  {'eye group':<18}{'base':>8}{'new':>8}{'Δ':>9}{'eyes':>7}")
    print(f"  {'-'*50}")
    for label, g in gh_m["severity_per_eye"].items():
        if baseline_m is not None:
            b = baseline_m["severity_per_eye"][label]["mae"]
            print(f"  {label:<18}{b:>8.3f}{g['mae']:>8.3f}"
                  f"{g['mae'] - b:>+9.3f}{g['n_eyes']:>7d}")
        else:
            print(f"  {label:<18}{'—':>8}{g['mae']:>8.3f}{'—':>9}{g['n_eyes']:>7d}")


# ==============================================================
# CLI
# ==============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Garway–Heath sector-weighted loss: sanity checks + evaluation/figures")
    ap.add_argument("--selftest", action="store_true",
                    help="Run §5 sanity checks and exit (no model needed).")
    ap.add_argument("--evaluate", action="store_true",
                    help="Evaluate the sector-weighted model (+ baseline) and save all results.")
    ap.add_argument("--model", default=GH_MODEL,
                    help="Sector-weighted checkpoint to evaluate.")
    ap.add_argument("--baseline", default=BASELINE_MODEL,
                    help="Baseline (value-only) checkpoint for the comparison.")
    ap.add_argument("--json", default=VAL_JSON, help="Eval dataset JSON.")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.evaluate:
        selftest()
        if not args.evaluate:
            return

    evaluate(model_path=args.model, baseline_path=args.baseline,
             json_path=args.json, out_dir=args.out_dir, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()
