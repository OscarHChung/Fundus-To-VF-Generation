"""Task 5 (P-B3 fusion) + Task 6 (P-C1 ordinal) unit tests -- synthetic-data checks only (no torch,
no real bake-off cache required): both probes must run end-to-end on synthetic data and return
finite numbers. Repo convention: no decoder/__init__.py, no conftest -- put decoder/ on sys.path and
import bare.

Run:  python -m pytest decoder/tests_fusion_ordinal.py -q
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_fusion_ordinal_probe as FO


def _synthetic_bakeoff_npz(path, n=100, d_spat=24, d_sev=16, n_folds=5, seed=0, severity_scale=6.0):
    """Write a synthetic bake-off-shaped npz (real keys: sev, spat, md, vf52, lat, pid, fold) with an
    actual eye-specific spatial signal (a per-eye latent correlated with both a `spat` direction and
    a within-eye residual pattern in `vf52`), plus a severity signal linking `sev` to `md` -- so
    spatial_pcorr / ridge / CORAL have real signal to recover, not just noise. That's what makes
    'probe runs and returns a finite, non-degenerate number' a meaningful assertion."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n // 2), 2)[:n]
    lat = np.array(['OD', 'OS'] * (n // 2))[:n]
    fold = np.array([i % n_folds for i in range(n)])
    eye_sig = rng.normal(size=n)                        # per-eye within-eye "shape" signal
    spat_dir = rng.normal(size=d_spat)
    spat = rng.normal(size=(n, d_spat)) * 0.5 + eye_sig[:, None] * spat_dir
    sev_dir = rng.normal(size=d_sev)
    md = severity_scale + severity_scale * 0.6 * rng.normal(size=n) + severity_scale * 0.1 * eye_sig
    md = np.clip(md, 0.5, 29.0)
    sev = rng.normal(size=(n, d_sev)) * 0.5 + (md[:, None] / severity_scale) * sev_dir
    point_dir = rng.normal(size=52)
    vf52 = md[:, None] + eye_sig[:, None] * point_dir[None, :] * 0.3 + rng.normal(size=(n, 52)) * 0.2
    np.savez(path, sev=sev.astype(np.float64), spat=spat.astype(np.float64), md=md.astype(np.float64),
             vf52=vf52.astype(np.float64), lat=lat, pid=pid.astype(np.int64), fold=fold.astype(np.int64))


# --------------------------------------------------------------------------- PROBE A: fusion
def test_fusion_spatial_probe_runs_and_returns_finite(tmp_path):
    p_disc = tmp_path / "disc.npz"
    p_full = tmp_path / "full.npz"
    _synthetic_bakeoff_npz(str(p_disc), seed=0)
    _synthetic_bakeoff_npz(str(p_full), seed=1)      # different spat draw, SAME n/pid/lat/fold scheme
    r = FO.fusion_spatial_probe(str(p_disc), str(p_full), k=8, lam=5.0, boot=50, seed=0)
    assert np.isfinite(r['fused_pcorr'])
    assert np.isfinite(r['disc_alone_pcorr'])
    assert np.isfinite(r['fused_ci'][0]) and np.isfinite(r['fused_ci'][1])
    assert np.isfinite(r['delta_vs_disc_alone'])
    assert r['verdict'] in ("PASS (>=0.35) -> build fusion head (Task 8b)", "DEAD (<=0.25)",
                             "HOLD (0.25-0.35)")


def test_fusion_spatial_probe_raises_on_misaligned_caches(tmp_path):
    """pid/lat/fold must match between the two caches -- a shuffled second cache must raise rather
    than silently hstacking mismatched eyes."""
    p_disc = tmp_path / "disc.npz"
    p_full = tmp_path / "full_shuffled.npz"
    _synthetic_bakeoff_npz(str(p_disc), seed=0, n=100)
    _synthetic_bakeoff_npz(str(p_full), seed=1, n=100)
    d = dict(np.load(p_full))
    perm = np.random.default_rng(0).permutation(len(d['pid']))
    d['pid'] = d['pid'][perm]
    np.savez(p_full, **d)
    with pytest.raises(ValueError):
        FO.fusion_spatial_probe(str(p_disc), str(p_full), k=8, lam=5.0, boot=50, seed=0)


# --------------------------------------------------------------------------- PROBE B: ordinal
def test_coral_fit_predict_finite_and_in_range():
    rng = np.random.default_rng(0)
    n, d = 80, 10
    ytr = rng.uniform(1, 29, size=n)
    Xtr = rng.normal(size=(n, d)) + (ytr[:, None] / 10.0)
    Xte = rng.normal(size=(20, d))
    edges = np.linspace(0, 30, 11)
    pred = FO.coral_fit_predict(Xtr, ytr, Xte, edges, C=0.5)
    assert np.isfinite(pred).all()
    assert (pred >= edges[0]).all() and (pred <= edges[-1]).all()


def test_ordinal_severity_probe_runs_and_returns_finite(tmp_path):
    p = tmp_path / "disc.npz"
    _synthetic_bakeoff_npz(str(p), n=150, seed=2)
    r = FO.ordinal_severity_probe(str(p), n_bins=6, lo=0.0, hi=30.0, lam=5.0, C=0.5)
    for key in ('corr_ridge', 'corr_coral', 'delta_corr', 'spear_ridge', 'spear_coral', 'delta_spear',
                'mae_ridge', 'mae_coral'):
        assert np.isfinite(r[key]), key
    assert r['verdict'] in ("PASS (>=+0.02) -> carry to Task 11", "DROP")
