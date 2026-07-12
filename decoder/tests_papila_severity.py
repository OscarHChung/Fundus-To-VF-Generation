"""Task 3 (P-A2) unit tests -- synthetic-data checks only (no torch, no real PAPILA images/model,
no real GRAPE cache required): the probe's numpy machinery must run end-to-end and return a finite
sev_corr for BOTH the baseline and +PAPILA conditions. Repo convention: no decoder/__init__.py, no
conftest -- put decoder/ on sys.path and import bare.

Run:  python -m pytest decoder/tests_papila_severity.py -q
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_papila_severity as P


def _synthetic_grape(n=200, d=16, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    true_sev = rng.normal(size=n)
    signal_dir = rng.normal(size=d)
    X = rng.normal(size=(n, d)) * 0.5 + true_sev[:, None] * signal_dir
    pid = np.repeat(np.arange(n // 2), 2)[:n]            # 2 rows/patient (fellow eyes)
    fold = np.array([i % n_folds for i in range(n)])
    return X.astype(np.float64), true_sev, pid, fold


def _synthetic_papila(n=60, d=16, seed=1):
    rng = np.random.default_rng(seed)
    true_sev = rng.normal(size=n)
    signal_dir = rng.normal(size=d)
    X = rng.normal(size=(n, d)) * 0.5 + true_sev[:, None] * signal_dir
    return X.astype(np.float64), true_sev


def test_zscore_mean_zero_std_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = P.zscore(x)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-6   # zscore adds a 1e-8 epsilon to the denominator


def test_pooled_oof_baseline_finite_and_correlated():
    X, y, pid, fold = _synthetic_grape()
    pred = P.pooled_oof(X, y, fold, lam=5.0)
    assert np.isfinite(pred).all()
    corr, mae = P.sev_corr_mae(pred, y)
    assert np.isfinite(corr) and np.isfinite(mae)
    assert corr > 0.3   # synthetic signal must be recoverable, else the harness itself is broken


def test_pooled_oof_with_extra_papila_rows_finite():
    """extra_X/extra_y (the PAPILA rows) are TRAIN-ONLY: concatenated into every fold's training
    set but never inserted into a held-out fold's prediction target."""
    X, y, pid, fold = _synthetic_grape()
    Xp, yp = _synthetic_papila()
    pred = P.pooled_oof(X, y, fold, extra_X=Xp, extra_y=yp, lam=5.0)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    corr, mae = P.sev_corr_mae(pred, y)
    assert np.isfinite(corr) and np.isfinite(mae)


def test_pooled_oof_center_separately_finite():
    """Fallback (b): per-dataset feature-mean centering before the ridge must also stay finite."""
    X, y, pid, fold = _synthetic_grape()
    Xp, yp = _synthetic_papila()
    pred = P.pooled_oof(X, y, fold, extra_X=Xp, extra_y=yp, lam=5.0, center_separately=True)
    assert np.isfinite(pred).all()


def test_bootstrap_delta_finite_ci():
    X, y, pid, fold = _synthetic_grape()
    Xp, yp = _synthetic_papila()
    pred_base = P.pooled_oof(X, y, fold, lam=5.0)
    pred_plus = P.pooled_oof(X, y, fold, extra_X=Xp, extra_y=yp, lam=5.0)
    dmean, lo, hi = P.bootstrap_delta(pred_base, pred_plus, y, pid, boot=200, seed=0)
    assert np.isfinite(dmean) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= dmean <= hi


def test_run_condition_end_to_end_returns_finite_sev_corr_both_conditions(tmp_path):
    """The assertion required by the task-3 brief: the probe runs and returns a finite sev_corr for
    BOTH the baseline (GRAPE-only) and +PAPILA conditions, exercising run_condition()'s full path
    (npz load -> per-dataset z-score -> fold-wise ridge -> pooled OOF -> bootstrap) on a synthetic
    GRAPE-shaped cache (real sev/md/pid/fold keys, arbitrary raw MD scale/offset to also cover the
    z-scoring step)."""
    X, y_true, pid, fold = _synthetic_grape()
    npz_path = tmp_path / "synthetic_grape.npz"
    np.savez(npz_path, sev=X, md=y_true * 5.0 + 20.0, pid=pid, fold=fold)   # raw-scale target
    Xp, yp_raw = _synthetic_papila()
    yp = P.zscore(yp_raw)
    r = P.run_condition("synthetic", str(npz_path), Xp, yp, lam=5.0, boot=200)
    assert np.isfinite(r["corr_base"])   # baseline condition: finite sev_corr
    assert np.isfinite(r["corr_plus"])   # +PAPILA condition: finite sev_corr
    assert np.isfinite(r["delta"])
