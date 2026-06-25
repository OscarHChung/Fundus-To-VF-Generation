"""Session-3 correctness tests — run before trusting a lever in a long training run.

Validates: (1) pooled metrics math, (2) per-patient folds have no leakage, (3) the new
dispersion-match loss penalizes UNDER-dispersed (flat) predictions and ~vanishes when the
field spread already matches. Run: decoder/tests_session3.py  (loads RETFound for compute_loss).
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # decoder/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repo root
import diagnostics as D

P = lambda name: print(f"  PASS  {name}")


# ── Longitudinal-expansion correctness (Tasks 1–4) ────────────────────────────
def test_g1_to_hvf_shape_and_mask():
    import vf_test_converter as C
    grid = C.g1_to_hvf([20.0] * 61, "OD")
    assert len(grid) == 8 and all(len(r) == 9 for r in grid), "must be 8x9"
    flat = [v for row in grid for v in row]
    assert flat.count(100.0) == 72 - 52, "exactly 20 masked cells (72-52)"
    valid = [v for v in flat if v != 100.0]
    assert len(valid) == 52 and all(abs(v - 20.0) < 1e-6 for v in valid), "constant in → constant out"
    P("g1_to_hvf: 8x9 grid, 52 valid cells, constant input maps to constant field")


def test_followup_visit1_matches_baseline():
    """g1_to_hvf on each eye's Follow-up VISIT 1 must equal the existing Baseline record."""
    import json, build_longitudinal_grape as B, vf_test_converter as C
    rows = B.read_followup(B.XLSX)
    base = {(int(r["PatientID"]), r["Laterality"]): np.array(r["hvf"])
            for r in json.load(open(B.BASELINE_JSON))}
    checked = 0
    for r in rows:
        if r["visit"] != 1:
            continue
        key = (r["subject"], r["laterality"])
        if key not in base:
            continue
        got = np.array(C.g1_to_hvf(r["g1"], r["laterality"]))
        assert np.allclose(got, base[key], atol=1e-6), f"visit-1 != baseline for {key}"
        checked += 1
    assert checked >= 200, f"expected to validate most eyes, only {checked}"
    P(f"followup visit-1 == baseline for {checked} eyes (G1 column alignment confirmed)")


def test_longitudinal_build_schema():
    import build_longitudinal_grape as B
    recs = B.build(B.XLSX, B.FUNDUS_DIR, "/tmp/_grape_long_test.json")
    assert len(recs) >= 600, f"expected ~631 paired visits, got {len(recs)}"
    n_sev = sum(r["mean_db"] < 15 for r in recs)
    assert n_sev >= 90, f"expected ~100 severe visits, got {n_sev}"
    for r in recs:
        assert isinstance(r["PatientID"], int)
        assert r["Laterality"] in ("OD", "OS")
        assert isinstance(r["FundusImage"], list) and len(r["FundusImage"]) == 1
        assert os.path.exists(os.path.join(B.FUNDUS_DIR, r["FundusImage"][0]))
        assert len(r["hvf"]) == 8 and len(r["hvf"][0]) == 9
    P(f"longitudinal build: {len(recs)} paired visits, {n_sev} severe, schema OK")


def test_longitudinal_folds_no_patient_leak():
    import build_longitudinal_grape as B
    recs = D.load_records(B.OUT)
    folds = D.build_patient_folds(recs, k=5)
    seen = {}
    for j, f in enumerate(folds):
        for i in f:
            pid = recs[i]["PatientID"]
            assert pid not in seen or seen[pid] == j, f"patient {pid} leaks across folds"
            seen[pid] = j
    assert sum(len(f) for f in folds) == len(recs), "every visit assigned to exactly one fold"
    P(f"longitudinal folds: {len(recs)} visits, no patient leaks across {len(folds)} folds")


def test_stratified_report_buckets():
    severe = [np.full(52, 5.0)]
    mild = [np.full(52, 28.0)]
    rep = D.stratified_report([np.full(52, 6.0), np.full(52, 27.0)],
                              severe + mild, verbose=False)
    assert rep["severe"]["n_eyes"] == 1 and rep["mild"]["n_eyes"] == 1
    assert abs(rep["severe"]["mae"] - 1.0) < 1e-6 and abs(rep["mild"]["mae"] - 1.0) < 1e-6
    P("stratified_report: buckets eyes by true mean dB, per-bucket MAE correct")


def test_pooled_metrics_perfect():
    t = np.linspace(0, 30, 52); preds = [t.copy()]; trues = [t.copy()]
    m = D.pooled_metrics(preds, trues)
    assert m['mae'] < 1e-6 and abs(m['slope'] - 1) < 1e-6 and abs(m['corr'] - 1) < 1e-6, m
    assert abs(m['sig_ratio'] - 1) < 1e-6, m
    P("pooled_metrics: perfect prediction → MAE0 slope1 corr1 σratio1")


def test_pooled_metrics_flat():
    t = np.linspace(0, 30, 52); preds = [np.full(52, t.mean())]; trues = [t.copy()]
    m = D.pooled_metrics(preds, trues)
    assert abs(m['slope']) < 1e-6 and m['sig_ratio'] < 1e-6, m   # flat pred → slope 0, no spread
    P("pooled_metrics: flat prediction → slope0 σratio0 (the shrinkage failure mode)")


def test_folds_no_leak():
    recs = D.load_records()
    folds = D.build_patient_folds(recs, k=5)
    seen = {}
    for j, f in enumerate(folds):
        for i in f:
            pid = recs[i]['PatientID']
            assert pid not in seen or seen[pid] == j, f"patient {pid} leaks across folds"
            seen[pid] = j
    assert sum(len(f) for f in folds) == len(recs)
    P("folds: every patient in exactly one fold; all eyes covered (no leakage)")


def test_dispersion_loss():
    import training as T
    vi = T.valid_indices_od
    vals = torch.linspace(0, 30, 52)
    target = torch.full((1, 72), 100.0)
    target[0, vi] = vals                                   # valid points spread 0..30
    flat  = torch.full((1, 52), float(vals.mean()))        # under-dispersed (shrunk to mean)
    match = vals.unsqueeze(0).clone()                      # spread matches target
    def loss(pred, w):
        l, _, _ = T.compute_loss(pred, target, ['OD'], epoch=10, dispersion_weight=w)
        return float(l)
    pen_flat  = loss(flat,  1.0) - loss(flat,  0.0)
    pen_match = loss(match, 1.0) - loss(match, 0.0)
    assert pen_flat > 0.5, f"flat pred should incur a real dispersion penalty, got {pen_flat:.3f}"
    assert pen_match < 0.05, f"matched-spread pred penalty should ~vanish, got {pen_match:.3f}"
    assert pen_flat > 10 * (pen_match + 1e-6)
    P(f"dispersion loss: flat penalty {pen_flat:.2f} ≫ matched penalty {pen_match:.3f} (anti-shrinkage works)")


def test_mean_residual_head():
    import training as T
    torch.manual_seed(0)
    m = T.PerPointVFModel(T.base_model, mean_residual=True).to('cpu').eval()
    lat = torch.randn(2, 197, 1024)
    with torch.no_grad():
        out = m.decode_latent(lat, ['OD', 'OS'], average_multi=False)
        mh = m.mean_head(lat[:, 0, :]).squeeze(-1)
    assert out.shape == (2, 52), out.shape
    # residual is zero-mean by construction → at init (refinement≈identity) each eye's pred mean
    # equals the mean-head output: the decomposition routes severity through the mean head.
    assert torch.allclose(out.mean(1), mh, atol=0.3), (out.mean(1), mh)
    P(f"mean+residual: out{tuple(out.shape)}; per-eye pred-mean≈mean_head (Δ<0.3) — decomposition routes severity")


def test_global_head():
    import training as T
    torch.manual_seed(0)
    m = T.PerPointVFModel(T.base_model, global_head=True).to('cpu').eval()
    lat = torch.randn(2, 197, 1024)
    with torch.no_grad():
        out = m.decode_latent(lat, ['OD', 'OS'], average_multi=False)
        g = m._global_residual(lat[:, 0, :], lat[:, 1:, :])
    assert out.shape == (2, 52), out.shape
    assert m._last_attn_weights is not None, "additive global head keeps the attention path active"
    assert torch.allclose(g.mean(1), torch.zeros(2), atol=1e-5), "global residual must be zero-mean"
    assert g.abs().mean() < 0.5, "zero-init global head must be ~no-op at start (degrades to control)"
    P(f"global head: additive, zero-mean residual {float(g.abs().mean()):.3f}≈0 at init; attention kept")


def test_bitfit_encoder():
    import training as T
    m = T.PerPointVFModel(T.base_model, finetune_norm=True, global_head=True).to('cpu')
    enc_train = [n for n, p in m.encoder.named_parameters() if p.requires_grad]
    assert len(enc_train) > 0, "BitFit unfroze no encoder params"
    assert all(('norm' in n or n.endswith('.bias')) for n in enc_train), "BitFit unfroze non-norm/bias"
    x = torch.randn(1, 3, 224, 224)
    out = m(x, laterality=['OD'], average_multi=False)
    out.sum().backward()
    grads = [p.grad for n, p in m.encoder.named_parameters() if 'norm' in n and p.grad is not None]
    assert len(grads) > 0, "BitFit: no gradient reached encoder norm params"
    # frozen (non-norm/bias) encoder params must stay frozen
    frozen = [n for n, p in m.encoder.named_parameters() if not p.requires_grad]
    assert any('attn' in n for n in frozen), "expected attention weight matrices to stay frozen"
    P(f"BitFit: {len(enc_train)} enc tensors trainable (norm+bias only); grad reaches encoder norms")


if __name__ == "__main__":
    print("Session-3 tests:")
    test_g1_to_hvf_shape_and_mask()
    test_followup_visit1_matches_baseline()
    test_longitudinal_build_schema()
    test_longitudinal_folds_no_patient_leak()
    test_stratified_report_buckets()
    test_pooled_metrics_perfect()
    test_pooled_metrics_flat()
    test_folds_no_leak()
    test_dispersion_loss()
    test_mean_residual_head()
    test_global_head()
    test_bitfit_encoder()
    print("ALL PASSED")
