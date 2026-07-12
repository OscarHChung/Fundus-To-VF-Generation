"""Pre-committed decision rule: is P1 (disc-only) a real pooled win over the m1sev champion?

Reads the two tags' cached OOF val predictions from a shared cache dir (must be the SAME
--no-tta cache so the comparison is apples-to-apples), aligns each fold's rows to its
fold{f}_val.json (for PatientID → patient-clustered bootstrap and severity strata), and applies
the rule from p1_disc_roi_handoff.md §3 / design §6.5:

  PROMOTE iff  pooled paired ΔMAE ≤ −0.12 dB
          AND  patient-bootstrap 95% CI of ΔMAE excludes 0 (i.e. upper bound < 0)
          AND  ΔMAE negative in ≥ 4/5 folds
          AND  severe-band (true mean < 15 dB) MAE not worse.

ΔMAE = MAE(disc) − MAE(m1sev), pooled over valid points; negative = disc better.
Fold-0 alone can KILL but never PROMOTE (the M2 lesson: a single fold washes out).

  python decoder/paired_decision.py                        # disc vs m1sev, oof_cache_notta
  python decoder/paired_decision.py --new p1discX --ref m1sev --cache-dir <dir>
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D

AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")


def load_tag(tag, cache_dir, cv_dir):
    """Per-record arrays aligned across folds: preds (52,), trues (52,), pid, fold."""
    preds, trues, pids, folds = [], [], [], []
    for f in range(5):
        npz = os.path.join(cache_dir, f"{tag}_f{f}.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(f"{npz} — run eval_oof_cached.py --tag {tag} --no-tta "
                                    f"--cache-dir {cache_dir} first")
        d = np.load(npz)
        vp, vt = d['vp'], d['vt']                       # (n,52) float64, nan at masked
        items = json.load(open(os.path.join(cv_dir, f"fold{f}_val.json")))
        if len(items) != vp.shape[0]:
            raise ValueError(f"{tag} fold{f}: {vp.shape[0]} preds vs {len(items)} val records — "
                             f"row alignment broken")
        for i in range(vp.shape[0]):
            preds.append(vp[i]); trues.append(vt[i])
            pids.append(int(items[i].get('PatientID', -1))); folds.append(f)
    return preds, trues, np.array(pids), np.array(folds)


def pooled_point_mae(preds, trues):
    """MAE over all valid (non-nan) points, pooled across the given records."""
    ae = np.concatenate([np.abs(p - t)[~np.isnan(t)] for p, t in zip(preds, trues)])
    return float(ae.mean()), ae.size


def per_record_ae(preds, trues):
    """List of 1-D abs-error arrays over each record's valid points (for clustered bootstrap)."""
    return [np.abs(p - t)[~np.isnan(t)] for p, t in zip(preds, trues)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', default='p1disc', help='candidate tag')
    ap.add_argument('--ref', default='m1sev', help='champion/reference tag')
    ap.add_argument('--cache-dir', default=os.path.join(AUTO, 'oof_cache_notta'))
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    np_new, nt_new, pid_new, fold_new = load_tag(a.new, a.cache_dir, CV_DIR)
    np_ref, nt_ref, pid_ref, fold_ref = load_tag(a.ref, a.cache_dir, CV_DIR)

    # Same fold val sets → identical targets & order. Verify before pairing.
    if not (np.array_equal(pid_new, pid_ref) and np.array_equal(fold_new, fold_ref)):
        raise ValueError("candidate and reference are not record-aligned (different splits?)")
    for a_t, b_t in zip(nt_new, nt_ref):
        if not np.array_equal(np.isnan(a_t), np.isnan(b_t)):
            raise ValueError("mask mismatch between tags — targets differ, not comparable")

    n = len(np_new)
    pids, folds = pid_new, fold_new
    true_mean = np.array([np.nanmean(t) for t in nt_new])

    # ---- pooled metrics (full metric set for each) ----
    m_new = D.pooled_metrics(np_new, nt_new)
    m_ref = D.pooled_metrics(np_ref, nt_ref)
    print("=" * 96)
    print(f"CANDIDATE {a.new:>10s}  {D.fmt(m_new)}")
    print(f"REFERENCE {a.ref:>10s}  {D.fmt(m_ref)}")
    print("=" * 96)

    # ---- pooled paired ΔMAE (point-level) ----
    mae_new, npts = pooled_point_mae(np_new, nt_new)
    mae_ref, _ = pooled_point_mae(np_ref, nt_ref)
    dmae = mae_new - mae_ref
    print(f"pooled MAE  candidate {mae_new:.4f}  |  reference {mae_ref:.4f}  |  ΔMAE {dmae:+.4f} dB "
          f"({'candidate better' if dmae < 0 else 'candidate worse'})  over {npts} points, {n} eyes")

    # ---- per-fold ΔMAE signs ----
    print("\nper-fold ΔMAE (candidate − reference):")
    neg_folds = 0
    for f in range(5):
        idx = [i for i in range(n) if folds[i] == f]
        mf_new, _ = pooled_point_mae([np_new[i] for i in idx], [nt_new[i] for i in idx])
        mf_ref, _ = pooled_point_mae([np_ref[i] for i in idx], [nt_ref[i] for i in idx])
        d = mf_new - mf_ref
        neg_folds += (d < 0)
        print(f"  fold {f}: {mf_new:.3f} vs {mf_ref:.3f}  Δ {d:+.3f}  {'▼ better' if d < 0 else '▲ worse'}")

    # ---- severe band (true mean < 15) ----
    sev_idx = [i for i in range(n) if true_mean[i] < 15]
    sev_new, sev_np = pooled_point_mae([np_new[i] for i in sev_idx], [nt_new[i] for i in sev_idx])
    sev_ref, _ = pooled_point_mae([np_ref[i] for i in sev_idx], [nt_ref[i] for i in sev_idx])
    sev_d = sev_new - sev_ref
    print(f"\nsevere band (<15 dB, n={len(sev_idx)} eyes): candidate {sev_new:.3f} vs reference "
          f"{sev_ref:.3f}  Δ {sev_d:+.3f}  {'not worse ✓' if sev_d <= 0.05 else 'WORSE ✗'}")

    # ---- patient-clustered bootstrap of pooled AND severe-band ΔMAE (identical resamples) ----
    ae_new = per_record_ae(np_new, nt_new)
    ae_ref = per_record_ae(np_ref, nt_ref)
    is_sev = true_mean < 15
    uniq = np.unique(pids)
    by_pat = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(a.seed)
    boot = np.empty(a.boot); boot_sev = np.full(a.boot, np.nan); boot_abs = np.empty(a.boot)
    for b in range(a.boot):
        rows = np.concatenate([by_pat[p] for p in rng.choice(uniq, size=uniq.size, replace=True)])
        e_new = np.concatenate([ae_new[i] for i in rows])
        boot_abs[b] = e_new.mean()                                   # candidate's OWN absolute MAE
        boot[b] = e_new.mean() - np.concatenate([ae_ref[i] for i in rows]).mean()   # paired Δ
        srows = rows[is_sev[rows]]
        if srows.size:
            boot_sev[b] = (np.concatenate([ae_new[i] for i in srows]).mean()
                           - np.concatenate([ae_ref[i] for i in srows]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    slo, shi = np.nanpercentile(boot_sev, [2.5, 97.5])
    abs_lo, abs_hi = np.percentile(boot_abs, [2.5, 97.5])
    p_ge0 = float((boot >= 0).mean())      # one-sided: prob candidate not better
    print(f"\npatient-bootstrap pooled ΔMAE 95% CI [{lo:+.4f}, {hi:+.4f}] dB "
          f"({len(uniq)} patients, {a.boot} resamples)  P(Δ≥0)={p_ge0:.4f}")
    print(f"patient-bootstrap severe ΔMAE 95% CI [{slo:+.4f}, {shi:+.4f}] dB  (point {sev_d:+.3f})")
    print(f"patient-bootstrap candidate absolute MAE 95% CI [{abs_lo:.4f}, {abs_hi:.4f}] dB")

    # ---- decision rule (design §6.5 rule 2, verbatim) ----
    c1 = dmae <= -0.12
    c2 = hi < 0                     # pooled CI excludes 0 on the improvement side
    c3 = neg_folds >= 4
    c4 = shi < 0.15                 # severe-band paired ΔMAE 95% CI UPPER BOUND < +0.15 dB
    c5 = m_new['slope'] >= m_ref['slope'] - 1e-9   # raw slope not worse than baseline
    print("\n" + "-" * 96)
    print("PRE-COMMITTED DECISION RULE (§6.5 rule 2 — all required to PROMOTE):")
    print(f"  [{'✓' if c1 else '✗'}] pooled ΔMAE ≤ −0.12              : {dmae:+.4f}")
    print(f"  [{'✓' if c2 else '✗'}] pooled 95% CI upper < 0           : {hi:+.4f}")
    print(f"  [{'✓' if c3 else '✗'}] ΔMAE negative in ≥4/5 folds       : {neg_folds}/5")
    print(f"  [{'✓' if c4 else '✗'}] severe ΔMAE 95% CI upper < +0.15  : {shi:+.4f}")
    print(f"  [{'✓' if c5 else '✗'}] raw slope not worse               : {m_new['slope']:.3f} vs {m_ref['slope']:.3f}")
    promote = c1 and c2 and c3 and c4 and c5
    print(f"\n  VERDICT: {'PROMOTE ✅' if promote else 'DO NOT PROMOTE ❌'}")
    native_ok = abs_hi < 4.0   # §6.5 rule 5: candidate's OWN patient-bootstrap 95% upper bound < 4.0
    print(f"  native pooled MAE = {mae_new:.3f} dB (95% CI upper {abs_hi:.3f})  → native <4.0 claim "
          f"{'ALLOWED' if native_ok else 'NOT allowed (§6.5 rule 5)'}")
    print("-" * 96)

    json.dump({'new': a.new, 'ref': a.ref, 'mae_new': mae_new, 'mae_ref': mae_ref, 'dmae': dmae,
               'ci': [float(lo), float(hi)], 'severe_ci': [float(slo), float(shi)],
               'native_ci': [float(abs_lo), float(abs_hi)], 'native_ok': bool(native_ok),
               'p_ge0': p_ge0, 'neg_folds': int(neg_folds), 'severe_delta': sev_d,
               'slope_new': m_new['slope'], 'slope_ref': m_ref['slope'], 'promote': bool(promote)},
              open(os.path.join(AUTO, f"paired_{a.new}_vs_{a.ref}.json"), 'w'), indent=2, default=float)


if __name__ == "__main__":
    main()
