"""D2 gate (design §D2 / §5 #1): is a model's within-eye spatial prediction EYE-SPECIFIC, or is it
just the population template that every eye shares?

`decompose.py`'s res_corr does NOT answer this — it is ~entirely the template (§5 #1: model 0.471
vs no-image template 0.473). This script separates them by computing, on the pooled OOF held-out
val, the PARTIAL correlation of the model's predicted within-eye residual with the true residual,
controlling for a fixed population template built leak-free from each fold's TRAIN targets only:

    partial-corr(pred_resid, true_resid | template_resid)

Decision gate for P2 (Dir 2 feature fusion + Dir 4 disc-angular prior):
  ≥ 0.35 for some model/feature set  → P2 becomes the priority (design §D2)
  ≤ 0.25                             → eye-specific spatial channel empirically absent → P2 dead
                                       (design §6.5 stop rule); do NOT spend GPU-hours retraining.
Templates are per-laterality (OD/OS residual patterns mirror), so residual orders never mix.

  python decoder/d2_template_partial.py --tag p1disc --cache-dir decoder/results/auto/oof_cache_notta
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")


def _lat(x):
    return 'OD' if str(x).startswith('OD') else 'OS'


def _resid(v):
    """(52,) → within-eye residual (point − eye mean), nan preserved at masked."""
    m = ~np.isnan(v)
    r = np.full(v.shape, np.nan)
    r[m] = v[m] - v[m].mean()
    return r


def build_template(tt, lats):
    """Population within-eye residual template per laterality, from TRAIN targets (leak-free)."""
    acc = {'OD': [], 'OS': []}
    for t, lat in zip(tt, lats):
        acc[_lat(lat)].append(_resid(t))
    return {k: np.nanmean(np.vstack(v), axis=0) for k, v in acc.items() if v}


def eye_triples(vp, vt, lats, templ):
    """Per-eye (pred_resid, true_resid, template_resid) over that eye's valid points."""
    out = []
    for p, t, lat in zip(vp, vt, lats):
        m = ~np.isnan(t)
        pr = (p - np.nanmean(p[m] if m.any() else p))[m]        # pred residual on valid pts
        tr = _resid(t)[m]
        zr = templ[_lat(lat)][m]
        good = ~np.isnan(zr)
        out.append((pr[good], tr[good], zr[good]))
    return out


def partial_corr(x, y, z):
    """corr(x,y | z): residualize x and y on [1,z], correlate the residuals."""
    Z = np.vstack([np.ones_like(z), z]).T
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def incr_r2(pred, true, templ):
    """Incremental R² of the model's pred residual OVER the template, predicting the true residual."""
    def r2(cols):
        X = np.vstack([np.ones_like(true)] + cols).T
        beta = np.linalg.lstsq(X, true, rcond=None)[0]
        resid = true - X @ beta
        return 1 - resid.var() / true.var()
    return r2([templ, pred]) - r2([templ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='p1disc')
    ap.add_argument('--cache-dir', default=os.path.join(AUTO, 'oof_cache_notta'))
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    eyes = []          # (pid, pr, tr, zr) pooled over folds
    for f in range(5):
        d = np.load(os.path.join(a.cache_dir, f"{a.tag}_f{f}.npz"))
        vp, vt, tp, tt = list(d['vp']), list(d['vt']), list(d['tp']), list(d['tt'])
        tr_items = json.load(open(os.path.join(CV_DIR, f"fold{f}_train.json")))
        va_items = json.load(open(os.path.join(CV_DIR, f"fold{f}_val.json")))
        assert len(tr_items) == len(tt) and len(va_items) == len(vt), f"fold{f} misalignment"
        templ = build_template(tt, [r['Laterality'] for r in tr_items])
        va_lat = [r['Laterality'] for r in va_items]
        va_pid = [int(r.get('PatientID', -1)) for r in va_items]
        for (pr, tr, zr), pid in zip(eye_triples(vp, vt, va_lat, templ), va_pid):
            eyes.append((pid, pr, tr, zr))

    PR = np.concatenate([e[1] for e in eyes])
    TR = np.concatenate([e[2] for e in eyes])
    ZR = np.concatenate([e[3] for e in eyes])

    res_corr = float(np.corrcoef(PR, TR)[0, 1])
    templ_corr = float(np.corrcoef(ZR, TR)[0, 1])
    pcorr = partial_corr(PR, TR, ZR)
    dr2 = incr_r2(PR, TR, ZR)

    # patient-clustered bootstrap of the partial corr
    pids = np.array([e[0] for e in eyes])
    uniq = np.unique(pids)
    by_pat = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(a.seed)
    boot = np.empty(a.boot)
    for b in range(a.boot):
        rows = np.concatenate([by_pat[p] for p in rng.choice(uniq, uniq.size, replace=True)])
        boot[b] = partial_corr(np.concatenate([eyes[i][1] for i in rows]),
                               np.concatenate([eyes[i][2] for i in rows]),
                               np.concatenate([eyes[i][3] for i in rows]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    print("=" * 92)
    print(f"D2 template-partial spatial gate — tag={a.tag}  ({len(eyes)} eyes, {PR.size} points)")
    print("-" * 92)
    print(f"  raw within-eye res_corr          corr(pred_resid, true_resid)          = {res_corr:.3f}")
    print(f"  population template res_corr     corr(template_resid, true_resid)      = {templ_corr:.3f}")
    print(f"  EYE-SPECIFIC partial corr        corr(pred, true | template)           = {pcorr:.3f}"
          f"   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  incremental R² of model / templ  ΔR²(true ~ template+pred vs template)  = {dr2:.4f}")
    print("-" * 92)
    gate = ("BUILD P2 — eye-specific spatial signal present (≥0.35)" if pcorr >= 0.35 else
            "P2 DEAD — spatial channel empirically absent (≤0.25)" if pcorr <= 0.25 else
            "AMBIGUOUS (0.25–0.35) — borderline; weigh the severe-band CI before spending GPU-hours")
    print(f"  GATE: {gate}")
    print("=" * 92)
    json.dump({'tag': a.tag, 'res_corr': res_corr, 'template_corr': templ_corr,
               'partial_corr': pcorr, 'ci': [float(lo), float(hi)], 'incr_r2': float(dr2)},
              open(os.path.join(AUTO, f"d2_{a.tag}.json"), 'w'), indent=2, default=float)


if __name__ == "__main__":
    main()
