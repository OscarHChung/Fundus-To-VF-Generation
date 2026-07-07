"""One-process fold scoreboard: pooled + severity-stratified + severity/spatial decomposition
for several checkpoints on the SAME val fold (loads RETFound once). Apples-to-apples TTA eval.

  python decoder/eval_fold.py <ckpt...> --val-json <fold_val.json>
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E
import diagnostics as D
import decompose as DEC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--no-tta', action='store_true')
    a = ap.parse_args()
    print(f"val={os.path.basename(a.val_json)}  TTA={not a.no_tta}\n")
    for c in a.ckpts:
        model = E.load_model(c)
        vp, vt = E.per_eye_preds(model, a.val_json, use_tta=not a.no_tta)
        m = D.pooled_metrics(vp, vt)
        strat = D.stratified_report(vp, vt, verbose=False)
        d = DEC.decompose(vp, vt)
        sev = strat['severe']; mod = strat['moderate']; mild = strat['mild']
        print(f"◆ {os.path.basename(c)}")
        print(f"   {D.fmt(m)}")
        print(f"   strata: severe(n{sev['n_eyes']}) {sev['mae']:.3f}  "
              f"moderate(n{mod['n_eyes']}) {mod['mae']:.3f}  mild(n{mild['n_eyes']}) {mild['mae']:.3f}")
        print(f"   severity: sev_corr {d['sev_corr']:.3f}  sev_shrink {d['sev_shrink']:.2f}  "
              f"sev_mae {d['sev_mae']:.2f} | res_corr {d['res_corr']:.3f}  res_shrink {d['res_shrink']:.2f}\n")


if __name__ == "__main__":
    main()
