"""M2 side-car: per-eye baseline OCT-RNFL (+ age/CCT/IOP) lookup for the train-only aux head.

The GRAPE `grape_data.xlsx` **Baseline** sheet has, per eye, a 5-value OCT RNFL vector
[Mean, S, N, I, T] (cols 11-15; row 0 is the S/N/I/T sub-header) plus Age (col2), IOP (col4),
CCT (col5). This is the learnable structure-function substrate (Medeiros "M2M": fundus->RNFL is
learnable from a fundus). We attach it as a **training-only auxiliary target** to sharpen the
fundus features toward structure — inference stays fundus-only (the aux head is unused at test).

This does NOT touch the frozen longitudinal dataset / folds / eval: it writes a separate
`data/vf_tests/grape_rnfl_lookup.json` keyed by "PatientID_Laterality", consumed like the Method-B
denoised lookup. Records whose eye lacks RNFL are simply masked out of the aux loss.

Output json:
  {
    "norm": {"rnfl_mean": [5], "rnfl_std": [5], "aux_mean": [age,cct,iop], "aux_std": [...]},
    "eyes": {"<pid>_<lat>": {"rnfl": [Mean,S,N,I,T], "age":.., "cct":.., "iop":..}, ...}
  }
Only eyes with a COMPLETE 5-value RNFL are emitted (244/263; ~94% of the 631 records).

  python scripts/build_rnfl_lookup.py
"""
import os, json, math
import pandas as pd

XLSX = os.path.join("data", "vf_tests", "grape_data.xlsx")
OUT = os.path.join("data", "vf_tests", "grape_rnfl_lookup.json")


def _num(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def main():
    df = pd.ExcelFile(XLSX).parse("Baseline", header=0)
    eyes = {}
    for _, row in df.iterrows():
        pid = _num(row.iloc[0])
        if pid is None:                      # skip the S/N/I/T sub-header row
            continue
        lat = str(row.iloc[1]).strip()
        rnfl = [_num(row.iloc[c]) for c in range(11, 16)]   # Mean, S, N, I, T
        if any(v is None for v in rnfl):     # require the complete vector for the aux target
            continue
        eyes[f"{int(pid)}_{lat}"] = {
            "rnfl": rnfl,
            "age": _num(row.iloc[2]),
            "cct": _num(row.iloc[5]),
            "iop": _num(row.iloc[4]),
        }

    # Per-channel normalization stats (population; RNFL/age/cct/iop are never scored, so using
    # global stats to z-score the AUX target leaks no VF-label information).
    def col_stats(getter, k):
        vals = [getter(v) for v in eyes.values() if getter(v) is not None]
        n = len(vals); mu = sum(vals) / n
        sd = (sum((x - mu) ** 2 for x in vals) / max(n - 1, 1)) ** 0.5
        return mu, (sd if sd > 1e-6 else 1.0)

    rnfl_mean, rnfl_std = [], []
    for j in range(5):
        mu, sd = col_stats(lambda v: v["rnfl"][j], j)
        rnfl_mean.append(mu); rnfl_std.append(sd)
    aux_mean, aux_std = [], []
    for key in ("age", "cct", "iop"):
        mu, sd = col_stats(lambda v: v[key], key)
        aux_mean.append(mu); aux_std.append(sd)

    out = {"norm": {"rnfl_mean": rnfl_mean, "rnfl_std": rnfl_std,
                    "aux_mean": aux_mean, "aux_std": aux_std},
           "eyes": eyes}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Wrote {OUT}: {len(eyes)} eyes with full RNFL")
    print(f"  RNFL[Mean,S,N,I,T] mean = {[round(x,1) for x in rnfl_mean]}")
    print(f"  RNFL[Mean,S,N,I,T] std  = {[round(x,1) for x in rnfl_std]}")
    print(f"  aux[age,cct,iop] mean   = {[round(x,1) for x in aux_mean]}")

    # coverage report against the longitudinal records
    recs = json.load(open(os.path.join("data", "vf_tests", "grape_longitudinal.json")))
    if isinstance(recs, dict):
        recs = list(recs.values())
    keys = [f"{r.get('PatientID')}_{r.get('Laterality')}" for r in recs]
    m = sum(1 for k in keys if k in eyes)
    print(f"  coverage: {m}/{len(recs)} records ({100*m/len(recs):.0f}%) joinable to full RNFL")


if __name__ == "__main__":
    main()
