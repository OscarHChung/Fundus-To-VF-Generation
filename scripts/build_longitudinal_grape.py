"""Build the longitudinal GRAPE dataset from the unused 'Follow-up' sheet.

Each Follow-up row is one visit: Subject, Laterality, Visit Number, Interval Years, IOP,
Corresponding CFP (fundus filename), then 61 raw G1 VF values (cols 9..69). We pair each
visit's fundus photo with ITS OWN contemporaneous VF (fixing the expand_GRAPE label-noise
bug where every image was paired with the single baseline VF). Rows whose CFP is "/" have a
VF but no photo and are skipped. Output schema matches training.MultiImageDataset.

  python scripts/build_longitudinal_grape.py     # writes data/vf_tests/grape_longitudinal.json
"""
import os, re, json, zipfile
import numpy as np
from scipy.stats import theilslopes
import vf_test_converter as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file lives in scripts/)
XLSX = os.path.join(ROOT, "data", "vf_tests", "grape_data.xlsx")
FUNDUS_DIR = os.path.join(ROOT, "data", "fundus", "grape_fundus_images")
BASELINE_JSON = os.path.join(ROOT, "data", "vf_tests", "grape_new_vf_tests.json")
OUT = os.path.join(ROOT, "data", "vf_tests", "grape_longitudinal.json")


def _colnum(ref):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return col, int(m.group(2))


def read_followup(xlsx_path):
    """Parse the 'Follow-up' worksheet with the stdlib (no openpyxl dependency).

    Returns list of dicts: {PatientID:int, subject:int, laterality:str, visit:int,
    interval:float, cfp:str, g1:list[61], hvf:8x9}."""
    z = zipfile.ZipFile(xlsx_path)
    shared = re.findall(r"<t[^>]*>([^<]*)</t>",
                        z.read("xl/sharedStrings.xml").decode("utf8", "ignore"))
    wb = z.read("xl/workbook.xml").decode("utf8", "ignore")
    names = re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wb)
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf8", "ignore")
    relmap = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="([^"]+)"', rels))
    rid = next(r for n, r in names if n == "Follow-up")
    sheet_path = "xl/" + relmap[rid].lstrip("/")
    xml = z.read(sheet_path).decode("utf8", "ignore")
    cells = re.findall(r'<c r="([A-Z]+\d+)"([^>]*)>(.*?)</c>', xml)
    rows = {}
    for ref, attr, inner in cells:
        col, row = _colnum(ref)
        v = re.search(r"<v>([^<]*)</v>", inner)
        if v is None:
            continue
        val = v.group(1)
        if 't="s"' in attr:
            val = shared[int(val)]
        rows.setdefault(row, {})[col] = val
    out = []
    for rno in sorted(rows):
        if rno < 3:                       # row1 header, row2 G1-index sub-header
            continue
        d = rows[rno]
        subj = d.get(1)
        if subj is None:
            continue
        g1 = [float(d[c]) for c in range(9, 70) if c in d]
        if len(g1) != 61:
            continue
        lat = str(d.get(2, "OD")).strip().upper()
        lat = "OD" if lat.startswith("OD") else "OS"
        out.append({
            "PatientID": int(float(subj)),
            "subject": int(float(subj)),
            "laterality": lat,
            "visit": int(float(d.get(3, 0) or 0)),
            "interval": float(d.get(4) or 0.0),
            "cfp": str(d.get(6, "/")).strip(),
            "g1": g1,
            "hvf": C.g1_to_hvf(g1, lat),
        })
    return out


MASK = 99.0


def compute_hvf_denoised(eye_visits, target_visit_no):
    """Task 7 — per-point robust TRAJECTORY fit over an eye's FULL visit timeline (`eye_visits`:
    ALL that eye's follow-up rows, photo or not, each with 'visit'/'interval'/'hvf'), evaluated at
    `target_visit_no`'s own interval. TRAIN-target denoising only; inference/eval are unaffected.

    Per point: if the target's raw value is masked, stay masked. Else gather (interval, value)
    across the eye's visits where that point is valid (masked>=99.0 excluded). With >=2 valid
    timepoints, fit scipy.stats.theilslopes (median-pairwise-slope + median intercept) and
    evaluate at the target's own interval, clipped to the point's observed range +/-3 dB (an
    interpolation guard against a wild two-point slope). With <2 valid timepoints (i.e. only the
    target's own visit has this point valid) there is nothing to fit against, so it falls back to
    the point's raw value (the median of the single available observation, which IS the raw
    value). Returns an 8x9 nested list.
    """
    target = next(v for v in eye_visits if v["visit"] == target_visit_no)
    raw = np.array(target["hvf"], dtype=float).flatten()          # (72,)
    t_tgt = float(target["interval"])
    intervals = np.array([v["interval"] for v in eye_visits], dtype=float)
    fields = np.array([np.array(v["hvf"], dtype=float).flatten() for v in eye_visits])  # (n,72)
    out = raw.copy()
    for p in range(raw.shape[0]):
        if raw[p] >= MASK:
            continue                                              # masked in raw -> stays masked
        col = fields[:, p]
        valid = col < MASK
        if valid.sum() < 2:
            continue                                              # <2 timepoints -> raw (unchanged)
        tv, vv = intervals[valid], col[valid]
        slope, intercept, _, _ = theilslopes(vv, tv)
        pred = slope * t_tgt + intercept
        lo, hi = vv.min() - 3.0, vv.max() + 3.0                   # interpolation guard
        out[p] = float(np.clip(pred, lo, hi))
    return out.reshape(8, 9).tolist()


def denoised_lookup_from_records(recs):
    """Build the {"PatientID_Laterality_VisitNumber": hvf_denoised} lookup consumed by
    `--denoise-target` (Task 7). Keys match the fold-train JSONs under decoder/results/cv_long/
    (they carry the same PatientID/Laterality/VisitNumber fields), so the fold files themselves
    never need to be regenerated — the trainer joins hvf_denoised onto them by this key at load
    time (TRAIN split only; val/eval keep raw hvf — see decoder/train_lora_cached.py)."""
    out = {}
    for r in recs:
        hd = r.get("hvf_denoised")
        if hd is None:
            continue
        key = f"{int(r['PatientID'])}_{r['Laterality']}_{int(r['VisitNumber'])}"
        out[key] = hd
    return out


def load_denoise_target_lookup(json_path=None):
    recs = json.load(open(json_path or OUT))
    return denoised_lookup_from_records(recs)


def build(xlsx_path, fundus_dir, out_path):
    """Write per-visit records for every visit whose fundus photo exists on disk.

    Each record also carries `interval_years` (cumulative years from baseline) and its
    most-recent CAUSAL prior VF (`prior_hvf`, from ANY earlier follow-up visit of the same
    eye — photo-bearing or not, since interim visits are temporally closer), with the
    visit-to-visit gap `delta_t`. Visit-1 records have prior_hvf=None / has_prior=False.
    """
    from collections import defaultdict
    have = set(os.listdir(fundus_dir))
    rows = read_followup(xlsx_path)
    # full per-eye VF timeline (ALL visits, photo or not), sorted by visit
    timeline = defaultdict(list)
    for r in rows:
        timeline[(r["subject"], r["laterality"])].append(r)
    for k in timeline:
        timeline[k].sort(key=lambda x: x["visit"])
    recs = []
    for r in rows:
        if r["cfp"] not in have:          # "/" or a photo we don't have
            continue
        flat = [v for row in r["hvf"] for v in row if v < 99.0]
        eye_timeline = timeline[(r["subject"], r["laterality"])]
        # most-recent strictly-earlier visit of the same eye (any-VF prior)
        hist = [p for p in eye_timeline if p["visit"] < r["visit"]]
        prior = hist[-1] if hist else None
        # Task 7 — TRAIN-only trajectory-denoised target (per-point Theil-Sen trend over the
        # eye's full timeline, evaluated at this visit's own date). Reuses the same timeline
        # already built above for prior_hvf. Eval/inference always use the raw `hvf` above.
        hvf_denoised = compute_hvf_denoised(eye_timeline, r["visit"])
        recs.append({
            "PatientID": int(r["subject"]),
            "Laterality": r["laterality"],
            "VisitNumber": r["visit"],
            "FundusImage": [r["cfp"]],
            "hvf": r["hvf"],
            "hvf_denoised": hvf_denoised,
            "mean_db": float(np.mean(flat)),
            "interval_years": float(r["interval"]),
            "has_prior": prior is not None,
            "prior_hvf": prior["hvf"] if prior else None,
            "prior_visit": prior["visit"] if prior else None,
            "delta_t": float(r["interval"] - prior["interval"]) if prior else 0.0,
        })
    json.dump(recs, open(out_path, "w"), indent=2)
    n_prior = sum(x["has_prior"] for x in recs)
    print(f"records with a causal prior VF: {n_prior}/{len(recs)}")
    sev = sum(x["mean_db"] < 15 for x in recs)
    mod = sum(15 <= x["mean_db"] < 22 for x in recs)
    n_pat = len(set(x["PatientID"] for x in recs))
    n_eye = len(set((x["PatientID"], x["Laterality"]) for x in recs))
    print(f"{len(recs)} paired visits / {n_pat} patients / {n_eye} eyes")
    print(f"severity: severe<15={sev}  moderate={mod}  mild={len(recs) - sev - mod}")
    print(f"-> {out_path}")
    return recs


if __name__ == "__main__":
    build(XLSX, FUNDUS_DIR, OUT)
