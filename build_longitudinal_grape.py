"""Build the longitudinal GRAPE dataset from the unused 'Follow-up' sheet.

Each Follow-up row is one visit: Subject, Laterality, Visit Number, Interval Years, IOP,
Corresponding CFP (fundus filename), then 61 raw G1 VF values (cols 9..69). We pair each
visit's fundus photo with ITS OWN contemporaneous VF (fixing the expand_GRAPE label-noise
bug where every image was paired with the single baseline VF). Rows whose CFP is "/" have a
VF but no photo and are skipped. Output schema matches training.MultiImageDataset.

  python build_longitudinal_grape.py     # writes data/vf_tests/grape_longitudinal.json
"""
import os, re, json, zipfile
import numpy as np
import vf_test_converter as C

ROOT = os.path.dirname(os.path.abspath(__file__))
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
        # most-recent strictly-earlier visit of the same eye (any-VF prior)
        hist = [p for p in timeline[(r["subject"], r["laterality"])] if p["visit"] < r["visit"]]
        prior = hist[-1] if hist else None
        recs.append({
            "PatientID": int(r["subject"]),
            "Laterality": r["laterality"],
            "VisitNumber": r["visit"],
            "FundusImage": [r["cfp"]],
            "hvf": r["hvf"],
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
