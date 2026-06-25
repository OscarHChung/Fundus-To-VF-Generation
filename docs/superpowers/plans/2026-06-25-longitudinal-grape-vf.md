# Longitudinal GRAPE Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the GRAPE training set from 263 baseline eyes to ~631 per-visit (fundus → same-visit VF) pairs by reading the unused `Follow-up` sheet, then retrain the proven decoder recipe to push honest per-patient-CV MAE from ~4.46 to ≤3.9 with a strong calibrated scatterplot line.

**Architecture:** Frozen RETFound encoder → PerPointAttention (Garway–Heath anatomical prior, mandatory) → PointHead + additive zero-init global-spatial head → post-hoc variance-match calibration. No architecture rewrite for Stages 0–2; the only new code is the longitudinal data builder, a per-patient longitudinal split, and stratified reporting. Encoder fine-tuning and manifold-decoder init are reserve levers, built only if Stages 0–2 fall short.

**Tech Stack:** Python 3.10, PyTorch (MPS), the existing `decoder/training.py` model + CLI, stdlib `zipfile`/`re` for xlsx parsing (no new dependency), scipy (already used by `vf_test_converter.py`).

## Global Constraints

- Encoder→decoder architecture and **Garway–Heath sectoring are always on** (`--weighting garway_heath`). Never remove them.
- The **patch-shuffle fix** (`PerPointVFModel._encode`, no `random_masking`) stays — it is a correctness fix.
- The pretrained VF auto-decoder (`decoder/pretrained_vf_decoder.pth`, ~2.0 MAE) must **not** be overwritten.
- **Headline metric = leak-free per-patient CV.** Every reported number is accompanied by severity-stratified (mild/moderate/severe) and per-eye breakdowns. A fast single split may be used for dev velocity only, always labelled as such.
- Per-patient grouping is mandatory in every split: **all visits of a patient (both eyes, all timepoints) live in exactly one fold.**
- VF record schema consumed by `training.py`: `{"PatientID": int, "Laterality": "OD"|"OS", "FundusImage": [filenames], "hvf": [8×9 lists, 100.0 = masked]}`. New records MUST match this exactly (plus optional `VisitNumber`, `mean_db`).
- Run all commands from the repo root: `/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation`.
- Prefix shell commands with `rtk` per the repo CLAUDE.md (e.g. `rtk python ...`). Use `rtk proxy <cmd>` when you need raw, unfiltered output.

---

## File Structure

- `vf_test_converter.py` (modify) — extract the G1→24-2 mapping into a reusable `g1_to_hvf(g1_61, laterality)` function; the `__main__` baseline behavior is preserved.
- `build_longitudinal_grape.py` (create) — stdlib xlsx reader for the `Follow-up` sheet + `g1_to_hvf` → `data/vf_tests/grape_longitudinal.json` (one record per paired visit).
- `decoder/diagnostics.py` (modify) — parametrize `cmd_split` (data path + output dir + dev-prefix) and add a `split-long` subcommand; add a `stratified_report` helper.
- `decoder/run_cv.py` (modify) — add `--cv-dir` so it can score the longitudinal folds; print the stratified report after pooling.
- `decoder/tests_session3.py` (modify) — add longitudinal correctness tests.
- `decoder/results/cv_long/` (create, generated) — the 5 longitudinal per-patient folds.
- `data/vf_tests/grape_longitudinal.json`, `grape_long_train.json`, `grape_long_val.json` (create, generated).

---

## Task 1: Extract a reusable G1→24-2 mapping

**Files:**
- Modify: `vf_test_converter.py`
- Test: `decoder/tests_session3.py`

**Interfaces:**
- Produces: `g1_to_hvf(g1_61, laterality) -> list[8][9]` — takes the 61 raw G1 sensitivity values (in GRAPE column order, including the two blind-spot columns at indices 21 and 32), returns an 8×9 HVF grid (Python lists) with `100.0` at masked cells, identical to what the current Baseline pipeline produces.

- [ ] **Step 1: Write the failing test**

Add to `decoder/tests_session3.py`:

```python
def test_g1_to_hvf_shape_and_mask():
    import vf_test_converter as C
    g1 = [20.0] * 61
    grid = C.g1_to_hvf(g1, "OD")
    assert len(grid) == 8 and all(len(r) == 9 for r in grid), "must be 8x9"
    flat = [v for row in grid for v in row]
    assert flat.count(100.0) == 72 - 52, "exactly 20 masked cells (72-52)"
    valid = [v for v in flat if v != 100.0]
    assert len(valid) == 52 and all(abs(v - 20.0) < 1e-6 for v in valid), "constant input → constant valid output"
    P("g1_to_hvf: 8x9 grid, 52 valid cells, constant input maps to constant field")
```

Note: the file's tests run via a bare `__main__`; register the new test by adding `test_g1_to_hvf_shape_and_mask()` to the call list at the bottom (done in Step 5 of Task 2 to batch the test-registration edits).

- [ ] **Step 2: Run it to verify it fails**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); import tests_session3 as t; t.test_g1_to_hvf_shape_and_mask()"`
Expected: FAIL with `AttributeError: module 'vf_test_converter' has no attribute 'g1_to_hvf'`.

- [ ] **Step 3: Refactor the mapping into a function**

In `vf_test_converter.py`, keep all the existing constants (`G1_LOCATIONS_RIGHT/LEFT`, `VF24_2_RIGHT/LEFT`, `spiral_order`, `mask_OD`) at module scope. Wrap the per-row mapping body (currently inside the `for i, pid` loop, lines ~144–193) into:

```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

_KD_RIGHT = cKDTree(VF24_2_RIGHT)
_KD_LEFT  = cKDTree(VF24_2_LEFT)
_MASK_OS  = mask_OD[:, ::-1]

def g1_to_hvf(g1_61, laterality):
    """61 raw G1 values (GRAPE column order, incl. blind-spot cols 21 & 32) -> 8x9 HVF grid."""
    eye = str(laterality).strip().upper()
    vf = np.asarray(g1_61, dtype=float)
    keep = np.ones(vf.shape[0], dtype=bool); keep[21] = False; keep[32] = False
    vf = vf[keep]                                  # 61 -> 59
    vf = vf[spiral_order(eye)]                      # reorder to G1_LOCATIONS order
    if eye == "OD":
        g1_pts, kd, mask = G1_LOCATIONS_RIGHT, _KD_RIGHT, mask_OD
    else:
        g1_pts, kd, mask = G1_LOCATIONS_LEFT, _KD_LEFT, _MASK_OS
    _, idx = kd.query(g1_pts)
    buckets = [[] for _ in range(len(kd.data))]
    for val, j in zip(vf, idx):
        buckets[j].append(val)
    mapped = np.array([np.mean(b) if b else np.nan for b in buckets])
    nan = np.isnan(mapped)
    if nan.any():
        mapped[nan] = griddata(kd.data[~nan], mapped[~nan], kd.data[nan], method="nearest")
    grid = np.full(mask.shape, 100.0)
    grid[mask] = mapped
    return grid.tolist()
```

Then have the existing `__main__` Baseline loop call `g1_to_hvf(grape_vf_row_61, eye)` instead of the inline body, so its output is unchanged. (Guard the file's top-level `pd.read_excel(...)` so importing the module does not require the spreadsheet — move it inside an `if __name__ == "__main__":` block.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); sys.path.insert(0,'.'); import tests_session3 as t; t.test_g1_to_hvf_shape_and_mask()"`
Expected: `PASS  g1_to_hvf: 8x9 grid, 52 valid cells, constant input maps to constant field`.

- [ ] **Step 5: Commit**

```bash
rtk git add vf_test_converter.py decoder/tests_session3.py
rtk git commit -m "refactor: extract reusable g1_to_hvf mapping from vf_test_converter"
```

---

## Task 2: Build the longitudinal dataset from the Follow-up sheet

**Files:**
- Create: `build_longitudinal_grape.py`
- Test: `decoder/tests_session3.py`

**Interfaces:**
- Consumes: `vf_test_converter.g1_to_hvf` (Task 1).
- Produces:
  - `read_followup(xlsx_path) -> list[dict]` with keys `subject:int, laterality:str, visit:int, interval:float, cfp:str, g1:list[61]`.
  - `build(xlsx_path, fundus_dir, out_path) -> list[dict]` writing per-visit records `{PatientID, Laterality, VisitNumber, FundusImage:[cfp], hvf, mean_db}` for every visit whose `cfp` exists on disk (skip `"/"`/missing). Writes `out_path`.

- [ ] **Step 1: Write the failing test**

Add to `decoder/tests_session3.py`:

```python
def test_followup_visit1_matches_baseline():
    """g1_to_hvf on each eye's Follow-up VISIT 1 must equal the existing Baseline record."""
    import json, numpy as np, build_longitudinal_grape as B, vf_test_converter as C
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
    import os, build_longitudinal_grape as B
    recs = B.build(B.XLSX, B.FUNDUS_DIR, "/tmp/_grape_long_test.json")
    assert len(recs) >= 600, f"expected ~631 paired visits, got {len(recs)}"
    n_sev = sum(r["mean_db"] < 15 for r in recs)
    assert n_sev >= 90, f"expected ~104 severe visits, got {n_sev}"
    for r in recs:
        assert isinstance(r["PatientID"], int)
        assert r["Laterality"] in ("OD", "OS")
        assert isinstance(r["FundusImage"], list) and len(r["FundusImage"]) == 1
        assert os.path.exists(os.path.join(B.FUNDUS_DIR, r["FundusImage"][0]))
        assert len(r["hvf"]) == 8 and len(r["hvf"][0]) == 9
    P(f"longitudinal build: {len(recs)} paired visits, {n_sev} severe, schema OK")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); sys.path.insert(0,'.'); import tests_session3 as t; t.test_followup_visit1_matches_baseline()"`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_longitudinal_grape'`.

- [ ] **Step 3: Implement the builder**

Create `build_longitudinal_grape.py`:

```python
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
    m = re.match(r"([A-Z]+)(\d+)", ref); col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return col, int(m.group(2))


def read_followup(xlsx_path):
    """Parse the 'Follow-up' worksheet with the stdlib (no openpyxl dependency)."""
    z = zipfile.ZipFile(xlsx_path)
    shared = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf8", "ignore"))
    # locate the Follow-up sheet's worksheet file via workbook rels
    wb = z.read("xl/workbook.xml").decode("utf8", "ignore")
    names = re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wb)
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf8", "ignore")
    relmap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    target = next(t for n, rid in names if n == "Follow-up" for t in [relmap[rid]])
    sheet_path = "xl/" + target.lstrip("/")
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
        if rno < 3:               # row1 header, row2 G1-index sub-header
            continue
        d = rows[rno]
        subj = d.get(1)
        if subj is None:
            continue
        g1 = [float(d[c]) for c in range(9, 70) if c in d]
        if len(g1) != 61:
            continue
        out.append({"PatientID": int(float(subj)),
                    "subject": int(float(subj)),
                    "laterality": str(d.get(2, "OD")).strip().upper(),
                    "visit": int(float(d.get(3, 0))),
                    "interval": float(d.get(4) or 0.0),
                    "cfp": str(d.get(6, "/")).strip(),
                    "g1": g1,
                    "hvf": C.g1_to_hvf(g1, str(d.get(2, "OD")).strip().upper())})
    return out


def build(xlsx_path, fundus_dir, out_path):
    have = set(os.listdir(fundus_dir))
    rows = read_followup(xlsx_path)
    recs = []
    for r in rows:
        if r["cfp"] not in have:          # "/" or missing photo
            continue
        hvf = r["hvf"]
        flat = [v for row in hvf for v in row if v < 99.0]
        recs.append({"PatientID": int(r["subject"]),
                     "Laterality": r["laterality"],
                     "VisitNumber": r["visit"],
                     "FundusImage": [r["cfp"]],
                     "hvf": hvf,
                     "mean_db": float(np.mean(flat))})
    json.dump(recs, open(out_path, "w"), indent=2)
    sev = sum(x["mean_db"] < 15 for x in recs)
    mod = sum(15 <= x["mean_db"] < 22 for x in recs)
    print(f"{len(recs)} paired visits / {len(set(x['PatientID'] for x in recs))} patients "
          f"/ {len(set((x['PatientID'], x['Laterality']) for x in recs))} eyes")
    print(f"severity: severe<15={sev}  moderate={mod}  mild={len(recs)-sev-mod}")
    print(f"-> {out_path}")
    return recs


if __name__ == "__main__":
    build(XLSX, FUNDUS_DIR, OUT)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); sys.path.insert(0,'.'); import tests_session3 as t; t.test_followup_visit1_matches_baseline(); t.test_longitudinal_build_schema()"`
Expected: both PASS (≥200 eyes validated; ~631 records, ~104 severe).

- [ ] **Step 5: Register the new tests and generate the dataset**

Add `test_g1_to_hvf_shape_and_mask()`, `test_followup_visit1_matches_baseline()`, and `test_longitudinal_build_schema()` to the `__main__` call list at the bottom of `decoder/tests_session3.py`. Then build:

Run: `rtk proxy python build_longitudinal_grape.py`
Expected: prints ~631 paired visits / 144 patients / 263 eyes; severe ~104; writes `data/vf_tests/grape_longitudinal.json`.

Run the full suite: `rtk proxy python decoder/tests_session3.py`
Expected: `ALL PASSED`.

- [ ] **Step 6: Commit**

```bash
rtk git add build_longitudinal_grape.py decoder/tests_session3.py data/vf_tests/grape_longitudinal.json
rtk git commit -m "feat: build longitudinal GRAPE dataset (631 per-visit fundus->VF pairs)"
```

---

## Task 3: Per-patient longitudinal split

**Files:**
- Modify: `decoder/diagnostics.py`
- Test: `decoder/tests_session3.py`

**Interfaces:**
- Consumes: `diagnostics.build_patient_folds(records, k, seed)` (already groups by `PatientID` — keeps all of a patient's visits/eyes together).
- Produces: `cmd_split(path, cv_dir, dev_prefix)` writing `fold{0..4}_{train,val}.json` into `cv_dir` and `{dev_prefix}_train.json`/`{dev_prefix}_val.json` into `data/vf_tests/`; new CLI subcommand `split-long`.

- [ ] **Step 1: Write the failing test**

Add to `decoder/tests_session3.py`:

```python
def test_longitudinal_folds_no_patient_leak():
    import json, os, diagnostics as D, build_longitudinal_grape as B
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); sys.path.insert(0,'.'); import tests_session3 as t; t.test_longitudinal_folds_no_patient_leak()"`
Expected: PASS already if `grape_longitudinal.json` exists (build_patient_folds groups by PatientID). If `D.load_records` cannot take a path arg, it FAILS with a `TypeError` — proceed to Step 3. (It already accepts `path=FULL_JSON`, so this test should pass; the real change is `cmd_split` parametrization below.)

- [ ] **Step 3: Parametrize the split writer**

In `decoder/diagnostics.py`, change `cmd_split` to accept parameters and add a longitudinal entry point. Replace the `cmd_split` signature and body's hard-coded paths:

```python
def cmd_split(k=5, path=FULL_JSON, cv_dir=CV_DIR, dev_prefix="grape_ppat"):
    os.makedirs(cv_dir, exist_ok=True)
    records = load_records(path)
    folds = build_patient_folds(records, k=k)
    fold_pat = [set(records[i]['PatientID'] for i in f) for f in folds]
    for a in range(k):
        for b in range(a + 1, k):
            assert not (fold_pat[a] & fold_pat[b]), f"patient leak across folds {a},{b}"
    print(f"Per-patient stratified {k}-fold over {len(records)} records / "
          f"{len(set(r['PatientID'] for r in records))} patients:")
    for j, f in enumerate(folds):
        sev = [eye_severity(records[i]) for i in f]
        n_sev = sum(s < 15 for s in sev)
        print(f"  fold {j}: {len(f):3d} recs | {len(fold_pat[j]):3d} patients | "
              f"severe(<15)={n_sev:2d} | meandB {np.mean(sev):.1f}")
        json.dump([records[i] for i in f], open(os.path.join(cv_dir, f"fold{j}_val.json"), 'w'))
        json.dump([records[i] for jj, ff in enumerate(folds) if jj != j for i in ff],
                  open(os.path.join(cv_dir, f"fold{j}_train.json"), 'w'))
    json.dump([records[i] for jj in range(1, k) for i in folds[jj]],
              open(os.path.join(VF_DIR, f"{dev_prefix}_train.json"), 'w'))
    json.dump([records[i] for i in folds[0]],
              open(os.path.join(VF_DIR, f"{dev_prefix}_val.json"), 'w'))
    print(f"\nWrote {k} folds -> {cv_dir}; dev split -> {dev_prefix}_train/val.json")
```

Add to the `__main__` dispatch:

```python
    elif cmd == "split-long":
        import build_longitudinal_grape as B
        cmd_split(path=B.OUT,
                  cv_dir=os.path.join(CURRENT_DIR, "results", "cv_long"),
                  dev_prefix="grape_long")
```

- [ ] **Step 4: Generate the longitudinal folds and verify**

Run: `rtk proxy python decoder/diagnostics.py split-long`
Expected: prints 5 folds over ~631 records / 144 patients, each fold with a proportional severe share; writes `decoder/results/cv_long/fold{0..4}_{train,val}.json` and `data/vf_tests/grape_long_{train,val}.json`.

Run: `rtk proxy python decoder/tests_session3.py`
Expected: `ALL PASSED`.

- [ ] **Step 5: Commit**

```bash
rtk git add decoder/diagnostics.py decoder/tests_session3.py decoder/results/cv_long data/vf_tests/grape_long_train.json data/vf_tests/grape_long_val.json
rtk git commit -m "feat: per-patient longitudinal 5-fold split (split-long)"
```

---

## Task 4: Severity-stratified honest reporting

**Files:**
- Modify: `decoder/diagnostics.py`
- Modify: `decoder/run_cv.py`
- Test: `decoder/tests_session3.py`

**Interfaces:**
- Produces: `diagnostics.stratified_report(preds, trues) -> dict` — buckets each per-eye record by its true mean dB into `severe(<15)`, `moderate(15-22)`, `mild(>=22)` and returns each bucket's pooled metrics + record count. Prints a table.

- [ ] **Step 1: Write the failing test**

Add to `decoder/tests_session3.py`:

```python
def test_stratified_report_buckets():
    import numpy as np, diagnostics as D
    severe = [np.full(52, 5.0)]; mild = [np.full(52, 28.0)]
    rep = D.stratified_report([np.full(52, 6.0)] + [np.full(52, 27.0)],
                              severe + mild, verbose=False)
    assert rep["severe"]["n_eyes"] == 1 and rep["mild"]["n_eyes"] == 1
    assert abs(rep["severe"]["mae"] - 1.0) < 1e-6 and abs(rep["mild"]["mae"] - 1.0) < 1e-6
    P("stratified_report: buckets eyes by true mean dB, per-bucket MAE correct")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); import tests_session3 as t; t.test_stratified_report_buckets()"`
Expected: FAIL with `AttributeError: module 'diagnostics' has no attribute 'stratified_report'`.

- [ ] **Step 3: Implement the helper**

Add to `decoder/diagnostics.py`:

```python
def stratified_report(preds, trues, verbose=True):
    """Bucket per-eye records by TRUE mean dB and report pooled metrics per severity band.
    Required alongside any pooled number so a milder per-visit distribution can't mislead."""
    bands = {"severe": (-1, 15), "moderate": (15, 22), "mild": (22, 99)}
    out = {}
    for name, (lo, hi) in bands.items():
        idx = [i for i, t in enumerate(trues) if lo <= np.nanmean(t) < hi]
        if not idx:
            out[name] = {"n_eyes": 0, "mae": float("nan")}
            continue
        m = pooled_metrics([preds[i] for i in idx], [trues[i] for i in idx])
        m["n_eyes"] = len(idx); out[name] = m
    if verbose:
        print("  severity-stratified (per-eye true mean dB):")
        for name in ("severe", "moderate", "mild"):
            r = out[name]
            if r["n_eyes"]:
                print(f"    {name:8s} n={r['n_eyes']:3d} | MAE {r['mae']:.3f} | "
                      f"floor {r.get('floor', float('nan')):.2f} | slope {r.get('slope', float('nan')):.3f}")
    return out
```

In `decoder/run_cv.py`, add a `--cv-dir` argument and call the report after pooling. Change the `CV_DIR` reference in `train_fold`/`main` to use `a.cv_dir`, and after computing `raw`/`cal`:

```python
    ap.add_argument('--cv-dir', default=CV_DIR)   # near the other add_argument calls
    ...
    import diagnostics as D
    print("\nRAW pooled:")
    D.stratified_report(vp_all, vt_all)
    print("CALIB pooled:")
    D.stratified_report(vp_cal, vt_all)
```

(Replace the two `os.path.join(CV_DIR, ...)` in `train_fold` and `main` with `a.cv_dir`; pass `a.cv_dir` into `train_fold`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `rtk proxy python -c "import sys; sys.path.insert(0,'decoder'); import tests_session3 as t; t.test_stratified_report_buckets()"`
Expected: `PASS  stratified_report: ...`.

- [ ] **Step 5: Commit**

```bash
rtk git add decoder/diagnostics.py decoder/run_cv.py decoder/tests_session3.py
rtk git commit -m "feat: severity-stratified honest reporting in CV harness"
```

---

## Task 5: Stage 1 — train the proven recipe on the longitudinal data (the read-out)

This is an execution/benchmark task. The recipe is the honest best from Session 3 (lever1b2 = control + additive global head), now on ~5× the paired data with clean per-visit labels. Keep all regularizers (Session-3 finding: removing entropy/label-noise overfits).

**Files:**
- Generated logs: `decoder/results/auto/long_global_f{0..4}.log`, checkpoints `decoder/results/auto/long_global_f{0..4}_best.pth`, summary `decoder/results/auto/long_global_cv.json`.

- [ ] **Step 1: Sanity dev run on fold 0 (fast signal before the full CV)**

Run (logs streamed to a file so a failure is fully diagnosable):

```bash
rtk proxy python decoder/training.py \
  --train-json decoder/results/cv_long/fold0_train.json \
  --val-json   decoder/results/cv_long/fold0_val.json \
  --out-tag long_global_f0 --no-champion --epochs 60 \
  --weighting garway_heath --sector-combine sector_only --reweight value \
  --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head \
  2>&1 | tee decoder/results/auto/long_global_f0.log
```

Watch the log: confirm `Train: NNN eyes -> NNN images` shows ~500 train records, val MAE prints each epoch, and the run does not diverge. **Early-stop criterion:** if val MAE is still > 4.4 by epoch 25 with no downward trend, stop and debug (see Step 4) rather than burning the full CV.

- [ ] **Step 2: Decompose fold 0 to confirm the mechanism**

Run: `rtk proxy python decoder/decompose.py decoder/results/auto/long_global_f0_best.pth --val-json decoder/results/cv_long/fold0_val.json`
Expected/looking for: `sevMAE` materially below the 2.58 baseline (the longitudinal severity-range is doing its job). Record the numbers in the iterations log.

- [ ] **Step 3: Full 5-fold CV**

Run:

```bash
rtk proxy python decoder/run_cv.py --tag long_global --cv-dir decoder/results/cv_long --epochs 60 -- \
  --weighting garway_heath --sector-combine sector_only --reweight value \
  --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head \
  2>&1 | tee decoder/results/auto/long_global_cv.log
```

Expected: prints per-fold metrics, then `5-FOLD OOF (... records) RAW` and `CALIB`, followed by the severity-stratified tables. **Success gate for Stage 1:** RAW or CALIB pooled MAE < 4.2 with a clear severe-bucket improvement. (≤3.9 may already be hit here; if so, jump to Task 6 only for the scatterplot deliverable.)

- [ ] **Step 4: If it underperforms — debug, don't thrash**

Use superpowers:systematic-debugging. The most likely issues and checks:
- **Stale/empty fold files:** `rtk proxy python -c "import json;print(len(json.load(open('decoder/results/cv_long/fold0_train.json'))))"` — expect ~500.
- **Image not found at train time:** the log raises `FileNotFoundError`; confirm `FundusImage[0]` basenames exist in `data/fundus/grape_fundus_images/`.
- **Distribution shift masking a real gain:** read the stratified table — if severe MAE improved but pooled didn't, that is still progress; report both.
- **Divergence/NaN:** lower `--lr` to 5e-4 for the longitudinal size and rerun fold 0.
Log the diagnosis and fix in `decoder/results/auto/iterations.md` before re-running the full CV.

- [ ] **Step 5: Record the result and commit logs**

Append a new iteration entry to `decoder/results/auto/iterations.md` (hypothesis, config, RAW/CALIB pooled MAE, stratified table, decomposition). Then:

```bash
rtk git add decoder/results/auto/iterations.md decoder/results/auto/long_global_cv.json
rtk git commit -m "results: Stage 1 longitudinal CV (global-head recipe)"
```

(Checkpoints `*.pth` are large — do not commit them unless the repo already tracks model files; confirm with `rtk git status` first.)

---

## Task 6: Stage 2 — push the scatterplot line (dispersion + calibration deliverable)

**Files:**
- Generated: `decoder/results/auto/long_disp_cv.json`, scatterplot PNGs `decoder/results/auto/long_*_scatter_calibrated.png`.

- [ ] **Step 1: Add the training-time dispersion term and re-run CV**

```bash
rtk proxy python decoder/run_cv.py --tag long_disp --cv-dir decoder/results/cv_long --epochs 60 -- \
  --weighting garway_heath --sector-combine sector_only --reweight value \
  --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head --dispersion-weight 0.1 \
  2>&1 | tee decoder/results/auto/long_disp_cv.log
```

Compare to Stage 1: dispersion should raise calibrated slope and lower the severe floor; accept it only if pooled MAE does not regress beyond ~0.05 dB (the anti-shrinkage/MAE trade — keep the variant that best serves severe-first + slope).

- [ ] **Step 2: Produce the deliverable scatterplot (calibrated)**

Pick the better Stage-1/Stage-2 recipe; for a representative fold (e.g. fold 0) render the calibrated line of best fit:

```bash
rtk proxy python decoder/scatter_clean.py decoder/results/auto/<best_tag>_f0_best.pth \
  --train-json decoder/results/cv_long/fold0_train.json \
  --val-json   decoder/results/cv_long/fold0_val.json --calib \
  -o decoder/results/auto/<best_tag>_f0_scatter_calibrated.png
```

**Success gate for Stage 2:** calibrated slope ≥ ~0.7 with severe points sitting near the oracle floor (~8.4 dB), and the printed `eyeCorr`/`floor` improved vs the prior best. This is the "strong line of best fit" deliverable.

- [ ] **Step 3: Update champions if beaten**

If the honest pooled MAE beats the recorded honest baseline (~4.46) and/or the calibrated severe floor beats the prior, update `decoder/results/champion/` records (overall + severe) with the new honest numbers and note "longitudinal" provenance. Keep both an overall-MAE champion and a severe/scatterplot champion (the existing two-champion convention).

- [ ] **Step 4: Commit**

```bash
rtk git add decoder/results/auto/long_disp_cv.json decoder/results/auto/iterations.md decoder/results/champion
rtk git commit -m "results: Stage 2 longitudinal + dispersion + calibrated scatterplot"
```

- [ ] **Step 5: Decision gate**

If both gates met (MAE ≤ 3.9 honest **and** strong calibrated line) → **stop; ship.** Otherwise proceed to the reserve levers (Task 7).

---

## Task 7 (RESERVE — only if Stages 0–2 miss ≤3.9): encoder adaptation + manifold init

Do **not** start this unless Task 6 Step 5 says to. Each is a separate brainstorm/spec-worthy sub-effort; sketches only here.

- [ ] **Lever 3a — BitFit encoder fine-tune on longitudinal data.** The "encoder overfits → data-limited" verdict was measured on 211 eyes; ~631 cleaner pairs change the overfit math. Reuse the existing `--finetune-norm --enc-lr 1e-4 --heavy-aug --batch-size 16` flags (already implemented, `test_bitfit_encoder` passes). Run on fold 0 first; keep only if val eyeCorr exceeds the frozen ~0.45 cap AND severity does not regress. Caching is disabled when the encoder trains → ~4× slower; budget overnight.

- [ ] **Lever 3b — LoRA on the last 1–2 ViT blocks.** If BitFit is too weak, add low-rank adapters (rank 4–8) to the last blocks' attention/MLP; heavier capacity than BitFit, still small. New code; spec it before building.

- [ ] **Lever 4 — Manifold-decoder init / consistency loss.** Initialize the VF-prediction path from `pretrained_vf_decoder.pth` (the ~2.0-MAE manifold) and/or add a training-time manifold-consistency regularizer. The prior failure was a post-hoc projection on OOD smooth inputs; the correct version retrains the auto-decoder on non-zeroed inputs first (allowed: decoder pretraining may be edited, must stay a masked-point spatial mapping) and applies it as a soft coherence prior during training, not post-hoc. Spec before building.

---

## Self-Review

**Spec coverage:**
- Data expansion (spec §3) → Tasks 1–2. ✓
- Label-noise fix (per-visit pairing) → Task 2 `build`. ✓
- Per-patient leak-free CV (spec §1, §5 Stage 0) → Task 3. ✓
- Honest stratified reporting (spec §1) → Task 4. ✓
- Stage 1 proven recipe (spec §5) → Task 5. ✓
- Stage 2 dispersion + calibration + scatterplot (spec §5) → Task 6. ✓
- Reserve encoder/manifold levers (spec §5) → Task 7. ✓
- Built-in visit-1==baseline correctness check (spec §3) → Task 2 `test_followup_visit1_matches_baseline`. ✓
- Risks: distribution-shift → stratified report (Task 4); G1 alignment → visit-1 test (Task 2); visit non-independence → per-patient folds (Task 3). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command and expected output. ✓

**Type consistency:** `g1_to_hvf(g1_61, laterality)` defined in Task 1 and consumed in Task 2; `read_followup`/`build`/`B.OUT`/`B.XLSX`/`B.FUNDUS_DIR`/`B.BASELINE_JSON` defined in Task 2 and consumed in Task 3 test; `cmd_split(path, cv_dir, dev_prefix)` and `stratified_report(preds, trues, verbose)` signatures consistent across Tasks 3–4 and run_cv. Record schema (`PatientID:int, Laterality, FundusImage:list, hvf:8×9`) matches `training.MultiImageDataset` (verified at training.py:285–289). ✓

## Notes on parallelism (per user request to use subagents)

- Tasks 1–4 are pure-CPU code/data/tests with no GPU and can be built **in parallel by separate subagents** (they touch different files: `vf_test_converter.py`/`build_longitudinal_grape.py` vs `diagnostics.py`/`run_cv.py`), then integrated. Tasks 3 and 4 both edit `diagnostics.py`, so serialize those two or have one subagent own that file.
- Tasks 5–7 are **GPU/MPS training and are serial** — one MPS device means parallel fold training contends and is not faster. Run folds sequentially (the existing `run_cv.py` already does). Use a background process + log tailing to monitor; do not spawn parallel training subagents.
- Always stream training output to a `.log` file (`2>&1 | tee ...`) so any failure is fully diagnosable after the fact.
