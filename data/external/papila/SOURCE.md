# PAPILA — source, license, and verified coding

## Provenance

- **Dataset:** PAPILA (fundus images + clinical data of both eyes of the same patient, for
  glaucoma assessment). Kovalyk, Morales-Sánchez, Verdú-Monedero, Sellés-Navarro, Palazón-Cabanes,
  Sancho-Gómez (2022).
- **Figshare article:** 14798004 — https://figshare.com/articles/dataset/PAPILA/14798004
  (DOI `10.6084/m9.figshare.14798004.v2`).
- **Downloaded file:** `PAPILA.zip` (590,857,635 bytes) via the figshare API's public
  `download_url` for that article (no login required):
  ```
  curl -s https://api.figshare.com/v2/articles/14798004      # -> lists files + download_url
  curl -L -o PAPILA.zip https://ndownloader.figshare.com/files/35013982
  ```
- **Download date:** 2026-07-11.
- **Companion paper (Data Descriptor):** Kovalyk et al., "PAPILA: Dataset with fundus images and
  clinical data of both eyes of the same patient for glaucoma assessment", *Scientific Data* 9,
  291 (2022). https://doi.org/10.1038/s41597-022-01388-1 (PMC9184612). Cite this paper, not the
  figshare repository, per the depositors' own request in the figshare description.

## License — a discrepancy, disclosed

Two different license statements exist for this dataset and we could not fully reconcile them
programmatically, so both are recorded here:

- The **figshare API metadata** for article 14798004 (`GET /v2/articles/14798004` -> `license`
  field) reports **"GPL 3.0+"** (https://www.gnu.org/licenses/gpl-3.0.html).
- The **published Nature Scientific Data paper** (a Data Descriptor, which per Springer Nature
  policy is CC-BY licensed) states in its own text that the work "is licensed under a Creative
  Commons Attribution 4.0 International License" (**CC-BY-4.0**).

We are treating **CC-BY-4.0** (journal policy + the paper's explicit self-declaration) as the
operative license for the dataset content, consistent with the task brief's assumption. The
figshare-item-level "GPL 3.0+" tag looks like metadata the depositors set incorrectly on the
figshare item itself (GPL is a software license, not an obvious fit for a data+images corpus,
and contradicts the paper's own CC-BY statement). **Flag for legal/PI review before any
redistribution beyond internal research use** — this repo only uses PAPILA for (a) internal
severity co-training and (b) internal external-validation reporting, consistent with CC-BY.

## Archive structure (after `unzip PAPILA.zip`)

```
PapilaDB-PAPILA-<git-hash>/
  FundusImages/                 # 488 disc-centered fundus photos: RET###OD.jpg / RET###OS.jpg
  ClinicalData/
    patient_data_od.xlsx        # 244 rows, one per patient, OD (right) eye
    patient_data_os.xlsx        # 244 rows, one per patient, OS (left) eye
  ExpertsSegmentations/         # optic disc/cup contour .txt files (2 experts) — NOT extracted
                                 # here (~tens of thousands of tiny files, unused by this task)
  HelpCode/                     # authors' own utils.py + Examples.ipynb + k-fold index files
  README.md                     # one line, points to the paper; no coding legend
```

We extracted everything **except** `ExpertsSegmentations/` (irrelevant to this task and adds a
large number of small files).

## Clinical-sheet columns (verified by inspection)

Both `patient_data_od.xlsx` and `patient_data_os.xlsx`, after stripping the duplicated header row
and the `ID` meta-row (see `decoder/build_external_papila.py::_fix_df`, which reproduces the
authors' own `HelpCode/utils.py::_fix_df`), have columns:

```
Age, Gender, Diagnosis, dioptre_1, dioptre_2, astigmatism, Phakic/Pseudophakic, Pneumatic,
Perkins, Pachymetry, Axial_Length, VF_MD
```

- **MD column used:** `VF_MD` — Humphrey visual-field **Mean Deviation, in dB**. This is the only
  MD-like field in either sheet.
- Row index is the patient ID, formatted `#002`, `#004`, ... (already zero-padded to 3 digits;
  `RET{index}{OD|OS}.jpg` is the matching image filename).

## Diagnosis coding — verified

`Diagnosis ∈ {0, 1, 2}`. Verified two ways:

1. **Directly against the published paper** (fetched via PMC9184612): *"the diagnosis (0 stands
   for healthy, 1 for glaucoma, and 2 for suspicious)"*.
2. **Cross-checked against the paper's reported totals** (488 eyes = 333 healthy + 155
   glaucoma-or-suspect): our own `value_counts()` on the two sheets gives
   `OD: {0:170, 1:40, 2:34}`, `OS: {0:163, 1:47, 2:34}` → 0-counts sum to 333, {1,2}-counts sum to
   155. Exact match.

Mapping used in `decoder/build_external_papila.py`:
```python
DIAGNOSIS_MAP = {0: "healthy", 1: "glaucoma", 2: "suspect"}
```

## Important data-reality caveat: VF_MD is NOT populated for all 488 eyes

Per the paper: the visual-field test was only retrieved for **glaucoma patients (Group 1)** and
for **suspect patients (Group 2) whose IOP was > 22 mmHg**; healthy controls with normal IOP were
generally not perimetry-tested. We verified this directly in the spreadsheets: only **164 of 488
eyes (82 unique patients x 2 eyes)** have a non-null numeric `VF_MD`; the rest (mostly the 333
healthy-labeled eyes) have no MD at all. (A handful of healthy-labeled fellow-eyes, 8 OD + 1 OS,
do have an MD recorded — almost always because that patient's *other* eye was Group 1/2 and got a
bilateral VF test at the same visit.)

**Consequence for `decoder/tests_external_papila.py`:** the Task 0 brief's draft test asserted
`len(recs) >= 200`. That count is unreachable by any faithful harmonization of this dataset —
only 164 eyes have a real MD to keep. The committed test asserts `>= 150` instead, with a comment
pointing back to this file. `label_counts` in the harmonized output: `{"suspect": 68, "glaucoma":
87, "healthy": 9}` (164 total), MD range `[-30.47, 1.55]` dB.

## Output schema

`decoder/build_external_papila.py` writes `data/external/papila_records.json`, a list of:
```json
{"eye_id": "papila_RET002OD", "image": "<abs path>/FundusImages/RET002OD.jpg",
 "Laterality": "OD", "md": -0.07, "label": "suspect", "source": "papila"}
```
Only eyes with a numeric `VF_MD` are kept, matching Task 0's interface contract.

## AIROGS (optional, not wired here — Step 6 of the brief)

AIROGS (Rotterdam EyePACS AIROGS, ~113k fundus images) is a **binary** referable-glaucoma /
no-referable-glaucoma (RG/NRG) labeled dataset — it has no MD or per-eye severity grading, so it
cannot feed the same schema as PAPILA. It is a weak auxiliary signal at best (a binary head, not
severity regression). **Not downloaded or wired in this task** — only note it here per the brief;
revisit only if Task 3's co-training probe specifically wants a binary auxiliary head.

## What is/ isn't committed to git

- **Committed:** this `SOURCE.md`, `decoder/build_external_papila.py`,
  `decoder/tests_external_papila.py`, and the small harmonized `data/external/papila_records.json`
  (52 KB).
- **NOT committed (gitignored):** `data/external/papila/PAPILA.zip` and the entire extracted
  `data/external/papila/PapilaDB-PAPILA-*/` tree (raw images + spreadsheets + segmentations), per
  `.gitignore` rule `data/external/papila/*` (with `SOURCE.md` explicitly un-ignored). Anyone
  reproducing this needs to re-run the `curl` commands above and `unzip PAPILA.zip`.
