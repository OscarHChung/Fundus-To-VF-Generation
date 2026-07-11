"""Harmonize PAPILA (figshare 14798004, Kovalyk et al. 2022) to our external-eval schema.

PAPILA ships disc-centered fundus photos (`FundusImages/RET###OD|OS.jpg`) plus two per-laterality
clinical spreadsheets (`ClinicalData/patient_data_od.xlsx` / `patient_data_os.xlsx`) whose columns
include Age, Gender, Diagnosis, and VF_MD (Humphrey visual-field mean deviation, dB).

Verified ground truth (see data/external/papila/SOURCE.md for the full trail):
  - Diagnosis coding: 0=healthy, 1=glaucoma, 2=suspect (confirmed against the published Data
    Descriptor, Kovalyk et al., Scientific Data 2022, and cross-checked: counts of 0/1/2 per sheet
    sum to the paper's reported 333 healthy + 155 glaucoma-or-suspect eyes).
  - VF_MD (the MD column) is NOT populated for all 488 eyes: per the paper, perimetry was only
    retrieved for glaucoma patients (Group 1) and suspect patients (Group 2) with IOP > 22 mmHg —
    i.e. healthy controls mostly never got a VF test. Only 164 of 488 eyes (82 patients x 2 eyes)
    have a numeric VF_MD. This is a real property of the source data, not a bug in this script.

Output: data/external/papila_records.json = list of
  {"eye_id", "image", "Laterality", "md", "label", "source":"papila"}
Only eyes with a numeric VF_MD are kept (per Task 0 brief).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPILA_ROOT_GLOB = os.path.join(REPO_ROOT, "data", "external", "papila", "PapilaDB-PAPILA-*")
OUT_PATH = os.path.join(REPO_ROOT, "data", "external", "papila_records.json")

DIAGNOSIS_MAP = {0: "healthy", 1: "glaucoma", 2: "suspect"}


def _find_papila_root():
    matches = sorted(p for p in glob.glob(PAPILA_ROOT_GLOB) if os.path.isdir(p))
    if not matches:
        raise FileNotFoundError(
            f"PAPILA extracted folder not found under {PAPILA_ROOT_GLOB}. "
            "Download+extract PAPILA.zip first -- see data/external/papila/SOURCE.md."
        )
    return matches[0]


def _fix_df(df):
    """Reproduce PAPILA's own HelpCode/utils.py::_fix_df: the raw sheet has a duplicated header
    row and an 'ID' meta-row; this drops both and promotes the real header."""
    df_new = df.drop(["ID"], axis=0)
    df_new.columns = df_new.iloc[0, :]
    df_new.drop([np.nan], axis=0, inplace=True)
    df_new.columns.name = "ID"
    return df_new


def _load_clinical(root):
    df_od_raw = pd.read_excel(os.path.join(root, "ClinicalData", "patient_data_od.xlsx"), index_col=[0])
    df_os_raw = pd.read_excel(os.path.join(root, "ClinicalData", "patient_data_os.xlsx"), index_col=[0])
    return _fix_df(df_od_raw), _fix_df(df_os_raw)


def build_records(root=None):
    root = root or _find_papila_root()
    df_od, df_os = _load_clinical(root)
    fundus_dir = os.path.join(root, "FundusImages")

    records = []
    for laterality, df in (("OD", df_od), ("OS", df_os)):
        for patient_id, row in df.iterrows():
            md = pd.to_numeric(row.get("VF_MD"), errors="coerce")
            if pd.isna(md):
                continue
            try:
                diag_int = int(float(row.get("Diagnosis")))
            except (TypeError, ValueError):
                continue
            label = DIAGNOSIS_MAP.get(diag_int)
            if label is None:
                continue

            num = str(patient_id).lstrip("#").zfill(3)
            image_name = f"RET{num}{laterality}.jpg"
            image_path = os.path.join(fundus_dir, image_name)
            if not os.path.exists(image_path):
                continue

            records.append({
                "eye_id": f"papila_RET{num}{laterality}",
                "image": image_path,
                "Laterality": laterality,
                "md": float(md),
                "label": label,
                "source": "papila",
            })
    return records


def main():
    records = build_records()
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    mds = [r["md"] for r in records]
    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"Wrote {len(records)} records to {OUT_PATH}")
    print(f"MD range: [{min(mds):.2f}, {max(mds):.2f}]  mean={sum(mds) / len(mds):.2f}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    main()
