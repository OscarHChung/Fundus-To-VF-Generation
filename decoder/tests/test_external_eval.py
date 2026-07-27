"""Task 13 -- data-hygiene + report-schema tests for the PAPILA external validation
(decoder/eval_external_papila.py). TDD: written before that script's report JSON existed.

Two checks:
  (1) PAPILA eye_ids are disjoint from every GRAPE cv_long fold record. Trivially true (PAPILA
      ids are strings "papila_RET###OD|OS"; GRAPE cv_long records key on an int PatientID + eye
      Laterality/VisitNumber -- different datasets, different id namespaces) but asserted
      explicitly per the task-13 brief so this is a real regression guard, not just a comment.
  (2) once eval_external_papila.py has been run, its saved report JSON has the keys the brief
      requires: overall Pearson/Spearman r (+ CI) and calibrated MAE, stratified by PAPILA label
      (healthy/suspect/glaucoma).

Run: python -m pytest decoder/tests/test_external_eval.py -q   (pure JSON/glob checks -- no torch)
"""
import glob
import json
import os

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPILA_JSON = os.path.join(REPO_ROOT, "data", "external", "papila_records.json")
CV_LONG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "cv_long")
REPORT_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "auto", "papila_external_eval.json")

REQUIRED_OVERALL_KEYS = {"r_pearson", "r_pearson_ci", "r_spearman", "r_spearman_ci", "mae_calibrated"}
LABELS = ("healthy", "suspect", "glaucoma")


def _cv_long_ids():
    """Every GRAPE cv_long record's identifier, across all 5 folds' train+val (the train/val
    split doesn't matter for a namespace-disjointness check -- we want the union of every GRAPE
    eye-visit that could ever be used to train or score the champion)."""
    ids = set()
    paths = glob.glob(os.path.join(CV_LONG_DIR, "fold*_*.json"))
    assert paths, f"no cv_long fold files found under {CV_LONG_DIR}"
    for path in paths:
        for r in json.load(open(path)):
            ids.add(f"{r['PatientID']}_{r['Laterality']}_{r.get('VisitNumber')}")
    return ids


def test_papila_ids_disjoint_from_cv_long():
    papila = json.load(open(PAPILA_JSON))
    papila_ids = {r["eye_id"] for r in papila}
    grape_ids = _cv_long_ids()
    assert len(papila_ids) > 0 and len(grape_ids) > 0

    overlap = papila_ids & grape_ids
    assert not overlap, f"PAPILA eye_ids collide with GRAPE cv_long ids: {overlap}"

    # belt-and-braces on the namespace itself: PAPILA ids all carry the 'papila_' prefix, no
    # GRAPE cv_long id ever could (GRAPE ids are built from an int PatientID).
    assert all(pid.startswith("papila_") for pid in papila_ids)
    assert not any(gid.startswith("papila_") for gid in grape_ids)


def test_report_schema():
    if not os.path.exists(REPORT_JSON):
        import pytest
        pytest.skip(f"run `python decoder/eval_external_papila.py` first to produce {REPORT_JSON}")
    report = json.load(open(REPORT_JSON))

    assert "overall" in report and "by_label" in report
    assert REQUIRED_OVERALL_KEYS <= set(report["overall"]), \
        f"missing overall keys: {REQUIRED_OVERALL_KEYS - set(report['overall'])}"

    for label in LABELS:
        assert label in report["by_label"], f"missing per-label breakdown for {label!r}"
        assert REQUIRED_OVERALL_KEYS <= set(report["by_label"][label]), \
            f"{label} missing keys: {REQUIRED_OVERALL_KEYS - set(report['by_label'][label])}"


if __name__ == "__main__":
    test_papila_ids_disjoint_from_cv_long()
    print("PASS: test_papila_ids_disjoint_from_cv_long")
    test_report_schema()
    print("PASS: test_report_schema")
