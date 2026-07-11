import json, os
def test_papila_records_schema():
    recs = json.load(open("data/external/papila_records.json"))
    # NOTE: the Task 0 brief's draft test asserted >= 200. Verified ground truth (both by
    # inspecting the spreadsheets directly and cross-checking the published Data Descriptor,
    # Kovalyk et al. 2022, Scientific Data): PAPILA only performed the visual-field/MD test on
    # 164 of its 488 eyes (82 patients x 2 eyes) -- glaucoma patients (Group 1) and IOP>22mmHg
    # suspects (Group 2); healthy controls were not perimetry-tested. 200 is unreachable by any
    # faithful harmonization. See data/external/papila/SOURCE.md for the full verification trail.
    assert len(recs) >= 150
    r = recs[0]
    assert set(r) >= {"eye_id","image","Laterality","md","label","source"}
    assert r["Laterality"] in ("OD","OS") and isinstance(r["md"], float)
    assert os.path.exists(r["image"])
    # glaucoma-range MD present (not all healthy) so severity transfer is meaningful
    assert any(x["md"] < -6 for x in recs)
