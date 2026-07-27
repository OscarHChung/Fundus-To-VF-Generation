"""Task 7 — trajectory target-denoising unit tests.

hvf_denoised = per-(PatientID,Laterality) per-point Theil-Sen trajectory fit over the eye's FULL
visit timeline (all follow-up visits, photo or not), evaluated at each record's own visit date.
TRAIN-ONLY: val/eval always score against the raw observed VF. See also
decoder/tests/test_session3.py::test_denoised_target_swap, which already proves the generic
MultiImageDataset train/val gate (denoised_lookup applied iff mode=='train'); this file proves
(1) build_longitudinal_grape.py produces a sane hvf_denoised on every record, (2) the
--denoise-target lookup joins onto fold-train records by PatientID_Laterality_VisitNumber, and
(3) that lookup, fed through the real dataset class, never leaks into a val/eval sample.

Run:  python -m pytest decoder/tests/test_denoise_target.py -q
  or: python decoder/tests/test_denoise_target.py
"""
import os, sys, json, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                       # decoder/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))  # scripts/
import build_longitudinal_grape as BL

P = lambda name: print(f"  PASS  {name}")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LONG_JSON = os.path.join(ROOT, "data", "vf_tests", "grape_longitudinal.json")


def test_denoised_target_present_and_reasonable():
    recs = json.load(open(LONG_JSON))
    with_d = [r for r in recs if r.get("hvf_denoised") is not None]
    assert len(with_d) == len(recs), f"only {len(with_d)}/{len(recs)} records have hvf_denoised"
    diffs = []
    for r in with_d:
        a = np.array(r["hvf"], dtype=float)
        b = np.array(r["hvf_denoised"], dtype=float)
        assert a.shape == (8, 9) and b.shape == (8, 9), (a.shape, b.shape)
        valid = a < 99.0
        assert (b[valid] < 99.0).all(), "masking not preserved: a valid raw cell went masked"
        assert (b[~valid] >= 99.0).all(), "masking not preserved: a masked raw cell went valid"
        diffs.append(np.abs(a[valid] - b[valid]).mean())
    mean_diff = float(np.mean(diffs))
    assert mean_diff < 6.0, f"mean |raw-denoised| {mean_diff:.2f} dB — looks like wild extrapolation"
    P(f"hvf_denoised present on {len(with_d)}/{len(recs)} records; "
      f"mean |raw-denoised| = {mean_diff:.3f} dB")


def test_visit1_baseline_invariant_unaffected():
    """Adding hvf_denoised must not touch the raw hvf field (visit-1 == Baseline invariant)."""
    import vf_test_converter as C
    rows = BL.read_followup(BL.XLSX)
    base = {(int(r["PatientID"]), r["Laterality"]): np.array(r["hvf"])
            for r in json.load(open(BL.BASELINE_JSON))}
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
    P(f"raw hvf unaffected: visit-1 == baseline for {checked} eyes")


def test_denoise_target_lookup_join():
    """--denoise-target plumbing: the lookup keyed PatientID_Laterality_VisitNumber must expose
    exactly hvf_denoised, so train_lora_cached.py can join it onto fold-train JSON records
    (which share PatientID/Laterality/VisitNumber with grape_longitudinal.json)."""
    recs = json.load(open(LONG_JSON))
    lookup = BL.denoised_lookup_from_records(recs)
    assert len(lookup) == len(recs), (len(lookup), len(recs))
    for r in recs[:5]:
        key = f"{int(r['PatientID'])}_{r['Laterality']}_{int(r['VisitNumber'])}"
        assert key in lookup, key
        assert np.allclose(np.array(lookup[key]), np.array(r["hvf_denoised"]))
    P("denoise-target lookup joins by PatientID_Laterality_VisitNumber")


def test_eval_uses_raw_hvf():
    """Critical leak guard: a val/eval MultiImageDataset fed the --denoise-target lookup must
    IGNORE it and keep scoring against the RAW observed hvf; only mode='train' swaps to
    hvf_denoised. Exercises the real production dataset class (decoder.training.MultiImageDataset)
    with the actual lookup this task wires into train_lora_cached.py."""
    import training as T   # heavy (loads the frozen RETFound checkpoint once) — same precedent as
                            # tests_session3.py::test_denoised_target_swap; no training/GPU forward.
    recs_all = json.load(open(LONG_JSON))
    lookup = BL.denoised_lookup_from_records(recs_all)
    diff_recs = [r for r in recs_all
                 if not np.allclose(np.array(r["hvf"]), np.array(r["hvf_denoised"]))]
    assert diff_recs, "need at least one record where denoising actually changed something"
    recs = diff_recs[:5]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
        path = f.name
    try:
        ds_train = T.MultiImageDataset(path, "/nonexistent", T.val_transform, mode='train',
                                       denoised_lookup=lookup)
        ds_val = T.MultiImageDataset(path, "/nonexistent", T.val_transform, mode='val',
                                     denoised_lookup=lookup)
        assert len(ds_train.samples) == len(recs) == len(ds_val.samples)
        for i, r in enumerate(recs):
            raw = np.array(r["hvf"], dtype=np.float32)
            den = np.array(r["hvf_denoised"], dtype=np.float32)
            val_hvf = np.array(ds_val.samples[i]['hvf'], dtype=np.float32)
            train_hvf = np.array(ds_train.samples[i]['hvf'], dtype=np.float32)
            assert np.allclose(val_hvf, raw), "val dataset must use RAW hvf, not denoised"
            assert np.allclose(train_hvf, den), "train dataset must use the denoised target"
    finally:
        os.unlink(path)
    P("val/eval MultiImageDataset ignores --denoise-target lookup (raw hvf); "
      "train mode swaps to hvf_denoised")


if __name__ == "__main__":
    print("Task 7 (trajectory target-denoising) tests:")
    test_denoised_target_present_and_reasonable()
    test_visit1_baseline_invariant_unaffected()
    test_denoise_target_lookup_join()
    test_eval_uses_raw_hvf()
    print("ALL PASSED")
