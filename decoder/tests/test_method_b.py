"""Method B (per-eye target denoising) fast unit tests — NO encoder needed.

Core property: fitting a robust per-point trend over an eye's visits and evaluating it at the
target visit's date recovers a target CLOSER to the noise-free truth than the single raw VF.
Fallbacks: <3-visit eyes and per-point-sparse / masked cells keep the raw value exactly.

Run:  python decoder/tests/test_method_b.py    (instant; pure numpy)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))  # scripts/
import build_denoised_targets as B

P = lambda name: print(f"  PASS  {name}")
MASK = 99.0


def test_denoise_closer_to_truth():
    """52 points, 4 visits with a real linear decline + test-retest noise; the denoised target
    at the target date must have lower mean |error| vs the noise-free truth than the raw label."""
    rng = np.random.RandomState(0)
    n_pts, intervals = 52, [0.0, 1.0, 2.0, 3.0]
    tgt_idx = 1                                    # denoise the t=1 visit (interpolation)
    base = rng.uniform(5, 30, n_pts)              # per-point baseline sensitivity
    decline = rng.uniform(0.0, 2.0, n_pts)        # dB/yr loss (glaucoma progresses down)
    truth = [base - decline * t for t in intervals]          # noise-free field per visit
    fields = [truth[i] + rng.randn(n_pts) * 2.4 for i in range(len(intervals))]  # + noise
    den = B.denoise_field(intervals, fields, tgt_idx)
    raw_err = np.abs(fields[tgt_idx] - truth[tgt_idx]).mean()
    den_err = np.abs(den - truth[tgt_idx]).mean()
    assert den_err < raw_err - 0.2, f"denoised {den_err:.2f} not clearly < raw {raw_err:.2f}"
    P(f"denoise: mean |err| raw {raw_err:.2f} → denoised {den_err:.2f} dB (closer to truth)")


def test_fallback_few_visits_returns_raw():
    rng = np.random.RandomState(1)
    fields = [rng.uniform(5, 30, 72), rng.uniform(5, 30, 72)]   # only 2 visits (<3)
    out = B.denoise_field([0.0, 1.0], fields, 1)
    assert np.array_equal(out, fields[1]), "‹3-visit eye must return the raw target unchanged"
    P("fallback: <3-visit eye returns raw target exactly (no denoising)")


def test_masked_and_sparse_points_kept():
    rng = np.random.RandomState(2)
    intervals = [0.0, 1.0, 2.0]
    fields = [rng.uniform(5, 30, 72) for _ in range(3)]
    fields[0][10] = MASK; fields[1][10] = MASK; fields[2][10] = MASK   # always-masked point
    fields[0][20] = MASK; fields[1][20] = MASK                          # only 1 valid obs (sparse)
    tgt = 2
    out = B.denoise_field(intervals, fields, tgt)
    assert out[10] == fields[tgt][10], "masked target point must stay masked"
    assert out[20] == fields[tgt][20], "point with <3 valid obs must keep its raw value"
    P("masked/sparse: masked-at-target and <3-obs points keep raw value")


if __name__ == "__main__":
    print("Method B (target denoising) unit tests:")
    test_denoise_closer_to_truth()
    test_fallback_few_visits_returns_raw()
    test_masked_and_sparse_points_kept()
    print("ALL PASSED")
