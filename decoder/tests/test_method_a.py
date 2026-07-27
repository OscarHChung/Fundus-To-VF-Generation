"""Method A (Balanced MSE / BMC) fast unit tests — NO encoder needed.

The core property: Balanced-MSE (BMC, Ren et al. CVPR'22) de-shrinks a regression that
ordinary MSE collapses toward the mean. On a noisy-feature regression whose MSE-optimal
slope is ~0.5 (regression dilution), one optimization under BMC must recover a fitted slope
that is (a) higher than MSE's and (b) closer to the true slope of 1.

Run:  python decoder/tests/test_method_a.py    (instant; pure torch, no RETFound)
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # decoder/
from losses import balanced_mse_loss

P = lambda name: print(f"  PASS  {name}")


def _fit_slope(loss_fn, x, t, steps=600, lr=0.02, a0=0.5):
    """Fit pred = a*x + b minimizing loss_fn(pred, t); return fitted slope of pred vs t."""
    a = torch.tensor(a0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(a * x + b, t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(np.polyfit(t.numpy(), (a * x + b).numpy(), 1)[0])


def _regression_dilution_data(n=2000, sig=5.0, seed=0):
    """z ~ N(0,sig); target t = z; feature x = z + N(0,sig) so MSE-optimal slope ≈ 0.5."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=g) * sig
    return z + torch.randn(n, generator=g) * sig, z.clone()   # (x, t)


def test_bmc_deshrinks_slope():
    x, t = _regression_dilution_data()
    mse = _fit_slope(lambda p, tt: ((p - tt) ** 2).mean(), x, t)
    # σ=2 (< target spread) gives a robust de-shrink margin; assert both up-vs-MSE and nearer-1.
    bmc = _fit_slope(lambda p, tt: balanced_mse_loss(p, tt, sigma=2.0), x, t)
    assert mse < 0.62, f"sanity: MSE should regress to the mean (~0.5), got {mse:.3f}"
    assert bmc > mse + 0.05, f"BMC must de-shrink slope above MSE: bmc {bmc:.3f} vs mse {mse:.3f}"
    assert abs(bmc - 1.0) < abs(mse - 1.0), f"BMC slope {bmc:.3f} must be nearer 1 than MSE {mse:.3f}"
    P(f"BMC de-shrinks slope: MSE {mse:.3f} → BMC(σ=2) {bmc:.3f} (nearer true slope 1)")


def test_bmc_sigma_monotone():
    """Larger σ ⇒ stronger de-shrink (this data/formula); guards the σ direction we tune on."""
    x, t = _regression_dilution_data()
    s1 = _fit_slope(lambda p, tt: balanced_mse_loss(p, tt, sigma=1.0), x, t)
    s3 = _fit_slope(lambda p, tt: balanced_mse_loss(p, tt, sigma=3.0), x, t)
    assert s3 > s1 + 0.05, f"σ=3 slope {s3:.3f} should exceed σ=1 slope {s1:.3f} (raise σ to de-shrink more)"
    P(f"BMC σ-monotonicity: slope σ1 {s1:.3f} < σ3 {s3:.3f} (tune σ UP for more slope)")


def test_bmc_weights_and_shape():
    """Scalar, finite, ≥0; all-ones weights == unweighted mean; perfect pred → ~0 loss."""
    g = torch.Generator().manual_seed(1)
    p = torch.randn(40, generator=g)
    t = torch.randn(40, generator=g)
    l_plain = balanced_mse_loss(p, t, sigma=1.5)
    l_ones = balanced_mse_loss(p, t, sigma=1.5, weights=torch.ones(40))
    assert l_plain.dim() == 0 and torch.isfinite(l_plain) and l_plain >= 0, l_plain
    assert torch.allclose(l_plain, l_ones, atol=1e-6), (float(l_plain), float(l_ones))
    # a well-separated perfect predictor is uniquely nearest its own target → near-zero CE
    tt = torch.arange(40).float() * 3.0
    assert balanced_mse_loss(tt.clone(), tt, sigma=0.5) < 0.05
    P("BMC helper: scalar/finite/≥0; ones-weight == unweighted; perfect pred → ~0 loss")


if __name__ == "__main__":
    print("Method A (Balanced MSE) unit tests:")
    test_bmc_deshrinks_slope()
    test_bmc_sigma_monotone()
    test_bmc_weights_and_shape()
    print("ALL PASSED")
