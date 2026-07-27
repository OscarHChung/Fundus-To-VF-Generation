"""Task 11 (CORAL ordinal per-point head) unit tests.

Encoder-free: decode_latent runs the decoder on a pre-computed latent (no encoder forward), and the
CORAL helpers (coral_decode / coral_bce_per_point / CoralPointHead) are pure tensor functions.
Importing training loads RETFound once for base_model, but no encoder forward runs here. Run:
  python decoder/tests/test_ordinal_head.py   (ONE process at a time)

Context: a frozen-feature probe (P-C1, diag_fusion_ordinal_probe.py — removed in the 2026-07 cleanup; see git history) showed a CORAL ordinal readout
beat plain ridge by +0.040 sev_corr. This wires a REAL (rank-consistent, shared-weight) CORAL head
into the trained per-point decoder as a default-OFF flag (--ordinal-head).

Validates:
  (1) --ordinal-head OFF (default) is a no-op: the ordinal_head submodule exists (built
      unconditionally but LAST in __init__, after every other module, so its RNG draws can never
      perturb any other module's init) but is never called when OFF -- scrambling its weights must
      not change decode_latent's output.
  (2) Building with ordinal_head=True vs False (same seed) leaves EVERY other module's weights
      identical -- proves there is no RNG-order leak from adding the new module.
  (3) --ordinal-head ON: decode_latent output stays (B,52), finite, and within the model's existing
      output clamp range -- the continuous interface is unchanged even though the internal
      representation is K-1 ordinal logits.
  (4) CoralPointHead's K-1 logits are RANK-MONOTONE (non-increasing across the threshold index) for
      arbitrary input -- the Cao et al. 2020 rank-consistency guarantee from one shared trunk logit
      + strictly-ordered biases.
  (5) coral_decode matches its closed form at the two extremes (uninformative / saturated logits).
  (6) compute_loss: ordinal_logits=None (default) is byte-identical to the current Huber loss;
      supplying ordinal_logits/ordinal_cfg gives a finite CORAL loss that still respects the SAME
      per-point weighting path, while MAE (computed from pred/target only) is unaffected.
  (7) compute_loss ordinal ON-path fix: the CORAL BCE is an ADDITIVE auxiliary term (scaled by
      ordinal_cfg['weight'], default ORDINAL_WEIGHT) on top of the SAME Huber primary-fit term used
      OFF -- NOT a replacement. Proof: ordinal_weight=0.0 ON reduces byte-identical to the OFF/Huber
      loss, and the loss is LINEAR in ordinal_weight.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

P = lambda name: print(f"  PASS  {name}")


def test_ordinal_head_off_is_noop():
    """OFF: the ordinal_head submodule exists but is never called -- perturbing its weights must not
    change decode_latent output (proves default-OFF == the pre-Task-11 continuous head)."""
    import training as T
    torch.manual_seed(0)
    m = T.PerPointVFModel(T.base_model, ordinal_head=False).to('cpu').eval()
    lat = torch.randn(2, 197, 1024)
    with torch.no_grad():
        out0 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False).clone()
        for p in m.ordinal_head.parameters():
            p.add_(torch.randn_like(p) * 5.0)
        out1 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False)
    assert torch.allclose(out0, out1, atol=1e-6), (out0 - out1).abs().max()
    assert m._last_ordinal_logits is None, "OFF must never populate _last_ordinal_logits"
    P("ordinal-head OFF: output independent of ordinal_head weights (byte-identical to baseline path)")


def test_ordinal_head_off_matches_seeded_baseline():
    """Two models built with the SAME seed, one ordinal_head=True and one False, must have IDENTICAL
    weights in every OTHER module -- proving ordinal_head's construction (placed LAST in __init__)
    never perturbs the RNG stream consumed by earlier modules, so OFF is truly unaffected by this
    feature's mere existence in the code."""
    import training as T
    torch.manual_seed(7)
    m_off = T.PerPointVFModel(T.base_model, ordinal_head=False).to('cpu').eval()
    torch.manual_seed(7)
    m_on = T.PerPointVFModel(T.base_model, ordinal_head=True).to('cpu').eval()
    n_compared = 0
    for (n0, p0), (n1, p1) in zip(m_off.named_parameters(), m_on.named_parameters()):
        if n0.startswith('ordinal_head.') or n1.startswith('ordinal_head.'):
            continue
        assert n0 == n1, (n0, n1)
        assert torch.equal(p0, p1), f"{n0} diverged between ordinal_head=False/True (RNG-order leak)"
        n_compared += 1
    assert n_compared > 0
    P(f"ordinal_head=True/False builds are IDENTICAL in all {n_compared} non-ordinal_head "
      f"parameter tensors (no RNG-order leak)")


def test_ordinal_head_on_shape_and_range():
    """ON: decode_latent output stays (B,52), finite, and within the model's existing clamp range."""
    import training as T
    torch.manual_seed(1)
    m = T.PerPointVFModel(T.base_model, ordinal_head=True).to('cpu').eval()
    lat = torch.randn(3, 197, 1024)
    with torch.no_grad():
        out = m.decode_latent(lat, ['OD', 'OS', 'OD'], average_multi=False)
    assert out.shape == (3, 52), out.shape
    assert torch.isfinite(out).all()
    lo, hi = T.OUTLIER_CLIP_RANGE
    assert (out >= lo - 1e-4).all() and (out <= hi + 1e-4).all(), (out.min().item(), out.max().item())
    assert m._last_ordinal_logits.shape == (3, 52, T.ORDINAL_N_THRESH)
    P(f"ordinal-head ON: output (3,52) finite in [{out.min():.2f},{out.max():.2f}] "
      f"(clamp [{lo},{hi}]), logits shape {tuple(m._last_ordinal_logits.shape)}")


def test_coral_logits_rank_monotone():
    """CoralPointHead's K-1 logits must be non-increasing across the threshold index for ANY input
    -- the CORAL rank-consistency guarantee (shared trunk + strictly ordered biases), independent of
    what the trunk happens to output."""
    import training as T
    torch.manual_seed(2)
    head = T.CoralPointHead(input_dim=16, hidden=8, dropout=0.0, n_thresh=T.ORDINAL_N_THRESH).eval()
    x = torch.randn(5, 7, 16) * 3.0   # arbitrary batch/point/feature shape
    with torch.no_grad():
        logits = head(x)
    assert logits.shape == (5, 7, T.ORDINAL_N_THRESH)
    diffs = logits[..., 1:] - logits[..., :-1]
    assert (diffs <= 1e-5).all(), f"logits not rank-monotone, max increase={diffs.max().item()}"
    P("CORAL logits are rank-monotone (non-increasing across ordered thresholds) for arbitrary input")


def test_coral_decode_matches_closed_form():
    """coral_decode's expected-dB value must equal the closed form
    bin_width * (0.5 + sum(sigmoid(logits))) at the two extremes (all-0 logits -> mid-scale;
    saturated-positive logits -> the top-bin center)."""
    import training as T
    K = T.ORDINAL_N_THRESH
    zeros = torch.zeros(1, K)
    huge = torch.full((1, K), 50.0)
    lo = T.coral_decode(zeros)
    hi = T.coral_decode(huge)
    expect_lo = T.ORDINAL_BIN_WIDTH * (0.5 + 0.5 * K)     # sigmoid(0) == 0.5 everywhere
    expect_hi = T.ORDINAL_BIN_WIDTH * (0.5 + K)           # sigmoid(50) ~saturates to 1.0
    assert torch.allclose(lo, torch.tensor([[expect_lo]]), atol=1e-3), lo
    assert torch.allclose(hi, torch.tensor([[expect_hi]]), atol=1e-2), hi
    P(f"coral_decode closed-form: all-0 logits -> {lo.item():.2f} dB; saturated logits -> "
      f"{hi.item():.2f} dB (top bin center {T.ORDINAL_BIN_WIDTH * (T.ORDINAL_N_BINS - 0.5):.1f})")


def test_compute_loss_ordinal_gating_and_finite():
    """compute_loss: ordinal_logits=None (default / absent) must be byte-identical to the current
    Huber loss. With ordinal_logits/ordinal_cfg set, the loss is finite and MAE (computed purely from
    pred/target) is unaffected -- proving CCC/variance/bias/severity terms are untouched."""
    import training as T
    torch.manual_seed(3)
    B = 6
    pred = torch.rand(B, 52) * 30 + 4
    target = torch.full((B, 72), 99.0)
    for i in range(B):
        for j, vi in enumerate(T.valid_indices_od):
            target[i, vi] = pred[i, j].item() + float(torch.randn(1) * 2.0)
    lat = ['OD'] * B

    l_a, mae_a, n_a = T.compute_loss(pred, target, lat, epoch=10)
    l_b, mae_b, n_b = T.compute_loss(pred, target, lat, epoch=10, ordinal_logits=None, ordinal_cfg=None)
    assert abs(float(l_a) - float(l_b)) < 1e-9 and mae_a == mae_b and n_a == n_b, \
        "ordinal_logits=None/absent must not change the loss"

    logits = torch.randn(B, 52, T.ORDINAL_N_THRESH)
    cfg = {'bin_width': T.ORDINAL_BIN_WIDTH, 'n_thresh': T.ORDINAL_N_THRESH}
    l_c, mae_c, n_c = T.compute_loss(pred, target, lat, epoch=10, ordinal_logits=logits, ordinal_cfg=cfg)
    assert np.isfinite(float(l_c))
    assert n_c == n_a
    assert abs(mae_c - mae_a) < 1e-9, "MAE must be identical (it never looks at the loss term used)"
    assert abs(float(l_c) - float(l_a)) > 1e-6, "CORAL loss should differ from the Huber baseline"
    P(f"compute_loss: ordinal_logits=None byte-identical to baseline (loss {float(l_a):.4f}); "
      f"ON gives a finite CORAL loss {float(l_c):.4f}, MAE unaffected ({mae_c:.4f})")


def test_compute_loss_ordinal_additive_not_replacing():
    """Fix (final code review, task-11 follow-up): the ordinal ON-path must be an ADDITIVE
    auxiliary term (ordinal_weight * CORAL BCE) on top of the SAME Huber primary-fit term used by
    the OFF path -- NOT a replacement of it. A replacement would (a) starve the global-spatial/M1
    severity heads of gradient from the primary point-fit term (they only touch `pred`, the FINAL
    prediction, not the pre-decode logits) and (b) put a ~10-13 nat/point BCE on a completely
    different scale than the single-digit-dB^2 Huber it replaced, silently down-weighting every
    other pre-tuned aux term (CCC/variance/bias/severity/entropy) by ~10x.

    Proof:
      (1) ordinal_weight=0.0 ON reduces BYTE-IDENTICAL to the OFF/Huber-only loss on a fixed batch
          -- only possible if the Huber term is still the primary fit (a replacement would give a
          totally different loss value even at weight 0, since the "replaced" Huber would be gone).
      (2) The loss is LINEAR in ordinal_weight: loss(w) - loss(weight=0) scales proportionally
          with w for two different w's -- the signature of an ADDED term, not a substituted one.
    """
    import training as T
    torch.manual_seed(4)
    B = 6
    pred = torch.rand(B, 52) * 30 + 4
    target = torch.full((B, 72), 99.0)
    for i in range(B):
        for j, vi in enumerate(T.valid_indices_od):
            target[i, vi] = pred[i, j].item() + float(torch.randn(1) * 2.0)
    lat = ['OD'] * B
    logits = torch.randn(B, 52, T.ORDINAL_N_THRESH)

    l_off, mae_off, n_off = T.compute_loss(pred, target, lat, epoch=10)

    cfg0 = {'bin_width': T.ORDINAL_BIN_WIDTH, 'n_thresh': T.ORDINAL_N_THRESH, 'weight': 0.0}
    l_w0, mae_w0, n_w0 = T.compute_loss(pred, target, lat, epoch=10,
                                        ordinal_logits=logits, ordinal_cfg=cfg0)
    assert abs(float(l_off) - float(l_w0)) < 1e-6, \
        ("ordinal_weight=0.0 ON must reduce to the OFF/Huber-only loss -- proves the Huber "
         f"primary-fit term is still present when ordinal-head is ON (off={float(l_off):.6f} "
         f"w0={float(l_w0):.6f})")
    assert mae_w0 == mae_off and n_w0 == n_off

    cfg1 = {'bin_width': T.ORDINAL_BIN_WIDTH, 'n_thresh': T.ORDINAL_N_THRESH, 'weight': 0.3}
    cfg2 = {'bin_width': T.ORDINAL_BIN_WIDTH, 'n_thresh': T.ORDINAL_N_THRESH, 'weight': 0.6}
    l_w1, _, _ = T.compute_loss(pred, target, lat, epoch=10, ordinal_logits=logits, ordinal_cfg=cfg1)
    l_w2, _, _ = T.compute_loss(pred, target, lat, epoch=10, ordinal_logits=logits, ordinal_cfg=cfg2)
    assert np.isfinite(float(l_w1)) and np.isfinite(float(l_w2))
    d1 = float(l_w1) - float(l_w0)
    d2 = float(l_w2) - float(l_w0)
    assert d1 > 1e-6, "ordinal term at weight>0 must add positive loss on top of the Huber baseline"
    assert abs(d2 - 2 * d1) < 1e-4, \
        f"loss must be LINEAR in ordinal_weight (additive term), got d1={d1:.6f} d2={d2:.6f}"
    P(f"compute_loss ordinal ON is ADDITIVE: weight=0 == OFF loss ({float(l_off):.4f}); "
      f"loss linear in ordinal_weight (Δ@0.3={d1:.4f}, Δ@0.6={d2:.4f})")


if __name__ == "__main__":
    print("Task 11 (CORAL ordinal head) tests:")
    test_ordinal_head_off_is_noop()
    test_ordinal_head_off_matches_seeded_baseline()
    test_ordinal_head_on_shape_and_range()
    test_coral_logits_rank_monotone()
    test_coral_decode_matches_closed_form()
    test_compute_loss_ordinal_gating_and_finite()
    test_compute_loss_ordinal_additive_not_replacing()
    print("ALL PASSED")
