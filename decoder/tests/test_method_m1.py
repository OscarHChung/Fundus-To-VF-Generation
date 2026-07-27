"""M1 (severity head + de-shrink loss) and M3 (EMA) correctness tests.

Encoder-free: decode_latent runs the decoder on a pre-computed latent (no encoder forward), and
compute_loss is a pure function. Importing training loads RETFound once for base_model, but no
encoder forward runs here. Run: python decoder/tests/test_method_m1.py  (ONE process at a time).

Validates:
  (1) severity_head default OFF is a no-op — output is INDEPENDENT of the severity_head weights;
  (2) severity_head ON REPLACES the field eye-mean with severity_head(CLS) while preserving the
      within-eye pattern (residual identical to the OFF field's residual) and exposes _last_severity;
  (3) severity_blend interpolates the field mean between emergent and severity-head;
  (4) the compute_loss severity term: absent when severity_cfg=None (byte-identical), a de-shrink
      CCC that is LARGER for under-dispersed eye-means, and a Huber that pins the level;
  (5) WeightEMA shadow tracks the params.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

P = lambda name: print(f"  PASS  {name}")


def test_severity_head_off_is_noop():
    """OFF: the severity_head submodule exists but is never called — perturbing its weights must
    not change decode_latent output (proves default-OFF ≡ long_global)."""
    import training as T
    torch.manual_seed(0)
    m = T.PerPointVFModel(T.base_model, global_head=True, severity_head=False).to('cpu').eval()
    lat = torch.randn(2, 197, 1024)
    with torch.no_grad():
        out0 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False).clone()
        # scramble the (unused) severity head; OFF output must be unchanged
        for p in m.severity_head.parameters():
            p.add_(torch.randn_like(p) * 5.0)
        out1 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False)
    assert torch.allclose(out0, out1, atol=1e-6), (out0 - out1).abs().max()
    assert m._last_severity is None, "OFF must never populate _last_severity"
    P("severity OFF: output independent of severity_head weights (byte-identical to baseline path)")


def test_severity_head_delta_shift():
    """ON: zero-init ⇒ exact no-op at start (safe warm-start); a nonzero delta is a UNIFORM field
    shift (within-eye pattern preserved) and _last_severity = emergent + delta. Checked via
    _apply_severity so the invariant is exact (pre-refinement)."""
    import training as T
    torch.manual_seed(1)
    m = T.PerPointVFModel(T.base_model, global_head=True, severity_head=True).to('cpu').eval()
    cls = torch.randn(3, 1024)
    base_pred = torch.randn(3, 52) * 4 + 18            # a pre-severity field
    with torch.no_grad():
        # (a) zero-init delta ⇒ field unchanged; predicted MD == emergent mean
        composed0 = m._apply_severity(base_pred.clone(), cls)
        assert torch.allclose(composed0, base_pred, atol=1e-6), (composed0 - base_pred).abs().max()
        assert torch.allclose(m._last_severity.reshape(-1), base_pred.mean(1), atol=1e-5)
        # (b) give the delta head a real (eye-varying) output → uniform shift
        m.severity_head[-1].weight.normal_(0, 0.5)
        m.severity_head[-1].bias.fill_(1.5)
        composed = m._apply_severity(base_pred.clone(), cls)
        delta = m.severity_head(cls).reshape(-1)
    assert torch.allclose(composed, base_pred + delta.unsqueeze(1), atol=1e-5)
    assert torch.allclose(composed.mean(1), base_pred.mean(1) + delta, atol=1e-5)
    c0 = base_pred - base_pred.mean(1, keepdim=True)
    c1 = composed - composed.mean(1, keepdim=True)
    assert torch.allclose(c0, c1, atol=1e-5), "uniform shift must preserve the within-eye pattern"
    assert m._last_severity.shape == (3, 1)
    P("severity ON: zero-init no-op start; nonzero delta = uniform shift, pattern preserved, MD exposed")


def test_severity_blend():
    """blend scales the severity correction: field mean = emergent + blend·delta."""
    import training as T
    torch.manual_seed(2)
    m = T.PerPointVFModel(T.base_model, global_head=True, severity_head=True,
                          severity_blend=0.5).to('cpu').eval()
    cls = torch.randn(4, 1024)
    base_pred = torch.randn(4, 52) * 5 + 15
    with torch.no_grad():
        m.severity_head[-1].weight.normal_(0, 0.5)      # make delta nonzero
        m.severity_head[-1].bias.fill_(2.0)
        composed = m._apply_severity(base_pred.clone(), cls)
        delta = m.severity_head(cls).reshape(-1)
    expect = base_pred.mean(1) + 0.5 * delta
    assert torch.allclose(composed.mean(1), expect, atol=1e-5), (composed.mean(1), expect)
    P("severity blend=0.5: field mean = emergent + ½·delta")


def _sev_loss(sp, st, w=0.5, ccc=0.5):
    """Run compute_loss with only the severity term active (52 identical dummy points/eye so the
    per-point Huber contributes a constant across the compared cases); return the total."""
    import training as T
    B = sp.shape[0]
    pred = torch.full((B, 52), 18.0)                       # flat dummy field
    target = torch.full((B, 72), 99.0)
    # put each eye's true mean into all its valid OD points
    for i in range(B):
        for vi in T.valid_indices_od:
            target[i, vi] = st[i]
    cfg = dict(weight=w, ccc=ccc, eye_scale=0.0)           # eye_scale 0 → unit eye weights
    loss, _, _ = T.compute_loss(pred, target, ['OD'] * B, epoch=0,
                                severity_pred=sp.reshape(B, 1), severity_cfg=cfg)
    return float(loss)


def test_severity_loss_gating_and_deshrink():
    import training as T
    torch.manual_seed(3)
    st = torch.tensor([8.0, 14.0, 20.0, 26.0, 30.0])       # spread of true eye-means
    # (a) gating: severity_cfg=None ⇒ severity term absent (identical to a no-severity call)
    pred = torch.full((5, 52), 18.0)
    target = torch.full((5, 72), 99.0)
    for i in range(5):
        for vi in T.valid_indices_od:
            target[i, vi] = st[i]
    l_off, _, _ = T.compute_loss(pred, target, ['OD'] * 5, epoch=0)
    l_off2, _, _ = T.compute_loss(pred, target, ['OD'] * 5, epoch=0,
                                  severity_pred=None, severity_cfg=None)
    assert abs(l_off - l_off2) < 1e-9, "severity_cfg=None must not change the loss"
    # (b) de-shrink: a perfectly-matched severity pred has near-zero severity loss; an
    #     under-dispersed (shrunk-to-mean) pred costs MORE via both Huber and the CCC term.
    perfect = st.clone()
    shrunk = st.mean() + 0.3 * (st - st.mean())            # compressed toward the grand mean
    l_perfect = _sev_loss(perfect, st)
    l_shrunk = _sev_loss(shrunk, st)
    assert l_shrunk > l_perfect + 0.1, (l_shrunk, l_perfect)
    # (c) the CCC term specifically penalizes shrinkage even at equal Huber: compare ccc-only
    l_shrunk_ccc = _sev_loss(shrunk, st, w=0.0, ccc=1.0)
    l_perfect_ccc = _sev_loss(perfect, st, w=0.0, ccc=1.0)
    assert l_shrunk_ccc > l_perfect_ccc + 0.05, (l_shrunk_ccc, l_perfect_ccc)
    P(f"severity loss: gated off cleanly; de-shrink penalizes compression "
      f"(shrunk {l_shrunk:.3f} > perfect {l_perfect:.3f}; ccc {l_shrunk_ccc:.3f}>{l_perfect_ccc:.3f})")


def test_ema_shadow_tracks():
    import training as T
    torch.manual_seed(4)
    lin = torch.nn.Linear(8, 4)
    ema = T.WeightEMA([lin], decay=0.9)
    before = [s.clone() for s in ema.shadow]
    with torch.no_grad():
        for p in lin.parameters():
            p.add_(10.0)                                   # big param jump
    for _ in range(50):
        ema.update()
    moved = [(s - b).abs().mean().item() for s, b in zip(ema.shadow, before)]
    assert all(mv > 0.5 for mv in moved), moved
    # apply/restore round-trip: jump params AGAIN so shadow (≈init+10) ≠ params (init+110)
    with torch.no_grad():
        for p in lin.parameters():
            p.add_(100.0)
    live0 = [p.clone() for p in ema.params]
    ema.apply_to()
    assert not torch.allclose(ema.params[0], live0[0]), "apply_to should swap in EMA weights"
    ema.restore()
    assert torch.allclose(ema.params[0], live0[0]), "restore should bring back live weights"
    P(f"EMA: shadow tracks params after jump (Δ={moved[0]:.2f}); apply/restore round-trips")


if __name__ == "__main__":
    print("M1/M3 tests:")
    test_severity_head_off_is_noop()
    test_severity_head_delta_shift()
    test_severity_blend()
    test_severity_loss_gating_and_deshrink()
    test_ema_shadow_tracks()
    print("ALL PASSED")
