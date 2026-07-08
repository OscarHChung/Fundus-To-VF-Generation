"""M2 (fundus→RNFL structural-surrogate aux head) correctness tests.

Encoder-free: decode_latent runs the decoder on a pre-computed latent (no encoder forward).
Importing training loads RETFound once for base_model, but no encoder forward runs here.
Run: python decoder/tests_method_m2.py  (ONE process at a time).

The core guarantee: the aux head is TRAIN-ONLY and NEVER touches `pred`, so inference stays
fundus-only and byte-identical whether or not it exists. Validates:
  (1) rnfl_aux OFF ≡ current model — no rnfl_head is built and _last_rnfl stays None;
  (2) rnfl_aux ON populates _last_rnfl (B,5) BUT perturbing rnfl_head weights leaves `pred`
      byte-identical (the aux head is a pure side-output, not in the prediction path);
  (3) the masked aux loss: a fully-masked batch contributes exactly 0 (and 0 gradient to the
      aux head), while an unmasked batch drives a real gradient into rnfl_head.
"""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

P = lambda name: print(f"  PASS  {name}")


def test_rnfl_off_is_noop():
    """OFF: no rnfl_head is built; decode_latent leaves _last_rnfl None; output is deterministic."""
    import training as T
    torch.manual_seed(0)
    m = T.PerPointVFModel(T.base_model, global_head=True, rnfl_aux=False).to('cpu').eval()
    assert m.use_rnfl_aux is False
    assert not hasattr(m, 'rnfl_head'), "rnfl_head must NOT be built when OFF (≡ current model)"
    lat = torch.randn(2, 197, 1024)
    with torch.no_grad():
        out0 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False).clone()
        out1 = m.decode_latent(lat, ['OD', 'OS'], average_multi=False)
    assert m._last_rnfl is None, "OFF must never populate _last_rnfl"
    assert torch.allclose(out0, out1, atol=1e-6)
    P("rnfl_aux OFF: no head built, _last_rnfl stays None, output deterministic (≡ current)")


def test_rnfl_on_does_not_touch_pred():
    """ON: _last_rnfl is (B,5); scrambling rnfl_head weights changes _last_rnfl but NOT pred —
    proves the aux head is a pure side-output (inference stays fundus-only & identical)."""
    import training as T
    torch.manual_seed(1)
    m = T.PerPointVFModel(T.base_model, global_head=True, rnfl_aux=True).to('cpu').eval()
    assert hasattr(m, 'rnfl_head')
    lat = torch.randn(3, 197, 1024)
    with torch.no_grad():
        pred0 = m.decode_latent(lat, ['OD', 'OS', 'OD'], average_multi=False).clone()
        rnfl0 = m._last_rnfl.clone()
        assert rnfl0.shape == (3, 5), rnfl0.shape
        # scramble ONLY the aux head; pred must be unchanged, _last_rnfl must change
        for p in m.rnfl_head.parameters():
            p.add_(torch.randn_like(p) * 5.0)
        pred1 = m.decode_latent(lat, ['OD', 'OS', 'OD'], average_multi=False)
        rnfl1 = m._last_rnfl
    assert torch.allclose(pred0, pred1, atol=1e-6), (pred0 - pred1).abs().max()
    assert not torch.allclose(rnfl0, rnfl1), "aux-head weights must change _last_rnfl"
    P("rnfl_aux ON: _last_rnfl (B,5) exposed; pred byte-identical under aux-head perturbation")


def _masked_aux_loss(pred_rnfl, target_rnfl, mask):
    """Replicates the trainer's masked RNFL aux loss (train_lora_cached.py)."""
    aux = F.smooth_l1_loss(pred_rnfl, target_rnfl, reduction='none').mean(dim=1)   # (B,)
    return (aux * mask).sum() / mask.sum().clamp_min(1.0)


def test_masked_aux_loss_and_grad():
    """A fully-masked batch → 0 aux loss and 0 aux-head gradient; an unmasked batch → real grad."""
    import training as T
    torch.manual_seed(2)
    m = T.PerPointVFModel(T.base_model, global_head=True, rnfl_aux=True).to('cpu').train()
    lat = torch.randn(4, 197, 1024)
    target = torch.randn(4, 5)

    # (a) fully masked → loss exactly 0, no gradient reaches rnfl_head
    m.zero_grad()
    m.decode_latent(lat, ['OD'] * 4, average_multi=False)
    loss0 = _masked_aux_loss(m._last_rnfl, target, torch.zeros(4))
    assert float(loss0) == 0.0, float(loss0)
    loss0.backward()
    g_masked = m.rnfl_head[-1].weight.grad
    assert g_masked is None or g_masked.abs().sum() == 0, "masked batch must give 0 aux grad"

    # (b) unmasked → positive loss and a real gradient into the aux head
    m.zero_grad()
    m.decode_latent(lat, ['OD'] * 4, average_multi=False)
    loss1 = _masked_aux_loss(m._last_rnfl, target, torch.ones(4))
    assert float(loss1) > 0
    loss1.backward()
    g = m.rnfl_head[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "unmasked batch must drive aux-head gradient"
    # partial mask = mean over the in-mask rows only
    m.eval()
    with torch.no_grad():
        pr = m.rnfl_head(torch.randn(4, 1024))
    lp = _masked_aux_loss(pr, target, torch.tensor([1., 0., 1., 0.]))
    per = F.smooth_l1_loss(pr, target, reduction='none').mean(dim=1)
    assert abs(float(lp) - float((per[0] + per[2]) / 2)) < 1e-5
    P(f"masked aux loss: full-mask=0 (no grad); unmask drives grad; partial = in-mask mean")


if __name__ == "__main__":
    print("M2 tests:")
    test_rnfl_off_is_noop()
    test_rnfl_on_does_not_touch_pred()
    test_masked_aux_loss_and_grad()
    print("ALL PASSED")
