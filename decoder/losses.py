"""Loss helpers that need no encoder (import-cheap, unit-testable).

balanced_mse_loss — the BMC (Batch-based Monte-Carlo) variant of Balanced MSE
(Ren et al., "Balanced MSE for Imbalanced Visual Regression", CVPR 2022). It treats the
batch's targets as the candidate set: each prediction must be *uniquely nearest* its own
target, which penalises the mean-collapse (variance shrinkage) that ordinary MSE rewards on
imbalanced regression. This raises the predicted-vs-true slope without a label prior.
"""
import torch
import torch.nn.functional as F


def balanced_mse_loss(pred, target, sigma=1.0, weights=None):
    """Balanced-MSE (BMC) over a flat batch of matched (pred, target) points.

    Args:
        pred:    1-D tensor (N,) of predictions (already masked to valid points).
        target:  1-D tensor (N,) of the matching targets, same order as ``pred``.
        sigma:   noise scale (same units as the target, e.g. dB). Larger σ ⇒ stronger
                 de-shrinkage on this formulation (σ scales with the target spread).
        weights: optional (N,) per-point weights (e.g. Garway-Heath sector weights);
                 applied to the per-point cross-entropy before averaging.

    Returns:
        Scalar loss = mean_i w_i · CE( -(pred_i - target_j)^2 / 2σ² , label=i ).
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    n = pred.shape[0]
    if n == 0:
        return pred.sum() * 0.0
    # logits[i, j] = -(pred_i - target_j)^2 / (2 σ²);  correct class for row i is i.
    logits = -(pred[:, None] - target[None, :]) ** 2 / (2.0 * sigma * sigma)
    idx = torch.arange(n, device=pred.device)
    ce = F.cross_entropy(logits, idx, reduction='none')   # (N,)
    if weights is not None:
        w = weights.reshape(-1).to(ce.dtype)
        return (ce * w).sum() / (w.sum() + 1e-8)
    return ce.mean()
