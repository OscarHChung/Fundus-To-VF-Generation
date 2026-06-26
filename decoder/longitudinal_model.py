"""LongitudinalVFModel — predicts a visit's VF from fundus_t + the eye's prior VF + interval.

Subclasses PerPointVFModel (frozen RETFound + Garway-Heath per-point attention + global head, all
reused). Two prediction paths, selected per-record by `has_prior`:
  * has_prior : pred = prior_field + alpha(interval) * delta(fundus, prior_latent, interval)
                with delta zero-initialised -> starts EXACTLY at the prior field (persistence 3.185)
                and can only earn corrections. prior_field = observed prior values (AE-infilled where
                the prior was masked).
  * visit-1   : pred = the existing absolute fundus prediction (point+global head) -> the ~4.29 model.
The frozen VF autoencoder embeds the prior field into a manifold latent z_prior and infills/decodes it.
All 52-vectors are OD/OS query order (compute_loss / pooled_metrics apply unchanged).
"""
import os, sys, math
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T
from vf_autoencoder import load_ae

NUM = T.NUM_VALID_POINTS  # 52


class IntervalEmbed(nn.Module):
    """Δt (years since prior) -> (B, d) embedding. Fourier features (multi-scale) + raw + has_prior."""
    def __init__(self, d=64, n_freq=8):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(0.5), math.log(8.0), n_freq))
        self.register_buffer("freqs", freqs)
        self.net = nn.Sequential(nn.Linear(2 * n_freq + 3, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, dt, has_prior):           # (B,),(B,)
        a = dt[:, None] * self.freqs[None, :] * 2 * math.pi          # (B,n_freq)
        feats = torch.cat([torch.sin(a), torch.cos(a),
                           dt[:, None], torch.log1p(dt.clamp(min=0))[:, None],
                           has_prior[:, None]], dim=-1)
        return self.net(feats)


class LongitudinalVFModel(T.PerPointVFModel):
    def __init__(self, encoder, ae_path, d_int=64, global_head=True, **kw):
        super().__init__(encoder, global_head=global_head, **kw)
        ae = load_ae(ae_path)
        for p in ae.parameters():
            p.requires_grad = False
        self.ae = ae
        self.d_latent = ae.d
        self.interval = IntervalEmbed(d_int)
        self.alpha = nn.Linear(d_int, 1)            # scalar delta gate
        nn.init.zeros_(self.alpha.weight); nn.init.constant_(self.alpha.bias, -2.0)  # start small
        self.no_prior_token = nn.Parameter(torch.zeros(self.d_latent))
        in_dim = self.embed_dim * 2 + self.d_latent + d_int        # attended ‖ cls ‖ z_prior ‖ ie
        self.delta_head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(T.HEAD_DROPOUT),
            nn.Linear(256, 1))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        self._last_pred_fundus = None
        print("✓ Longitudinal head ON (prior-VF residual on persistence; zero-init delta; "
              f"interval-gated). AE latent d={self.d_latent}, frozen.")

    def _prior_pieces(self, prior_vec, prior_mask, dt, has_prior):
        B = prior_vec.shape[0]
        v_imp = torch.where(prior_mask.bool(), prior_vec,
                            self.ae.point_mean.to(prior_vec.device)[None, :].expand(B, -1))
        z_obs = self.ae.encode(v_imp, prior_mask)                              # (B,d)
        z_prior = torch.where(has_prior[:, None].bool(), z_obs,
                              self.no_prior_token[None, :].expand(B, -1))
        ie = self.interval(dt, has_prior)                                      # (B,d_int)
        decoded = self.ae.decode(z_prior)                                      # (B,52)
        # observed prior values kept exact; masked points (and visit-1) use AE decode
        prior_field = torch.where((prior_mask.bool()) & (has_prior[:, None].bool()),
                                  prior_vec, decoded)
        return z_prior, ie, prior_field

    def _long_decode(self, latent, prior_vec, prior_mask, dt, has_prior, laterality):
        B = latent.shape[0]
        cls = latent[:, 0, :]; patches = latent[:, 1:, :]
        attended, attn_w = self.attention(patches, laterality)                # (B,52,1024)
        pred_fundus = self._apply_heads(attended, cls, B)
        if self.use_global_head:
            pred_fundus = pred_fundus + self._global_residual(cls, patches)
        self._last_pred_fundus = pred_fundus
        z_prior, ie, prior_field = self._prior_pieces(prior_vec, prior_mask, dt, has_prior)
        h = torch.cat([attended,
                       cls[:, None, :].expand(B, NUM, -1),
                       z_prior[:, None, :].expand(B, NUM, -1),
                       ie[:, None, :].expand(B, NUM, -1)], dim=-1)             # (B,52,in_dim)
        delta = torch.sigmoid(self.alpha(ie)) * self.delta_head(h).squeeze(-1)  # (B,52)
        self._last_delta = delta * has_prior[:, None]
        pred_long = prior_field + delta
        pred = torch.where(has_prior[:, None].bool(), pred_long, pred_fundus)
        self._last_attn_weights = attn_w
        return pred

    def forward(self, x, prior_vec, prior_mask, dt, has_prior, laterality='OD', average_multi=True):
        latent = self._encode(x)
        pred = self._long_decode(latent, prior_vec, prior_mask, dt, has_prior, laterality)
        return self._finish(pred, average_multi)

    def decode_latent_long(self, latent, prior_vec, prior_mask, dt, has_prior,
                           laterality='OD', average_multi=True):
        pred = self._long_decode(latent, prior_vec, prior_mask, dt, has_prior, laterality)
        return self._finish(pred, average_multi)
