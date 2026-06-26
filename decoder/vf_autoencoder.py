"""Bottleneck VF autoencoder = a learned manifold of realistic 24-2 fields.

Unlike decoder/pretrained_vf_decoder.pth (a 52->...->52 denoiser trained with ZEROS at masked
points, no latent), this is a true bottleneck AE with a non-zeroed input (mean-impute + a binary
valid-mask channel) so it can ENCODE a partial/observed prior field into a compact latent z and
INFILL missing points. Used by the longitudinal model to embed the eye's prior VF.

  python decoder/vf_autoencoder.py            # trains on UWHVF -> decoder/pretrained_vf_ae.pth

Reuse at inference:
  ae = load_ae('decoder/pretrained_vf_ae.pth')        # frozen
  z   = ae.encode(v_imp, mask)                          # (B,52),(B,52) -> (B,d)
  rec = ae.decode(z)                                    # (B,d) -> (B,52)
All vectors are in 52-d OD/OS QUERY order (same as diagnostics.vec52 / training).
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

CUR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(CUR, "..")
UWHVF_JSON = os.path.join(BASE, "data", "vf_tests", "uwhvf_vf_tests_standardized.json")
OUT = os.path.join(CUR, "pretrained_vf_ae.pth")
MASKED = 99.0
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]], dtype=bool)
valid_indices_od = [i for i, v in enumerate(mask_OD.flatten()) if v]
valid_indices_os = list(reversed(valid_indices_od))


def vec52(record):
    hvf = np.array(record["hvf"], dtype=np.float32).flatten()
    lat = str(record.get("Laterality", record.get("laterality", "OD"))).strip().upper()
    vi = valid_indices_od if lat.startswith("OD") else valid_indices_os
    v = hvf[vi].astype(np.float32)
    v[v >= MASKED] = np.nan
    return v


# ============== model ==============
class VFEncoder(nn.Module):
    def __init__(self, d=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(104, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, d))

    def forward(self, v_imp, mask):           # (B,52),(B,52)
        return self.net(torch.cat([v_imp, mask], dim=-1))


class VFDecoder(nn.Module):
    def __init__(self, d=64, drop=0.1, bias_init=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 52))
        if bias_init is not None:
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.as_tensor(bias_init, dtype=torch.float32))

    def forward(self, z):
        return self.net(z)


class VFAutoencoder(nn.Module):
    def __init__(self, d=64, point_mean=None):
        super().__init__()
        self.d = d
        self.enc = VFEncoder(d)
        self.dec = VFDecoder(d, bias_init=point_mean)
        self.register_buffer("point_mean", torch.as_tensor(
            point_mean if point_mean is not None else np.zeros(52), dtype=torch.float32))

    def encode(self, v_imp, mask):
        return self.enc(v_imp, mask)

    def decode(self, z):
        return self.dec(z)

    def forward(self, v_imp, mask):
        z = self.enc(v_imp, mask)
        return self.dec(z), z


def load_ae(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    ae = VFAutoencoder(d=ck["d_latent"], point_mean=ck["point_mean"])
    ae.load_state_dict(ck["state"])
    return ae


# ============== data ==============
class UWHVFManifold(Dataset):
    """Returns (v_imp, in_mask, clean, valid_mask): in_mask/v_imp are the OBSERVED (post-dropout)
    input; clean+valid_mask are the full field for the reconstruction/infill target."""
    def __init__(self, json_path, point_mean, drop_max=0.4):
        data = json.load(open(json_path))
        self.recs = data if isinstance(data, list) else list(data.values())
        self.point_mean = point_mean.astype(np.float32)
        self.drop_max = drop_max

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        v = vec52(self.recs[i])                          # (52,) nan at masked
        valid = ~np.isnan(v)
        clean = np.where(valid, v, self.point_mean).astype(np.float32)
        # observed = valid points minus a random dropout (simulate a partial prior)
        obs = valid.copy()
        vi = np.where(valid)[0]
        if len(vi) > 1:
            k = int(np.random.rand() * self.drop_max * len(vi))
            if k > 0:
                obs[np.random.choice(vi, k, replace=False)] = False
        v_imp = np.where(obs, clean, self.point_mean).astype(np.float32)
        return (torch.from_numpy(v_imp), torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(clean), torch.from_numpy(valid.astype(np.float32)))


def compute_point_mean(json_path):
    recs = json.load(open(json_path))
    recs = recs if isinstance(recs, list) else list(recs.values())
    acc = np.zeros(52); cnt = np.zeros(52)
    for r in recs:
        v = vec52(r); m = ~np.isnan(v)
        acc[m] += v[m]; cnt[m] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def masked_huber(pred, target, mask, delta=1.0):
    e = (pred - target).abs()
    h = torch.where(e < delta, 0.5 * e * e, delta * (e - 0.5 * delta))
    return (h * mask).sum() / mask.sum().clamp(min=1)


def main():
    torch.manual_seed(42); np.random.seed(42)
    d_latent = 64
    pm = compute_point_mean(UWHVF_JSON)
    ds = UWHVFManifold(UWHVF_JSON, pm)
    n_val = max(1, int(0.1 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=512, shuffle=False, num_workers=0)
    ae = VFAutoencoder(d=d_latent, point_mean=pm).to(DEVICE)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-5)
    print(f"UWHVF manifold AE | {len(ds)} fields | d={d_latent} | dev {DEVICE}")
    best = float("inf")
    for ep in range(80):
        ae.train()
        for v_imp, in_m, clean, valid in tl:
            v_imp, in_m, clean, valid = [t.to(DEVICE) for t in (v_imp, in_m, clean, valid)]
            rec, z = ae(v_imp, in_m)
            loss = masked_huber(rec, clean, valid) + 0.3 * (((rec - clean) ** 2) * valid).sum() / valid.sum() \
                + 1e-4 * (z ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0); opt.step()
        sched.step()
        # val: recon MAE on observed pts + infill MAE on dropped-but-valid pts
        ae.eval(); obs_e = obs_n = inf_e = inf_n = 0.0
        with torch.no_grad():
            for v_imp, in_m, clean, valid in vl:
                v_imp, in_m, clean, valid = [t.to(DEVICE) for t in (v_imp, in_m, clean, valid)]
                rec, _ = ae(v_imp, in_m)
                ae_err = (rec - clean).abs()
                obs_e += (ae_err * in_m).sum().item(); obs_n += in_m.sum().item()
                dropped = valid * (1 - in_m)
                inf_e += (ae_err * dropped).sum().item(); inf_n += dropped.sum().item()
        obs_mae = obs_e / max(obs_n, 1); inf_mae = inf_e / max(inf_n, 1)
        if ep % 10 == 0 or ep == 79:
            print(f"  ep{ep:3d} | recon(observed) {obs_mae:.3f} | infill(dropped) {inf_mae:.3f}")
        score = obs_mae + 0.5 * inf_mae
        if score < best:
            best = score
            torch.save({"state": ae.state_dict(), "d_latent": d_latent,
                        "point_mean": pm, "config": {"drop_max": 0.4}}, OUT)
    print(f"saved -> {OUT} (best recon+0.5infill {best:.3f})")


if __name__ == "__main__":
    main()
