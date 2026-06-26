"""Dataset for the longitudinal model. Each record (from grape_longitudinal.json) yields the
target fundus image + target VF (72-flat, masked=100) + the eye's most-recent causal prior VF
(52-d query order + mask) + Δt + has_prior. Train = 1 image; val = TTA stack.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T

MASKED = T.MASKED_VALUE_THRESHOLD


def prior_vec_mask(prior_hvf, laterality):
    """8x9 prior grid (or None) -> (vec52 query order with masked->0, mask 1/0)."""
    if prior_hvf is None:
        return np.zeros(52, np.float32), np.zeros(52, np.float32)
    v = np.array(prior_hvf, dtype=np.float32).flatten()
    vi = T.valid_indices_od if laterality.startswith("OD") else T.valid_indices_os
    v = v[vi]
    mask = (v < MASKED).astype(np.float32)
    v = np.where(mask > 0, v, 0.0).astype(np.float32)
    return v, mask


class LongitudinalDataset(Dataset):
    def __init__(self, json_path, mode="train", use_tta=False):
        self.data = json.load(open(json_path))
        self.mode = mode
        self.use_tta = use_tta
        n_prior = sum(r.get("has_prior", False) for r in self.data)
        print(f"  Longitudinal {mode}: {len(self.data)} records ({n_prior} with prior, "
              f"{len(self.data) - n_prior} visit-1)")

    def __len__(self):
        return len(self.data)

    def get_sample_severity(self):
        ws = []
        for r in self.data:
            w = 1.0 + T.WEIGHT_SCALE * (T.MAX_DB - r["mean_db"]) / T.MAX_DB
            ws.append(max(w, 1.0))
        return ws

    def __getitem__(self, idx):
        r = self.data[idx]
        lat = str(r["Laterality"]).strip().upper()
        pv, pm = prior_vec_mask(r.get("prior_hvf"), lat)
        dt = np.float32(r.get("delta_t", 0.0))
        hp = np.float32(1.0 if r.get("has_prior") else 0.0)
        hvf = np.array(r["hvf"], dtype=np.float32).flatten()
        img = Image.open(os.path.join(T.FUNDUS_DIR, r["FundusImage"][0])).convert("RGB")
        if self.mode == "train":
            x = T.train_transform(img)
        elif self.use_tta:
            x = torch.stack([t(img) for t in T.get_tta_transforms()])
        else:
            x = T.val_transform(img).unsqueeze(0)
        return (x, torch.tensor(hvf), lat, torch.tensor(pv), torch.tensor(pm),
                torch.tensor(dt), torch.tensor(hp))


def collate_train(batch):
    xs, hvf, lats, pv, pm, dt, hp = zip(*batch)
    return (torch.stack(xs), torch.stack(hvf), list(lats), torch.stack(pv),
            torch.stack(pm), torch.stack(dt), torch.stack(hp))


def collate_val(batch):
    return batch[0]   # batch_size=1; (xs[V,3,H,W], hvf[72], lat, pv[52], pm[52], dt, hp)
