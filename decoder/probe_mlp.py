"""Nonlinear (MLP) probe on the CACHED frozen features — decoder-vs-encoder bottleneck test.

The linear probe capped within-eye eyeCorr at ~0.46 (≈ the population template). Does a
properly-regularized NONLINEAR readout do better? If an MLP on the same frozen features also
caps ~0.46, the FROZEN FEATURES are the bottleneck → encoder adaptation is the justified lever.
If it reaches ~0.6+, the decoder/training is at fault → fixable with no encoder change.

Runs on CPU (tiny MLP, cached features) so it doesn't contend with a training run on MPS.
Reuses /tmp/grape_probe_features.pt (built by probe_features.py).
"""
import os, sys
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D

torch.manual_seed(0); np.random.seed(0)
CACHE = "/tmp/grape_probe_features.pt"
DEV = torch.device("cpu")


def feat_allpatch(f):   # best linear feature: mean-pooled patches ‖ CLS  (2048)
    return np.concatenate([f['patch'].mean(0), f['cls']])


def feat_combined(f):   # full-image + disc + multi-layer CLS (7168) — max frozen signal
    layers = np.concatenate([f['layers'][b] for b in sorted(f['layers'])])
    return np.concatenate([f['patch'].mean(0), f['cls'], f['disc'].mean(0), layers])


class MLP(nn.Module):
    def __init__(self, din, hidden=256, dropout=0.5, bias_init=20.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 52))
        nn.init.constant_(self.net[-1].bias, bias_init)        # start at the population mean
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        return self.net(x)


def _masked_row_std(pred, Mt):
    n = Mt.sum(1, keepdim=True).clamp(min=1)
    mean = (pred * Mt).sum(1, keepdim=True) / n
    var = ((pred - mean) ** 2 * Mt).sum(1, keepdim=True) / n
    return var.clamp(min=1e-6).sqrt().squeeze(1)


def train_mlp(Xtr, Ytr, Xva, dispersion=0.0, epochs=400, lr=2e-3, wd=2e-3, batch=32):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xall = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xva  = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    Yall = torch.tensor(np.nan_to_num(Ytr), dtype=torch.float32)
    Mall = torch.tensor(~np.isnan(Ytr), dtype=torch.float32)
    Tsig = torch.tensor([np.nanstd(row) for row in Ytr], dtype=torch.float32)
    n = len(Xall); k = max(8, int(0.15 * n)); idx = torch.randperm(n)
    va_i, tr_i = idx[:k], idx[k:]
    model = MLP(Xall.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(perm), batch):
            bi = perm[b:b + batch]
            pred = model(Xall[bi]); Mb = Mall[bi]
            diff = (pred - Yall[bi]) * Mb
            huber = (torch.where(diff.abs() < 1, 0.5 * diff ** 2, diff.abs() - 0.5).sum()
                     / Mb.sum().clamp(min=1))
            loss = huber
            if dispersion > 0:
                loss = loss + dispersion * ((_masked_row_std(pred, Mb) - Tsig[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            d = (model(Xall[va_i]) - Yall[va_i]) * Mall[va_i]
            vloss = (d.abs().sum() / Mall[va_i].sum()).item()
        if vloss < best - 1e-3:
            best, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xva).numpy()


def run(featfn=feat_allpatch, dispersion=0.0):
    feats = torch.load(CACHE, weights_only=False)
    recs = D.load_records()
    folds = D.build_patient_folds(recs, k=5)
    preds, trues = [], []
    for j, val_idx in enumerate(folds):
        val = [feats[i] for i in val_idx]
        train = [feats[i] for jj, f in enumerate(folds) if jj != j for i in f]
        Xtr = np.vstack([featfn(f) for f in train]); Ytr = np.vstack([f['y'] for f in train])
        Xva = np.vstack([featfn(f) for f in val])
        P = train_mlp(Xtr, Ytr, Xva, dispersion=dispersion)
        for f, p in zip(val, P):
            preds.append(p); trues.append(f['y'])
    return D.pooled_metrics(preds, trues)


if __name__ == "__main__":
    print("Nonlinear MLP probe on frozen features (per-patient 5-fold OOF):")
    print(f"  ALLPATCH (2048)   {D.fmt(run(feat_allpatch))}")
    print(f"  COMBINED (7168)   {D.fmt(run(feat_combined))}   [full+disc+multilayer]")
    print("\nrefs: linear ALLPATCH eyeCorr 0.461 ; template 0.488 ; in-pipeline model eyeCorr 0.42.")
    print("Read: COMBINED eyeCorr ≫0.51 ⇒ more frozen signal to exploit (richer decoder input). "
          "COMBINED ≈0.51 ⇒ frozen features capped → ENCODER LEVER required for sub-3.74.")
