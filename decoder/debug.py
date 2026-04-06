"""
debug.py — Diagnose why training is not reaching sub-3.74 dB MAE.

Run this BEFORE starting a training run, or paste epoch logs into
the prompts below to get targeted fixes.

Usage:
    python debug.py
    python debug.py --log path/to/log.txt   (parse a saved log)
"""

import os, sys, json, numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

# ── Resolve paths ──────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE'))

CHECKPOINT_PATH    = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
TRAIN_JSON         = os.path.join(CURRENT_DIR, "..", "data", "vf_tests", "grape_train.json")
FUNDUS_DIR         = os.path.join(CURRENT_DIR, "..", "data", "fundus", "grape_fundus_images")

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

issues   = []
warnings = []

# ==============================================================
# CHECK 1 — Files exist
# ==============================================================
print("\n" + "="*60)
print("CHECK 1: Required files")
print("="*60)

for label, path in [
    ("RETFound weights",     CHECKPOINT_PATH),
    ("Pretrained decoder",   PRETRAINED_DECODER),
    ("Train JSON",           TRAIN_JSON),
    ("Fundus image dir",     FUNDUS_DIR),
]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1e6
        print(f"  {PASS} {label}: {path}  ({size:.1f} MB)")
    else:
        print(f"  {FAIL} {label} MISSING: {path}")
        issues.append(f"Missing file: {label} at {path}")

# ==============================================================
# CHECK 2 — Pretrained decoder quality
# ==============================================================
print("\n" + "="*60)
print("CHECK 2: Pretrained decoder")
print("="*60)

if os.path.exists(PRETRAINED_DECODER):
    try:
        with torch.serialization.safe_globals([np.dtype]):
            dec_ckpt = torch.load(PRETRAINED_DECODER, map_location='cpu', weights_only=False)
        dec_mae = dec_ckpt.get('val_mae', None)
        if dec_mae is not None:
            print(f"  {PASS} Decoder val MAE: {dec_mae:.2f} dB")
            if dec_mae > 4.0:
                warnings.append(
                    f"Decoder pretrained MAE is {dec_mae:.2f} dB — quite high. "
                    "A decoder pretrained below 3 dB would give a better starting point."
                )
                print(f"  {WARN} MAE > 4.0 dB — decoder may not be providing useful signal")
        else:
            print(f"  {WARN} val_mae key not found in decoder checkpoint")
            warnings.append("Decoder checkpoint has no val_mae key — cannot verify quality.")
    except Exception as e:
        print(f"  {FAIL} Could not load decoder: {e}")
        issues.append(f"Decoder load error: {e}")
else:
    print(f"  {FAIL} Decoder file not found")
    issues.append("Pretrained decoder missing — model will train from scratch with no warm start on refinement.")

# ==============================================================
# CHECK 3 — Dataset statistics
# ==============================================================
print("\n" + "="*60)
print("CHECK 3: Dataset statistics")
print("="*60)

try:
    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    all_hvf    = []
    lat_counts = {'OD': 0, 'OS': 0, 'OTHER': 0}
    missing_imgs = 0

    for item in train_data:
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        hvf_valid = hvf[hvf < 99.0]
        all_hvf.extend(hvf_valid.tolist())

        lat = item.get('Laterality', 'OD').strip().upper()
        if lat.startswith('OD'):
            lat_counts['OD'] += 1
        elif lat.startswith('OS'):
            lat_counts['OS'] += 1
        else:
            lat_counts['OTHER'] += 1

        images = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
        for img in images:
            full_path = os.path.join(FUNDUS_DIR, img)
            if not os.path.exists(full_path):
                missing_imgs += 1

    all_hvf = np.array(all_hvf)
    print(f"  {PASS} Training eyes: {len(train_data)}")
    print(f"  {PASS} Laterality split — OD: {lat_counts['OD']}, OS: {lat_counts['OS']}, Other: {lat_counts['OTHER']}")
    print(f"  {PASS} VF sensitivity range: {all_hvf.min():.1f} – {all_hvf.max():.1f} dB")
    print(f"  {PASS} VF mean: {all_hvf.mean():.1f} dB  |  std: {all_hvf.std():.1f} dB")

    low_pct = (all_hvf < 10).mean() * 100
    print(f"  {PASS} Points < 10 dB (scotoma): {low_pct:.1f}%")

    if all_hvf.mean() < 15 or all_hvf.mean() > 22:
        warnings.append(
            f"VF mean is {all_hvf.mean():.1f} dB — far from the assumed 18 dB warm-start bias. "
            f"Update PROJ_INIT_BIAS in train.py to {all_hvf.mean():.0f} for better initialisation."
        )
        print(f"  {WARN} Mean {all_hvf.mean():.1f} dB deviates from PROJ_INIT_BIAS=18. "
              f"Consider setting PROJ_INIT_BIAS={all_hvf.mean():.0f}")

    if missing_imgs > 0:
        issues.append(f"{missing_imgs} image file(s) listed in JSON not found on disk.")
        print(f"  {FAIL} {missing_imgs} image files MISSING on disk")
    else:
        print(f"  {PASS} All image files found on disk")

    if len(train_data) < 150:
        warnings.append(
            f"Only {len(train_data)} training eyes — very small. "
            "LoRA rank-8 may still overfit; consider reducing to rank-4."
        )
        print(f"  {WARN} Small dataset ({len(train_data)} eyes) — LoRA rank-8 may overfit")

except Exception as e:
    print(f"  {FAIL} Dataset check failed: {e}")
    issues.append(f"Dataset error: {e}")

# ==============================================================
# CHECK 4 — Encoder forward pass + LoRA injection
# ==============================================================
print("\n" + "="*60)
print("CHECK 4: Encoder + LoRA forward pass")
print("="*60)

try:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    from models_mae import mae_vit_large_patch16_dec512d8b
    import math

    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    enc = mae_vit_large_patch16_dec512d8b()
    enc.load_state_dict(ckpt['model'], strict=False)
    enc.eval()

    # Inject LoRA
    lora_params = []
    for block in enc.blocks:
        attn = block.attn
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            lin = attn.qkv
            rank, alpha = 8, 16
            scale = alpha / rank
            in_f, out_f = lin.in_features, lin.out_features

            class _LoRA(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = lin
                    self.lora_A = nn.Parameter(torch.empty(rank, in_f))
                    self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                    self.linear.weight.requires_grad = False
                    if self.linear.bias is not None:
                        self.linear.bias.requires_grad = False
                def forward(self, x):
                    return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * scale

            lora_layer  = _LoRA()
            attn.qkv    = lora_layer
            lora_params += [lora_layer.lora_A, lora_layer.lora_B]

    n_lora = len(lora_params) // 2
    total_trainable = sum(p.numel() for p in lora_params)
    print(f"  {PASS} LoRA injected into {n_lora} blocks")
    print(f"  {PASS} LoRA trainable params: {total_trainable:,}")

    enc.to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = enc.forward_encoder(dummy, mask_ratio=0.0)[0]
        cls = out[:, 0, :]
    print(f"  {PASS} Encoder forward pass OK — CLS token shape: {cls.shape}")

    if cls.shape[-1] != 1024:
        issues.append(f"Encoder CLS token dim is {cls.shape[-1]}, expected 1024. Projection head dim wrong.")
        print(f"  {FAIL} CLS dim {cls.shape[-1]} != 1024")

except Exception as e:
    print(f"  {FAIL} Encoder/LoRA check failed: {e}")
    issues.append(f"Encoder forward pass error: {e}")

# ==============================================================
# CHECK 5 — Projection warm-start sanity
# ==============================================================
print("\n" + "="*60)
print("CHECK 5: Projection warm-start output")
print("="*60)

try:
    proj = nn.Sequential(
        nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
        nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(),
        nn.Linear(256, 52)
    )
    final = proj[-1]
    nn.init.constant_(final.bias, 18.0)
    nn.init.normal_(final.weight, mean=0.0, std=0.01)

    proj.eval()
    with torch.no_grad():
        dummy_feat = torch.randn(4, 1024)
        out        = proj(dummy_feat)

    mean_out = out.mean().item()
    std_out  = out.std().item()
    print(f"  {PASS} Projection output mean: {mean_out:.2f} dB  (target: ~18 dB)")
    print(f"  {PASS} Projection output std:  {std_out:.2f} dB")

    if abs(mean_out - 18.0) > 3.0:
        issues.append(
            f"Warm-start not working — projection output mean {mean_out:.1f} dB, expected ~18 dB. "
            "Check that final linear layer init is applied AFTER the Sequential is built."
        )
        print(f"  {FAIL} Mean {mean_out:.1f} is far from 18 dB — warm-start failed")

except Exception as e:
    print(f"  {FAIL} Projection check failed: {e}")
    issues.append(f"Projection warm-start error: {e}")

# ==============================================================
# CHECK 6 — Log analyser (optional)
# ==============================================================
print("\n" + "="*60)
print("CHECK 6: Log analysis")
print("="*60)

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default=None)
args, _ = parser.parse_known_args()

if args.log and os.path.exists(args.log):
    with open(args.log) as f:
        lines = f.readlines()

    epochs, val_maes, train_maes, val_corrs, train_corrs = [], [], [], [], []
    for line in lines:
        if line.strip().startswith('Train:'):
            parts = line.split('|')
            try:
                train_maes.append(float(parts[0].split(':')[1].strip().split(' ')[0]))
                train_corrs.append(float(parts[1].strip().split(':')[1].strip()))
            except:
                pass
        if line.strip().startswith('Val:') and 'Gap' in line:
            parts = line.split('|')
            try:
                val_maes.append(float(parts[0].split(':')[1].strip().split(' ')[0]))
                val_corrs.append(float(parts[1].strip().split(':')[1].strip()))
            except:
                pass

    if val_maes:
        print(f"  Parsed {len(val_maes)} val checkpoints from log")
        print(f"  Best val MAE: {min(val_maes):.2f} dB at checkpoint {np.argmin(val_maes)+1}")
        print(f"  Final val MAE: {val_maes[-1]:.2f} dB")
        print(f"  Final val Corr: {val_corrs[-1]:.3f}")

        # Diagnose
        if len(val_corrs) >= 3 and val_corrs[2] < 0.35:
            issues.append(
                "Correlation < 0.35 at epoch 15. LoRA is likely not adapting. "
                "Try increasing LORA_RANK to 16, or raise LORA LR from 0.3x to 0.5x BASE_LR."
            )
            print(f"  {FAIL} Corr < 0.35 at epoch 15 — LoRA not adapting")

        gaps = [t - v for t, v in zip(train_maes, val_maes)]
        if any(g < -0.5 for g in gaps):
            worst = min(gaps)
            issues.append(
                f"Overfitting detected (gap reached {worst:.2f} dB). "
                "Increase WEIGHT_DECAY from 5e-4 to 1e-3, or reduce LORA_RANK to 4."
            )
            print(f"  {FAIL} Overfitting: worst gap = {worst:.2f} dB")

        if min(val_maes) > 3.74:
            delta = min(val_maes) - 3.74
            print(f"  {WARN} Best val MAE {min(val_maes):.2f} dB still above 3.74 baseline by {delta:.2f} dB")
            if val_corrs[-1] > 0.6:
                warnings.append(
                    "Correlation is good (>0.6) but MAE is still above baseline. "
                    "The model captures spatial structure but overestimates sensitivity. "
                    "Try removing label smoothing (set LABEL_SMOOTH=0.0) — it may be biasing predictions toward the mean."
                )
                print(f"  {WARN} High corr but high MAE — label smoothing may be biasing predictions upward")
    else:
        print(f"  No [Epoch N] blocks found in log — check format")
else:
    print(f"  No log file provided. Run with --log path/to/log.txt to analyse a run.")

# ==============================================================
# SUMMARY
# ==============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if not issues and not warnings:
    print(f"  {PASS} All checks passed — environment looks correct.")
    print(f"  {PASS} Ready to run train.py")
else:
    if issues:
        print(f"\n  {FAIL} ISSUES ({len(issues)}) — must fix before training:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    if warnings:
        print(f"\n  {WARN} WARNINGS ({len(warnings)}) — consider fixing:")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")

print("\n" + "="*60)
print("QUICK REFERENCE — what to change if training stalls")
print("="*60)
print("""
  Symptom                          Fix
  ─────────────────────────────────────────────────────────────
  Epoch 1 MAE > 10 dB             Warm-start not applied.
                                   Check PROJ_INIT_BIAS=18 and
                                   that init runs after Sequential.

  Corr flat at 0.15-0.25          LoRA not adapting.
  through epoch 15                 Try LORA_RANK=16 or raise
                                   LoRA LR from 0.3x to 0.5x.

  Val gap worse than -0.5 dB      Overfitting.
  after epoch 20                   Raise WEIGHT_DECAY to 1e-3
                                   or lower LORA_RANK to 4.

  Corr good (>0.6) but MAE        Label smoothing biasing
  stuck above baseline             predictions. Set
                                   LABEL_SMOOTH=0.0.

  Val MAE bouncing ±0.5 each      Projection LR too high.
  epoch, not converging            Lower BASE_LR from 5e-4
                                   to 2e-4.

  Train MAE << val MAE            Encoder LoRA overfitting.
  (gap > 1.0 dB)                   Lower LORA_RANK to 4.

  Val MAE plateaus at 4.3-4.5     Decoder warm-start not
  and never improves               helping. Check decoder
                                   loads correctly (should
                                   print 2.09 dB on startup).
""")