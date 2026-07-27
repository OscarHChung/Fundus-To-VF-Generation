#!/bin/zsh
# Train the 5 CV folds of the reported model (p1disc_denoise) — or any variant of the recipe.
#
#   ./scripts/train_folds.sh                          # reported model, folds 0-4
#   ./scripts/train_folds.sh p1disc ""                # the p1disc comparator (no denoising)
#   ./scripts/train_folds.sh mytag "--disc-jitter" 2  # custom variant, folds 2-4
#
# Recipe = disc-crop-as-sole-view (--disc-only, tight crop 0.27) + severity head, LoRA r8 over the
# last 8 blocks, warm-started per fold from the fundus-only base (long_global_f{f}); the reported
# model adds --denoise-target (Theil-Sen per-point trajectory targets, TRAIN-ONLY — validation is
# always scored against the raw HVF).
#
# PREREQUISITE: the per-fold warm-start bases decoder/results/auto/long_global_f{0..4}_best.pth.
# They are NOT in the repo (~6 GB, pruned). Rebuild them first:
#   python decoder/run_cv.py --tag long_global --cv-dir decoder/results/cv_long --epochs 60 -- \
#     --weighting garway_heath --sector-combine sector_only --reweight value \
#     --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head
#
# Runs SERIALLY on purpose: this box OOM-kills concurrent torch processes.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

TAG="${1:-p1disc_denoise}"
EXTRA="${2:---denoise-target}"
FIRST="${3:-0}"
LOGS=decoder/results/auto/logs
mkdir -p "$LOGS"

for f in $(seq "$FIRST" 4); do
  WARM=decoder/results/auto/long_global_f${f}_best.pth
  if [[ ! -f $WARM ]]; then
    echo "MISSING warm-start $WARM — see the prerequisite block at the top of this script." >&2
    exit 1
  fi
  echo "===== FOLD $f start $(date) =====" > "$LOGS/${TAG}_f${f}.log"
  python decoder/train_lora_cached.py \
    --train-json decoder/results/cv_long/fold${f}_train.json \
    --val-json   decoder/results/cv_long/fold${f}_val.json \
    --out-tag ${TAG}_f${f} --epochs 40 \
    --lora-rank 8 --lora-blocks 8 --lora-alpha 16 --lora-dropout 0.1 --lora-lr 2e-4 \
    --warm-start "$WARM" --select mae_slope \
    --severity-head --severity-weight 0.5 --severity-ccc 0.5 --severity-eye-scale 2.0 \
    --disc-only ${=EXTRA} >> "$LOGS/${TAG}_f${f}.log" 2>&1
  echo "===== FOLD $f done $(date) =====" >> "$LOGS/${TAG}_f${f}.log"
done

echo "All folds done. Score them with:"
echo "  python decoder/eval_oof_cached.py --tag ${TAG} --no-tta"
echo "  python decoder/composition_report.py --tag ${TAG}"
