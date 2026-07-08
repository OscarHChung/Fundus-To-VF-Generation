#!/bin/zsh
# P1 disc-only, folds 1-4 (fold 0 already done). Serialized: ONE encoder process at a time
# (this box OOM-kills concurrent torch). Same recipe as m1sev + --disc-only (tight crop 0.27).
cd "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation"
export PYTORCH_ENABLE_MPS_FALLBACK=1
for f in 1 2 3 4; do
  echo "===== FOLD $f start $(date) =====" > decoder/results/auto/p1disc_f${f}.log
  python decoder/train_lora_cached.py \
    --train-json decoder/results/cv_long/fold${f}_train.json \
    --val-json   decoder/results/cv_long/fold${f}_val.json \
    --out-tag p1disc_f${f} --epochs 40 \
    --lora-rank 8 --lora-blocks 8 --lora-alpha 16 --lora-dropout 0.1 --lora-lr 2e-4 \
    --warm-start decoder/results/auto/long_global_f${f}_best.pth --select mae_slope \
    --severity-head --severity-weight 0.5 --severity-ccc 0.5 --severity-eye-scale 2.0 \
    --disc-only >> decoder/results/auto/p1disc_f${f}.log 2>&1
  echo "===== FOLD $f done $(date) =====" >> decoder/results/auto/p1disc_f${f}.log
done
echo "ALL FOLDS DONE $(date)" > decoder/results/auto/p1disc_folds_complete.flag
