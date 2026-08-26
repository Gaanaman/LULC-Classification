#!/bin/bash
# Runs after the seed sweep: 10-band surface-only ablation.
# Tests whether the 12-band gain depends on the atmospheric bands (B01, B09).
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
while pgrep -f "configs/seeds" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] seed sweep finished; starting surface ablation"
$PY -u scripts/train.py --config configs/scratch_cnn_ms_surface.yaml 2>&1 | grep -E "^Epoch|Train Loss" | tail -2
$PY -u scripts/evaluate.py --config configs/scratch_cnn_ms_surface.yaml \
    --ckpt outputs/checkpoints/scratch_cnn_ms_surface/best_model.pth 2>&1 | grep -E "accuracy|f1_macro"
echo "[$(date +%H:%M:%S)] ABLATION_COMPLETE"
