#!/bin/bash
# Matched extra seeds for the RGB vs k=2 comparison.
# Logs per-run so progress is visible while it runs.
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
for cfg in configs/seeds/*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "[$(date +%H:%M:%S)] START $name"
  $PY -u scripts/train.py --config "$cfg" 2>&1 | grep -E "^Epoch|Train Loss" | tail -2
  $PY -u scripts/evaluate.py --config "$cfg" \
      --ckpt "outputs/checkpoints/${name}/best_model.pth" 2>&1 | grep -E "accuracy|f1_macro"
  echo "[$(date +%H:%M:%S)] DONE  $name"
done
echo "SWEEP_COMPLETE $(date +%H:%M:%S)"
