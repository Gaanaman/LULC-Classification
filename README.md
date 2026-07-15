# LULC-Classification

Land use / land cover classification on EuroSAT (Sentinel-2, 10 classes),
studying what actually drives accuracy in the 98%+ regime: model capacity,
input spectrum, or training signal. Role for this repo: model training.

## Studies in this repo

1. **Capacity / distillation sweep** — scratch CNNs at 94K / 391K / 1.56M
   parameters, each with and without knowledge distillation from a fine-tuned
   EfficientNet-V2-S teacher. Finding: accuracy tracks capacity; distillation
   from the 20M teacher adds nothing at any student size.
2. **Multispectral input** — the same 1.56M model on RGB vs. all 12 Sentinel-2
   bands vs. bands + spectral indices (NDVI/NDWI/NDBI). Finding: 12 raw bands
   reach 98.5% (matching ResNet-18); indices add nothing beyond raw bands.
3. **Task-optimized virtual sensors** — a learnable 1x1 spectral mixing layer
   compresses the 12 bands into k channels, swept k=1..6. Finding: two learned
   channels beat 3-channel RGB; six match all 12 bands.

Baselines: scratch CNN, ResNet-18, EfficientNet-V2-S (fine-tuned from ImageNet,
98.6% / 98.9% test accuracy). Combined results in
`outputs/reports/master_results.md`; figures in `outputs/reports/figures/`.

## Dataset

- EuroSAT, 27,000 labeled images, 10 classes.
- RGB version auto-downloads via torchvision to `data/raw/EuroSAT/` on first run.
- Multispectral (13-band GeoTIFF) via torchgeo to `data/raw/EuroSATMS/` (~2 GB).
- Original dataset: https://github.com/phelber/EuroSAT (Helber et al., 2019).

The dataset and checkpoints are gitignored — this repo holds code and
lightweight result tables/figures. Re-running the commands below regenerates
everything else.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the studies

All commands run from this directory. Configs in `configs/` are the single
source of truth; `run_name` in each config sets its output directory.

```bash
# Baselines
python scripts/train.py    --config configs/resnet18.yaml
python scripts/evaluate.py --config configs/resnet18.yaml --ckpt outputs/checkpoints/resnet18/best_model.pth

# Study 1: distillation (needs a trained teacher checkpoint)
python scripts/train.py    --config configs/scratch_cnn_m_distill.yaml

# Study 2: multispectral
python scripts/train.py    --config configs/scratch_cnn_ms_all.yaml

# Study 3: virtual sensor (k learned channels)
python scripts/train.py    --config configs/scratch_cnn_ms_proj3.yaml

# Reporting
python scripts/benchmark.py    --runs <config.yaml>:<ckpt.pth> ...   # params/speed/acc/F1
python scripts/master_table.py                                       # combined results
python scripts/ms_compare.py                                         # per-class F1 across MS inputs
python scripts/make_figures.py                                       # figures for the report
```

## Layout

```
configs/    experiment configs (one per run)
scripts/    thin CLI wrappers -> functions in src/
src/
  data/         eurosat.py (RGB), eurosat_ms.py (13-band via torchgeo)
  models/       scratch_cnn.py, finetune.py, spectral.py (virtual sensor)
  training/     train.py (train_from_config, optional KD)
  eval/         evaluate.py, benchmark.py, metrics.py
  visualization/ plots.py, grad_cam.py
outputs/    checkpoints, reports, figures, logs (mostly gitignored)
```

## Notes for collaborators

- Apple Silicon (MPS): launch long runs with an absolute interpreter path and
  wrap in `caffeinate -dims`; scripts that spawn DataLoaders need a `__main__`
  guard or `num_workers=0`.
- The RGB studies use a random 80/10/10 split (seed 42); the multispectral
  studies use torchgeo's official 60/20/20 split. Keep splits matched when
  comparing across the two.
