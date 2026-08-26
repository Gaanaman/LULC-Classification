# LULC-Classification

Land use and land cover classification on EuroSAT (Sentinel-2, 10 classes). The
question is what drives accuracy once a benchmark sits above 98%: model
capacity, the input spectrum, or the training signal. The backbone family is
held fixed and one factor is varied at a time.

## Experiments

**1. Capacity and distillation.** Scratch CNNs at 0.094M / 0.39M / 1.56M
parameters, each with and without knowledge distillation from a fine-tuned
EfficientNet-V2-S teacher. Accuracy rises with model scale (92.26% to 96.41%).
Distillation changes accuracy by -0.82%, +0.18% and +0.07% at the three sizes,
so it does not help at any student size.

**2. Input bands.** The same 1.56M model on RGB, on all 12 non-cirrus Sentinel-2
bands, and on bands plus spectral indices (NDVI, NDWI, NDBI). RGB gives 97.59%,
12 bands give 98.50%, indices give 98.44%. A 10-band ablation dropping the
atmospheric bands B01 and B09 gives 98.52%, so the gain does not depend on them.

**3. Learned spectral projection.** A 1x1 convolution maps the 12 bands to k
channels, trained end to end, swept over k = 1, 2, 3, 4, 6. Three learned
channels give 98.17% against 97.59% for three-channel RGB. Six give 98.48%,
within 0.02% of all twelve bands. RGB and k=2 were repeated over three seeds:
97.54 +/- 0.06% against 97.96 +/- 0.12%, with no overlap between the runs.

Reference points: ResNet-18 and EfficientNet-V2-S fine-tuned from ImageNet reach
98.63% and 98.93%. Combined results are in `outputs/reports/master_results.md`,
figures in `outputs/reports/figures/`.

## Dataset

EuroSAT, 27,000 labelled 64x64 patches, 10 classes.

- RGB release downloads via torchvision to `data/raw/EuroSAT/` on first run.
- 13-band release downloads via torchgeo to `data/raw/EuroSATMS/` (about 2 GB).
- Source: https://github.com/phelber/EuroSAT (Helber et al., 2019).

RGB runs use a random 80/10/10 split at seed 42. Multispectral runs use
torchgeo's official 60/20/20 split, so RGB results are compared only within an
experiment.

`data/`, `outputs/checkpoints/` and `models/` are gitignored. The repository
holds code plus the result tables and figures.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Configs in `configs/` are the single source of truth; `run_name` in each config
sets its output directory.

```bash
python scripts/train.py    --config configs/resnet18.yaml
python scripts/evaluate.py --config configs/resnet18.yaml \
                           --ckpt outputs/checkpoints/resnet18/best_model.pth
```

Analysis, once the runs exist:

```bash
python scripts/master_table.py   # combined results table
python scripts/ms_compare.py     # per-class F1 across band configurations
python scripts/make_figures.py   # figures
python scripts/mcnemar.py        # paired significance tests
python scripts/seed_stats.py     # multi-seed means and Welch test
python scripts/benchmark.py --runs configs/resnet18.yaml:<ckpt> ...
```

Multi-seed and ablation runs:

```bash
bash scripts/run_seed_sweep.sh   # RGB and k=2 at seeds 43, 44
bash scripts/run_ablation.sh     # 10-band surface-only ablation
```

Set `PYTHON` to pick an interpreter, for example `PYTHON=python3.12 bash
scripts/run_seed_sweep.sh`.

## Interactive explorer

```bash
streamlit run app.py
```

Loads the saved metrics and confusion matrices and lets you step through the
three experiments, the per-class errors and the learned spectral weights.

## Layout

```
src/data/        EuroSAT loaders, RGB and 13-band
src/models/      ScratchCNN, fine-tuned backbones, spectral projection
src/training/    training loop, optional distillation
src/eval/        evaluation, metrics, throughput benchmark
src/visualization/ plots
scripts/         thin CLI wrappers, one per task
configs/         one YAML per run
```

## Notes

Training used the MPS backend on Apple silicon; a 40-epoch run of the 1.56M
model takes about 32 minutes. Checkpoints are plain state dicts, loaded with
`torch.load(..., weights_only=True)`. Scripts that spawn DataLoader workers need
a `__main__` guard or `num_workers=0` on macOS.
