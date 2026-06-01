# LULC-Classification

Land Use and Land Cover (LULC) classification project for satellite imagery, focused on practical urban-planning and environmental-monitoring relevance for fast-growing regions such as Greater Accra.

## Project Goal
Build and compare three CNN-based approaches for 10-class land cover classification:
1. A CNN trained from scratch (baseline)
2. ResNet-50 fine-tuned from ImageNet
3. EfficientNet-B0 fine-tuned from ImageNet

## Dataset
- EuroSAT (27,000 labeled satellite images, 10 classes)
- Kaggle mirror: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
- Original dataset repo: https://github.com/phelber/EuroSAT
- Planned data loading utility: TorchGeo (https://torchgeo.readthedocs.io)

## References
- Torchvision pretrained model docs: https://pytorch.org/vision/stable/models.html
- Benchmark paper: https://arxiv.org/abs/1709.00029

## Team Roles (Pick One)
- Data and baseline: dataset setup, scratch CNN training, data section of report
- Model training: fine-tune ResNet-50 and EfficientNet-B0, run experiments, tune hyperparameters
- Evaluation and writing: metrics, Grad-CAM visualizations, plots, final report and presentation

## Selected Role for This Repo
Model training

## Timeline
- Finalize and submit as soon as possible.
- Stated deadline in project brief: Friday, May 29, 11:59 PM GMT.

## Expected Deliverables
- Reproducible training pipeline for all three models
- Comparative performance table versus benchmark values
- Evaluation outputs (metrics/plots, including interpretability visuals)
- Final written report and presentation materials

## Layout
```
.
├── configs/                # experiment configs
├── data/                   # raw and processed datasets
├── outputs/                # checkpoints, logs, plots
├── reports/                # report drafts and figures
├── scripts/                # CLI entry points
└── src/                    # core code
```

## Setup
1. Create and activate a virtual environment.
2. Install requirements: `pip install -r requirements.txt`.
3. Download EuroSAT to `data/raw/EuroSAT/` so each class folder is inside that directory.

## Notes
- The scripts and training pipeline are minimal placeholders so you can fill in the final logic quickly.
- Config files are in `configs/` and are meant to drive training, evaluation, and visualization.
