"""Build the combined master results table across all studies.

Reads every run's metrics.json + parameter count and writes a grouped markdown
table to outputs/reports/master_results.md. Independent of benchmark.py's
comparison.md (which gets overwritten per invocation).
"""
import json
import sys
from pathlib import Path

# Allow running as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_yaml
from src.models.spectral import build_scratch_from_config
from src.models.finetune import build_finetune_model

# (label, config, checkpoint dir under outputs/checkpoints|reports)
GROUPS = {
    "Distillation / capacity sweep (JPG-RGB, random 80/10/10 split)": [
        ("ScratchCNN 94K",            "configs/scratch_cnn_long.yaml",       "scratch_cnn_long"),
        ("ScratchCNN 94K + KD",       "configs/distill_scratch_cnn.yaml",    "scratch_cnn_distilled"),
        ("ScratchCNN-S 391K",         "configs/scratch_cnn_s_plain.yaml",    "scratch_cnn_s_plain"),
        ("ScratchCNN-S 391K + KD",    "configs/scratch_cnn_s_distill.yaml",  "scratch_cnn_s_distill"),
        ("ScratchCNN-M 1.56M",        "configs/scratch_cnn_m_plain.yaml",    "scratch_cnn_m_plain"),
        ("ScratchCNN-M 1.56M + KD",   "configs/scratch_cnn_m_distill.yaml",  "scratch_cnn_m_distill"),
        ("ResNet-18 (ImageNet)",      "configs/resnet18.yaml",               "resnet18"),
        ("EfficientNet-V2-S (teacher)", "configs/efficientnet_v2_s.yaml",    "efficientnet_v2_s"),
    ],
    "Multispectral input study (raw reflectance, official 60/20/20 split, fixed 1.56M model)": [
        ("RGB (3 bands)",             "configs/scratch_cnn_ms_rgb.yaml",     "scratch_cnn_ms_rgb"),
        ("All bands (12)",            "configs/scratch_cnn_ms_all.yaml",     "scratch_cnn_ms_all"),
        ("All bands + indices (15)",  "configs/scratch_cnn_ms_indices.yaml", "scratch_cnn_ms_indices"),
    ],
    "Task-optimized virtual sensors (learned k-channel spectral mixing, same model/split)": [
        ("k=1 learned channel",   "configs/scratch_cnn_ms_proj1.yaml", "scratch_cnn_ms_proj1"),
        ("k=2 learned channels",  "configs/scratch_cnn_ms_proj2.yaml", "scratch_cnn_ms_proj2"),
        ("k=3 learned channels",  "configs/scratch_cnn_ms_proj3.yaml", "scratch_cnn_ms_proj3"),
        ("k=4 learned channels",  "configs/scratch_cnn_ms_proj4.yaml", "scratch_cnn_ms_proj4"),
        ("k=6 learned channels",  "configs/scratch_cnn_ms_proj6.yaml", "scratch_cnn_ms_proj6"),
    ],
}


def param_count(config):
    mc = config["model"]
    if mc["name"] == "scratch_cnn":
        m = build_scratch_from_config(config)
    else:
        m = build_finetune_model(mc["name"], config["data"]["num_classes"], pretrained=False)
    return sum(p.numel() for p in m.parameters())


def main():
    lines = ["# Master results — EuroSAT land-use classification\n"]
    for group, runs in GROUPS.items():
        lines.append(f"## {group}\n")
        lines.append("| Model / input | Params | Accuracy | Macro F1 |")
        lines.append("|---|---|---|---|")
        for label, cfg_path, run in runs:
            metrics_path = Path("outputs/reports") / run / "metrics.json"
            if not metrics_path.exists():
                lines.append(f"| {label} | - | (missing) | - |")
                continue
            m = json.loads(metrics_path.read_text())
            n = param_count(read_yaml(cfg_path))
            lines.append(f"| {label} | {n:,} | {m['accuracy']*100:.2f}% | {m['f1_macro']:.4f} |")
        lines.append("")
    table = "\n".join(lines)
    out = Path("outputs/reports/master_results.md")
    out.write_text(table)
    print(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
