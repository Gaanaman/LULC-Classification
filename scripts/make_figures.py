"""Generate write-up figures for both innovations.

Outputs to outputs/reports/figures/:
  - pareto_accuracy_vs_params.png   (distillation / capacity sweep)
  - multispectral_per_class_f1.png  (RGB vs all-bands per-class F1)
  - confusion_<run>.png             (multispectral runs)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plots import plot_confusion_matrix

FIG_DIR = Path("outputs/reports/figures")
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def _acc(run):
    return json.loads((Path("outputs/reports") / run / "metrics.json").read_text())["accuracy"] * 100


def pareto_plot():
    # (label, params, run, marker)
    pts = [
        ("ScratchCNN 94K", 94_762, "scratch_cnn_long"),
        ("ScratchCNN-S 391K", 391_466, "scratch_cnn_s_plain"),
        ("ScratchCNN-M 1.56M", 1_557_066, "scratch_cnn_m_plain"),
        ("ResNet-18 11M", 11_181_642, "resnet18"),
        ("EffNet-V2-S 20M", 20_190_298, "efficientnet_v2_s"),
    ]
    xs = [p for _, p, _ in pts]
    ys = [_acc(r) for _, _, r in pts]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ys, "o-", color="#2b6cb0", markersize=9)
    # highlight the multispectral 1.56M result as a star
    ms_acc = _acc("scratch_cnn_ms_all")
    ax.scatter([1_562_250], [ms_acc], marker="*", s=420, color="#dd6b20", zorder=5,
               label=f"ScratchCNN-M + 12-band ({ms_acc:.1f}%)")
    for (label, p, _), y in zip(pts, ys):
        ax.annotate(f"{label}\n{y:.1f}%", (p, y), textcoords="offset points",
                    xytext=(8, -18 if "94K" in label else 8), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. model size on EuroSAT\n(multispectral input lets a 1.56M model rival 11–20M pretrained backbones)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "pareto_accuracy_vs_params.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def multispectral_plot():
    rgb = json.loads(Path("outputs/reports/scratch_cnn_ms_rgb/classification_report.json").read_text())
    allb = json.loads(Path("outputs/reports/scratch_cnn_ms_all/classification_report.json").read_text())
    rgb_f1 = [rgb[c]["f1-score"] for c in CLASSES]
    all_f1 = [allb[c]["f1-score"] for c in CLASSES]
    order = np.argsort([a - r for r, a in zip(rgb_f1, all_f1)])[::-1]
    classes = [CLASSES[i] for i in order]
    rgb_f1 = [rgb_f1[i] for i in order]
    all_f1 = [all_f1[i] for i in order]

    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, rgb_f1, w, label="RGB (3 bands)", color="#a0aec0")
    ax.bar(x + w / 2, all_f1, w, label="All bands (12)", color="#dd6b20")
    for i, (r, a) in enumerate(zip(rgb_f1, all_f1)):
        d = a - r
        if d > 0.003:
            ax.annotate(f"+{d:.3f}", (i + w / 2, a), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color="#276749")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0.93, 1.005)
    ax.set_title("Multispectral bands rescue the weak classes\n(same 1.56M model; RGB 97.6% → all-bands 98.5%)")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "multispectral_per_class_f1.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def virtual_sensor_curve():
    ks = [1, 2, 3, 4, 6]
    accs = [_acc(f"scratch_cnn_ms_proj{k}") for k in ks]
    rgb = _acc("scratch_cnn_ms_rgb")
    full = _acc("scratch_cnn_ms_all")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, accs, "o-", color="#dd6b20", markersize=9, label="Learned k-channel virtual sensor")
    ax.axhline(full, color="#2b6cb0", ls="--", lw=1.5, label=f"All 12 raw bands ({full:.2f}%)")
    ax.scatter([3], [rgb], marker="s", s=110, color="#718096", zorder=5,
               label=f"Fixed RGB, 3 channels ({rgb:.2f}%)")
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.2f}%", (k, a), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.set_xlabel("Spectral channel budget k")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(ks)
    ax.set_title("Spectral rate–distortion: accuracy vs. learned channel budget\n"
                 "(2 learned channels beat RGB; 6 match all 12 bands; same 1.56M model)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "virtual_sensor_rate_distortion.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def response_function_heatmap():
    import torch
    # Kept-band order (torchgeo order minus B10/cirrus) and center wavelengths (nm).
    bands = ["B01\n443", "B02\n490", "B03\n560", "B04\n665", "B05\n705", "B06\n740",
             "B07\n783", "B08\n842", "B09\n945", "B11\n1610", "B12\n2190", "B8A\n865"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), sharey=False)
    for ax, k in zip(axes, [1, 2, 3]):
        sd = torch.load(f"outputs/checkpoints/scratch_cnn_ms_proj{k}/best_model.pth",
                        map_location="cpu", weights_only=True)
        w = sd["projection.mix.weight"].squeeze(-1).squeeze(-1).numpy()  # [k, 12]
        # sign/scale-normalize each channel for display (direction chosen so max |w| is positive)
        for i in range(w.shape[0]):
            if abs(w[i].min()) > abs(w[i].max()):
                w[i] = -w[i]
            w[i] = w[i] / np.abs(w[i]).max()
        im = ax.imshow(w, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(bands, fontsize=7)
        ax.set_yticks(range(w.shape[0]))
        ax.set_yticklabels([f"ch{i+1}" for i in range(w.shape[0])], fontsize=9)
        ax.set_title(f"k={k} virtual sensor", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="normalized weight")
    fig.suptitle("Learned spectral response functions (weights over standardized bands)", y=1.04)
    out = FIG_DIR / "virtual_sensor_response_functions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pareto_plot()
    multispectral_plot()
    virtual_sensor_curve()
    response_function_heatmap()
    for run in ["scratch_cnn_ms_rgb", "scratch_cnn_ms_all", "scratch_cnn_ms_indices"]:
        cm = Path("outputs/reports") / run / "confusion_matrix.npy"
        if cm.exists():
            plot_confusion_matrix(str(cm), str(FIG_DIR), CLASSES)
            (FIG_DIR / "confusion_matrix.png").rename(FIG_DIR / f"confusion_{run}.png")
    print("Figures complete.")


if __name__ == "__main__":
    main()
