import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

def plot_history(history_path: str, output_dir: str):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.title("Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    out_path = Path(output_dir) / "training_history.png"
    plt.savefig(out_path)
    print(f"Saved training history plot to {out_path}")

def plot_confusion_matrix(cm_path: str, output_dir: str, class_names: list):
    cm = np.load(cm_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix plot to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, help="Path to history.json")
    parser.add_argument("--cm", type=str, help="Path to confusion_matrix.npy")
    parser.add_argument("--out", type=str, default="outputs/reports", help="Output directory")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.history:
        plot_history(args.history, args.out)

    if args.cm:
        class_names = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]
        plot_confusion_matrix(args.cm, args.out, class_names)
