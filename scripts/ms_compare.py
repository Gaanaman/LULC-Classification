"""Per-class F1 comparison across the three multispectral input representations.

Reads the classification reports produced by evaluate.py for ms_rgb / ms_all /
ms_indices and writes a per-class table with deltas to outputs/reports/ms_comparison.md.
"""
import json
from pathlib import Path

RUNS = [
    ("RGB (3ch)", "scratch_cnn_ms_rgb"),
    ("All bands (12ch)", "scratch_cnn_ms_all"),
    ("Bands+indices (15ch)", "scratch_cnn_ms_indices"),
]
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def main():
    reports = {}
    for label, run in RUNS:
        p = Path("outputs/reports") / run / "classification_report.json"
        if not p.exists():
            print(f"missing: {p}")
            return
        reports[label] = json.loads(p.read_text())

    labels = [l for l, _ in RUNS]
    lines = []
    header = "| Class | " + " | ".join(labels) + " | Δ(indices−rgb) |"
    lines.append(header)
    lines.append("|" + "---|" * (len(labels) + 2))
    for c in CLASSES:
        f1s = [reports[l][c]["f1-score"] for l in labels]
        delta = f1s[-1] - f1s[0]
        row = f"| {c} | " + " | ".join(f"{v:.3f}" for v in f1s) + f" | {delta:+.3f} |"
        lines.append(row)
    # macro row
    macro = [reports[l]["macro avg"]["f1-score"] for l in labels]
    lines.append(f"| **macro F1** | " + " | ".join(f"**{v:.3f}**" for v in macro)
                 + f" | **{macro[-1]-macro[0]:+.3f}** |")
    acc = [reports[l]["accuracy"] for l in labels]
    lines.append(f"| **accuracy** | " + " | ".join(f"**{v*100:.2f}%**" for v in acc)
                 + f" | **{(acc[-1]-acc[0])*100:+.2f}pt** |")

    table = "\n".join(lines)
    print("\n" + table)
    out = Path("outputs/reports/ms_comparison.md")
    out.write_text(table + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
