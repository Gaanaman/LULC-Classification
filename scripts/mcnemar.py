"""Paired McNemar tests between runs that share the 60/20/20 test split."""
import sys, itertools, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.stats import binomtest
from src.utils.io import read_yaml
from src.data.eurosat import build_dataloaders
from src.models.spectral import build_scratch_from_config

RUNS = {"RGB": "scratch_cnn_ms_rgb", "k=2": "scratch_cnn_ms_proj2",
        "k=3": "scratch_cnn_ms_proj3", "12 bands": "scratch_cnn_ms_all"}

def predictions(cfg_name, run):
    cfg = read_yaml(f"configs/{cfg_name}.yaml")
    cfg["data"]["num_workers"] = 0
    model = build_scratch_from_config(cfg)
    model.load_state_dict(torch.load(f"outputs/checkpoints/{run}/best_model.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    correct, loader = [], build_dataloaders(cfg)["test"]
    with torch.no_grad():
        for x, y in loader:
            correct.append((model(x).argmax(1) == y).numpy())
    return np.concatenate(correct)

if __name__ == "__main__":
    corr = {}
    for label, run in RUNS.items():
        corr[label] = predictions(run, run)
        print(f"{label:9s} acc {corr[label].mean()*100:.2f}%  n={len(corr[label])}")
    print("\nMcNemar, exact binomial on discordant pairs:")
    rows = []
    for a, b in [("k=2","RGB"), ("k=3","RGB"), ("12 bands","RGB"), ("k=3","k=2"), ("12 bands","k=3")]:
        A, B = corr[a], corr[b]
        n01 = int((~A & B).sum())   # b right, a wrong
        n10 = int((A & ~B).sum())   # a right, b wrong
        p = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
        sig = "yes" if p < 0.05 else "no"
        print(f"  {a:9s} vs {b:9s}  {a} only right {n10:3d} | {b} only right {n01:3d} "
              f"| p={p:.4f} | significant at .05: {sig}")
        rows.append({"a": a, "b": b, "n10": n10, "n01": n01, "p": round(p, 5)})
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open("outputs/reports/mcnemar.json", "w"), indent=2)
    print("\nwrote outputs/reports/mcnemar.json")
