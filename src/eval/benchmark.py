"""Compare runs: parameter count, inference latency, and test metrics side by side."""
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.utils.io import read_yaml
from src.data.eurosat import build_dataloaders
from src.models.finetune import build_finetune_model
from src.models.scratch_cnn import ScratchCNN


def _build_model(config: Dict) -> torch.nn.Module:
    model_config = config["model"]
    if model_config["name"] == "scratch_cnn":
        from src.models.spectral import build_scratch_from_config
        return build_scratch_from_config(config)
    return build_finetune_model(
        name=model_config["name"],
        num_classes=config["data"]["num_classes"],
        pretrained=False,
    )


def benchmark_run(config_path: str, checkpoint_path: str, n_batches: int = 20) -> Dict:
    config = read_yaml(config_path)
    run_name = config.get("run_name", config["model"]["name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = _build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())

    test_loader = build_dataloaders(config)["test"]
    images = 0
    elapsed = 0.0
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            model(inputs)
            if device.type == "mps":
                torch.mps.synchronize()
            if i > 0:  # skip warmup batch
                elapsed += time.time() - t0
                images += inputs.size(0)

    metrics_path = Path("outputs/reports") / run_name / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    return {
        "run": run_name,
        "params": n_params,
        "images_per_sec": images / elapsed if elapsed > 0 else None,
        **metrics,
    }


def benchmark_table(entries: List[Dict]) -> str:
    header = "| Run | Params | Img/s | Accuracy | Macro F1 |"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]
    for e in entries:
        acc = f"{e.get('accuracy', float('nan')) * 100:.2f}%" if "accuracy" in e else "-"
        f1 = f"{e.get('f1_macro', float('nan')):.4f}" if "f1_macro" in e else "-"
        ips = f"{e['images_per_sec']:.0f}" if e.get("images_per_sec") else "-"
        rows.append(f"| {e['run']} | {e['params']:,} | {ips} | {acc} | {f1} |")
    return "\n".join(rows)


