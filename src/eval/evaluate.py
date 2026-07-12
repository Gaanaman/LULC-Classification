import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.utils.io import read_yaml
from src.data.eurosat import build_dataloaders
from src.models.finetune import build_finetune_model
from src.models.scratch_cnn import ScratchCNN
from src.eval.metrics import compute_classification_metrics


def evaluate_from_config(config_path: str, checkpoint_path: str) -> None:
    config = read_yaml(config_path)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloaders = build_dataloaders(config)
    test_loader = dataloaders["test"]
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
        "Industrial", "Pasture", "PermanentCrop", "Residential", 
        "River", "SeaLake"
    ] # Standard EuroSAT class names in alphabetical order (ImageFolder default)

    # Model
    model_config = config["model"]
    if model_config["name"] == "scratch_cnn":
        from src.models.spectral import build_scratch_from_config
        model = build_scratch_from_config(config)
    else:
        model = build_finetune_model(
            name=model_config["name"], 
            num_classes=config["data"]["num_classes"],
            pretrained=False
        )
    
    # Load weights
    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    
    # Inference
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Metrics
    metrics = compute_classification_metrics(all_targets, all_preds)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Save outputs (run_name keeps e.g. distilled runs separate from baselines)
    out_dir = Path("outputs/reports") / config.get("run_name", model_config["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    np.save(out_dir / "confusion_matrix.npy", cm)
    
    print(f"Evaluation complete. Reports saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()
    evaluate_from_config(args.config, args.checkpoint)
