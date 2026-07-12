import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import json

from src.utils.io import read_yaml
from src.utils.seed import set_seed
from src.data.eurosat import build_dataloaders
from src.models.finetune import build_finetune_model
from src.models.scratch_cnn import ScratchCNN
from src.eval.metrics import compute_classification_metrics


def train_from_config(config_path: str) -> None:
    config = read_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloaders = build_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Model
    model_config = config["model"]
    if model_config["name"] == "scratch_cnn":
        from src.models.spectral import build_scratch_from_config
        model = build_scratch_from_config(config)
    else:
        model = build_finetune_model(
            name=model_config["name"], 
            num_classes=config["data"]["num_classes"],
            pretrained=model_config.get("pretrained", True)
        )
    model = model.to(device)

    # Optional knowledge-distillation teacher (frozen, eval-only).
    # Assumes teacher and student take the same input channels/normalization
    # (the distill configs pair an RGB teacher with an RGB student).
    distill_config = config.get("distill")
    teacher = None
    if distill_config is not None:
        teacher = build_finetune_model(
            name=distill_config["teacher_name"],
            num_classes=config["data"]["num_classes"],
            pretrained=False,
        )
        teacher.load_state_dict(
            torch.load(distill_config["teacher_ckpt"], map_location=device, weights_only=True)
        )
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        temperature = float(distill_config.get("temperature", 4.0))
        alpha = float(distill_config.get("alpha", 0.7))
        teacher_image_size = int(distill_config.get("teacher_image_size", 224))
        print(f"Distilling from {distill_config['teacher_name']} (T={temperature}, alpha={alpha})")

    # Training Setup
    train_config = config["train"]
    epochs = train_config.get("epochs", 15)
    lr = train_config.get("lr", 3e-4)
    weight_decay = train_config.get("weight_decay", 1e-4)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = None
    if train_config.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Checkpointing (run_name keeps e.g. distilled runs separate from baselines)
    run_name = config.get("run_name", model_config["name"])
    out_dir = Path("outputs/checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    
    # History
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if teacher is not None:
                # Teacher sees the same augmented view, upsampled to its input size
                with torch.no_grad():
                    teacher_inputs = inputs
                    if inputs.shape[-1] != teacher_image_size:
                        teacher_inputs = F.interpolate(
                            inputs, size=(teacher_image_size, teacher_image_size),
                            mode="bilinear", align_corners=False, antialias=True,
                        )
                    teacher_logits = teacher(teacher_inputs)
                kd_loss = F.kl_div(
                    F.log_softmax(outputs / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss = alpha * kd_loss + (1.0 - alpha) * loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        if scheduler:
            scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        metrics = compute_classification_metrics(all_targets, all_preds)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1_macro']:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics['accuracy'])
        history["val_f1"].append(metrics['f1_macro'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            
    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    train_from_config(args.config)
