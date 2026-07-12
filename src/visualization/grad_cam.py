import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import transforms

from src.utils.io import read_yaml
from src.models.finetune import build_finetune_model
from src.models.scratch_cnn import ScratchCNN

def run_grad_cam(config_path: str, checkpoint_path: str, image_path: str) -> None:
    config = read_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Model
    model_config = config["model"]
    if model_config["name"] == "scratch_cnn":
        model = ScratchCNN(num_classes=config["data"]["num_classes"])
        target_layers = [model.features[-3]] # Last conv layer before MaxPool
    elif model_config["name"] == "resnet18":
        model = build_finetune_model(name=model_config["name"], num_classes=config["data"]["num_classes"], pretrained=False)
        target_layers = [model.layer4[-1]]
    elif model_config["name"] == "efficientnet_v2_s":
        model = build_finetune_model(name=model_config["name"], num_classes=config["data"]["num_classes"], pretrained=False)
        target_layers = [model.features[-1]]
    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = config["data"]["image_size"]

    img_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Visualization
    img_float = cv2.resize(np.array(img_pil), (img_size, img_size)) / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    out_dir = Path("outputs/reports") / model_config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_float)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM")
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = out_dir / f"gradcam_{Path(image_path).stem}.png"
    plt.savefig(save_path)
    print(f"Saved Grad-CAM to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    run_grad_cam(args.config, args.checkpoint, args.image)
