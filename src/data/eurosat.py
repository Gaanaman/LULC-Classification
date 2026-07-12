import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import EuroSAT
from torchvision import transforms
from typing import Dict

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def build_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders for EuroSAT.

    Routes to the 13-band multispectral loader when data.bands == 'multispectral';
    otherwise uses the RGB (torchvision) path below.
    """
    if config["data"].get("bands", "rgb") == "multispectral":
        from src.data.eurosat_ms import build_ms_dataloaders
        return build_ms_dataloaders(config)

    data_config = config["data"]
    root = data_config["root"]
    image_size = data_config["image_size"]
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 4)
    seed = config.get("seed", 42)

    # ImageNet stats for pretrained models (also good enough for scratch CNN)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset without transforms initially
    try:
        base_dataset = EuroSAT(root=root, download=False)
    except Exception as e:
        print(f"Error loading EuroSAT: {e}. Attempting download...")
        base_dataset = EuroSAT(root=root, download=True)

    # Calculate split sizes (80/10/10)
    total_len = len(base_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=generator
    )

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_test_transform)
    test_dataset = TransformSubset(test_subset, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
