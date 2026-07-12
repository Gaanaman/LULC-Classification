import argparse
from pathlib import Path
from torchvision.datasets import EuroSAT

def download_dataset(root_dir: str):
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EuroSAT to {root_path}...")
    dataset = EuroSAT(root=str(root_path), download=True)
    print(f"Downloaded {len(dataset)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EuroSAT dataset.")
    parser.add_argument("--root", type=str, default="data/raw/EuroSAT", help="Root directory for dataset")
    args = parser.parse_args()
    download_dataset(args.root)
