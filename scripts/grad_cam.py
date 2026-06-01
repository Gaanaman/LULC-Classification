import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.grad_cam import run_grad_cam


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM visualization.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--ckpt", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    args = parser.parse_args()

    run_grad_cam(args.config, args.ckpt, args.image)


if __name__ == "__main__":
    main()
