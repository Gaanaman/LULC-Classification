import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    train_from_config(args.config)


if __name__ == "__main__":
    main()
