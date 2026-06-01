import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluate import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--ckpt", required=True, help="Path to a model checkpoint.")
    args = parser.parse_args()

    evaluate_from_config(args.config, args.ckpt)


if __name__ == "__main__":
    main()
