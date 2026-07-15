import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark import benchmark_run, benchmark_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs: params, speed, test metrics.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Pairs of config:checkpoint, e.g. configs/x.yaml:outputs/checkpoints/x/best_model.pth")
    args = parser.parse_args()

    results = []
    for spec in args.runs:
        config_path, ckpt = spec.split(":")
        print(f"Benchmarking {config_path} ...")
        results.append(benchmark_run(config_path, ckpt))

    table = benchmark_table(results)
    print("\n" + table)
    out = Path("outputs/reports/comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
