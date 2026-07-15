import subprocess
import sys
from pathlib import Path

def run_command(cmd, desc):
    print(f"\n{'='*50}\n{desc}\n{'='*50}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    configs = ["configs/scratch_cnn.yaml", "configs/resnet18.yaml", "configs/efficientnet_v2_s.yaml"]

    # 1. Download EuroSAT
    run_command([sys.executable, "-m", "src.data.download_eurosat"], "Downloading EuroSAT Dataset")

    for config in configs:
        model_name = Path(config).stem

        # 2. Train
        run_command([sys.executable, "-m", "src.training.train", "--config", config],
                    f"Training {model_name}")

        # 3. Evaluate
        checkpoint = f"outputs/checkpoints/{model_name}/best_model.pth"
        run_command([sys.executable, "-m", "src.eval.evaluate", "--config", config, "--checkpoint", checkpoint],
                    f"Evaluating {model_name}")

        # 4. Plot History & CM
        history = f"outputs/checkpoints/{model_name}/history.json"
        cm = f"outputs/reports/{model_name}/confusion_matrix.npy"
        out_dir = f"outputs/reports/{model_name}"
        run_command([sys.executable, "-m", "src.visualization.plots",
                     "--history", history, "--cm", cm, "--out", out_dir],
                    f"Plotting metrics for {model_name}")

        # Grad-CAM is run separately via scripts/grad_cam.py with an explicit image path.

    print("\nAll models trained and evaluated.")

if __name__ == "__main__":
    main()
