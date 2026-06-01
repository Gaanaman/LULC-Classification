from pathlib import Path


def run_grad_cam(config_path: str, checkpoint_path: str, image_path: str) -> None:
    _ = Path(config_path)
    _ = Path(checkpoint_path)
    _ = Path(image_path)
    raise NotImplementedError("TODO: implement Grad-CAM visualization.")
