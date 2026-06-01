from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_from_config(config_path: str, checkpoint_path: str) -> None:
    _ = load_config(config_path)
    _ = Path(checkpoint_path)
    raise NotImplementedError("TODO: implement evaluation pipeline.")
