from pathlib import Path
from typing import Dict

import yaml

from src.utils.seed import set_seed


def load_config(path: str) -> Dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_from_config(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    raise NotImplementedError("TODO: implement training loop.")
