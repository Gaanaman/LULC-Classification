from pathlib import Path
from typing import Dict

import yaml


def read_yaml(path: str) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
