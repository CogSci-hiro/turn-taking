
from pathlib import Path
from typing import Any, Dict

import yaml

from turntaking.config.schema import ProjectConfig


def load_config(path: Path) -> ProjectConfig:
    """
    Load a YAML config file into a typed ProjectConfig.
    """
    raw: Dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ProjectConfig.from_dict(raw)
