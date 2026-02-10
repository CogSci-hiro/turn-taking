
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project configuration loaded from YAML.

    Notes
    -----
    This is a minimal placeholder. Expand as needed.
    """
    raw: Dict[str, Any]


def load_project_config(path: Path) -> ProjectConfig:
    """Load project config from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return ProjectConfig(raw=data)
