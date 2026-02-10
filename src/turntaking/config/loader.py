
from pathlib import Path
import yaml
from turntaking.config.analysis_schema import TurntakingConfig

def load_config(path: Path) -> TurntakingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return TurntakingConfig.from_dict(raw)

