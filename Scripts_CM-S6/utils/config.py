from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {path}")

    required = ("dataset", "data_root", "output_root", "num_classes", "channels")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    if config["dataset"] not in {"houston", "augsburg", "muufl"}:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if len(config["channels"]) not in {2, 3}:
        raise ValueError("The channels field must describe two or three modalities")
    if any(int(value) < 1 for value in config["channels"]):
        raise ValueError("All modality channel counts must be positive")
    if len(config.get("class_names", [])) != int(config["num_classes"]):
        raise ValueError("class_names must match num_classes")

    config = deepcopy(config)
    for key in ("data_root", "output_root"):
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        config[key] = str(value.resolve())
    config["config_path"] = str(path)
    return config
