"""
src/utils/config.py
YAML configuration loader with safe defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "backbone": "mobilenet_v2",
        "pretrained": True,
        "input_size": [224, 224, 3],
        "dropout": 0.4,
    },
    "training": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "early_stopping_patience": 5,
    },
    "data": {
        "augmentation": True,
        "val_split": 0.2,
        "test_split": 0.1,
        "num_workers": 4,
    },
    "paths": {
        "data_dir"   : "data/",
        "models_dir" : "models/saved/",
        "outputs_dir": "outputs/",
    },
}


def load_config(path: str = "src/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Falls back to DEFAULT_CONFIG if file not found.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    cfg = DEFAULT_CONFIG.copy()

    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Shallow merge per top-level key
        for key, val in user_cfg.items():
            if key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(val)
            else:
                cfg[key] = val

    return cfg
