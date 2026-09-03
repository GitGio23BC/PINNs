import logging
import os
import sys
from pathlib import Path

import yaml


def init_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_name = Path(sys.argv[0]).stem
    log_path = logs_dir / f"{script_name}.log"

    root.setLevel(log_level)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    """ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    """

def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
