from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    data["_config_path"] = str(path.resolve())
    return data


def require_dataset_config(cfg: dict[str, Any]) -> None:
    required = ("task", "dataset", "model", "training", "evaluation")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Dataset config missing sections: {missing}")
    for key in ("name", "head_id"):
        if key not in cfg["task"]:
            raise ValueError(f"task.{key} is required")
    ds = cfg["dataset"]
    for key in ("root", "classes", "label_map", "splits"):
        if key not in ds:
            raise ValueError(f"dataset.{key} is required")
    classes = ds["classes"]
    mapped = set(int(v) for v in ds["label_map"].values())
    valid = set(range(len(classes))) | {int(ds.get("ignore_index", 255))}
    if not mapped <= valid:
        raise ValueError("label_map contains an output outside classes or ignore_index")


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(cfg["_config_path"]).parent.joinpath(value).resolve()
