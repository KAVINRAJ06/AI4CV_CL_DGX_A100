from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import resolve_path


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    mask: Path
    tile_index: int | None = None


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _openearthmap(root: Path, item: str) -> Sample:
    stem = Path(item).stem
    city = stem.rsplit("_", 1)[0]
    return Sample(item, root / city / "images" / item, root / city / "labels" / item)


def _landcover(root: Path, item: str) -> Sample:
    match = re.fullmatch(r"(.+)_(\d+)", item)
    if not match:
        raise ValueError(f"LandCover.ai manifest entry is not a tiled ID: {item}")
    stem, tile = match.groups()
    return Sample(item, root / "images" / f"{stem}.tif", root / "masks" / f"{stem}.tif", int(tile))


def discover_split(cfg: dict, split: str) -> list[Sample]:
    ds = cfg["dataset"]
    root = resolve_path(cfg, ds["root"])
    manifest = resolve_path(cfg, ds["splits"][split])
    kind = ds["kind"]
    resolver = _openearthmap if kind == "openearthmap" else _landcover if kind == "landcoverai" else None
    if resolver is None:
        raise ValueError(f"Unsupported dataset.kind: {kind}")
    samples = [resolver(root, item) for item in _lines(manifest)]
    missing = [s.sample_id for s in samples if not s.image.is_file() or not s.mask.is_file()]
    if missing:
        raise FileNotFoundError(f"{split}: {len(missing)} missing pairs; first: {missing[:3]}")
    return samples


def validate_dataset(cfg: dict, splits: Iterable[str] = ("train", "val", "test")) -> dict:
    all_ids: dict[str, set[str]] = {}
    hist: dict[int, int] = {}
    raw_allowed = {int(k) for k in cfg["dataset"]["label_map"]}
    for split in splits:
        records = discover_split(cfg, split)
        all_ids[split] = {record.sample_id for record in records}
        for record in records[: cfg["dataset"].get("validation_scan_limit", 32)]:
            labels = _read_mask(record, cfg)
            values, counts = np.unique(labels, return_counts=True)
            unknown = set(values.tolist()) - raw_allowed
            if unknown:
                raise ValueError(f"{record.sample_id} has unmapped raw labels: {sorted(unknown)}")
            for value, count in zip(values, counts):
                hist[int(value)] = hist.get(int(value), 0) + int(count)
    names = list(all_ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = all_ids[left] & all_ids[right]
            if shared:
                raise ValueError(f"Split leakage between {left}/{right}: {next(iter(shared))}")
    return {"samples": {name: len(ids) for name, ids in all_ids.items()}, "raw_label_histogram": hist}


def _read_mask(record: Sample, cfg: dict) -> np.ndarray:
    mask = np.asarray(Image.open(record.mask))
    if record.tile_index is not None:
        size = int(cfg["dataset"]["tiling"]["size"])
        cols = mask.shape[1] // size
        y, x = divmod(record.tile_index, cols)
        mask = mask[y * size : (y + 1) * size, x * size : (x + 1) * size]
    return mask


class SegmentationDataset(Dataset):
    def __init__(self, cfg: dict, split: str, augment: bool = False):
        self.cfg, self.samples, self.augment = cfg, discover_split(cfg, split), augment
        self.size = int(cfg["dataset"]["transforms"]["image_size"])
        self.label_map = {int(k): int(v) for k, v in cfg["dataset"]["label_map"].items()}
        self.ignore = int(cfg["dataset"].get("ignore_index", 255))
        self.mean = torch.tensor(cfg["dataset"]["transforms"].get("mean", [0.485, 0.456, 0.406]))[:, None, None]
        self.std = torch.tensor(cfg["dataset"]["transforms"].get("std", [0.229, 0.224, 0.225]))[:, None, None]

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, record: Sample) -> Image.Image:
        image = Image.open(record.image).convert("RGB")
        if record.tile_index is not None:
            size = int(self.cfg["dataset"]["tiling"]["size"])
            cols = image.width // size
            y, x = divmod(record.tile_index, cols)
            image = image.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
        return image

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        image = self._read_image(record).resize((self.size, self.size), Image.Resampling.BILINEAR)
        raw = Image.fromarray(_read_mask(record, self.cfg)).resize((self.size, self.size), Image.Resampling.NEAREST)
        mask = np.asarray(raw, dtype=np.int64)
        mapped = np.full_like(mask, self.ignore)
        for source, target in self.label_map.items():
            mapped[mask == source] = target
        if self.augment and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mapped = np.fliplr(mapped).copy()
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        return {"image": (tensor - self.mean) / self.std, "mask": torch.from_numpy(mapped), "id": record.sample_id}
