from pathlib import Path

from qcl_sam_seg.config import load_yaml, require_dataset_config
from qcl_sam_seg.data import discover_split


ROOT = Path(__file__).resolve().parents[1]


def test_openearthmap_manifest_resolves_existing_pairs():
    cfg = load_yaml(ROOT / "configs/datasets/openearthmap.yaml")
    require_dataset_config(cfg)
    records = discover_split(cfg, "train")
    assert records and records[0].image.exists() and records[0].mask.exists()


def test_landcover_manifest_resolves_existing_pairs():
    cfg = load_yaml(ROOT / "configs/datasets/landcoverai.yaml")
    require_dataset_config(cfg)
    records = discover_split(cfg, "train")
    assert records and records[0].tile_index is not None and records[0].image.exists()
