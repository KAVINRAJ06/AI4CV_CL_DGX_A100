from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from .config import load_yaml, require_dataset_config
from .data import validate_dataset
from .engine import loaders, run_epoch, train_task, write_forgetting, write_tsne
from .ewc import OnlineEWC
from .model import make_model
from .visualize import prediction_panel, predict_image


def _configs(stream_path: str):
    stream = load_yaml(stream_path); base = Path(stream_path).resolve().parent
    return stream, [load_yaml((base / item).resolve()) for item in stream["tasks"]]


def main() -> None:
    parser = argparse.ArgumentParser(prog="qcl_sam_seg")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "evaluate", "predict", "visualize-tsne"):
        p = sub.add_parser(command); p.add_argument("--config", required=True); p.add_argument("--checkpoint")
        if command == "predict": p.add_argument("--image")
    train = sub.add_parser("train"); train.add_argument("--stream", required=True)
    args = parser.parse_args()
    if args.command == "prepare-data":
        cfg = load_yaml(args.config); require_dataset_config(cfg); print(json.dumps(validate_dataset(cfg), indent=2)); return
    if args.command == "train":
        stream, configs = _configs(args.stream)
        [require_dataset_config(cfg) for cfg in configs]
        head_classes = {cfg["task"]["head_id"]: len(cfg["dataset"]["classes"]) for cfg in configs}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(configs[0]["model"], head_classes).to(device)
        ewc, best, task_configs = OnlineEWC(float(configs[0]["continual"].get("ewc_decay", 1.0))), {}, {}
        root = Path(configs[0]["output"]["root"]) / stream["name"]
        for cfg in configs:
            task_configs[cfg["task"]["name"]] = cfg
            train_task(model, cfg, ewc, device, root / cfg["task"]["name"], best)
            scores = {}
            for name, old in task_configs.items():
                classes, ignore = len(old["dataset"]["classes"]), int(old["dataset"].get("ignore_index", 255))
                scores[name] = run_epoch(model, loaders(old)["test"], old["task"]["head_id"], torch.nn.CrossEntropyLoss(ignore_index=ignore), device, classes=classes, ignore_index=ignore)
            write_forgetting(root / cfg["task"]["name"], scores, best)
            write_tsne(model, {name: (loaders(old)["test"], old["task"]["head_id"]) for name, old in task_configs.items()}, root / cfg["task"]["name"] / "tsne.png", int(cfg["evaluation"].get("tsne_samples", 1000)), int(cfg["evaluation"].get("seed", 42)))
        return
    cfg = load_yaml(args.config); require_dataset_config(cfg)
    if not args.checkpoint: parser.error("--checkpoint is required")
    model = make_model(cfg["model"], {cfg["task"]["head_id"]: len(cfg["dataset"]["classes"])})
    checkpoint = torch.load(args.checkpoint, map_location="cpu"); model.load_state_dict(checkpoint["model"], strict=False)
    if args.command == "evaluate":
        result = run_epoch(model, loaders(cfg)["test"], cfg["task"]["head_id"], torch.nn.CrossEntropyLoss(ignore_index=int(cfg["dataset"].get("ignore_index", 255))), torch.device("cpu"), classes=len(cfg["dataset"]["classes"]), ignore_index=int(cfg["dataset"].get("ignore_index", 255)))
        print(json.dumps(result, indent=2))
    elif args.command == "predict":
        if args.image:
            result = predict_image(model, args.image, Path(cfg["output"]["root"]) / cfg["task"]["name"] / "predictions" / f"{Path(args.image).stem}_prediction.png", int(cfg["dataset"]["transforms"]["image_size"]), cfg["dataset"]["transforms"].get("mean", [0.485, 0.456, 0.406]), cfg["dataset"]["transforms"].get("std", [0.229, 0.224, 0.225]), cfg["task"]["head_id"])
            print(result)
        else:
            prediction_panel(model, loaders(cfg)["test"], cfg["task"]["head_id"], Path(cfg["output"]["root"]) / cfg["task"]["name"] / "predictions", int(cfg["evaluation"].get("num_visualizations", 12)))
    else:
        write_tsne(model, {cfg["task"]["name"]: (loaders(cfg)["test"], cfg["task"]["head_id"])}, Path(cfg["output"]["root"]) / cfg["task"]["name"] / "tsne.png", int(cfg["evaluation"].get("tsne_samples", 1000)), int(cfg["evaluation"].get("seed", 42)))


if __name__ == "__main__":
    main()
