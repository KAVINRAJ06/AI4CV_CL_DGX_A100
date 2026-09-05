"""Small fixed-subset diagnostic; does not change production training."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_yaml, require_dataset_config
from .data import SegmentationDataset, _read_mask
from .model import FrozenSAM, QuantumBottleneck
from .metrics import SegmentationMetrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adaptation(features, quantum, mode, scale):
    if mode == "sam_only":
        return features
    value = quantum(features)
    return value if mode == "quantum_only" else features + scale * value


def panel(image, mask, classes, path, prediction=None):
    count = 3 if prediction is None else 4
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    rgb = image.permute(1, 2, 0).numpy().clip(0, 1)
    labels = np.ma.masked_where((mask.numpy() < 0) | (mask.numpy() >= len(classes)), mask.numpy())
    cmap = plt.get_cmap("tab20", len(classes))
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=len(classes)-1, interpolation="nearest")
    axes[1].set_title("Mapped labels")
    axes[2].imshow(rgb)
    axes[2].imshow(labels, cmap=cmap, vmin=0, vmax=len(classes)-1, alpha=0.4, interpolation="nearest")
    axes[2].set_title("Alignment overlay")
    if prediction is not None:
        axes[3].imshow(prediction.numpy(), cmap=cmap, vmin=0, vmax=len(classes)-1, interpolation="nearest")
        axes[3].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(" | ".join(f"{i}: {name}" for i, name in enumerate(classes)), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def logits_for(head, quantum, mode, item, device, scale):
    features = item["features"].to(device)
    priors = item["priors"].to(device)
    adapted = adaptation(features, quantum, mode, scale)
    output = head(torch.cat((adapted, priors), dim=1))
    return F.interpolate(output, size=item["mask"].shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(head, quantum, mode, items, device, classes, ignore, scale):
    head.eval()
    if quantum is not None:
        quantum.eval()
    metric = SegmentationMetrics(len(classes), ignore)
    loss = 0.0
    for item in items:
        logits = logits_for(head, quantum, mode, item, device, scale)
        mask = item["mask"].to(device)
        loss += float(F.cross_entropy(logits, mask, ignore_index=ignore))
        metric.update(logits, mask)
    result = metric.compute()
    result["loss"] = loss / len(items)
    cm = np.asarray(result["confusion_matrix"])
    result["target_fraction"] = (cm.sum(1) / max(cm.sum(), 1)).tolist()
    result["predicted_fraction"] = (cm.sum(0) / max(cm.sum(), 1)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.samples, args.val_samples, args.steps) < 1 or args.lr <= 0:
        parser.error("sample counts, steps and learning rate must be positive")
    # Set before this process first initializes CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_yaml(args.config)
    require_dataset_config(cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    classes = cfg["dataset"]["classes"]
    ignore = int(cfg["dataset"].get("ignore_index", 255))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = cfg["model"]
    if settings.get("backbone", "sam_vit_b") != "sam_vit_b":
        raise ValueError("This diagnostic requires the real frozen SAM backbone")
    print(f"Diagnostic on {device}; {args.steps} optimizer steps per variant", flush=True)
    backbone = FrozenSAM(settings["sam_checkpoint"], settings.get("sam_model_type", "vit_b"),
                         settings.get("sam_precision", "float32")).to(device)
    cached, selection = {}, {}
    generator = torch.Generator().manual_seed(args.seed)
    for split, count in (("train", args.samples), ("val", args.val_samples)):
        dataset = SegmentationDataset(cfg, split, augment=False)
        if len(dataset) < count:
            raise ValueError(f"{split}: requested {count}, available {len(dataset)}")
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        cached[split], selection[split] = [], []
        for number, index in enumerate(indices):
            record = dataset.samples[index]
            source_image = dataset._read_image(record)
            raw = _read_mask(record, cfg)
            if raw.shape != (source_image.height, source_image.width):
                raise ValueError(f"Image/mask size mismatch: {record.sample_id}, {source_image.size}, {raw.shape}")
            unknown = set(np.unique(raw).tolist()) - set(dataset.label_map)
            if unknown:
                raise ValueError(f"Unmapped raw labels in {record.sample_id}: {sorted(unknown)}")
            item = dataset[index]
            if not (item["mask"] != ignore).any():
                raise ValueError(f"All labels ignored: {record.sample_id}")
            with torch.no_grad():
                features, priors = backbone(item["image"].unsqueeze(0).to(device))
                priors = F.interpolate(priors, size=features.shape[-2:], mode="bilinear", align_corners=False)
            rgb = item["image"] * dataset.std + dataset.mean
            cached[split].append({"features": features.cpu(), "priors": priors.cpu(),
                                  "mask": item["mask"].unsqueeze(0), "rgb": rgb})
            selection[split].append({"index": index, "id": item["id"], "image": str(record.image),
                                     "mask": str(record.mask), "raw_labels": sorted(np.unique(raw).tolist())})
            panel(rgb, item["mask"], classes, output / f"{split}_{number:02d}_labels.png")
            print(f"Cached SAM {split} {number+1}/{count}: {item['id']}", flush=True)
    train_ids = {row["id"] for row in selection["train"]}
    if train_ids & {row["id"] for row in selection["val"]}:
        raise ValueError("Selected train and validation sample IDs overlap")
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {"arguments": vars(args), "config": cfg, "selection": selection,
                "torch": torch.__version__, "device": str(device),
                "note": "Cached fixed features; augmentation disabled; validation subset is diagnostic, not final evaluation. Deterministic algorithms warn on unsupported operations."}
    (output / "setup.json").write_text(json.dumps(metadata, indent=2))
    channels = cached["train"][0]["features"].shape[1]
    seed_all(args.seed)
    initial_head = nn.Sequential(nn.Conv2d(channels+3, channels, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(channels, len(classes), 1)).state_dict()
    schedule = torch.randint(args.samples, (args.steps,), generator=torch.Generator().manual_seed(args.seed+1)).tolist()
    (output / "schedule.json").write_text(json.dumps(schedule))
    summary = {}
    scale = float(settings.get("residual_scale", 0.1)) or 0.1
    for mode in ("sam_only", "quantum_only", "residual"):
        seed_all(args.seed)
        head = nn.Sequential(nn.Conv2d(channels+3, channels, 3, padding=1), nn.GELU(),
                             nn.Conv2d(channels, len(classes), 1))
        head.load_state_dict(copy.deepcopy(initial_head))
        head.to(device)
        # Reset independently: both quantum variants receive identical quantum initialization.
        seed_all(args.seed+2)
        quantum = None if mode == "sam_only" else QuantumBottleneck(
            channels, int(settings.get("qubits", 8)), int(settings.get("quantum_layers", 4)),
            int(settings.get("token_grid", 8))).to(device)
        parameters = list(head.parameters()) + ([] if quantum is None else list(quantum.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
        rows = []
        start = time.perf_counter()
        for step in range(args.steps+1):
            if step == 0 or step % 25 == 0 or step == args.steps:
                scores = {split: evaluate(head, quantum, mode, items, device, classes, ignore, scale)
                          for split, items in cached.items()}
                rows.append({"step": step, **scores})
                (output / f"{mode}_history.json").write_text(json.dumps(rows, indent=2))
                print(f"{mode} step {step}/{args.steps} | train loss {scores['train']['loss']:.4f}, "
                      f"train mIoU {scores['train']['miou']:.4f}, val mIoU {scores['val']['miou']:.4f}", flush=True)
            if step == args.steps:
                break
            head.train()
            if quantum is not None:
                quantum.train()
            item = cached["train"][schedule[step]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits_for(head, quantum, mode, item, device, scale),
                                   item["mask"].to(device), ignore_index=ignore)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{mode}: nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
        summary[mode] = {"seconds_including_evaluation": time.perf_counter()-start, **scores}
        with (output / f"{mode}_classes.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "class_id", "class", "iou", "target_fraction", "predicted_fraction"])
            for split in cached:
                for i, name in enumerate(classes):
                    result = scores[split]
                    writer.writerow([split, i, name, result["per_class_iou"][i],
                                     result["target_fraction"][i], result["predicted_fraction"][i]])
        with torch.no_grad():
            for split in cached:
                for number, item in enumerate(cached[split][:2]):
                    pred = logits_for(head, quantum, mode, item, device, scale).argmax(1)[0].cpu()
                    panel(item["rgb"], item["mask"][0], classes,
                          output / f"{mode}_{split}_{number:02d}_prediction.png", pred)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        del optimizer, head, quantum
    print(f"Diagnostic complete: {output}", flush=True)


if __name__ == "__main__":
    main()
