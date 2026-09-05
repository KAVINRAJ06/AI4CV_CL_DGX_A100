from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .ewc import OnlineEWC, estimate_fisher
from .metrics import SegmentationMetrics
from .visualize import prediction_panel, tsne_plot


def loaders(cfg: dict, workers: int = 0):
    batch = int(cfg["training"].get("batch_size", 2))
    return {split: DataLoader(SegmentationDataset(cfg, split, augment=split == "train"), batch_size=batch, shuffle=split == "train", num_workers=workers) for split in ("train", "val", "test")}


def run_epoch(model, loader, head_id, criterion, device, optimizer=None, ewc=None, ewc_lambda=0.0, classes=2, ignore_index=255):
    training = optimizer is not None
    model.train(training)
    metric, loss_total, samples = SegmentationMetrics(classes, ignore_index), 0.0, 0
    for batch in loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        with torch.set_grad_enabled(training):
            logits, _ = model(images, head_id)
            loss = criterion(logits, masks)
            if training and ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(model.quantum)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metric.update(logits, masks)
        loss_total += float(loss.detach()) * images.shape[0]
        samples += images.shape[0]
    result = metric.compute()
    result["loss"] = loss_total / max(samples, 1)
    return result


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_metrics_csv(path: Path, metrics: dict) -> None:
    scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar.keys()); writer.writeheader(); writer.writerow(scalar)


def train_task(model, cfg, ewc: OnlineEWC, device, output: Path, previous_best: dict[str, dict]) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    head_id = cfg["task"]["head_id"]
    classes, ignore = len(cfg["dataset"]["classes"]), int(cfg["dataset"].get("ignore_index", 255))
    model.add_head(head_id, classes)
    current = list(model.quantum.parameters()) + list(model.semantic_heads[head_id].parameters())
    optimizer = torch.optim.AdamW(current, lr=float(cfg["training"].get("lr", 1e-4)), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    criterion = nn.CrossEntropyLoss(ignore_index=ignore)
    ds_loaders = loaders(cfg)
    history, best = [], -float("inf")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train = run_epoch(model, ds_loaders["train"], head_id, criterion, device, optimizer, ewc, float(cfg["continual"]["ewc_lambda"]), classes, ignore)
        val = run_epoch(model, ds_loaders["val"], head_id, criterion, device, classes=classes, ignore_index=ignore)
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        line = f"{cfg['task']['name']} | epoch [{epoch}/{cfg['training']['epochs']}] : {train['accuracy']:.4f} / {val['accuracy']:.4f} | {train['loss']:.4f} / {val['loss']:.4f} | {train['miou']:.4f} / {val['miou']:.4f} | {train['dice']:.4f} / {val['dice']:.4f} | mIoU {val['miou']:.4f} | bIoU {val['biou']:.4f} | Dice {val['dice']:.4f}"
        print(line)
        with (output / "train.log").open("a", encoding="utf-8") as stream: stream.write(line + "\n")
        if val["miou"] > best:
            best = val["miou"]
            torch.save({"model": model.state_dict(), "ewc": ewc.state_dict(), "task": cfg["task"]}, output / "best.pt")
    fisher = estimate_fisher(model, ds_loaders["train"], head_id, criterion, device, int(cfg["continual"].get("fisher_batches", 32)))
    ewc.snapshot(model.quantum, fisher)
    test = run_epoch(model, ds_loaders["test"], head_id, criterion, device, classes=classes, ignore_index=ignore)
    previous_best[cfg["task"]["name"]] = {"miou": max(previous_best.get(cfg["task"]["name"], {}).get("miou", -1), test["miou"]), "biou": max(previous_best.get(cfg["task"]["name"], {}).get("biou", -1), test["biou"]), "dice": max(previous_best.get(cfg["task"]["name"], {}).get("dice", -1), test["dice"])}
    _write_json(output / "history.json", history)
    _write_json(output / "test_metrics.json", test)
    _write_metrics_csv(output / "test_metrics.csv", test)
    torch.save(ewc.fisher, output / "fisher.pt")
    torch.save(ewc.means, output / "theta_star.pt")
    torch.save({"model": model.state_dict(), "ewc": ewc.state_dict(), "task": cfg["task"]}, output / "latest.pt")
    prediction_panel(model, ds_loaders["test"], head_id, output / "predictions", int(cfg["evaluation"].get("num_visualizations", 12)))
    return test, previous_best


def write_forgetting(output: Path, scores: dict[str, dict], best: dict[str, dict]) -> dict:
    rows = []
    for task, metrics in scores.items():
        row = {"task": task, **metrics}
        for key in ("miou", "biou", "dice"):
            row[f"forgetting_{key}"] = best[task][key] - metrics[key]
        rows.append(row)
    with (output / "forgetting.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {"tasks": rows, "mean_forgetting_miou": sum(r["forgetting_miou"] for r in rows) / len(rows)}
    _write_json(output / "forgetting.json", summary)
    return summary


@torch.no_grad()
def write_tsne(model, task_loaders: dict[str, tuple], output: Path, limit: int = 1000, seed: int = 42) -> None:
    """Use a bounded deterministic feature sample, grouped by task, for stage comparisons."""
    import numpy as np
    device = next(model.parameters()).device
    samples, labels = [], []
    for task, (loader, head_id) in task_loaders.items():
        count = 0
        for batch in loader:
            _, embedding = model(batch["image"].to(device), head_id)
            vectors = embedding.mean((-1, -2)).cpu().numpy()
            available = max(0, limit - count)
            vectors = vectors[:available]
            samples.append(vectors); labels.extend([task] * len(vectors)); count += len(vectors)
            if count >= limit: break
    if samples:
        rng = np.random.default_rng(seed)
        order = rng.permutation(sum(len(x) for x in samples))
        points = np.concatenate(samples)[order]
        names = [labels[i] for i in order]
        tsne_plot([points], names, output, seed)
