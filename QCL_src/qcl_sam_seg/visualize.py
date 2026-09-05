from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image


def prediction_panel(model, loader, head_id: str, output: Path, max_images: int = 12) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    model.eval(); count = 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["image"].to(next(model.parameters()).device), head_id)
            prediction = logits.argmax(1).cpu()
            for i, sample_id in enumerate(batch["id"]):
                image = batch["image"][i].permute(1, 2, 0).numpy()
                image = np.clip((image - image.min()) / max(image.max() - image.min(), 1e-6), 0, 1)
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for axis, value, title in zip(axes, (image, batch["mask"][i], prediction[i], prediction[i].ne(batch["mask"][i])), ("RGB", "ground truth", "prediction", "error")):
                    axis.imshow(value); axis.set_title(title); axis.axis("off")
                fig.tight_layout(); fig.savefig(output / f"{sample_id}.png", dpi=150); plt.close(fig)
                count += 1
                if count >= max_images: return


def tsne_plot(embeddings: list[np.ndarray], labels: list[str], output: Path, seed: int = 42) -> None:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    points = np.concatenate(embeddings)
    if len(points) < 2: return
    reduced = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(points)
    fig, axis = plt.subplots(figsize=(8, 6))
    for label in sorted(set(labels)):
        idx = np.array(labels) == label
        axis.scatter(reduced[idx, 0], reduced[idx, 1], s=8, label=label)
    axis.legend(); fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=160); plt.close(fig)


@torch.no_grad()
def predict_image(model, image_path: str | Path, output: Path, size: int, mean: list[float], std: list[float], head_id: str) -> Path:
    image = Image.open(image_path).convert("RGB")
    original = image.copy()
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div(255)
    tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    logits, _ = model(tensor.unsqueeze(0).to(next(model.parameters()).device), head_id)
    prediction = logits.argmax(1)[0].cpu().byte().numpy()
    mask = Image.fromarray(prediction).resize(original.size, Image.Resampling.NEAREST)
    output.parent.mkdir(parents=True, exist_ok=True); mask.save(output)
    return output
