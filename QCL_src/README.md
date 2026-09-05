# QCL-SAM satellite segmentation

Configuration-driven continual semantic segmentation with a frozen SAM ViT-B backbone, one shared variational quantum bottleneck, dataset-specific semantic heads, and online EWC.

Install `requirements.txt` and the package, provide a SAM ViT-B checkpoint, then run:

```powershell
pip install -r requirements.txt
pip install -e .
python -m qcl_sam_seg prepare-data --config configs/datasets/openearthmap.yaml
python -m qcl_sam_seg train --stream configs/streams/openearthmap_then_landcoverai.yaml
python -m qcl_sam_seg evaluate --config configs/datasets/openearthmap.yaml --checkpoint outputs/openearthmap_then_landcoverai/landcoverai/latest.pt
python -m qcl_sam_seg predict --config configs/datasets/openearthmap.yaml --checkpoint <checkpoint> --image <image.tif>
```

`prepare-data` validates labels and split isolation without changing source data. The `tiny_frozen` backbone exists only for CPU smoke tests; production configs use frozen `sam_vit_b`.

The supplied data configs preserve their native label spaces: OpenEarthMap has nine IDs (`0..8`) and LandCover.ai has five (`0..4`). Consequently, each task gets a distinct semantic head while the quantum bottleneck is shared and EWC-protected. Edit a dataset YAML—not source code—when introducing another satellite dataset or its label map.

The shipped configurations intentionally keep RGB tensors in `[0, 1]`: the frozen SAM wrapper applies SAM's own normalization and padding before encoding. Do not change these to ImageNet statistics unless replacing the SAM backbone.

## SAM checkpoint

Download the SAM ViT-B checkpoint once and set `SAM_CHECKPOINT` before validation or training. The dataset YAMLs intentionally resolve this environment variable, avoiding machine-specific paths:

```bash
export SAM_CHECKPOINT=/raid/workspace/AI4CV/models/sam_vit_b_01ec64.pth
```