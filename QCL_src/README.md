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

A nonempty `SAM_CHECKPOINT` takes precedence over `model.sam_checkpoint`, including when an older or locally customized YAML still contains a Windows path. If the variable is unset or empty, the YAML value is used. Environment variables and `~` in the selected path are expanded before checking that the file exists.

## DGX update blocked by local changes

An aborted `git pull` leaves the old code installed. Exporting `SAM_CHECKPOINT` cannot add override support to that old code. Preserve your tracked local edits before updating, and restore them afterward:

```bash
cd /raid/workspace/AI4CV/AI4CV_CL_DGX_A100
git status --short
git stash push -m "DGX local settings before checkpoint fix"
```

If Git reports that it saved your changes, run:

```bash
git pull --ff-only origin main && git stash apply 'stash@{0}'
```

If Git reports no local changes to save, run only `git pull --ff-only origin main`; do not apply an unrelated older stash. If the pull fails, stop and inspect its error. If applying the stash reports conflicts, resolve them before training, preserving your DGX dataset paths. The stash is deliberately retained as a backup. Untracked files are left in place; if they block the pull, preserve them separately before retrying.

After the update and any conflict resolution:

```bash
git log -1 --oneline
grep 'os.environ.get("SAM_CHECKPOINT")' QCL_src/qcl_sam_seg/model.py
cd QCL_src
source .venv/bin/activate
export SAM_CHECKPOINT=/raid/workspace/AI4CV/models/sam_vit_b_01ec64.pth
ls -lh "$SAM_CHECKPOINT"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -m qcl_sam_seg train \\
  --stream configs/streams/openearthmap_then_landcoverai.yaml \\
  2>&1 | tee dgx_training.log
```
