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


## DGX performance controls

The DGX notebook enables `model.sam_precision: bfloat16` for the frozen SAM forward pass and `training.num_workers: 4` with pinned memory and prefetching. SAM outputs are converted back to float32 before the quantum branch and semantic head. Ordinary YAML runs retain float32 SAM and zero workers unless configured. Set workers to zero if the server lacks sufficient shared memory.

Metrics now stay on the prediction device and retain the existing definitions (including the one-pixel boundary metric). Batch progress prints on the first batch, every 25 batches, and the last batch, with end-to-end seconds per batch and ETA. Epoch history is saved after each epoch. Fisher accumulators follow each parameter's device, fixing the CPU quantum/CUDA projection mismatch.

The notebook enables `model.residual_scale: 0.1`: the head receives SAM spatial features plus 0.1 times the quantum adaptation. The previous architecture discarded the direct SAM feature path after pooling to an 8x8 quantum grid. This is an architecture change intended to preserve detail; accuracy gains must be measured. Set residual_scale to 0 for the legacy architecture, including when evaluating old checkpoints. Evaluate new checkpoints with their saved run configs.

SAM remains frozen, but its features are recomputed; no feature cache or multi-GPU training is introduced. Batch size and image resolution are unchanged. BF16 and worker loading follow [PyTorch's performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [autocast documentation](https://docs.pytorch.org/docs/stable/amp.html).

Run regressions in the project environment:
```bash
cd QCL_src
python -m unittest discover -s tests -p test_dgx_optimization.py -v
```

The notebook runs these checks before training. They cover metric parity, residual gradient flow, frozen SAM behavior through a test double, and mixed-device Fisher estimation (CUDA tests skip without CUDA). A real SAM/A100 speed and quality benchmark is still required. Do not compare the residual architecture to an old checkpoint without its original config.


## Diagnose low segmentation accuracy

Open [Diagnose_DGX_Accuracy.ipynb](Diagnose_DGX_Accuracy.ipynb) with the existing QCL DGX kernel. Stop other training first. This creates a separate output directory and runs a fixed-seed comparison on 8 training and 8 validation images, with 200 one-image optimizer steps per variant by default.

The three variants use identical initial head weights, sample order, cached frozen SAM features and mask priors, learning rate, and update count. They differ only in adaptation: SAM-only, quantum-only, or SAM plus scaled quantum residual. Quantum weights start identically across the two quantum variants. Augmentation and EWC are disabled for this single-task capacity check. Deterministic algorithms are requested with warnings for unsupported operations; exact cross-hardware reproducibility is not promised.

Outputs include original-size/label checks, mapped-label overlays, learning curves, per-class IoU, target/predicted class frequencies, predictions and a report ZIP. setup.json records selected sample IDs, paths, config and seed; schedule.json records every sampled training index. The subset is not class-stratified. These are diagnostic metrics, not full validation/test estimates, and cached-feature timings are not production throughput.

If the SAM-only head can fit the subset but the quantum branch cannot, investigate the quantum representation and gradients. If neither can fit, inspect image/label overlays and optimization before full training. A small-subset experiment does not establish that one architecture is generally superior.

Terminal alternative, using an absolute-path config generated by the training notebook:
```bash
python -m qcl_sam_seg.diagnostics --config /path/to/run/configs/openearthmap.yaml --output outputs/diagnostic_new --samples 8 --val-samples 8 --steps 200 --seed 42
```
The output directory must not already exist. The diagnostic notebook runs its regression checks before loading datasets; local tests can also be run with `python -m unittest discover -s tests -p test_diagnostics.py -v`.
