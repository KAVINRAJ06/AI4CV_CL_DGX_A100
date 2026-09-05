"""DGX regression checks without datasets or a SAM checkpoint."""
import json
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from qcl_sam_seg.metrics import SegmentationMetrics
from qcl_sam_seg.model import UniversalQCLSAM, FrozenSAM
from qcl_sam_seg.ewc import estimate_fisher


class OptimizationTests(unittest.TestCase):
    def check_metrics(self, device):
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 12, 12)
        target = torch.randint(0, 3, (2, 12, 12))
        target[:, :2] = 255
        pred = logits.argmax(1)
        valid = target != 255
        cm = torch.bincount(3 * target[valid] + pred[valid], minlength=9).reshape(3, 3).double()
        intersections, unions = [], []
        for cls in range(3):
            p, t = pred == cls, target == cls
            pb = p ^ F.max_pool2d(p.float().unsqueeze(1), 3, 1, 1).squeeze(1).bool()
            tb = t ^ F.max_pool2d(t.float().unsqueeze(1), 3, 1, 1).squeeze(1).bool()
            intersections.append((pb & tb & valid).sum())
            unions.append(((pb | tb) & valid).sum())
        metric = SegmentationMetrics(3)
        metric.update(logits.to(device), target.to(device))
        self.assertEqual(metric.cm.device.type, device)
        torch.testing.assert_close(metric.cm.cpu(), cm)
        torch.testing.assert_close(metric.boundary_intersection.cpu(), torch.stack(intersections).double())
        torch.testing.assert_close(metric.boundary_union.cpu(), torch.stack(unions).double())
        self.assertAlmostEqual(metric.compute()["accuracy"], float(cm.diag().sum() / cm.sum()))

    def test_metrics_cpu(self):
        self.check_metrics("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_metrics_cuda(self):
        self.check_metrics("cuda")

    def test_residual_retains_features_and_quantum_gradients(self):
        class Backbone(nn.Module):
            def forward(self, image):
                return image, image.new_zeros(image.shape[0], 3, *image.shape[-2:])
        quantum = nn.Conv2d(4, 4, 1)
        model = UniversalQCLSAM(Backbone(), quantum, {"task": 3}, channels=4, residual_scale=0.1)
        image = torch.randn(2, 4, 8, 8)
        logits, adapted = model(image, "task")
        torch.testing.assert_close(adapted, image + 0.1 * quantum(image))
        logits.square().mean().backward()
        self.assertIsNotNone(quantum.weight.grad)
        self.assertGreater(quantum.weight.grad.abs().sum().item(), 0)
        model.residual_scale = 0
        _, legacy = model(image, "task")
        torch.testing.assert_close(legacy, quantum(image))

    def test_sam_wrapper_remains_frozen(self):
        # Avoid requiring segment-anything or downloading pretrained weights.
        model = FrozenSAM.__new__(FrozenSAM)
        nn.Module.__init__(model)
        model.sam = nn.Conv2d(3, 3, 1)
        model.sam.requires_grad_(False)
        model.precision = "float32"
        model._encode = lambda image: (model.sam(image), model.sam(image))
        model.train()
        self.assertFalse(model.training)
        self.assertFalse(model.sam.training)
        features, priors = model(torch.randn(1, 3, 4, 4, requires_grad=True))
        self.assertFalse(features.requires_grad)
        self.assertFalse(priors.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fisher_mixed_devices(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.quantum = nn.ModuleDict({
                    "cpu": nn.Conv2d(3, 3, 1),
                    "gpu": nn.Conv2d(3, 3, 1).cuda(),
                })
            def forward(self, image, head):
                value = self.quantum["cpu"](image.cpu()).cuda()
                value = self.quantum["gpu"](value)
                return value, value
        model = Mixed()
        loader = [{"image": torch.randn(2, 3, 4, 4),
                   "mask": torch.randint(0, 3, (2, 4, 4))}]
        fisher = estimate_fisher(model, loader, "task", nn.CrossEntropyLoss(), torch.device("cuda"), 1)
        self.assertEqual(set(fisher), set(dict(model.quantum.named_parameters())))
        for value in fisher.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(any(value.sum() > 0 for value in fisher.values()))

    def test_notebook_code_compiles(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "Run_DGX_Training.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
