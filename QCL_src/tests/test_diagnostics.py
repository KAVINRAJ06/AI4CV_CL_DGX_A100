import json
import unittest
from pathlib import Path

import torch
from torch import nn
from qcl_sam_seg.diagnostics import adaptation, evaluate, seed_all


class DiagnosticTests(unittest.TestCase):
    def test_sam_only_bypasses_quantum(self):
        def fail(_):
            raise AssertionError("Quantum must not execute")
        x = torch.randn(1, 4, 3, 3)
        self.assertIs(adaptation(x, fail, "sam_only", 0.1), x)

    def test_residual_and_quantum_gradients(self):
        layer = nn.Conv2d(4, 4, 1)
        x = torch.randn(1, 4, 3, 3)
        torch.testing.assert_close(adaptation(x, layer, "quantum_only", 0.1), layer(x))
        value = adaptation(x, layer, "residual", 0.1)
        torch.testing.assert_close(value, x + 0.1 * layer(x))
        value.square().mean().backward()
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_report_frequencies_and_iou(self):
        mask = torch.tensor([[[0, 1], [2, 0]]])
        features = torch.nn.functional.one_hot(mask, 3).permute(0, 3, 1, 2).float() * 10
        item = {"features": features, "priors": torch.zeros(1, 0, 2, 2), "mask": mask}
        result = evaluate(nn.Identity(), None, "sam_only", [item], torch.device("cpu"),
                          ["a", "b", "c"], 255, 0.1)
        self.assertEqual(result["miou"], 1.0)
        self.assertEqual(result["target_fraction"], [0.5, 0.25, 0.25])
        self.assertEqual(result["predicted_fraction"], [0.5, 0.25, 0.25])

    def test_seed_and_notebook_syntax(self):
        seed_all(42)
        a = torch.randn(4)
        seed_all(42)
        torch.testing.assert_close(a, torch.randn(4))
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "Diagnose_DGX_Accuracy.ipynb").read_text())
        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"diagnostic-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
