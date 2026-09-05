import pytest

torch = pytest.importorskip("torch")
from qcl_sam_seg.metrics import SegmentationMetrics


def test_metrics_perfect_prediction():
    logits = torch.tensor([[[[5., 0.], [0., 5.]], [[0., 5.], [5., 0.]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])
    metrics = SegmentationMetrics(2); metrics.update(logits, target)
    assert metrics.compute()["miou"] == pytest.approx(1.0)
