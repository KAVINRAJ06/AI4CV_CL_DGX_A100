import pytest

torch = pytest.importorskip("torch")
from qcl_sam_seg.ewc import OnlineEWC


def test_ewc_penalizes_changed_parameters():
    layer = torch.nn.Linear(2, 1, bias=False)
    ewc = OnlineEWC(); ewc.snapshot(layer, {"weight": torch.ones_like(layer.weight)})
    with torch.no_grad(): layer.weight.add_(1)
    assert ewc.penalty(layer).item() == pytest.approx(2.0)
