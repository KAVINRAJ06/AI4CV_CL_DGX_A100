import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from qcl_sam_seg.ewc import OnlineEWC
from qcl_sam_seg.model import QuantumBottleneck, TinyFrozenSAM, UniversalQCLSAM


def test_single_quantum_bottleneck_and_ewc_state():
    model = UniversalQCLSAM(TinyFrozenSAM(channels=8), QuantumBottleneck(channels=8, qubits=2, layers=1, token_grid=1), {"first": 2, "second": 3}, channels=8)
    assert sum(isinstance(module, QuantumBottleneck) for module in model.modules()) == 1
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    logits, embedding = model(torch.rand(1, 3, 16, 16), "first")
    assert logits.shape == (1, 2, 16, 16) and embedding.shape[1] == 8
    ewc = OnlineEWC()
    ewc.snapshot(model.quantum, {name: torch.ones_like(value) for name, value in model.quantum.named_parameters()})
    assert ewc.state_dict()["fisher"] and ewc.penalty(model.quantum).item() == pytest.approx(0.0)
