from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class QuantumBottleneck(nn.Module):
    """One shared VQC. It is applied to all pooled spatial tokens, never copied per task."""
    def __init__(self, channels: int = 256, qubits: int = 8, layers: int = 4, token_grid: int = 8):
        super().__init__()
        self.qubits, self.layers, self.token_grid = qubits, layers, token_grid
        self.encode = nn.Linear(channels, qubits)
        self.decode = nn.Linear(qubits, channels)
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install pennylane to use QuantumBottleneck") from exc
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(qubits))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]

        self.q_layer = qml.qnn.TorchLayer(circuit, {"weights": (layers, qubits, 3)})

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = features.shape[-2:]
        tokens = F.adaptive_avg_pool2d(features, (self.token_grid, self.token_grid)).flatten(2).transpose(1, 2)
        angles = torch.tanh(self.encode(tokens)) * math.pi
        measured = self.q_layer(angles.flatten(0, 1)).reshape_as(angles)
        restored = self.decode(measured).transpose(1, 2).reshape(features.shape[0], -1, self.token_grid, self.token_grid)
        return F.interpolate(restored, size=size, mode="bilinear", align_corners=False)


class TinyFrozenSAM(nn.Module):
    """Frozen shape-compatible substitute used solely by the CPU smoke test."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Conv2d(3, channels, 3, 2, 1), nn.GELU(), nn.Conv2d(channels, channels, 3, 2, 1))
        self.mask_decoder = nn.Conv2d(channels, 3, 1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.image_encoder(image)
        return embedding, self.mask_decoder(embedding)


class FrozenSAM(nn.Module):
    def __init__(self, checkpoint: str, model_type: str = "vit_b"):
        super().__init__()
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError("Install segment-anything to use SAM ViT-B") from exc
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dataset tensors are RGB floats in [0, 1]. SAM owns its pixel
        # normalization/padding, so do not feed it ImageNet-normalized data.
        embedding = self.sam.image_encoder(self.sam.preprocess(image * 255.0))
        batch = image.shape[0]
        sparse = embedding.new_zeros(batch, 0, embedding.shape[1])
        dense = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand_as(embedding)
        masks, _ = self.sam.mask_decoder(embedding, self.sam.prompt_encoder.get_dense_pe(), sparse, dense, multimask_output=True)
        return embedding, masks


class UniversalQCLSAM(nn.Module):
    def __init__(self, backbone: nn.Module, quantum: QuantumBottleneck, head_classes: dict[str, int], channels: int = 256):
        super().__init__()
        self.backbone, self.quantum = backbone, quantum
        self.semantic_heads = nn.ModuleDict({name: nn.Sequential(nn.Conv2d(channels + 3, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, count, 1)) for name, count in head_classes.items()})

    def add_head(self, head_id: str, classes: int, channels: int = 256) -> None:
        if head_id not in self.semantic_heads:
            self.semantic_heads[head_id] = nn.Sequential(nn.Conv2d(channels + 3, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, classes, 1))

    def shared_parameters(self):
        return list(self.quantum.parameters())

    def forward(self, image: torch.Tensor, head_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        features, priors = self.backbone(image)
        adapted = self.quantum(features)
        priors = F.interpolate(priors, size=adapted.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.semantic_heads[head_id](torch.cat((adapted, priors), dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False), adapted


def make_model(model_cfg: dict, head_classes: dict[str, int]) -> UniversalQCLSAM:
    channels = int(model_cfg.get("channels", 256))
    if model_cfg.get("backbone", "sam_vit_b") == "tiny_frozen":
        backbone = TinyFrozenSAM(channels)
    else:
        backbone = FrozenSAM(model_cfg["sam_checkpoint"], model_cfg.get("sam_model_type", "vit_b"))
    quantum = QuantumBottleneck(channels, int(model_cfg.get("qubits", 8)), int(model_cfg.get("quantum_layers", 4)), int(model_cfg.get("token_grid", 8)))
    return UniversalQCLSAM(backbone, quantum, head_classes, channels)
