from __future__ import annotations

import torch
from torch import nn


class OnlineEWC:
    def __init__(self, decay: float = 1.0):
        self.decay = decay
        self.fisher: dict[str, torch.Tensor] = {}
        self.means: dict[str, torch.Tensor] = {}

    def penalty(self, module: nn.Module) -> torch.Tensor:
        if not self.fisher:
            return next(module.parameters()).new_zeros(())
        return sum((self.fisher[name].to(param.device) * (param - self.means[name].to(param.device)).pow(2)).sum() for name, param in module.named_parameters() if name in self.fisher)

    @torch.no_grad()
    def snapshot(self, module: nn.Module, fisher: dict[str, torch.Tensor]) -> None:
        for name, value in fisher.items():
            old = self.fisher.get(name)
            self.fisher[name] = value.cpu() if old is None else self.decay * old + value.cpu()
        self.means = {name: param.detach().cpu().clone() for name, param in module.named_parameters()}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "fisher": self.fisher, "means": self.means}

    def load_state_dict(self, state: dict) -> None:
        self.decay, self.fisher, self.means = state["decay"], state["fisher"], state["means"]


def estimate_fisher(model, loader, head_id: str, criterion, device: torch.device, max_batches: int = 32) -> dict[str, torch.Tensor]:
    model.eval()
    total = {name: torch.zeros_like(param, device=device) for name, param in model.quantum.named_parameters()}
    count = 0
    for batch in loader:
        model.zero_grad(set_to_none=True)
        logits, _ = model(batch["image"].to(device), head_id)
        loss = criterion(logits, batch["mask"].to(device))
        loss.backward()
        for name, param in model.quantum.named_parameters():
            if param.grad is not None:
                total[name] += param.grad.detach().pow(2)
        count += 1
        if count >= max_batches:
            break
    return {name: value.div(max(count, 1)).cpu() for name, value in total.items()}
