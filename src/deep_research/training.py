"""Training helpers kept intentionally small for experimentation."""

import torch
from torch import nn
from typing import Optional


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    """Run one regression step and return the detached loss value."""
    loss_fn = loss_fn or nn.MSELoss()
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
