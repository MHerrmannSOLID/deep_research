"""Simple models used by the playground."""

import torch
from torch import nn


class PlaygroundMLP(nn.Module):
    """A compact fully-connected network for quick experiments."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("layer sizes must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
