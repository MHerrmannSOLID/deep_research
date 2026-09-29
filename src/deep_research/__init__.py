"""Small building blocks for PyTorch experiments."""

from .model import PlaygroundMLP
from .training import train_step

__all__ = ["PlaygroundMLP", "train_step"]
