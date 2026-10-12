"""
Shared image data utilities for diffusion training scripts.

Provides CelebA dataset loading and common image post-processing used
across both unconditional and inpainting training.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Lambda


def postprocess(tensor_imgs):
    """Un-normalise from [-1, 1] to [0, 1] and clamp."""
    imgs = (tensor_imgs + 1) * 0.5
    return torch.clamp(imgs, 0.0, 1.0)


def load_celeba_dataloader(image_size, batch_size, num_workers=0):
    """
    Load the CelebA faces dataset and return a ready-to-use DataLoader.

    Images are resized to image_size x image_size, randomly flipped,
    and normalised to [-1, 1] (standard range for diffusion models).

    Args:
        image_size:  Target spatial resolution (square).
        batch_size:  Batch size for the DataLoader.
        num_workers: Number of data loading workers.

    Returns:
        A DataLoader yielding batches with key "pixel_values" of shape
        [batch_size, 3, image_size, image_size] in range [-1, 1].
    """
    dataset = load_dataset("nielsr/CelebA-faces")

    transform = Compose([
        Resize((image_size, image_size)),
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),  # scale to [-1, 1]
    ])

    def transform_images(examples):
        examples["pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["image"]
        ]
        del examples["image"]
        return examples

    transformed_dataset = dataset.with_transform(transform_images)

    return DataLoader(
        transformed_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
