"""
Pre-encode CelebA images to VAE latents and cache them to disk.

Run this once before training. The training script then loads the cached
latents directly, skipping the VAE encoder entirely and significantly
speeding up each epoch.
"""

from pathlib import Path

import torch
from tqdm import tqdm
from diffusers import AutoencoderKL

from diffusion.image_data_helper import load_celeba_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 192
batch_size = 32
output_path = Path("./celeba_latents.pt")

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
).to(device)
vae.eval()

scaling_factor = vae.config.scaling_factor

dataloader = load_celeba_dataloader(image_size, batch_size)

all_latents = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Encoding"):
        images = batch["pixel_values"].to(device)
        latents = vae.encode(images).latent_dist.mode() * scaling_factor
        all_latents.append(latents.cpu())

all_latents = torch.cat(all_latents, dim=0)
torch.save(all_latents, output_path)
print(f"Saved {all_latents.shape[0]} latents {tuple(all_latents.shape[1:])} to {output_path.resolve()}")
print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
