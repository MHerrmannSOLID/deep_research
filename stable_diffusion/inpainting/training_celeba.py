from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.utils import save_image
from diffusers import AutoencoderKL

from models.ngu_net import Unet, SinusoidalTimeFiLM
from diffusion.image_data_helper import postprocess, load_celeba_dataloader
from diffusion.unconditional.diffusion_noise_helper import q_sample, timesteps
from diffusion.inpainting.mask_helper import generate_random_rectangle_mask
from diffusion.inpainting.sampling_helper import sample
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 192          # Pixel resolution: masks are applied here (SD-style inpainting)
latent_size = 24          # 192x192 pixels → 24x24 latents (VAE downsamples 8x)
latent_channels = 4       # SD VAE produces 4-channel latents
input_channels = 9        # 4 noisy latent + 1 mask + 4 masked-image latent
micro_batch_size = 16
accumulation_steps = 8    # Effective batch size = 16 * 8 = 128
learning_rate = 2e-4
epochs = 200

results_folder = Path("./results_celeba_latent_inpaint")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 10000

# ---------------------------------------------------------
# 2. VAE (frozen, both directions needed here!)
# ---------------------------------------------------------
# Unlike unconditional latent diffusion (where latents are pre-encoded),
# inpainting needs the VAE *encoder* during training: the mask is applied
# in pixel space and the masked image must then be encoded per batch,
# because masks are random per iteration and cannot be cached.
vae_model_id = "stabilityai/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=torch.float32).to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)

vae_scaling_factor = vae.config.scaling_factor

@torch.no_grad()
def encode_to_latent(pixel_images):
    return vae.encode(pixel_images).latent_dist.sample() * vae_scaling_factor

@torch.no_grad()
def decode_latents(latents):
    decoded = vae.decode(latents / vae_scaling_factor).sample
    return (decoded.clamp(-1, 1) + 1) / 2

# ---------------------------------------------------------
# 3. Dataset (pixel images — masking happens in pixel space)
# ---------------------------------------------------------
dataloader = load_celeba_dataloader(image_size, micro_batch_size)

# ---------------------------------------------------------
# 4. Inpainting conditioning (SD-style, Option B)
# ---------------------------------------------------------
@torch.no_grad()
def create_latent_inpainting_conditioning(pixel_images):
    """
    Build the 5-channel latent-space conditioning tensor [B, 5, 24, 24].

    SD-style inpainting: the mask is applied to the *pixel* image, the masked
    image is VAE-encoded, and the binary mask is downsampled to the latent
    resolution. This matches exactly what happens at inference time with a
    real image + user-drawn mask.

    Channel layout:
        Channel 0:    Binary mask at latent resolution (1 = hole)
        Channels 1-4: VAE latent of the masked image (visible context)
    """
    b, _, h, w = pixel_images.shape

    # Random rectangle mask in pixel space (1 = hole to inpaint)
    mask_px = generate_random_rectangle_mask(b, h, w).to(pixel_images.device)

    # Black out the hole in pixel space, then encode the masked image
    masked_images = pixel_images * (1.0 - mask_px)
    masked_latents = encode_to_latent(masked_images)

    # Bring the mask down to latent resolution (192 → 24, nearest keeps it binary)
    mask_latent = F.interpolate(mask_px, size=(latent_size, latent_size), mode="nearest")

    return torch.cat([mask_latent, masked_latents], dim=1)

# ---------------------------------------------------------
# 5. Visualisation helpers
# ---------------------------------------------------------
def save_inpainting_samples(model, sample_images, milestone, folder):
    """
    Generate inpainting samples and save a side-by-side comparison grid.

    For each image the grid shows three columns:
        1. Masked input  (VAE decode of the masked-image latent — what the model "sees")
        2. Model output  (inpainted latent, VAE decoded)
        3. Original      (ground truth pixels for visual comparison)
    """
    images = sample_images.to(device)
    conditioning = create_latent_inpainting_conditioning(images)

    inpainted_latents = sample(model, conditioning, image_size=latent_size, channels=latent_channels)

    inpainted = decode_latents(inpainted_latents).cpu()
    masked_vis = decode_latents(conditioning[:, 1:, :, :]).cpu()
    originals = postprocess(images.cpu())

    # Build a row-interleaved grid: [masked_0, inpainted_0, original_0, masked_1, ...]
    rows = []
    for i in range(images.shape[0]):
        rows.append(masked_vis[i])
        rows.append(inpainted[i])
        rows.append(originals[i])

    grid = torch.stack(rows)
    save_image(grid, str(folder / f"sample-{milestone}.png"), nrow=3)

# ---------------------------------------------------------
# 6. Model, Optimizer
# ---------------------------------------------------------
dim = 96
film_dim = dim * 4
film = SinusoidalTimeFiLM(sinusoidal_dim=dim, film_dim=film_dim)

model = Unet(
    dim=dim,                         # Base feature width
    channels=input_channels,         # 9-channel input (noisy latent + mask + masked latent)
    out_dim=latent_channels,         # 4-channel output (predicted noise for the latent only)
    dim_mults=(2, 4, 6),             # Downsamples: 24 → 12 → 6, bottleneck at 6x6
    film_module=film,
    film_dim=film_dim,
    full_attention_count=3,          # Full attention at all 3 stages (no linear attention)
    attn_heads=6,
    attn_dim_head=32,
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=epochs)

# ---------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------
# Keep a fixed set of images for periodic sampling so we can visually
# track inpainting quality across training.
sample_images = None

global_step = 0

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training", dynamic_ncols=True)):
        model.train()

        batch_images = batch["pixel_values"].to(device)
        b_size = batch_images.shape[0]

        # Capture a fixed batch of images for periodic visualisation
        if sample_images is None:
            sample_images = batch_images[:4].cpu().clone()

        # ---- Encode clean images to latents ----
        latents = encode_to_latent(batch_images)

        # ---- Forward diffusion (identical to unconditional, but on latents) ----
        t = torch.randint(0, timesteps, (b_size,), device=device).long()
        noise = torch.randn_like(latents)
        x_noisy = q_sample(x_start=latents, t=t, noise=noise)

        # ---- Inpainting conditioning (mask in pixel space, encode, downsample mask) ----
        conditioning = create_latent_inpainting_conditioning(batch_images)

        # ---- Assemble 9-channel model input ----
        # Channels 0-3: noisy latent (to be denoised)
        # Channel  4:   binary mask at latent resolution (1 = hole to inpaint)
        # Channels 5-8: latent of the masked image (visible context)
        model_input = torch.cat([x_noisy, conditioning], dim=1)

        # ---- Predict noise and compute loss ----
        # The model outputs 4 channels: predicted noise for the latent only.
        predicted_noise = model(model_input, t)
        loss = F.mse_loss(noise, predicted_noise)

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 100 == 0:
                print(f" Step {global_step} | Loss: {loss.item() * accumulation_steps:.4f}")

            if global_step > 0 and global_step % save_and_sample_every == 0:
                model.eval()
                milestone = global_step // save_and_sample_every
                save_inpainting_samples(model, sample_images, milestone, results_folder)
    scheduler.step()

# ---------------------------------------------------------
# 8. Final Evaluation & Checkpoint
# ---------------------------------------------------------
print("\nTraining complete! Generating final inpainting samples...")
model.eval()
save_inpainting_samples(model, sample_images, "final", results_folder)
print(f"Saved final samples to {results_folder / 'sample-final.png'}")

checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
final_weights_path = checkpoint_dir / "celeba_latent_unet_inpaint_final.pt"
torch.save(model.state_dict(), final_weights_path)
print(f"Model weights saved to {final_weights_path}")
