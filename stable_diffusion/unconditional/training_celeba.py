from pathlib import Path

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.transforms import Resize
from torchvision.utils import save_image
from diffusers import AutoencoderKL

from models.ngu_net import Unet, SinusoidalTimeFiLM
from diffusion.image_data_helper import load_celeba_dataloader
from diffusion.unconditional.diffusion_noise_helper import p_losses, timesteps
from diffusion.unconditional.sampling_helper import sample
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 192          # 192x192 pixels → 24x24 latents (VAE downsamples 8x)
latent_size = image_size // 8
latent_channels = 4       # SD VAE produces 4-channel latents
micro_batch_size = 8      # Fits comfortably in 8 GB VRAM
accumulation_steps = 8    # Effective batch size = 8 * 8 = 64
learning_rate = 2e-4
epochs = 16

results_folder = Path("./results_celeba_latent")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

# ---------------------------------------------------------
# 2. VAE (frozen encoder/decoder for pre/post-processing)
# ---------------------------------------------------------
vae_model_id = "stabilityai/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=torch.float32).to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)

vae_scaling_factor = vae.config.scaling_factor

# ---------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------
dataloader = load_celeba_dataloader(image_size, micro_batch_size)

@torch.no_grad()
def encode_to_latent(pixel_images):
    return vae.encode(pixel_images).latent_dist.sample() * vae_scaling_factor

@torch.no_grad()
def decode_latents(latents):
    decoded = vae.decode(latents / vae_scaling_factor).sample
    return (decoded.clamp(-1, 1) + 1) / 2

def decode_and_unsquish(latents):
    imgs = decode_latents(latents)
    return Resize((218, 178))(imgs)

# ---------------------------------------------------------
# 4. Model, Optimizer
# ---------------------------------------------------------
# The UNet operates entirely in latent space (4 channels, 24x24).
# Compared to pixel-space diffusion we use fewer downsampling stages
# (spatial is already small) but wider channels and more attention.
dim = 128
film_dim = dim * 4
film = SinusoidalTimeFiLM(sinusoidal_dim=dim, film_dim=film_dim)

model = Unet(
    dim=dim,                         # Base feature width
    channels=latent_channels,        # 4 VAE latent channels
    dim_mults=(2, 4, 8),             # Downsamples: 24 → 12 → 6, bottleneck at 6x6
    film_module=film,
    film_dim=film_dim,
    full_attention_count=3,          # Full attention at all 3 stages (no linear attention)
    attn_heads=6,
    attn_dim_head=64,
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer,eta_min=1e-6, T_max=epochs)

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
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

        # Pre-processing: encode pixels to latent space using frozen VAE
        latents = encode_to_latent(batch_images)

        # Sample uniform timestep t for each image in batch
        t = torch.randint(0, timesteps, (b_size,), device=device).long()

        loss = p_losses(model, latents, t, loss_type="l2")
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass with gradients
        loss.backward()

        # Perform optimizer step every `accumulation_steps`
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 100 == 0:
                print(f" Step {global_step} | Loss: {loss.item() * accumulation_steps:.4f}")

            # Save sample outputs during training
            if global_step > 0 and global_step % save_and_sample_every == 0:
                model.eval()
                milestone = global_step // save_and_sample_every

                # Sample in latent space, then decode through VAE for display
                sampled_latents = sample(model, image_size=latent_size, batch_size=4, channels=latent_channels)

                # Post-processing: VAE decode + un-squish to 178x218 aspect ratio
                final_grid = decode_and_unsquish(sampled_latents)
                save_image(final_grid, str(results_folder / f'sample-{milestone}.png'), nrow=2)
    scheduler.step()

# ---------------------------------------------------------
# 6. Final Evaluation & Sampling
# ---------------------------------------------------------
print("\nTraining complete! Generating final sample batch...")
model.eval()

# Generate 8 final face samples in latent space
final_latents = sample(model, image_size=latent_size, batch_size=8, channels=latent_channels)

# Post-processing: VAE decode + restore 178x218 resolution
final_grid = decode_and_unsquish(final_latents)
save_image(final_grid, 'final_celeba_faces_latent.png', nrow=4)
print("Saved final un-squished samples to 'final_celeba_faces_latent.png'")


# Ensure checkpoint directory exists
checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Save only the model weights
final_weights_path = checkpoint_dir / "celeba_latent_unet_final.pt"
torch.save(model.state_dict(), final_weights_path)

print(f"Final model parameters saved successfully to {final_weights_path}")
