from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.utils import save_image

from models.ngu_net import Unet, SinusoidalTimeFiLM
from image_data_helper import postprocess, load_celeba_dataloader
from diffusion_noise_helper import q_sample, timesteps
from mask_helper import create_inpainting_conditioning
from sampling_helper import sample

# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 128
channels = 3              # RGB output channels (what the model predicts noise for)
input_channels = 7        # 3 noisy RGB + 1 mask + 3 masked original
micro_batch_size = 8
accumulation_steps = 8
learning_rate = 2e-4
epochs = 8

results_folder = Path("./results_celeba_inpaint")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

# ---------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------
dataloader = load_celeba_dataloader(image_size, micro_batch_size)

# ---------------------------------------------------------
# 3. Visualisation helpers
# ---------------------------------------------------------
def save_inpainting_samples(model, sample_images, milestone, folder):
    """
    Generate inpainting samples and save a side-by-side comparison grid.

    For each image the grid shows three columns:
        1. Masked input  (visible context with black hole)
        2. Model output   (the full inpainted image)
        3. Original       (ground truth for visual comparison)
    """
    conditioning = create_inpainting_conditioning(sample_images)
    conditioning = conditioning.to(device)

    inpainted = sample(model, conditioning, image_size=image_size, channels=channels)

    # Build a row-interleaved grid: [masked_0, inpainted_0, original_0, masked_1, ...]
    masked_imgs = sample_images * (1.0 - conditioning[:, 0:1, :, :].cpu())
    rows = []
    for i in range(sample_images.shape[0]):
        rows.append(postprocess(masked_imgs[i]))
        rows.append(postprocess(inpainted[i].cpu()))
        rows.append(postprocess(sample_images[i]))

    grid = torch.stack(rows)
    save_image(grid, str(folder / f"sample-{milestone}.png"), nrow=3)


# ---------------------------------------------------------
# 4. Model, Optimizer
# ---------------------------------------------------------
# Same FiLM conditioning on diffusion timestep as the unconditional model
dim = 96
film_dim = dim * 4
film = SinusoidalTimeFiLM(sinusoidal_dim=dim, film_dim=film_dim)

model = Unet(
    dim=dim,
    channels=input_channels,      # 7-channel input (noisy RGB + mask + masked original)
    out_dim=channels,             # 3-channel output (predicted noise for RGB only)
    dim_mults=(1, 2, 4, 8),
    film_module=film,
    film_dim=film_dim,
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# ---------------------------------------------------------
# 5. Training Loop
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

        # ---- Forward diffusion (identical to unconditional) ----
        t = torch.randint(0, timesteps, (b_size,), device=device).long()
        noise = torch.randn_like(batch_images)
        x_noisy = q_sample(x_start=batch_images, t=t, noise=noise)

        # ---- Inpainting conditioning ----
        # For each image in the batch, generate a random rectangular mask and
        # build the 4-channel conditioning tensor [mask | masked_original].
        conditioning = create_inpainting_conditioning(batch_images)

        # ---- Assemble 7-channel model input ----
        # Channels 0-2: noisy image (to be denoised)
        # Channel  3:   binary mask (1 = hole to inpaint)
        # Channels 4-6: original image with hole zeroed out (visible context)
        model_input = torch.cat([x_noisy, conditioning], dim=1)

        # ---- Predict noise and compute loss ----
        # The model outputs 3 channels: predicted noise for the RGB image only.
        # Loss is the same as unconditional: MSE between true and predicted noise.
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

# ---------------------------------------------------------
# 6. Final Evaluation & Checkpoint
# ---------------------------------------------------------
print("\nTraining complete! Generating final inpainting samples...")
model.eval()
save_inpainting_samples(model, sample_images, "final", results_folder)
print(f"Saved final samples to {results_folder / 'sample-final.png'}")

checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
final_weights_path = checkpoint_dir / "celeba_unet_inpaint_final.pt"
torch.save(model.state_dict(), final_weights_path)
print(f"Model weights saved to {final_weights_path}")
