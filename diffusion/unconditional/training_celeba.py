from pathlib import Path
import sys

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torchvision.transforms import Resize
from torchvision.utils import save_image

from models.ngu_net import Unet, SinusoidalTimeFiLM
from image_data_helper import postprocess, load_celeba_dataloader
from diffusion_noise_helper import p_losses, timesteps
from sampling_helper import sample

# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 128          # 176x176 is divisible by 16 (for dim_mults=(1, 2, 4, 8))
channels = 3              # RGB for CelebA
micro_batch_size = 8      # Fits comfortably in 8 GB VRAM
accumulation_steps = 8    # Effective batch size = 16 * 4 = 64
learning_rate = 2e-4
epochs = 8

results_folder = Path("./results_celeba")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

# ---------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------
dataloader = load_celeba_dataloader(image_size, micro_batch_size)

def postprocess_and_unsquish(tensor_imgs):
    """Post-process and resize from 128x128 back to original CelebA aspect ratio (178x218)."""
    imgs = postprocess(tensor_imgs)
    return Resize((218, 178))(imgs)

# ---------------------------------------------------------
# 4. Model, Optimizer, and AMP Setup
# ---------------------------------------------------------
# FiLM stands for Feature-wise Linear Modulation (Perez et al., 2017)
# this is conditioning (scaling ans shifting) the featuremaps of the Unet.
# In diffusion models, FiLM allows the Unet to adapt its feature processing 
# based on the current timestep, effectively conditioning the denoising process on time.
# This is what u find below based on the SinusoidalTime embeddings.
# For diffusion time embedding are the minimum requirement. This would then be called unconditional diffusion.
# However, if one also conditions on additional information (like class labels or text embeddings), 
# it becomes conditional diffusion and such conditions would also be injected into the Unet via the same FiLM mechanism below.
dim = 96
film_dim = dim * 4
film = SinusoidalTimeFiLM(sinusoidal_dim=dim, film_dim=film_dim)

model = Unet(
    dim=dim,                         # Base feature width
    channels=channels,               # 3 RGB channels
    dim_mults=(1, 2, 4, 8),         # Downsamples: 128 -> 64 -> 32 -> 16 -> 8
    film_module=film,                # Inject time-based FiLM conditioning for diffusion
    film_dim=film_dim,               # Must match the output dimension of the FiLM module
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        model.train()
        
        batch_images = batch["pixel_values"].to(device)
        b_size = batch_images.shape[0]

        # Sample uniform timestep t for each image in batch
        t = torch.randint(0, timesteps, (b_size,), device=device).long()
        
        loss = p_losses(model, batch_images, t, loss_type="l2")
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
                
                # Sample 4 images
                sampled_imgs = sample(model, image_size=image_size, batch_size=4, channels=channels)
                
                # Un-squish back to 178x218 aspect ratio before saving
                final_grid = postprocess_and_unsquish(sampled_imgs)
                save_image(final_grid, str(results_folder / f'sample-{milestone}.png'), nrow=2)


# ---------------------------------------------------------
# 6. Final Evaluation & Sampling
# ---------------------------------------------------------
print("\nTraining complete! Generating final sample batch...")
model.eval()

# Generate 8 final face samples
final_samples = sample(model, image_size=image_size, batch_size=8, channels=channels)

# Restore 178x218 resolution and save
final_grid = postprocess_and_unsquish(final_samples)
save_image(final_grid, 'final_celeba_faces.png', nrow=4)
print("Saved final un-squished samples to 'final_celeba_faces.png'")



# Ensure checkpoint directory exists
checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Save only the model weights (lightweight, ~143 MB file)
final_weights_path = checkpoint_dir / "celeba_unet_final.pt"
torch.save(model.state_dict(), final_weights_path)

print(f"Final model parameters saved successfully to {final_weights_path}")
