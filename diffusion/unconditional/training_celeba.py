from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Lambda
from torchvision.utils import save_image

from models.ngu_net import Unet, SinusoidalTimeFiLM
from diffusion_noise_helper import (
    extract,
    p_losses,
    timesteps,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
)

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
# 2. Dataset & Transformations
# ---------------------------------------------------------
# Load CelebA from Hugging Face Hub
dataset = load_dataset("nielsr/CelebA-faces")

# Anamorphic squish to 176x176 square without cropping face features
transform = Compose([
    Resize((image_size, image_size)),
    RandomHorizontalFlip(),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
])

def transform_images(examples):
    # Convert images to RGB (CelebA) and apply transformations
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

transformed_dataset = dataset.with_transform(transform_images)

dataloader = DataLoader(
    transformed_dataset["train"], 
    batch_size=micro_batch_size, 
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ---------------------------------------------------------
# 3. Sampling Utilities
# ---------------------------------------------------------
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    
    # Run reverse diffusion loop with AMP enabled to speed up sampling
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling timestep', total=timesteps, leave=False):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            
    return img  # Returns tensor directly on GPU

@torch.no_grad()
def sample(model, image_size, batch_size=4, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def postprocess_and_unsquish(tensor_imgs):
    """
    1. Un-normalize from [-1, 1] to [0, 1]
    2. Resize from 176x176 square back to original CelebA aspect ratio (178x218)
    """
    imgs = (tensor_imgs + 1) * 0.5
    imgs = torch.clamp(imgs, 0.0, 1.0)
    
    # Restore original aspect ratio: height=218, width=178
    unscale = Resize((218, 178))
    return unscale(imgs)

# ---------------------------------------------------------
# 4. Model, Optimizer, and AMP Setup
# ---------------------------------------------------------
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
