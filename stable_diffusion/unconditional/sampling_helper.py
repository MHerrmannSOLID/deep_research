"""
Sampling helper for unconditional diffusion.

Implements the reverse diffusion process (denoising) for models that take
a noisy image [B, 3, H, W] as their only spatial input. The model predicts
the noise component, and these functions iteratively remove it to recover
a clean image.

Used by both the training script (periodic sample generation) and standalone
inference scripts.
"""

import torch
from tqdm import tqdm

from diffusion_noise_helper import (
    extract,
    timesteps,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
)


@torch.no_grad()
def p_sample(model, x, t, t_index):
    """
    Single reverse diffusion step: denoise x_t to x_{t-1}.

    Uses the model's noise prediction to compute the posterior mean,
    then adds the appropriate amount of noise (except at the final step t=0).

    Args:
        model:   The denoising network. Takes (x, t) and returns predicted noise.
        x:       Current noisy image, shape [B, 3, H, W].
        t:       Timestep tensor, shape [B], same value for all items in the batch.
        t_index: Scalar int of the current timestep (used to detect the final step).

    Returns:
        x_{t-1}: The slightly less noisy image, same shape as x.
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Predicted mean of x_{t-1}: subtract the model's noise estimate from x_t
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        # At the last step, return the mean directly (no noise added)
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    """
    Full reverse diffusion loop: start from pure Gaussian noise and denoise
    step by step back to a clean image.

    Args:
        model: The denoising network.
        shape: Tuple (batch_size, channels, height, width) for the output images.

    Returns:
        Denoised images, shape [B, 3, H, W], on the model's device.
    """
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling timestep", total=timesteps, leave=False):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)

    return img


@torch.no_grad()
def sample(model, image_size, batch_size=4, channels=3):
    """
    Convenience wrapper: generate a batch of images from scratch.

    Args:
        model:      The denoising network.
        image_size: Spatial resolution (images are square: image_size x image_size).
        batch_size: How many images to generate.
        channels:   Number of image channels (3 for RGB).

    Returns:
        Generated images, shape [batch_size, channels, image_size, image_size].
    """
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
