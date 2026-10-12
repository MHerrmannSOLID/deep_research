"""
Sampling helper for inpainting diffusion.

Implements the reverse diffusion process for models that take a 7-channel
input: the noisy image [B, 3, H, W] concatenated with a conditioning tensor
[B, 4, H, W] (binary mask + masked original image).

The key difference from unconditional sampling: at every reverse step, the
conditioning channels stay fixed (they are not noised or denoised) and are
re-concatenated with the evolving noisy image before each model call.

Used by both the training script (periodic sample generation) and standalone
inference scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
def p_sample(model, x, t, t_index, conditioning):
    """
    Single reverse diffusion step for inpainting: denoise x_t to x_{t-1}.

    The model receives the 7-channel input [x_t | conditioning] but only
    predicts the 3-channel noise for the image portion. The conditioning
    (mask + masked original) is constant across all timesteps.

    Args:
        model:        The denoising network. Takes [B, 7, H, W] and returns [B, 3, H, W].
        x:            Current noisy image, shape [B, 3, H, W].
        t:            Timestep tensor, shape [B].
        t_index:      Scalar int of the current timestep (used to detect the final step).
        conditioning: The 4-channel conditioning tensor [B, 4, H, W]
                      (channel 0 = mask, channels 1-3 = masked original image).

    Returns:
        x_{t-1}: The slightly less noisy image, shape [B, 3, H, W].
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Concatenate noisy image with conditioning to form the 7-channel model input
    model_input = torch.cat([x, conditioning], dim=1)

    # Model predicts the noise (3 channels), same as unconditional
    predicted_noise = model(model_input, t)

    # Compute the posterior mean using the noise prediction
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape, conditioning):
    """
    Full reverse diffusion loop for inpainting.

    Starts from pure Gaussian noise for the image channels and iteratively
    denoises, passing the fixed conditioning at each step.

    Args:
        model:        The denoising network.
        shape:        Tuple (batch_size, 3, height, width) for the noisy image.
        conditioning: The 4-channel conditioning tensor [B, 4, H, W].

    Returns:
        Denoised images, shape [B, 3, H, W], on the model's device.
    """
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling timestep", total=timesteps, leave=False):
        img = p_sample(
            model, img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            conditioning,
        )

    return img


@torch.no_grad()
def sample(model, conditioning, image_size, channels=3):
    """
    Convenience wrapper: generate inpainted images for a given conditioning.

    Args:
        model:        The denoising network.
        conditioning: [B, 4, H, W] tensor (mask + masked original).
                      The batch size is inferred from this tensor.
        image_size:   Spatial resolution (square images).
        channels:     Number of image channels (3 for RGB).

    Returns:
        Inpainted images, shape [B, channels, image_size, image_size].
    """
    batch_size = conditioning.shape[0]
    return p_sample_loop(
        model,
        shape=(batch_size, channels, image_size, image_size),
        conditioning=conditioning,
    )
