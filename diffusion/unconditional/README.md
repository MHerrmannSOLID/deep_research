# Unconditional Diffusion

Generates CelebA 128x128 face images from pure noise, without any conditioning signal beyond the diffusion timestep.

## Files

```
unconditional/
├── diffusion_noise_helper.py   # Local copy of noise schedule and forward diffusion
├── sampling_helper.py          # Reverse diffusion loop (denoising) for 3-ch input
└── training_celeba.py          # Training script
```

### `sampling_helper.py`

Implements the reverse (denoising) process for unconditional generation:

- **`p_sample(model, x, t, t_index)`** -- single reverse step: predict noise, compute posterior mean, add variance
- **`p_sample_loop(model, shape)`** -- full T-step reverse chain from pure noise to image
- **`sample(model, image_size, batch_size, channels)`** -- convenience wrapper

The model receives only the noisy image `[B, 3, H, W]` and the timestep at each step.

### `training_celeba.py`

Trains the U-Net on CelebA 128x128 with gradient accumulation (effective batch size 64).

Periodic samples are saved to `results_celeba/` during training. Final checkpoint is saved to `checkpoints/celeba_unet_final.pt`.

```bash
cd diffusion/unconditional
python training_celeba.py
```

## How It Works

1. **Forward process**: `q_sample` adds Gaussian noise to a clean image according to the schedule at timestep `t`
2. **Training objective**: the U-Net predicts the noise that was added; loss = MSE(predicted_noise, true_noise)
3. **Sampling**: start from pure Gaussian noise and iteratively denoise for T steps using the trained model
