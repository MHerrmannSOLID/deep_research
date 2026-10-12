# Inpainting Diffusion

Fills in masked (cut-out) regions of CelebA 128x128 face images using a diffusion model conditioned on visible context.

## Files

```
inpainting/
├── mask_helper.py                  # Random rectangle mask generation + conditioning tensor
├── sampling_helper.py              # Reverse diffusion loop for 7-ch input
└── training_celeba_inpaint.py      # Training script
```

### `mask_helper.py`

Generates random rectangular masks and builds the 4-channel conditioning tensor:

- **`generate_random_rectangle_mask(batch_size, H, W)`** -- creates `[B, 1, H, W]` binary masks (1 = hole, 0 = keep)
- **`create_inpainting_conditioning(x_original)`** -- returns `[B, 4, H, W]` = `[mask | masked_original]`

Can be run standalone to visualise masks on CelebA images:

```bash
cd diffusion/inpainting
python mask_helper.py
```

### `sampling_helper.py`

Implements the reverse (denoising) process for inpainting. Same math as unconditional, but at every step the fixed conditioning is concatenated with the current noisy image before passing to the model:

- **`p_sample(model, x, t, t_index, conditioning)`** -- single reverse step with 7-ch model input
- **`p_sample_loop(model, shape, conditioning)`** -- full T-step reverse chain
- **`sample(model, conditioning, image_size, channels)`** -- convenience wrapper; batch size is inferred from the conditioning tensor

### `training_celeba_inpaint.py`

Trains the U-Net on CelebA 128x128 with random rectangular masks.

Periodic samples are saved to `results_celeba_inpaint/` as 3-column grids (masked input | inpainted result | original). Final checkpoint is saved to `checkpoints/celeba_unet_inpaint_final.pt`.

```bash
cd diffusion/inpainting
python training_celeba_inpaint.py
```

## How It Differs from Unconditional

The forward diffusion and loss are identical. The only changes are:

1. **Model input is 7 channels** instead of 3:

    | Channels | Content                                     |
    |----------|---------------------------------------------|
    | 0-2      | Noisy image x_t (same as unconditional)     |
    | 3        | Binary mask (1 = hole to inpaint, 0 = keep) |
    | 4-6      | Original image with masked pixels zeroed out |

2. **Conditioning stays fixed** across all reverse steps -- only the noisy image channels evolve during denoising.

3. **Training generates random masks each iteration** so the model learns to inpaint arbitrary rectangular regions.
