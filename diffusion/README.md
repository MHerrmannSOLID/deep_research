# Diffusion Models

DDPM-based image diffusion experiments on CelebA 128x128 faces.

## Structure

```
diffusion/
├── diffusion_noise_helper.py      # Forward diffusion: noise schedules, q_sample, p_losses
├── image_data_helper.py           # Shared: CelebA dataloader, postprocess [-1,1] -> [0,1]
├── unconditional/                 # Unconditional image generation (3-ch input)
└── inpainting/                    # Inpainting with mask conditioning (7-ch input)
```

## Shared Modules

### `diffusion_noise_helper.py`

Implements the forward diffusion process and training loss. Provides:

- **Linear beta schedule** from 1e-4 to 0.02 over 1000 timesteps
- **`q_sample(x_start, t)`** -- add noise to a clean image at timestep `t`
- **`p_losses(model, x_start, t)`** -- compute MSE/L1/Huber loss between true and predicted noise
- **`extract(a, t, x_shape)`** -- index into schedule tensors with proper broadcasting

Also contains demo utilities (`get_noisy_image`, `plot`) and can be run standalone to visualise the forward noising process on a sample image.

### `image_data_helper.py`

Shared data utilities used by both training scripts:

- **`load_celeba_dataloader(image_size, batch_size)`** -- loads CelebA from HuggingFace, applies resize + flip + normalisation to [-1, 1]
- **`postprocess(tensor_imgs)`** -- un-normalise from [-1, 1] to [0, 1] for display/saving

## Model

Both variants use the same U-Net architecture (`models/ngu_net.py`) with FiLM-based timestep conditioning via sinusoidal embeddings. The only difference is the number of input channels:

| Variant       | Input channels | Output channels | U-Net `channels` | U-Net `out_dim` |
|---------------|---------------|-----------------|-------------------|-----------------|
| Unconditional | 3 (noisy RGB) | 3 (noise)       | 3                 | default (3)     |
| Inpainting    | 7 (noisy RGB + mask + masked original) | 3 (noise) | 7 | 3 |
