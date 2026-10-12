# ngu_net.py
This is short for NextGen U-Net. It's a U-Net implemented after the Hugging Face blog [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion). However, I added quite a good amount of explanatory comments, which had not been added to the blog post for some reason.

## Modifications
* Comments which hopefully lead to a better understanding
* Flexible attention: Full attention is not limited to the bottleneck, but can also be added to the up-/downsampling path
* Configurability: many additional parameters to control the model size and behavior
* Injectable FiLM conditioning: The original implementation hardcoded a time-embedding MLP inside the U-Net, coupling it to diffusion. This has been refactored so that the conditioning module (`film_module`) is injected via the constructor. This makes the U-Net task-agnostic — pass a `SinusoidalTimeFiLM` for diffusion, a combined time+class module for class-conditional generation, or `None` for tasks like segmentation that need no conditioning at all

## Architecture Overview

The model follows the classic U-Net encoder-decoder structure with skip connections, adapted for diffusion:

```
Input (B, C, H, W)
  |
  v
[1x1 Conv] -------- initial projection to `init_dim` channels -------- (copy saved for final residual)
  |
  v
=== Encoder (Downsampling Path) ===
For each resolution level:
  [ResnetBlock] -> [ResnetBlock] -> [Attention] -> [Downsample]
  |                                                     |
  (skip connections stored)                        (spatial size halved)
  |
  v
=== Bottleneck ===
  [ResnetBlock] -> [Full Attention] -> [ResnetBlock]
  |
  v
=== Decoder (Upsampling Path) ===
For each resolution level (reversed):
  [cat skip] -> [ResnetBlock] -> [cat skip] -> [ResnetBlock] -> [Attention] -> [Upsample]
  |
  v
[cat initial residual] -> [ResnetBlock] -> [1x1 Conv] -> Output (B, out_dim, H, W)
```

**FiLM conditioning** (Feature-wise Linear Modulation) is handled via an injectable `film_module`. The U-Net itself is task-agnostic — the `film_module` transforms raw conditioning input (e.g. timesteps, class labels, or both) into an embedding vector, which each `ResnetBlock` then projects and splits into scale and shift vectors to modulate its feature maps. For diffusion, you inject a `SinusoidalTimeFiLM`; for segmentation or other unconditional tasks, you pass `film_module=None` and the U-Net operates without conditioning.

**Skip connections** work as a LIFO stack: encoder feature maps are pushed during downsampling and popped during upsampling. They are concatenated (not added) along the channel dimension, which is why decoder blocks have doubled input channels compared to their encoder counterparts.

## Constructor Parameters

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `dim` | `int` | *required* | Base channel width after the initial convolution. All subsequent layer widths are derived from this via `dim_mults`. Controls the overall capacity of the network. |
| `init_dim` | `int \| None` | `None` | Channel count after the very first 1x1 convolution. If `None`, defaults to `dim`. Useful if you want the input projection to differ from the base width. |
| `out_dim` | `int \| None` | `None` | Number of output channels. If `None`, defaults to `channels` (i.e., the model predicts a tensor with the same shape as the input image). |
| `dim_mults` | `tuple[int]` | `(1, 2, 4, 8)` | Multipliers applied to `dim` at each encoder/decoder level. Defines both the depth (number of resolution levels) and the channel progression. E.g., `dim=64` with `(1, 2, 4, 8)` gives channel widths of 64, 128, 256, 512. |
| `channels` | `int` | `3` | Number of input image channels (3 for RGB). Determines the input size of the initial convolution. |
| `self_condition` | `bool` | `False` | Enables self-conditioning: the model's prediction from the previous timestep is concatenated with the current noisy input along the channel dimension, doubling the effective input channels. |
| `resnet_block_groups` | `int` | `4` | Number of groups for GroupNorm inside each `ResnetBlock`. Must evenly divide the channel count at every level. Lower values = more groups = finer normalization granularity. |
| `full_attention_count` | `int` | `2` | How many of the deepest resolution levels use full (O(n^2)) attention instead of linear attention. `0` means only the bottleneck gets full attention. Higher values improve global coherence at the cost of VRAM. |
| `attn_heads` | `int` | `4` | Number of attention heads in both full and linear attention layers. More heads let the model attend to different representation subspaces in parallel. |
| `attn_dim_head` | `int` | `32` | Dimension per attention head. The total hidden dimension of each attention layer is `attn_heads * attn_dim_head`. Increasing this adds capacity per head. |
| `dropout` | `float` | `0.0` | Dropout probability applied inside each `ResnetBlock` between the first and second sub-block. Acts as regularization to reduce overfitting, especially important for smaller datasets. |
| `film_module` | `nn.Module \| None` | `None` | Injectable conditioning module. Must output a `(B, film_dim)` tensor. When `None`, the U-Net operates without any conditioning (e.g. for segmentation). |
| `film_dim` | `int \| None` | `None` | Output dimension of `film_module`. Must be provided when `film_module` is not `None`. Each `ResnetBlock` projects from `film_dim` to its local channel count internally. |

### Derived dimensions

A few internal dimensions are computed from the parameters above:

- **Channel widths per level**: `[init_dim, dim*m1, dim*m2, ...]` from `dim_mults`
- **Attention hidden dimension**: `attn_heads * attn_dim_head` — shared across all attention layers

### Usage example

```python
import torch
from models.ngu_net import Unet, SinusoidalTimeFiLM

# --- Diffusion (with time-based FiLM conditioning) ---
dim = 64
film_dim = dim * 4
film = SinusoidalTimeFiLM(sinusoidal_dim=dim, film_dim=film_dim)

model = Unet(
    dim=dim,
    channels=3,
    dim_mults=(1, 2, 4, 8),
    film_module=film,
    film_dim=film_dim,
)

x = torch.randn(4, 3, 128, 128)       # batch of 4 RGB images
t = torch.randint(0, 1000, (4,))       # random timesteps
predicted_noise = model(x, t)           # same shape as x: (4, 3, 128, 128)

# --- Segmentation (no conditioning) ---
seg_model = Unet(
    dim=64,
    channels=3,
    out_dim=10,                         # e.g. 10 segmentation classes
    dim_mults=(1, 2, 4, 8),
)

logits = seg_model(x)                   # (4, 10, 128, 128)
```

## Building Blocks

| Class | Role |
|---|---|
| `SinusoidalPositionEmbeddings` | Encodes scalar timestep into a fixed-frequency positional vector |
| `SinusoidalTimeFiLM` | Standalone FiLM module: sinusoidal encoding + 2-layer MLP. Inject into `Unet` for diffusion training |
| `WeightStandardizedConv2d` | Conv2d with weight standardization ([Qiao et al. 2019](https://arxiv.org/abs/1903.10520)), works synergistically with GroupNorm |
| `Block` | Single conv-norm-activation unit with optional FiLM conditioning |
| `ResnetBlock` | Two `Block`s + residual connection + FiLM projection + dropout |
| `Attention` | Standard multi-head self-attention (O(n^2)), used at low spatial resolutions |
| `LinearAttention` | Approximate attention (O(n)) via kernel factorization, used at high spatial resolutions |
| `PreNorm` | Applies GroupNorm before a wrapped module |
| `Residual` | Wraps a module with an additive skip connection |
