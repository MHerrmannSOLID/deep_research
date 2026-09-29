import math
from inspect import isfunction
from functools import partial

#matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_period=10_000):
        super().__init__()
        self.dim = dim
        self.ln_max_period = math.log(max_period)

    def forward(self, time):
        device = time.device    # securing the device (CPU or GPU) the parameter 'time' is on.
                                # So that internally created vectors are being created  on the same device.
        half_dim = self.dim // 2
        # Create a vector with the position indices [0 to half_dim-1]
        positions = torch.arange(half_dim, device=device)
        exponents = positions * self.ln_max_period / (half_dim - 1)
        omeags = torch.exp(-exponents)
        # time[:, None] has shape (batch_size, 1), while omeags[None, :] has
        # shape (1, half_dim). Broadcasting their element-wise product produces
        # one set of frequency samples per batch item: (batch_size, half_dim).
        samples = time[:, None] * omeags[None, :]

        embeddings = torch.cat((samples.sin(), samples.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    #"""
    #https://arxiv.org/abs/1903.10520
    #weight standardization purportedly works synergistically with group normalization
    #"""

    def forward(self, x):
        # The formula for weight standardization is:
        # $\hat{W} = \frac{W - \mu_W}{\sqrt{\sigma_W^2 + \epsilon}}$
        # source : https://arxiv.org/abs/1903.10520
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        # -=| Weights in standard Conv2d |=-
        # Weight dimension in traditional Conv2d is [out_channels, in_channels, H, W]
        # --> We convolve an HxW kernel which has learned weights for each input channel (a 3D volume: [in_channels, H, W])
        # --> This 3D setup exists for EVERY output channel! --> 4D tensor: [out_channels, in_channels, H, W]
        weight = self.weight

        # -=| Weight Standardization |=-
        # We reduce all DOFs other than the output channel dimension (o) to a single scalar value.
        # reduce takes a function used for reduction (e.g., "mean" or "var").
        # It collapses [in_channels, H, W] down to 1 value per output channel, formatted as [out_channels, 1, 1, 1]:
        #[
        #  [ [[ mean_val_0 ]] ],  # Filter 0 average
        #  [ [[ mean_val_1 ]] ],  # Filter 1 average
        #  ...
        #  [ [[ mean_val_n ]] ]   # Filter n average
        #]
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))

        # -=| Normalize & Convolve |=-
        # Subtract mean and scale by inverse standard deviation (broadcasting [out_channels, 1, 1, 1] over 4D weight)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            # FiLM / AdaGN (Feature-wise Linear Modulation):
            # Applies a time-dependent affine transformation: x_mod = x * (1 + scale) + shift
            #
            # Conv layers learn static spatial features (edges, textures, shapes). The time embedding t
            # acts as a dynamic switchboard via scale & shift to gate channels ON/OFF before activation:
            #   - High noise (t ~ T): Suppresses high-frequency channels, boosts global shape channels.
            #   - Low noise  (t ~ 0): Amplifies fine-detail edge detectors for sharpening.
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    # This module executes two blocks (see Block) with optional time embedding conditioning.
    # Execution uses a residual connection (input is added to the output of the two blocks).
    #
    # Parameters:
    #   dim         : Number of input channels to the ResnetBlock.
    #   dim_out     : Number of output channels from the ResnetBlock.
    #                 (If dim != dim_out, a 1x1 convolution aligns the residual path.)
    #   time_emb_dim: Dimension of the optional time embedding input.
    #   groups      : Number of groups for GroupNorm layers within the blocks.
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # Time embedding MLP projection:
        # Activates the embedding (SiLU) and projects it to twice the output dimension (dim_out * 2)
        # using a linear layer so it can later be split into scale and shift parameters.
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        # The two sequential feature blocks
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # Optional 1x1 convolution to match spatial channel dimensions for the residual addition
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None  # Holds the (scale, shift) tuple for FiLM modulation

        # Process the time embedding through the MLP projection if both exist
        if exists(self.mlp) and exists(time_emb):
            # 1. Project time embedding: shape goes from (B, time_emb_dim) -> (B, dim_out * 2)
            time_emb = self.mlp(time_emb)

            # 2. Expand to 4D tensor (B, dim_out * 2, 1, 1) to enable spatial broadcasting
            #    across feature maps (B, C, H, W) in Block
            time_emb = rearrange(time_emb, "b c -> b c 1 1")

            # 3. Split channels into two equal halves: scale (B, dim_out, 1, 1) and shift (B, dim_out, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        # Execute Block 1: Receives feature map x (4D tensor) and scale_shift conditioning parameters.
        # Channel gain (scale) and bias (shift) modulate feature activations based on timestep t.
        h = self.block1(x, scale_shift=scale_shift)

        # Execute Block 2: Refines features without re-injecting time embedding.
        h = self.block2(h)

        # Residual Connection: Adds input features (transformed via 1x1 conv if channels change)
        # to the block outputs. Blocks learn residual updates relative to input state x.
        return h + self.res_conv(x)
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)




class Unet(nn.Module):
    def __init__(
        self,
        dim, # We start with 'dim' channels after the first layer, and then we expand by `dim_mults`
             # when proceeding down the net. 'dim' sets the global "width" scale of the network.
        init_dim=None, # Just in case we want a differnt channel count at the input layer.
        out_dim=None, # The number of output channels. Typically matches the number of input channels
                      # (e.g., 3 for RGB images).
        dim_mults=(1, 2, 4, 8), # Multiplicative factors for the number of channels at each level of the U-Net.
        channels=3, # initial number if image channel usually 3 for RGB images.
        self_condition=False, # self conditioning is an optimization where previous time step predictions
                              # are fed back into the model to improve current predictions.
        resnet_block_groups=4, # Number of groups for GroupNorm in ResNet blocks.
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        # if we go with self condition, we need double the input channel count, to accommodate the
        # concatenated previous time step predictions. That means 2 image with e.g. RGB -->6 channels
        input_channels = channels * (2 if self_condition else 1)

        # If we want the first convolutional layer to have a different number of input channels than the default,
        # where the default would be the same as `dim`,  we handle it here. See parameter list, usually `init_dim=None`.
        init_dim = default(init_dim, dim)
        # set the initial convolutional layer's input and output dimensions.
        # This is a 1x1 convolution, but the original paper was having here a 7x7.
        # Also as u can see it uses `init_dim` for the output channels, So if it was defined in the
        # parameters i would take affect here, otherwhise is just equal to `dim`.
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        #here we convert the width paramert dim with the multiplicative factors specified in `dim_mults`
        # into the specific channel counts . e.g. if dim=64 and dim_mults=(1,2,4,8),
        # dims would be [init_dim (64), 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # dims[:-1] -> All elements except the last -> [64, 64, 128, 256]
        # dims[1:] -> All elements except the first -> [64, 128, 256, 512]
        # And zip pairs them tlement wise -> [(64, 64), (64, 128), (128, 256), (256, 512)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # This is patially presettings one constructor parameter for the ResnetBlock,
        # specifically the number of groups for GroupNorm. So when us later use `block_klass` to create ResnetBlocks,
        # it has already the `groups` parameter set to `resnet_block_groups`. It only needs to be provided with
        # the input and output dimensions when instantiated. see below : 'block_klass(dim_in, dim_in, time_emb_dim=time_dim)'
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings is the output size of the time MLP.
        time_dim = dim * 4

        # This is the nn.Module for processing the time embeddings. It takes a [b, 1] tensor as input which
        # is just one integer per batch element, representing the timestep. This integer is converted into
        # a sinusoidal positional embedding by the `SinusoidalPositionEmbeddings` module and pushed through a
        # small MLP to transform it into the final time embedding.
        # The result of this MLP will later be provided to each ResnetBlock as the time embedding,
        # where it is projected to match the local channel dimensions of that block.
        self.time_mlp = nn.Sequential(
            # First, calculate the deterministic sinusoidal position pattern of size `dim`.
            SinusoidalPositionEmbeddings(dim),
            # Expand the deterministic pattern into a higher-dimensional space (`time_dim`) with a learnable linear layer.
            nn.Linear(dim, time_dim),
            # Apply GELU non-linearity to allow complex non-linear temporal representations.
            nn.GELU(),
            # Output the processed time embedding vector of size `time_dim`, ready for distribution to all ResnetBlocks.
            nn.Linear(time_dim, time_dim),
        )

        # Here we define a sequential nn.Module list for the encoder. This will store all the downsampling blocks.
        self.downs = nn.ModuleList([])
        # Similarly, we define a sequential nn.Module list for the decoder, which will store all the upsampling blocks.
        self.ups = nn.ModuleList([])
        # `num_resolutions` represents the number of downsampling/upsampling stages, will be used to identify the
        # last stage for special handling in the upcomming forloops
        num_resolutions = len(in_out)

        # Now we iterate over each resolution level to construct the downsampling blocks.
        # `ind` is the current resolution index, `dim_in` and `dim_out`
        # are the input and output channel dimensions for this stage.
        # They are sourced from the `in_out` zip pairs from above
        # -> [(64, 64), (64, 128), (128, 256), (256, 512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # true if last downsampling stage
            is_last = ind >= (num_resolutions - 1)
            # the last 2 layer should have full attnetion
            use_full_attn = ind >= (num_resolutions - 2)
            attn_klass = Attention if use_full_attn else LinearAttention
            self.downs.append(
                nn.ModuleList(
                    [
                        # this is straightforward: two residual blocks followed come first
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # Followed by a linear attention block wrapped in a pre-normalization wrapper
                        # and a residual skip connection.
                        # PreNorm applies GroupNorm before passing features to LinearAttention.
                        # With num_groups=1, it normalizes across all channels and spatial pixels
                        # independently per sample (Instance/Layer Normalization), without mixing
                        # statistics across batch elements.
			Residual(PreNorm(dim_in, attn_klass(dim_in))),
                        #Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # Finally, if we are not in the last downsampling stage, we apply a downsampling operation.
                        # Otherwise, we use a simple convolution to match the output dimensions (regarding the channels).
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # Now we are in the middle of the U-Net ! This part will connect the encoder with the decoder
        # (the downsampling path with the upsampling path).
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        # A seciality o these center blocks is that due to the low spacial resolution,
        # we can use the  standard attention (which is O(n^2) in the number of pixels).
        # This standard attention is much more potent than linear attention. However,
        # with high spacial resolution, the standard attention becomes computationally too expensive.
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # This is same like the downsampling, but with an upsampling operation at the end of each stage instead of downsampling.
        # I won't repeat the detailed comments here, as the structure is analogous to the downsampling path.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            use_full_attn = ind < 2
            attn_klass = Attention if use_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        # Here we have one detail we should pay attentons to!
                        # The input size of the residual block is the concatenation of the
                        # corresponding encoder feature map and the current decoder feature map.
                        # This is a property of U-Net architectures, called skip connections.
                        # Therefore, the input channels for the decoder blocks are doubled compared
                        # to the output channels of the corresponding encoder blocks.
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
			Residual(PreNorm(dim_out, attn_klass(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        # if there was no specified output dimension (constructor argument),
        # we default to the number of input channels.
        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        # The final convolution layer maps the features to the desired output dimension.
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    # The forward method defines how input data flows through the network.
    # It takes the input image `x`, the timestep `time`, and an optional self-conditioning input `x_self_cond`.
    def forward(self, x, time, x_self_cond=None):

        # If self-conditioning is enabled, concatenate the previous prediction with the input along the channel dimension.
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # Start the forward pass by applying the initial 1x1 convolution to project input into `init_dim` channels.
        x = self.init_conv(x)
        r = x.clone()  # Keep a copy of the initial features for a final residual skip connection at the very end.

        # Process the timestep integer through the time MLP to produce a `time_dim` conditioning vector.
        t = self.time_mlp(time)

        # U-Net skip connections preserve high-resolution spatial details from the encoder (downsampling)
        # to the decoder (upsampling).
        # Unlike residual connections (which add features element-wise), U-Net skip connections concatenate
        # encoder feature maps directly onto decoder feature maps along the channel dimension (`dim=1`).
        # As we pass down the encoder, intermediate feature maps are pushed onto `h` like a LIFO stack,
        # and then popped off in reverse order during the upsampling phase.
        h = []  # LIFO stack storing intermediate feature maps for encoder-decoder skip connections

        # Downsampling path (encoder) with skip connectionsbeing stored in h.
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x) # Store the output of the first block for skip connection

            x = block2(x, t)
            x = attn(x)
            h.append(x) # Store the output of the second block for skip connection

            x = downsample(x)

        # Middle (bottleneck) part of the U-Net, where the feature maps have the lowest spatial resolution.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling path (decoder) with skip connections being retrieved from h in reverse order.
        for block1, block2, attn, upsample in self.ups:
            # Concatenate the corresponding encoder feature map for the first block in the decoder stage.
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # Concatenate the corresponding encoder feature map for the second block in the decoder stage.
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1) # Here we are at the very top again, and we concatenate
                                     # the initial features for a final residual skip connection.

        x = self.final_res_block(x, t)
        return self.final_conv(x)

######### Here we are ! The model is defined !! Now lets train  🙃
