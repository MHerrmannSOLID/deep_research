import torch


def generate_random_rectangle_mask(batch_size, height, width, min_ratio=0.2, max_ratio=0.6):
    """
    Generate random rectangular binary masks for a batch of images.

    Each mask contains a single rectangle where pixels are set to 1.0 (the "hole"
    that the diffusion model will learn to inpaint). Everything outside the
    rectangle is 0.0 (the region that stays visible).

    Mask convention (following the majority of inpainting literature):
        1 = inpaint this pixel  (hole / cut-out region)
        0 = keep this pixel     (visible context the model can use)

    Args:
        batch_size: Number of masks to generate.
        height:     Image height in pixels.
        width:      Image width in pixels.
        min_ratio:  Minimum fraction of each dimension that the rectangle can span.
                    E.g. 0.2 means the rectangle is at least 20% of the image width/height.
        max_ratio:  Maximum fraction of each dimension that the rectangle can span.

    Returns:
        masks: Tensor of shape [B, 1, H, W] with values in {0.0, 1.0}.
    """
    masks = torch.zeros(batch_size, 1, height, width)

    for i in range(batch_size):
        # Randomly choose the rectangle's height and width as a fraction of the image
        rect_h = torch.randint(int(height * min_ratio), int(height * max_ratio) + 1, (1,)).item()
        rect_w = torch.randint(int(width * min_ratio), int(width * max_ratio) + 1, (1,)).item()

        # Randomly place the rectangle's top-left corner so it fits within the image
        top = torch.randint(0, height - rect_h + 1, (1,)).item()
        left = torch.randint(0, width - rect_w + 1, (1,)).item()

        # Set the rectangular region to 1.0 (this is the hole to be inpainted)
        masks[i, :, top:top + rect_h, left:left + rect_w] = 1.0

    return masks


def create_inpainting_conditioning(x_original):
    """
    Build the 4-channel conditioning input for inpainting diffusion.

    Given a batch of original (clean) images, this function:
    1. Generates a random rectangular mask per image.
    2. Blacks out the masked region in the original image.
    3. Concatenates the mask and the masked image along the channel dimension.

    The resulting tensor is meant to be concatenated with the noisy image x_t
    (produced by the forward diffusion process) to form the full 7-channel
    model input:

        model_input = cat([x_noisy,      conditioning  ], dim=1)
                           [B,3,H,W]     [B,4,H,W]
                         = [B,7,H,W]

    Channel layout of the returned conditioning tensor:
        Channel 0:     Binary mask        (1 = hole, 0 = keep)
        Channels 1-3:  Masked original    (visible pixels kept, hole zeroed out)

    Args:
        x_original: Batch of clean images, shape [B, 3, H, W].
                    Expected value range is [-1, 1] (standard diffusion normalisation).

    Returns:
        conditioning: Tensor of shape [B, 4, H, W] on the same device as x_original.
    """
    batch_size, _, height, width = x_original.shape

    # Generate one random rectangle mask per image in the batch
    mask = generate_random_rectangle_mask(batch_size, height, width)
    mask = mask.to(x_original.device)

    # Zero out the hole in the original image.
    # Where mask == 1 (hole), the pixel becomes 0; where mask == 0 (keep), it stays.
    masked_image = x_original * (1.0 - mask)

    # Stack mask (1 channel) and masked image (3 channels) → 4 channels
    conditioning = torch.cat([mask, masked_image], dim=1)

    return conditioning


# ---------------------------------------------------------------------------
# Demo: visualise masks on real CelebA images
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from torchvision.transforms import Compose, Resize, ToTensor, Lambda

    image_size = 128

    # Same preprocessing as the unconditional training script
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),  # scale to [-1, 1]
    ])

    print("Loading CelebA dataset …")
    dataset = load_dataset("nielsr/CelebA-faces", split="train")

    # Grab a small batch of images
    num_samples = 4
    images = []
    for idx in range(num_samples):
        img = dataset[idx]["image"].convert("RGB")
        images.append(transform(img))
    batch = torch.stack(images)  # [4, 3, 128, 128]

    # Generate the 4-channel conditioning tensor
    conditioning = create_inpainting_conditioning(batch)

    # Split it back apart for visualisation
    mask = conditioning[:, 0:1, :, :]    # [B, 1, H, W]
    masked_image = conditioning[:, 1:, :, :]  # [B, 3, H, W]

    # Un-normalise from [-1, 1] back to [0, 1] for display
    def to_display(t):
        return ((t + 1) * 0.5).clamp(0, 1)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    column_titles = ["Original", "Mask (white = hole)", "Masked Image"]

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(to_display(batch[i]).permute(1, 2, 0).numpy())
        # Mask — single channel, displayed in grey-scale
        axes[i, 1].imshow(mask[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        # Masked image — hole is black
        axes[i, 2].imshow(to_display(masked_image[i]).permute(1, 2, 0).numpy())

        for j in range(3):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(column_titles[j])

    plt.tight_layout()
    plt.savefig("mask_helper_demo.png", dpi=120)
    plt.show()
    print("Demo image saved to mask_helper_demo.png")
