from io import BytesIO
from pathlib import Path

import requests
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMAGE_URL = "https://picsum.photos/800/600"
MODEL_ID = "stabilityai/sd-vae-ft-mse"
ORIGINAL_IMAGE_PATH = Path("original.png")
RECONSTRUCTION_IMAGE_PATH = Path("reconstruction.png")


def load_random_image() -> Image.Image:
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


@torch.inference_mode()
def reconstruct(image: Image.Image) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    autoencoder = AutoencoderKL.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    
    # Unsqueeze to adds a dimension at the beginning to create a batch dimension (of 1)
    image_tensor = pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    # Convert the image to a tensor and normalize it to the range [-1, 1]
    image_tensor = image_tensor / 127.5 - 1.0

    latent = autoencoder.encode(image_tensor).latent_dist.mode()
    decoded = autoencoder.decode(latent).sample
    decoded = ((decoded.clamp(-1, 1) + 1) / 2).cpu().squeeze(0)

    return Image.fromarray(
        (decoded.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    )


def main() -> None:
    original = load_random_image()
    decoded = reconstruct(original)

    original.save(ORIGINAL_IMAGE_PATH)
    decoded.save(RECONSTRUCTION_IMAGE_PATH)
    print(f"Saved original image to {ORIGINAL_IMAGE_PATH.resolve()}")
    print(f"Saved reconstruction to {RECONSTRUCTION_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()