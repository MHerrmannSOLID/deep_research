# Deep Research

This repository is a collection of deep learning architecture experiments, mostly implemented in PyTorch. The projects are intended both to build a deeper understanding of the models and to gain hands-on experience implementing them.

> You don't truly understand something until you build it.

Each project explores a different model or application and may have its own documentation. The repository is organized by topic: for example, `diffusion/` contains diffusion model experiments, grouped into subdirectories by application.

## Running a module

Run modules from the repository root so that imports resolve correctly. For example, to train the diffusion inpainting model, use:

```bash
python -m diffusion.inpainting.training_celeba_inpaint
```

## Project documentation

- [Diffusion Models](diffusion/README.md) — Overview of the diffusion experiments, shared modules, and U-Net variants.
- [Unconditional Diffusion](diffusion/unconditional/README.md) — Documentation for generating CelebA face images from noise.
- [Inpainting Diffusion](diffusion/inpainting/README.md) — Documentation for filling masked regions of CelebA face images.
