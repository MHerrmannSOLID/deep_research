# ngu_net.py
This is short for NextGen U-Net. It's a U-Net implmented after the hugging face blog [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion). However, I added quite a good amount of explainatory comments, which had not been added to the blog post for some reason.

## Modifications
* Comments which hopefully leas to a better understanding 
* Fexibale attention: Full attention is not limited to the bottleneck, but can also be added to the up- downsampling
* Configerability, many additional parameters to cpntrol the model size

