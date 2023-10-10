# Diffusion Probabilistic Model with DDPMSampler

This GitHub repository contains a collection of Python code for implementing various probabilistic generative models and embedding techniques. These models are designed for <b>image enhancement, generative tasks, and probabilistic modeling</b>, offering a versatile set of tools for working with <i>image data and text embeddings</i>. Below is a summary of the key components and models included in this repository:

## Variational Autoencoder (VAE) Architecture

The VAE architecture is designed for image enhancement, generative tasks, and probabilistic modeling. It includes components such as denoising, inpainting, and image generation:

- VAE Encoder: A module that includes convolutional layers, residual blocks, and attention blocks for encoding input images.

- VAE Decoder: A module for decoding and generating enhanced images from latent representations.


## Diffusion Probabilistic Model
The diffusion probabilistic model, implemented in the Diffusion class, is a powerful generative model for image data. It takes into account latent variables, context information, and time embeddings to generate images. Key features include:

- Diffusion Model: A module for initializing the diffusion model, which can be used for tasks like image denoising, enhancement, and generation.

- DDPMSampler: A class for facilitating sampling from the diffusion model, allowing users to control noise strength and generate images at specific timesteps.

## CLIP Embedding
The CLIP (Contrastive Language-Image Pre-training) embedding model is designed for text and image embeddings including natural language understanding, image-text matching, and cross-modal applications. It includes the following components:

- CLIPEmbedding: A module for embedding tokens (text or image) using a combination of token embeddings and learnable position embeddings.

- CLIPLayer: A layer that performs self-attention and feedforward operations, allowing for the learning of complex relationships between tokens.

- CLIP Model: A complete CLIP model that combines the embedding and multiple CLIP layers for generating embeddings from tokens.

This repository provides a comprehensive set of tools for working with probabilistic generative models and embeddings, making it a valuable resource for researchers and developers working on image and text-related tasks. Each component comes with its own API and usage examples, ensuring flexibility and ease of integration into different projects.

For more information regarding DDPM or the model, read these papers for more thorough understanding [here](https://arxiv.org/abs/2006.11239v2) and [here](https://arxiv.org/abs/1706.03762)




