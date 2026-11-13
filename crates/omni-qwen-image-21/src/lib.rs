//! Qwen-Image-2.1 text-to-image on one GPU.
//!
//! Qwen-Image-2.1 is a single-stream diffusion transformer over one joint
//! sequence: the prompt's text tokens and the picture's VAE latents, one token
//! per 16 x 16 pixel cell, all in the same 3-axis RoPE. The text encoder is a
//! Qwen3-VL sidecar ([`sidecar`]) — this crate owns neither its tokenizer nor
//! its weights, only the embeddings and the token count it hands over. Every
//! denoising step is one forward of that sequence, run as one kern manifest
//! over our own kernels (`kernels/qwen.cu`) and cuBLASLt ([`model`]).
//! [`config`] reads the pipeline directory, [`weights`] the transformer's
//! shards, [`prompt`] the latent grid and its positions, [`sampler`] the
//! flow-matching schedule, and [`engine`] serves requests behind the image
//! contract of `omni-engine`.

pub mod config;
pub mod engine;
pub mod gemm;
pub mod model;
pub mod prompt;
pub mod sampler;
pub mod sidecar;
pub mod weights;
