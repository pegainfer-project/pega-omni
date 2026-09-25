//! HiDream-O1-Image text-to-image on one GPU.
//!
//! HiDream-O1 is Qwen3-VL-8B's text tower trained as a pixel-space diffusion
//! transformer: no VAE and no separate text encoder. A picture is a grid of
//! 32 x 32 RGB patches, each one token in the same sequence as the prompt, and
//! every denoising step is one forward of that sequence. The model and its
//! sampler run as one kern manifest over our own kernels (`kernels/hidream.cu`)
//! and cuBLASLt ([`model`]). The distilled checkpoints (`-Dev`, `-Dev-2604`)
//! sample in 28 steps without guidance ([`sampler`]). [`prompt`] builds the
//! sequence and its M-RoPE positions, [`weights`] reads the f32 shards, and
//! [`engine`] serves requests behind the image contract of `omni-engine`.

pub mod config;
pub mod engine;
mod manifest;
pub mod model;
pub mod prompt;
pub mod sampler;
pub mod weights;
