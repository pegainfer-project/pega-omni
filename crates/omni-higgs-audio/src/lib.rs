//! Higgs TTS 3 (4B, 8 codebooks at 25 Hz) on one GPU.
//!
//! The model is a Qwen3 body over text and audio tokens: a frame is eight codes drawn from one
//! fused head, laid out by the [`delay`] pattern, and a codec turns frames into 24 kHz PCM.

pub mod config;
pub mod delay;
pub mod prompt;
