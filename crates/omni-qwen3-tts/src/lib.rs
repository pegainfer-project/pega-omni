//! Qwen3-TTS (12 Hz tokenizer, CustomVoice) on one GPU.
//!
//! [`model::Model`] is the whole synthesis path as one kern manifest:
//! [`talker`] turns a [`prompt`] into codec frames one call at a time,
//! [`codec`] turns frames into PCM in the same call. [`engine`] runs a batch
//! of requests through it behind an [`omni_engine::Inbox`].

mod codec;
pub mod config;
pub mod engine;
mod manifest;
pub mod model;
pub mod prompt;
mod stack;
mod talker;
pub mod weights;
