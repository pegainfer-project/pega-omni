//! Qwen3-TTS (12 Hz tokenizer, CustomVoice) on one GPU.
//!
//! [`Model`] is the whole synthesis path: [`talker`] turns a [`prompt`] into
//! codec frames one step at a time, [`codec`] turns frames into PCM.
//! [`engine`] runs a batch of requests through it behind an
//! [`omni_engine::Inbox`].

pub mod codec;
pub mod config;
pub mod engine;
pub mod prompt;
pub mod stack;
pub mod talker;
pub mod weights;
