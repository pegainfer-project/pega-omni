//! PersonaPlex-7B (a Moshi-architecture full-duplex speech model) on one GPU.
//!
//! Every 80 ms tick, every live session advances by one frame: [`mimi`]
//! encodes the caller's frame, [`helium`] and [`depformer`] draw the agent's
//! text token and eight codebooks, and Mimi decodes the agent's frame, all
//! in one kern manifest call ([`model::Model::tick`]) that is one CUDA graph
//! per batch bucket. A session's [`prompt`] (its [`voice`] and role prompt)
//! is prefilled once when it opens; its KV is a fixed ring of
//! [`config::CONTEXT`] positions, so a session's memory never grows.
//! [`engine`] is the clock around it.

pub mod config;
mod depformer;
pub mod engine;
mod helium;
mod mimi;
pub mod model;
pub mod prompt;
pub mod tokenizer;
pub mod voice;
