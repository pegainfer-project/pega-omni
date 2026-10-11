//! The Qwen3 decoder stack as kern manifest calls, shared by the model crates built on it.
//!
//! [`Stack`] registers a checkpoint's layers as weights and emits one pass over token rows; the
//! kernels it launches are [`CUBIN`], which a model's manifest lists as its `qwen3` module.

pub mod config;
mod stack;

pub use stack::HEAD_DIM;
pub use stack::Keep;
pub use stack::Kv;
pub use stack::PAGE;
pub use stack::Pass;
pub use stack::Scratch;
pub use stack::Stack;

/// The stack's kernels (`kernels/qwen3.cu`) compiled for `OMNI_CUDA_ARCH`.
pub const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen3.cubin"));
