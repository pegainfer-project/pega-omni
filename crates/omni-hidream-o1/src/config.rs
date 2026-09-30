//! The checkpoint's `config.json`, reduced to the fields the engine reads, and
//! the constants the reference implementation fixes in code.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;

/// Pixels per patch side: a token is one 32 x 32 RGB patch.
pub const PATCH: usize = 32;
/// Values per patch token, channel-major: `3 * 32 * 32`.
pub const PATCH_DIM: usize = 3 * PATCH * PATCH;
/// The first M-RoPE position of the image grid, on every axis.
pub const IMAGE_POSITION: i32 = 4096;
/// Width of the sinusoid the timestep embedder reads.
pub const TIMESTEP_FREQUENCIES: usize = 256;
/// The special tokens that close a prompt: begin-of-image, then the timestep slot.
pub const BOI_TOKEN: &str = "<|boi_token|>";
pub const TMS_TOKEN: &str = "<|tms_token|>";

/// The Qwen3-VL text tower the model generates with.
#[derive(Clone, Debug, Deserialize)]
pub struct Text {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    rope_scaling: RopeScaling,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeScaling {
    mrope_section: [usize; 3],
    #[serde(default)]
    mrope_interleaved: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct File {
    text_config: Text,
}

impl Text {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let text =
            serde_json::from_str::<File>(&raw).with_context(|| format!("parsing {}", path.display()))?.text_config;
        let [t, h, w] = text.rope_scaling.mrope_section;
        ensure!(text.head_dim == 128, "head_dim {} unsupported (attention is built for 128)", text.head_dim);
        ensure!(text.rope_scaling.mrope_interleaved, "only interleaved M-RoPE is implemented");
        ensure!(h == w && t + h + w == text.head_dim / 2, "mrope_section {:?} unsupported", [t, h, w]);
        Ok(text)
    }

    /// Rotary pairs the h axis (and, equally, the w axis) owns.
    pub fn mrope_hw(&self) -> usize {
        self.rope_scaling.mrope_section[1]
    }

    pub fn qkv_width(&self) -> usize {
        (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
    }
}
