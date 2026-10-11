//! A Qwen3 stack's shape, as the checkpoints that embed one spell it.

use anyhow::Result;
use serde::Deserialize;

/// The fields of a Qwen3 text config the stack reads. The rotary base is `rope_theta` in older
/// configs and `rope_parameters.rope_theta` in newer ones.
#[derive(Clone, Debug, Deserialize)]
pub struct Stack {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    rope_theta: Option<f32>,
    rope_parameters: Option<RopeParameters>,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

impl Stack {
    pub fn rope_theta(&self) -> Result<f32> {
        let theta = self.rope_theta.or(self.rope_parameters.as_ref().map(|p| p.rope_theta));
        theta.ok_or_else(|| anyhow::anyhow!("neither rope_theta nor rope_parameters.rope_theta is set"))
    }
}
