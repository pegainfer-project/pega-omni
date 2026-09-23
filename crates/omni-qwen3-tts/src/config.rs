//! The checkpoint's JSON configs, reduced to the fields the engine reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// A Qwen3 decoder stack's shape.
#[derive(Clone, Debug, Deserialize)]
pub struct Stack {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Talker {
    #[serde(flatten)]
    pub stack: Stack,
    pub num_code_groups: usize,
    pub code_predictor_config: Stack,
    pub codec_bos_id: i32,
    pub codec_eos_token_id: i32,
    pub codec_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    pub codec_language_id: BTreeMap<String, i32>,
    pub spk_id: BTreeMap<String, i32>,
    /// `false`, or the dialect a speaker defaults to.
    pub spk_is_dialect: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    pub talker_config: Talker,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Generation {
    pub temperature: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: i32,
    pub max_new_tokens: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Codec {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct CodecFile {
    output_sample_rate: u32,
    decode_upsample_rate: usize,
    decoder_config: Codec,
}

/// Everything read from a checkpoint directory's JSON.
#[derive(Clone, Debug)]
pub struct Config {
    pub model: Model,
    pub generation: Generation,
    pub codec: Codec,
    pub sample_rate: u32,
    /// PCM samples per codec frame.
    pub samples_per_frame: usize,
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let codec: CodecFile = read(&dir.join("speech_tokenizer/config.json"))?;
        Ok(Self {
            model: read(&dir.join("config.json"))?,
            generation: read(&dir.join("generation_config.json"))?,
            sample_rate: codec.output_sample_rate,
            samples_per_frame: codec.decode_upsample_rate,
            codec: codec.decoder_config,
        })
    }
}
