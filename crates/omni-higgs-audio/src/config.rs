//! The checkpoint's `config.json`, reduced to what the engine reads and checked
//! against the one shape the engine is built for.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;

/// Codebooks per frame, and the codes each draws from (the last two are the delay pattern's
/// begin and end of codec).
pub const CODEBOOKS: usize = 8;
pub const VOCAB: usize = 1026;
/// PCM samples per frame: 24 kHz at 25 frames per second.
pub const SAMPLES_PER_FRAME: usize = 960;
pub const SAMPLE_RATE: u32 = 24_000;

/// The Qwen3 body's shape.
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
    pub max_position_embeddings: usize,
    rope_parameters: Rope,
}

#[derive(Clone, Debug, Deserialize)]
struct Rope {
    rope_theta: f32,
}

impl Stack {
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Audio {
    encoder_type: String,
    num_codebooks: usize,
    vocab_size: usize,
    out_dim: usize,
    tie_word_embeddings: bool,
    use_delay_pattern: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct Raw {
    architectures: Vec<String>,
    text_config: Stack,
    audio_encoder_config: Audio,
}

/// A checkpoint the engine can run: the discrete-codebook Higgs TTS with a tied fused head.
#[derive(Clone, Debug)]
pub struct Config {
    pub stack: Stack,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("{}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let raw: Raw = serde_json::from_str(text)?;
        let (s, a) = (&raw.text_config, &raw.audio_encoder_config);
        ensure!(
            raw.architectures == ["HiggsMultimodalQwen3ForConditionalGeneration"],
            "architecture {:?}, the engine runs HiggsMultimodalQwen3ForConditionalGeneration",
            raw.architectures
        );
        ensure!(a.encoder_type == "discrete", "audio encoder `{}`, the engine runs `discrete`", a.encoder_type);
        ensure!(
            a.num_codebooks == CODEBOOKS && a.vocab_size == VOCAB,
            "{} codebooks of {} codes, the engine is built for {CODEBOOKS} of {VOCAB}",
            a.num_codebooks,
            a.vocab_size
        );
        ensure!(a.use_delay_pattern, "the engine runs the delay pattern");
        ensure!(a.tie_word_embeddings, "the engine reads the audio head from the embedding");
        ensure!(a.out_dim == s.hidden_size, "audio width {} against body width {}", a.out_dim, s.hidden_size);
        ensure!(s.head_dim == 128, "head_dim {}, attention is built for 128", s.head_dim);
        ensure!(
            s.num_attention_heads % s.num_key_value_heads == 0 && s.num_attention_heads / s.num_key_value_heads <= 4,
            "{} query heads over {} key heads",
            s.num_attention_heads,
            s.num_key_value_heads
        );
        ensure!(s.hidden_size % 256 == 0 && s.hidden_size <= 8192, "hidden size {}", s.hidden_size);
        Ok(Self { stack: raw.text_config })
    }
}
