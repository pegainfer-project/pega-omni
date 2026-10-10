//! The shape the engine is built for, and the checkpoint's `config.json` checked against it.

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

pub use omni_qwen3::config::Stack;

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

/// The body's shape, from a checkpoint that is the discrete-codebook Higgs TTS with a tied fused head.
pub fn load(dir: &Path) -> Result<Stack> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let raw: Raw = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let a = &raw.audio_encoder_config;
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
    ensure!(
        a.out_dim == raw.text_config.hidden_size,
        "audio width {} against body width {}",
        a.out_dim,
        raw.text_config.hidden_size
    );
    Ok(raw.text_config)
}
