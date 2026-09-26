//! PersonaPlex-7B's shape. The checkpoint carries no usable config, so these
//! are the reference's constants (`moshi/models/loaders.py`), and loading
//! checks every tensor against them.

/// Helium, the temporal transformer.
pub const DIM: usize = 4096;
pub const LAYERS: usize = 32;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const HIDDEN: usize = 11264;
/// Positions a query attends to, itself included: the KV ring's length.
pub const CONTEXT: usize = 3000;
pub const NORM_EPS: f32 = 1e-8;
pub const ROPE_PERIOD: f32 = 10_000.0;

/// The depformer, one small transformer pass per codebook of a frame.
pub const DEP_DIM: usize = 1024;
pub const DEP_LAYERS: usize = 6;
pub const DEP_HEADS: usize = 16;
pub const DEP_HEAD_DIM: usize = 64;
pub const DEP_HIDDEN: usize = 2816;
/// Depformer steps with weights in the checkpoint.
pub const DEP_STEPS_STORED: usize = 16;

/// Token streams of a step: text, the agent's 8 codebooks, the caller's 8.
pub const STREAMS: usize = 17;
pub const CODEBOOKS: usize = 8;
/// Tokens drawn per step: text, then the agent's codebooks.
pub const DRAWN: usize = 1 + CODEBOOKS;
pub const DELAYS: [usize; STREAMS] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
/// Slots of the reference's token ring (`max(DELAYS) + 3`).
pub const RING: usize = 4;
pub const TEXT_VOCAB: usize = 32_000;
pub const CARD: usize = 2048;
/// The text stream's "nothing said" token.
pub const TEXT_PAD: i32 = 3;
pub const SILENCE: [i32; CODEBOOKS] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// The caller stream during prompts: a quiet 440 Hz tone, encoded.
pub const SINE: [i32; CODEBOOKS] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];
/// Silence steps before and after the role prompt (0.5 s).
pub const SILENCE_STEPS: usize = 6;

/// Mimi, the codec.
pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME: usize = 1920;
pub const MIMI_DIM: usize = 512;
pub const MIMI_LAYERS: usize = 8;
pub const MIMI_HEADS: usize = 8;
pub const MIMI_HIDDEN: usize = 2048;
pub const MIMI_CONTEXT: usize = 250;
pub const MIMI_FILTERS: usize = 64;
/// SEANet's decoder ratios; the encoder runs them reversed.
pub const RATIOS: [usize; 4] = [8, 6, 5, 4];
pub const CODEBOOK_DIM: usize = 256;

/// The reference's sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub text_temperature: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_k: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { text_temperature: 0.7, text_top_k: 25, audio_temperature: 0.8, audio_top_k: 250 }
    }
}
