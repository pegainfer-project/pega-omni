//! The checkpoint's configuration, reduced to the fields the engine reads, and
//! the constants the reference implementation fixes in code.
//!
//! A checkpoint is a diffusers pipeline directory: `transformer/config.json`
//! sizes the single-stream transformer, `scheduler/scheduler_config.json` the
//! flow-matching schedule, and `model_index.json` carries the distilled
//! checkpoints' fixed sampling grid (`sample_sigmas`).

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// Channels of a VAE latent, and so of one image token.
pub const LATENT_DIM: usize = 64;
/// Pixels per latent cell: a picture is a `(height / 16, width / 16)` token grid.
pub const VAE_SCALE: usize = 16;
/// RoPE positions counted up from 0.
pub const ROPE_POS: usize = 8192;
/// RoPE positions counted down from -1, stored after the positive ones.
pub const ROPE_NEG: usize = 1024;
/// Rows of the RoPE table: a position `p < 0` reads row `p + ROPE_TABLE`.
pub const ROPE_TABLE: usize = ROPE_POS + ROPE_NEG;
/// Width of the sinusoid the timestep embedder reads.
pub const TIMESTEP_FREQUENCIES: usize = 256;

/// The transformer (`transformer/config.json`).
#[derive(Clone, Debug, Deserialize)]
pub struct Transformer {
    pub attention_head_dim: usize,
    /// Rotary channels per axis, `(frame, height, width)`; they sum to a head.
    pub axes_dims_rope: [usize; 3],
    /// The text prefix is modulated from `t = 0`, so its activations do not
    /// depend on the denoising step and its K/V are computed once.
    pub causal_condition: bool,
    /// Channels of the text encoder's embeddings.
    pub context_in_dim: usize,
    pub eps: f32,
    pub in_channels: usize,
    pub mlp_ratio: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub out_channels: usize,
    pub patch_size: usize,
}

impl Transformer {
    /// Reads `transformer/config.json` of the checkpoint at `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Self = read(&dir.join("transformer/config.json"))?;
        ensure!(
            cfg.attention_head_dim == 128,
            "attention_head_dim {} unsupported (the kernels are built for 128)",
            cfg.attention_head_dim
        );
        ensure!(cfg.patch_size == 1, "patch_size {} unsupported (a token is one latent cell)", cfg.patch_size);
        ensure!(cfg.causal_condition, "only causal_condition is implemented (the text prefix is modulated from t = 0)");
        ensure!(cfg.axes_dims_rope == [16, 56, 56], "axes_dims_rope {:?} unsupported", cfg.axes_dims_rope);
        ensure!(
            cfg.in_channels == LATENT_DIM && cfg.out_channels == LATENT_DIM,
            "latents are {LATENT_DIM} channels, got {} in and {} out",
            cfg.in_channels,
            cfg.out_channels
        );
        Ok(cfg)
    }

    /// Width of the residual stream: every projection reads and writes it.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Width of the SwiGLU hidden layer, each half of the fused gate|up weight.
    pub fn mlp_hidden(&self) -> usize {
        self.inner_dim() * self.mlp_ratio
    }
}

/// The flow-matching schedule (`scheduler/scheduler_config.json`).
#[derive(Clone, Debug, Deserialize)]
pub struct Scheduler {
    pub num_train_timesteps: usize,
    /// The static shift the sigma grid is built with.
    pub shift: f32,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default)]
    pub stochastic_sampling: bool,
}

impl Scheduler {
    /// Reads `scheduler/scheduler_config.json` of the checkpoint at `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Self = read(&dir.join("scheduler/scheduler_config.json"))?;
        ensure!(!cfg.use_dynamic_shifting, "only a static shift is implemented, not use_dynamic_shifting");
        ensure!(!cfg.stochastic_sampling, "only the deterministic Euler step is implemented, not stochastic_sampling");
        Ok(cfg)
    }
}

/// What the engine reads from a checkpoint directory.
#[derive(Clone, Debug)]
pub struct Config {
    pub transformer: Transformer,
    pub scheduler: Scheduler,
    /// The distilled checkpoints' fixed grid, one sigma per step; `None` leaves
    /// the grid to the sampler.
    pub sample_sigmas: Option<Vec<f32>>,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let transformer = Transformer::load(dir)?;
        let scheduler = Scheduler::load(dir)?;
        let index: ModelIndex = read(&dir.join("model_index.json"))?;
        if let Some(sigmas) = &index.sample_sigmas {
            ensure!(!sigmas.is_empty(), "model_index.json: sample_sigmas is empty");
        }
        Ok(Self { transformer, scheduler, sample_sigmas: index.sample_sigmas })
    }

    /// The effective sigma schedule: N steps plus terminal 0.
    pub fn sigmas(&self) -> Vec<f32> {
        let mut s = match &self.sample_sigmas {
            Some(grid) => grid.clone(),
            None => {
                let n = 50usize;
                (0..n).map(|i| 1.0 - i as f32 / n as f32).collect()
            }
        };
        s.push(0.0);
        s
    }
}

#[derive(Deserialize)]
struct ModelIndex {
    #[serde(default)]
    sample_sigmas: Option<Vec<f32>>,
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}
