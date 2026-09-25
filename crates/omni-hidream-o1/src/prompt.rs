//! The token sequence of one generation and where every token sits in M-RoPE.
//!
//! A sequence is the chat-templated prompt, `<|boi_token|>`, the timestep slot
//! `<|tms_token|>`, then one token per 32 x 32 patch in raster order. Text token
//! `k` sits at `(k, k, k)`; patch `(i, j)` at `(4096, 4096 + i, 4096 + j)`
//! (`get_rope_index_fix_point` of the reference, which anchors the grid at a
//! fixed point instead of after the text).
//!
//! Every text token but the last attends causally among the text. The
//! timestep slot and the patches attend to the whole sequence.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use omni_engine::image::Size;

use crate::config::BOI_TOKEN;
use crate::config::IMAGE_POSITION;
use crate::config::PATCH;
use crate::config::TMS_TOKEN;

/// The resolutions the model was trained at (`PREDEFINED_RESOLUTIONS`).
pub const SIZES: [(u32, u32); 11] = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
];

pub fn sizes() -> impl Iterator<Item = Size> {
    SIZES.iter().map(|&(width, height)| Size { width, height })
}

/// Patch rows and columns of a picture.
pub fn grid(size: Size) -> (usize, usize) {
    (size.height as usize / PATCH, size.width as usize / PATCH)
}

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
            .context("loading the tokenizer")?;
        Ok(Self(inner))
    }

    /// The prompt's text tokens, the timestep slot last.
    pub fn encode(&self, prompt: &str) -> Result<Vec<i32>> {
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{BOI_TOKEN}{TMS_TOKEN}");
        let enc = self.0.encode(text, false).map_err(|e| anyhow::anyhow!("tokenizing the prompt: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let tms = self.0.token_to_id(TMS_TOKEN).context("the tokenizer lacks the timestep token")?;
        ensure!(ids.last() == Some(&(tms as i32)), "the prompt did not end on the timestep token");
        Ok(ids)
    }
}

/// `(t, h, w)` of text token `k`.
pub fn text_position(k: usize) -> [i32; 3] {
    [k as i32; 3]
}

/// `(t, h, w)` of every patch of a `rows x cols` grid, raster order.
pub fn patch_positions(rows: usize, cols: usize) -> Vec<[i32; 3]> {
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| [IMAGE_POSITION, IMAGE_POSITION + i as i32, IMAGE_POSITION + j as i32]))
        .collect()
}
