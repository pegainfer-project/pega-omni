//! The latent grid of one generation and where every token sits in RoPE.
//!
//! Text and image latents share one sequence and one 3-axis rotary embedding
//! `(frame, height, width)`. Text token `k` sits at `(k, k, k)`; the picture's
//! token `(i, j)` freezes the frame axis at the text length and centres the
//! spatial axes on zero, so a grid's positions do not depend on where it sits
//! in the sequence (`QwenImage21Rope` of the reference). A negative position
//! `p` reads row `p + ROPE_TABLE`, in the tail of the table that counts down
//! from -1.
//!
//! The prompt's tokens are the sidecar's: it hands over the embeddings and
//! their count, and this module places the picture after them.

use half::bf16;
use omni_engine::image::Size;

use crate::config::ROPE_TABLE;
use crate::config::TIMESTEP_FREQUENCIES;
use crate::config::VAE_SCALE;

/// The resolutions the model generates, `(width, height)`.
pub const SIZES: [(u32, u32); 7] =
    [(1024, 1024), (1280, 768), (768, 1280), (1152, 896), (896, 1152), (768, 768), (512, 512)];

pub fn sizes() -> impl Iterator<Item = Size> {
    SIZES.iter().map(|&(width, height)| Size { width, height })
}

/// Latent rows and columns of a picture.
pub fn grid(size: Size) -> (usize, usize) {
    assert!(
        size.height.is_multiple_of(32) && size.width.is_multiple_of(32),
        "{size} is not a multiple of 32 on both sides"
    );
    (size.height as usize / VAE_SCALE, size.width as usize / VAE_SCALE)
}

/// `(frame, height, width)` of text token `k`.
pub fn text_position(k: usize) -> [i32; 3] {
    [k as i32; 3]
}

/// `(frame, height, width)` of every token of a `rows x cols` grid, raster
/// order, after `text_len` text tokens.
pub fn image_positions(text_len: usize, rows: usize, cols: usize) -> Vec<[i32; 3]> {
    // A position below zero reads the same row from the end of the table.
    let wrap = |p: i32| if p < 0 { p + ROPE_TABLE as i32 } else { p };
    let (top, left) = ((rows - rows / 2) as i32, (cols - cols / 2) as i32);
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| [text_len as i32, wrap(i as i32 - top), wrap(j as i32 - left)]))
        .collect()
}

/// The sinusoid the timestep embedder reads at `sigma`: `cos` then `sin` of
/// `1000 * sigma` over the frequencies `exp(-ln(10000) * i / 128)`
/// (`QwenImage21TemporalTimesteps` of the reference).
///
/// The angle is not `1000 * sigma`. The reference casts the scheduler's
/// timestep to the latent dtype, divides it by 1000 in bf16, and only then does
/// the embedder widen it back — both roundings survive. At sigma 0.95418 they
/// move the angle from 954.18 to 957.03, most of a radian on the highest
/// frequency, so skipping them decorrelates the timestep embedding.
pub fn timestep_sinusoid(sigma: f32) -> [f32; TIMESTEP_FREQUENCIES] {
    let half = TIMESTEP_FREQUENCIES / 2;
    let timestep = bf16::from_f32(bf16::from_f32(sigma * 1000.0).to_f32() / 1000.0).to_f32();
    let t = timestep * 1000.0;
    let mut out = [0.0; TIMESTEP_FREQUENCIES];
    for i in 0..half {
        let freq = (-10000f32.ln() * i as f32 / half as f32).exp();
        out[i] = (t * freq).cos();
        out[half + i] = (t * freq).sin();
    }
    out
}
