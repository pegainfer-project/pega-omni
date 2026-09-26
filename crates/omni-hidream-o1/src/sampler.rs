//! The distilled checkpoint's sampler: 28 fixed timesteps, no guidance, and
//! fresh noise at every step (`FlashFlowMatchEulerDiscreteScheduler` with the
//! reference's `--model_type dev` settings).
//!
//! The latent `z` lives in pixel space as patch rows. It starts as
//! `NOISE_SCALE * eps`. At timestep `t_k` (sigma `t_k / 1000`) the model
//! predicts x0, and the next latent is
//! `sigma_{k+1} * NOISE_SCALE * clip(eps) + (1 - sigma_{k+1}) * x0`, the noise
//! clipped to `CLIP_STD` of its own standard deviation; after the last step
//! (sigma 0) the latent is x0 itself. The reference forms the velocity
//! `(x0 - z) / sigma` and steps back by it, which is x0 up to rounding.
//!
//! The engine draws the standard normals on the GPU from the request's seed;
//! the golden test replays the reference's own ([`Noise`]).

use anyhow::Result;

use crate::model::Model;

/// `DEFAULT_TIMESTEPS` of the reference pipeline.
pub const TIMESTEPS: [u16; 28] = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737, 707, 675, 640, 602, 560, 515, 464, 409,
    347, 278, 199, 110, 8,
];
pub const NOISE_SCALE: f32 = 7.5;
pub const CLIP_STD: f32 = 2.5;

/// Where a picture's standard normals come from: drawn on the GPU from a
/// seed, or given (a reference run's, replayed).
#[derive(Clone, Copy)]
pub enum Noise<'a> {
    Seed(u64),
    Given(&'a dyn Fn(u32) -> Vec<f32>),
}

/// Denoises the prefilled sequence into the model's latent: draw 0 starts
/// the latent, draw `k + 1` follows step `k`. `keep_going` is asked before
/// every step; returns `false` when it said no.
pub fn sample(model: &mut Model, noise: Noise, keep_going: &dyn Fn() -> bool) -> Result<bool> {
    match noise {
        Noise::Seed(seed) => model.start(seed, NOISE_SCALE)?,
        Noise::Given(draw) => model.advance_with(&draw(0), 1.0, NOISE_SCALE, 0.0)?,
    }
    for (k, &t) in TIMESTEPS.iter().enumerate() {
        if !keep_going() {
            return Ok(false);
        }
        model.predict(f32::from(t))?;
        let sigma_next = TIMESTEPS.get(k + 1).map_or(0.0, |&next| f32::from(next) / 1000.0);
        let draw = k as u32 + 1;
        match noise {
            Noise::Seed(seed) => model.advance((seed, draw), sigma_next, NOISE_SCALE, CLIP_STD)?,
            Noise::Given(given) => model.advance_with(&given(draw), sigma_next, NOISE_SCALE, CLIP_STD)?,
        }
    }
    Ok(true)
}
