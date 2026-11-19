//! Flow-matching Euler sampler for the distilled Turbo checkpoint: 8 fixed
//! sigmas, deterministic (one noise draw), velocity prediction, no guidance.
//!
//! `z` starts as standard normal noise. At each sigma the DiT predicts
//! velocity, and `z += (sigma_next - sigma) * pred`. After the last step
//! (sigma_next = 0) the latent is the clean image representation.

use anyhow::Result;

use crate::model::Model;

/// Where the initial latent noise comes from.
#[derive(Clone, Copy)]
pub enum Noise<'a> {
    Seed(u64),
    Given(&'a [u8]),
}

/// Denoises the prefilled sequence. `sigmas` has N+1 entries (N steps plus
/// terminal 0). `keep_going` is polled before every step; returns `false`
/// when the client disconnected.
pub fn sample(model: &mut Model, sigmas: &[f32], noise: Noise, keep_going: &dyn Fn() -> bool) -> Result<bool> {
    match noise {
        Noise::Seed(seed) => model.start(seed)?,
        Noise::Given(z) => model.set_z(z)?,
    }
    for k in 0..sigmas.len() - 1 {
        if !keep_going() {
            return Ok(false);
        }
        model.decode(sigmas[k])?;
        model.step(sigmas[k], sigmas[k + 1])?;
    }
    Ok(true)
}
