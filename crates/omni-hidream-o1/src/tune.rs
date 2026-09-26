//! Measures the step's decoder GEMM algorithms on this card and writes the
//! winners for the engine to pin (`pega-omni hidream-o1-tune-gemms`).
//!
//! Candidates are cuBLASLt's algorithms without a split reduction that write
//! the same bytes: each runs alone on the same probe operands (magnitudes
//! spread over 2^-12..2^12 so a different summation order shows) and the
//! largest group of identical outputs stays, so a pin changes the speed and
//! not the picture, whichever member wins (not splitting k is not enough:
//! see kern's runtime.md).
//!
//! On a card held at its power cap a GEMM is as fast as the energy the whole
//! step spends around it: timing a GEMM or a few layers alone ranked the
//! candidates otherwise. So every candidate lives in one manifest, picked per
//! step by a var, and is timed as whole `predict` steps of real pictures: one
//! shape at a time (the largest first, the others held at their choice), its
//! candidates in turn over [`ROUNDS`] rounds, the lowest median wins.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use half::bf16;
use kern_manifest::types::GemmAlgo;
use kern_runtime::Capacity;
use kern_runtime::Runtime;
use serde_json::json;
use sha2::Digest;

use crate::config::Text;
use crate::gemm::Device;
use crate::gemm::Gemms;
use crate::gemm::Pin;
use crate::gemm::Pins;
use crate::gemm::Shape;
use crate::gemm::candidates;
use crate::gemm::name;
use crate::model::Limits;
use crate::model::Model;
use crate::prompt;
use crate::prompt::Tokenizer;
use crate::sampler;

const ROUNDS: usize = 3;
const PROMPT: &str = "A lighthouse on a rocky coast at dusk, waves breaking below, warm light in the window";

/// The step's decoder GEMM shapes, largest first.
pub fn step_shapes(cfg: &Text) -> Vec<Shape> {
    let (h, d, inter) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
    let mut shapes = vec![(cfg.qkv_width(), h), (h, cfg.num_attention_heads * d), (2 * inter, h), (h, inter)];
    shapes.sort_by_key(|&(n, k)| std::cmp::Reverse(n * k));
    shapes
}

/// A deterministic probe value in `-1..1`, times `2^e` with `e` in `-12..=12` when `wide`.
fn probe(i: usize, seed: u32, wide: bool) -> f32 {
    let mut h = (i as u32).wrapping_mul(2_654_435_761) ^ seed;
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    let u = (h & 0xffff) as f32 / 32768.0 - 1.0;
    if wide { u * 2f32.powi((h >> 16) as i32 % 25 - 12) } else { u }
}

/// The largest group of `algos` whose outputs at `m` rows are bitwise one (on
/// a tie, the group holding the earliest); an algorithm kern refuses at some
/// row count up to `max_rows` is left out.
fn largest_class(
    ordinal: usize,
    max_rows: usize,
    m: usize,
    (n, k): Shape,
    algos: &[GemmAlgo],
) -> Result<Vec<GemmAlgo>> {
    let a: Vec<u8> = (0..m * k).flat_map(|i| bf16::from_f32(probe(i, 11, true)).to_le_bytes()).collect();
    let w: Vec<u8> = (0..n * k).flat_map(|i| bf16::from_f32(probe(i, 97, false)).to_le_bytes()).collect();
    let vars = BTreeMap::from([("rows".to_string(), m as u64)]);
    let mut classes: Vec<(Vec<u8>, Vec<GemmAlgo>)> = Vec::new();
    for algo in algos {
        let manifest = json!({
            "schema_version": 5, "model": "hidream-o1-gemm-probe", "vars": {"rows": {"max": max_rows}}, "states": {},
            "buffers": {
                "a": {"kind": "input", "dtype": "bf16", "shape": ["rows", k]},
                "w": {"kind": "input", "dtype": "bf16", "shape": [n, k]},
                "y": {"kind": "output", "dtype": "bf16", "shape": ["rows", n]}
            },
            "modules": {},
            "ops": {"gemm": {
                "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn", "algo": algo}]}
            }},
            "programs": {"probe": {"calls": [{"op": "gemm", "args": [
                {"buf": "a"}, {"buf": "w"}, {"buf": "y"}, {"var": "rows"}, {"i32": n}, {"i32": k}
            ]}]}}
        });
        let verified = kern_manifest::Verified::from_json(&manifest.to_string()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut rt = match Runtime::load(&verified, None, ordinal, Some(Capacity { tokens: Some(1), seqs: 1 }), None) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::info!(?algo, "left out: {e}");
                continue;
            }
        };
        rt.write_input_at("a", &a, &vars)?;
        rt.write_input("w", &w)?;
        rt.run("probe", &vars)?;
        let y = rt.read_output("y")?;
        let digest = sha2::Sha256::digest(&y[..m * n * 2]).to_vec();
        match classes.iter_mut().find(|(d, _)| *d == digest) {
            Some((_, members)) => members.push(*algo),
            None => classes.push((digest, vec![*algo])),
        }
    }
    let sizes: Vec<usize> = classes.iter().map(|(_, c)| c.len()).collect();
    tracing::info!(shape = ?(n, k), candidates = algos.len(), classes = ?sizes, "bitwise classes");
    // `max_by_key` keeps the last of equals; the classes are in first-member order.
    Ok(classes.into_iter().rev().max_by_key(|(_, c)| c.len()).map(|(_, c)| c).unwrap_or_default())
}

/// Tunes the checkpoint at `dir` on GPU `ordinal` at 2048 x 2048 and writes the pins to `out`.
pub fn tune(ordinal: usize, dir: &Path, limits: Limits, out: &Path) -> Result<()> {
    let cfg = Text::load(dir)?;
    let size = omni_engine::image::Size { width: 2048, height: 2048 };
    let grid = prompt::grid(size);
    let rows = grid.0 * grid.1 + 1;
    let max_rows = limits.max_text.max(limits.max_patches + 1);
    let mut all = BTreeMap::new();
    for shape in step_shapes(&cfg) {
        let offered = candidates(ordinal, rows, shape)?;
        all.insert(shape, largest_class(ordinal, max_rows, rows, shape, &offered)?);
    }
    let mut model = Model::load(ordinal, dir, limits, &Gemms::Candidates(all.clone()))
        .with_context(|| format!("load {}", dir.display()))?;
    let ids = Tokenizer::load(dir)?.encode(PROMPT)?;
    let mut steps = Steps { model: &mut model, ids, grid, step: sampler::TIMESTEPS.len(), seed: 0 };
    // One picture to bring the card to its working temperature.
    for _ in 0..sampler::TIMESTEPS.len() {
        steps.next()?;
    }
    let mut gemms = Vec::new();
    for shape in step_shapes(&cfg) {
        let algos = &all[&shape];
        let mut times = vec![Vec::new(); algos.len()];
        if algos.len() > 1 {
            for trial in 0..ROUNDS * algos.len() {
                let pick = (trial % algos.len() + trial / algos.len()) % algos.len();
                steps.model.pick(&name(shape), pick + 1);
                times[pick].push(steps.next()?);
            }
        }
        let medians: Vec<f64> = times
            .iter_mut()
            .map(|t| {
                t.sort_by(f64::total_cmp);
                t.get(t.len() / 2).copied().unwrap_or(0.0)
            })
            .collect();
        let best = (0..algos.len()).min_by(|&a, &b| medians[a].total_cmp(&medians[b])).unwrap_or(0);
        steps.model.pick(&name(shape), best + 1);
        tracing::info!(?shape, chosen = ?algos[best], step_ms = ?medians, "gemm tuned");
        gemms.push(Pin { n: shape.0, k: shape.1, algo: algos[best], step_ms: medians });
    }
    let pins = Pins { device: Device::of(ordinal)?, gemms };
    std::fs::write(out, serde_json::to_string_pretty(&pins)? + "\n")
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

/// Denoising steps of pictures of one prompt, one after another.
struct Steps<'a> {
    model: &'a mut Model,
    ids: Vec<i32>,
    grid: (usize, usize),
    /// The next step of the current picture; a new picture past the last.
    step: usize,
    seed: u64,
}

impl Steps<'_> {
    /// Runs one step and returns its `predict` time in ms.
    fn next(&mut self) -> Result<f64> {
        if self.step == sampler::TIMESTEPS.len() {
            self.seed += 1;
            self.model.prefill(&self.ids, self.grid)?;
            self.model.start(self.seed, sampler::NOISE_SCALE)?;
            self.step = 0;
        }
        let k = self.step;
        self.model.synchronize()?;
        let t = Instant::now();
        self.model.predict(f32::from(sampler::TIMESTEPS[k]))?;
        self.model.synchronize()?;
        let took = t.elapsed().as_secs_f64() * 1e3;
        let sigma_next = sampler::TIMESTEPS.get(k + 1).map_or(0.0, |&t| f32::from(t) / 1000.0);
        self.model.advance((self.seed, k as u32 + 1), sigma_next, sampler::NOISE_SCALE, sampler::CLIP_STD)?;
        self.step += 1;
        Ok(took)
    }
}
