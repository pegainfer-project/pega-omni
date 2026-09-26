//! The depformer as manifest calls: from Helium's output to the agent's eight
//! codebooks of a frame, one small transformer pass per codebook.
//!
//! Step `k` reads its own projection of Helium's output (all eight come out
//! of one GEMM) plus the embedding of the token step `k - 1` drew (the text
//! token for step 0), runs six layers whose weights are step `k`'s own
//! (`weights_per_step`), attends to the frame's steps `0..=k` (a workspace of
//! eight slots per session, no positional embedding), and draws codebook `k`
//! from its own head. The reference runs sixteen steps; the last eight
//! predict the caller's codes, which are heard rather than drawn, so they are
//! not run.

use anyhow::Result;
use omni_kern::Gen;
use omni_kern::buf;
use omni_kern::buf_at;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::weights::File;
use serde_json::json;

use crate::config::CARD;
use crate::config::CODEBOOKS;
use crate::config::DEP_DIM;
use crate::config::DEP_HEAD_DIM;
use crate::config::DEP_HEADS;
use crate::config::DEP_HIDDEN;
use crate::config::DEP_LAYERS;
use crate::config::DEP_STEPS_STORED;
use crate::config::DIM;
use crate::config::NORM_EPS;
use crate::config::Sampling;
use crate::config::TEXT_VOCAB;
use crate::helium::sample;

const STEPS: usize = CODEBOOKS;

struct Layer {
    norm1: String,
    qkv: String,
    out: String,
    norm2: String,
    gate_in: String,
    gate_out: String,
}

pub struct Depformer {
    input: String,
    text_emb: String,
    emb: String,
    heads: String,
    layers: Vec<Layer>,
    sampling: Sampling,
}

/// The first `STEPS` of a `[DEP_STEPS_STORED * rows, cols]` per-step weight.
fn steps(file: &File, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let t = file.expect(name, &[DEP_STEPS_STORED * rows, cols])?;
    Ok(t.data[..STEPS * rows * cols].to_vec())
}

fn stacked(file: &File, name: &str, n: usize, shape: [usize; 2]) -> Result<Vec<f32>> {
    Ok((0..n)
        .map(|k| file.expect(&name.replace("{k}", &k.to_string()), &shape))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flat_map(|t| t.data)
        .collect())
}

impl Depformer {
    pub fn load(g: &mut Gen, file: &File, sampling: Sampling) -> Result<Self> {
        let layers = (0..DEP_LAYERS)
            .map(|i| {
                let p = format!("depformer.layers.{i}");
                let n = |s: &str| format!("d.l{i}.{s}");
                let norm = |s: &str| file.expect(&format!("{p}.{s}.alpha"), &[1, 1, DEP_DIM]);
                Ok(Layer {
                    norm1: g.weight(&n("norm1"), &[DEP_DIM], &norm("norm1")?.data),
                    qkv: g.weight(
                        &n("qkv"),
                        &[STEPS * 3 * DEP_DIM, DEP_DIM],
                        &steps(file, &format!("{p}.self_attn.in_proj_weight"), 3 * DEP_DIM, DEP_DIM)?,
                    ),
                    out: g.weight(
                        &n("out"),
                        &[STEPS * DEP_DIM, DEP_DIM],
                        &steps(file, &format!("{p}.self_attn.out_proj.weight"), DEP_DIM, DEP_DIM)?,
                    ),
                    norm2: g.weight(&n("norm2"), &[DEP_DIM], &norm("norm2")?.data),
                    gate_in: g.weight(
                        &n("gate_in"),
                        &[STEPS * 2 * DEP_HIDDEN, DEP_DIM],
                        &stacked(
                            file,
                            &format!("{p}.gating.{{k}}.linear_in.weight"),
                            STEPS,
                            [2 * DEP_HIDDEN, DEP_DIM],
                        )?,
                    ),
                    gate_out: g.weight(
                        &n("gate_out"),
                        &[STEPS * DEP_DIM, DEP_HIDDEN],
                        &stacked(file, &format!("{p}.gating.{{k}}.linear_out.weight"), STEPS, [DEP_DIM, DEP_HIDDEN])?,
                    ),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            input: g.weight(
                "d.input",
                &[STEPS * DEP_DIM, DIM],
                &stacked(file, "depformer_in.{k}.weight", STEPS, [DEP_DIM, DIM])?,
            ),
            text_emb: g.weight(
                "d.text_emb",
                &[TEXT_VOCAB + 1, DEP_DIM],
                &file.expect("depformer_text_emb.weight", &[TEXT_VOCAB + 1, DEP_DIM])?.data,
            ),
            emb: g.weight(
                "d.emb",
                &[(STEPS - 1) * (CARD + 1), DEP_DIM],
                &stacked(file, "depformer_emb.{k}.weight", STEPS - 1, [CARD + 1, DEP_DIM])?,
            ),
            heads: g.weight(
                "d.heads",
                &[STEPS * CARD, DEP_DIM],
                &stacked(file, "linears.{k}.weight", STEPS, [CARD, DEP_DIM])?,
            ),
            layers,
            sampling,
        })
    }

    /// `h.xn` (Helium's output) and the text draw in, the eight agent draws
    /// (columns 1..=8 of `drawn`) and their logits out.
    pub fn step(&self, g: &mut Gen, max_seqs: usize) {
        let seqs = json!("seqs");
        let n = count(&seqs);
        let per_row = [seqs.clone(), json!(1), json!(1)];
        let block = (DEP_DIM / 8) as u32;
        let width = DEP_HEADS * DEP_HEAD_DIM;
        for (name, w) in [
            ("d.proj", STEPS * DEP_DIM),
            ("d.x", DEP_DIM),
            ("d.xn", DEP_DIM),
            ("d.qkv", 3 * DEP_DIM),
            ("d.attn", DEP_DIM),
            ("d.y", DEP_DIM),
            ("d.gu", 2 * DEP_HIDDEN),
            ("d.act", DEP_HIDDEN),
        ] {
            g.need(name, w);
        }
        for l in 0..DEP_LAYERS {
            g.need(&format!("dkv{l}"), STEPS * 2 * width);
        }
        g.gemm_rows("d.input", (buf("d.proj"), buf("h.xn"), buf(&self.input)), n.clone(), (STEPS * DEP_DIM, DIM));
        let bytes = |rows: usize, cols: usize| rows * cols * 2;
        for k in 0..STEPS {
            let table = match k {
                0 => buf(&self.text_emb),
                k => buf_at(&self.emb, (k - 1) * bytes(CARD + 1, DEP_DIM)),
            };
            g.launch(
                &format!("d{k}.embed"),
                "lm_dep_embed",
                per_row.clone(),
                block,
                vec![inb("d.proj"), i32a(k), ("in buffer<bf16>", table), ini("drawn"), outb("d.x"), i32a(DEP_DIM)],
            );
            g.launch(
                &format!("d{k}.norm"),
                "lm_norm",
                per_row.clone(),
                block,
                vec![inb("d.x"), inb(&self.layers[0].norm1), outb("d.xn"), i32a(DEP_DIM), f32a(NORM_EPS)],
            );
            for (i, l) in self.layers.iter().enumerate() {
                let at = |s: &str| format!("d{k}.l{i}.{s}");
                let ws = format!("dkv{i}");
                let slice = |w: &str, rows: usize, cols: usize| buf_at(w, k * bytes(rows, cols));
                g.gemm_rows(
                    &at("qkv"),
                    (buf("d.qkv"), buf("d.xn"), slice(&l.qkv, 3 * DEP_DIM, DEP_DIM)),
                    n.clone(),
                    (3 * DEP_DIM, DEP_DIM),
                );
                g.launch(
                    &at("kv"),
                    "lm_dep_kv",
                    per_row.clone(),
                    128,
                    vec![inb("d.qkv"), outb(&ws), i32a(k), i32a(DEP_HEADS)],
                );
                g.launch(
                    &at("attn"),
                    "lm_dep_attend",
                    [seqs.clone(), json!(DEP_HEADS), json!(1)],
                    128,
                    vec![
                        inb("d.qkv"),
                        inb(&ws),
                        i32a(k),
                        outb("d.attn"),
                        i32a(DEP_HEADS),
                        f32a(1.0 / (DEP_HEAD_DIM as f32).sqrt()),
                    ],
                );
                g.gemm_rows(
                    &at("out"),
                    (buf("d.y"), buf("d.attn"), slice(&l.out, DEP_DIM, DEP_DIM)),
                    n.clone(),
                    (DEP_DIM, DEP_DIM),
                );
                g.launch(
                    &at("norm2"),
                    "lm_add_norm",
                    per_row.clone(),
                    block,
                    vec![inb("d.y"), io("d.x"), inb(&l.norm2), outb("d.xn"), i32a(DEP_DIM), f32a(NORM_EPS)],
                );
                g.gemm_rows(
                    &at("gate_in"),
                    (buf("d.gu"), buf("d.xn"), slice(&l.gate_in, 2 * DEP_HIDDEN, DEP_DIM)),
                    n.clone(),
                    (2 * DEP_HIDDEN, DEP_DIM),
                );
                g.each8(
                    &at("silu_mul"),
                    "lm_silu_mul",
                    &seqs,
                    DEP_HIDDEN,
                    vec![inb("d.gu"), outb("d.act"), i32a(DEP_HIDDEN)],
                );
                g.gemm_rows(
                    &at("gate_out"),
                    (buf("d.y"), buf("d.act"), slice(&l.gate_out, DEP_DIM, DEP_HIDDEN)),
                    n.clone(),
                    (DEP_DIM, DEP_HIDDEN),
                );
                match self.layers.get(i + 1) {
                    Some(next) => g.launch(
                        &at("next_norm"),
                        "lm_add_norm",
                        per_row.clone(),
                        block,
                        vec![inb("d.y"), io("d.x"), inb(&next.norm1), outb("d.xn"), i32a(DEP_DIM), f32a(NORM_EPS)],
                    ),
                    None => g.each8(&at("add"), "lm_add", &seqs, DEP_DIM, vec![inb("d.y"), io("d.x")]),
                }
            }
            let logits = buf_at("audio_logits", k * max_seqs * CARD * 2);
            g.gemm_rows(
                &format!("d{k}.head"),
                (logits.clone(), buf("d.x"), buf_at(&self.heads, k * bytes(CARD, DEP_DIM))),
                n.clone(),
                (CARD, DEP_DIM),
            );
            let s = &self.sampling;
            sample(g, &format!("d{k}.sample"), logits, CARD, s.audio_temperature, s.audio_top_k, 1 + k);
        }
    }
}
