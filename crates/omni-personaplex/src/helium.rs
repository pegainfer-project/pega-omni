//! Helium, the 7B temporal transformer, as manifest calls.
//!
//! - [`Helium::prefill`]: prompt rows (`tokens` of them, row `n` of session
//!   `t_seq[n]` at `t_pos[n]`) through every layer into the KV ring; nothing
//!   comes out.
//! - [`Helium::step`]: one row per session, its input read from its token
//!   state, through every layer and the output norm into `h.out` (what the
//!   depformer reads), then the text head and the text draw.
//!
//! A row's input embedding is a voice-prompt embedding or the sum of its 17
//! streams' token embeddings. Attention is full MHA over the session's ring of
//! [`CONTEXT`] positions; RoPE is the reference's interleaved form.

use anyhow::Result;
use omni_kern::Gen;
use omni_kern::buf;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::state_in;
use omni_kern::stride;
use omni_kern::weights::File;
use serde_json::Value;
use serde_json::json;

use crate::config::CARD;
use crate::config::CODEBOOKS;
use crate::config::CONTEXT;
use crate::config::DIM;
use crate::config::HEAD_DIM;
use crate::config::HEADS;
use crate::config::HIDDEN;
use crate::config::LAYERS;
use crate::config::NORM_EPS;
use crate::config::ROPE_PERIOD;
use crate::config::Sampling;
use crate::config::TEXT_VOCAB;

/// Tokens per page of the paged KV.
pub const PAGE: usize = 16;
pub const SAMPLE_THREADS: u32 = 1024;

struct Layer {
    norm1: String,
    qkv: String,
    o: String,
    norm2: String,
    gate_up: String,
    down: String,
}

pub struct Helium {
    layers: Vec<Layer>,
    out_norm: String,
    pub text_emb: String,
    pub audio_emb: String,
    text_linear: String,
    /// Offset of the token state in a session's slot of the `seq` state.
    pub state: u64,
    sampling: Sampling,
}

/// The `["tokens", width]` workspaces Helium's rows run through.
pub const SCRATCH: [(&str, usize); 7] = [
    ("h.x", DIM),
    ("h.xn", DIM),
    ("h.qkv", 3 * DIM),
    ("h.attn", DIM),
    ("h.y", DIM),
    ("h.gu", 2 * HIDDEN),
    ("h.act", HIDDEN),
];

/// `-2 ln(period) / d`, the rotary coefficient `kernels/attend.cuh` takes.
pub fn rope_coef(d: usize) -> f32 {
    (-(ROPE_PERIOD as f64).ln() * 2.0 / d as f64) as f32
}

/// Bytes of K and V one position takes in one layer.
pub const KV_BYTES: usize = 2 * HEADS * HEAD_DIM * 2;

impl Helium {
    pub fn load(g: &mut Gen, file: &File, sampling: Sampling) -> Result<Self> {
        let w = |n: &str, shape: &[usize]| file.expect(n, shape);
        let layers = (0..LAYERS)
            .map(|i| {
                let p = format!("transformer.layers.{i}");
                let n = |s: &str| format!("h.l{i}.{s}");
                Ok(Layer {
                    norm1: g.weight(&n("norm1"), &[DIM], &w(&format!("{p}.norm1.alpha"), &[1, 1, DIM])?.data),
                    qkv: g.weight(
                        &n("qkv"),
                        &[3 * DIM, DIM],
                        &w(&format!("{p}.self_attn.in_proj_weight"), &[3 * DIM, DIM])?.data,
                    ),
                    o: g.weight(&n("o"), &[DIM, DIM], &w(&format!("{p}.self_attn.out_proj.weight"), &[DIM, DIM])?.data),
                    norm2: g.weight(&n("norm2"), &[DIM], &w(&format!("{p}.norm2.alpha"), &[1, 1, DIM])?.data),
                    gate_up: g.weight(
                        &n("gate_up"),
                        &[2 * HIDDEN, DIM],
                        &w(&format!("{p}.gating.linear_in.weight"), &[2 * HIDDEN, DIM])?.data,
                    ),
                    down: g.weight(
                        &n("down"),
                        &[DIM, HIDDEN],
                        &w(&format!("{p}.gating.linear_out.weight"), &[DIM, HIDDEN])?.data,
                    ),
                })
            })
            .collect::<Result<_>>()?;
        let audio: Vec<f32> = (0..2 * CODEBOOKS)
            .map(|k| w(&format!("emb.{k}.weight"), &[CARD + 1, DIM]))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|t| t.data)
            .collect();
        Ok(Self {
            layers,
            out_norm: g.weight("h.out_norm", &[DIM], &w("out_norm.alpha", &[1, 1, DIM])?.data),
            text_emb: g.weight(
                "h.text_emb",
                &[TEXT_VOCAB + 1, DIM],
                &w("text_emb.weight", &[TEXT_VOCAB + 1, DIM])?.data,
            ),
            audio_emb: g.weight("h.audio_emb", &[2 * CODEBOOKS * (CARD + 1), DIM], &audio),
            text_linear: g.weight(
                "h.text_linear",
                &[TEXT_VOCAB, DIM],
                &w("text_linear.weight", &[TEXT_VOCAB, DIM])?.data,
            ),
            state: g.region(4 * 24),
            sampling,
        })
    }

    /// Voice rows or token rows in, KV written.
    pub fn prefill(&self, g: &mut Gen) {
        let rows = json!("tokens");
        g.launch(
            "prefill.embed",
            "lm_embed_prompt",
            [rows.clone(), json!(1), json!(1)],
            (DIM / 8) as u32,
            vec![
                ini("p_ids"),
                ini("p_voice"),
                inb("voices"),
                inb(&self.text_emb),
                inb(&self.audio_emb),
                outb("h.x"),
                i32a(DIM),
            ],
        );
        self.forward(g, "prefill", &rows, true);
        g.launch(
            "prefill.init",
            "lm_init",
            [json!("seqs"), json!(1), json!(1)],
            32,
            vec![("inout state", json!({"state": "seq", "offset": self.state})), ini("lines"), stride(), ini("init")],
        );
    }

    /// Each session's input row in, `h.out` and the text draw out.
    pub fn step(&self, g: &mut Gen) {
        let rows = json!("seqs");
        g.launch(
            "step.embed",
            "lm_embed_state",
            [rows.clone(), json!(1), json!(1)],
            (DIM / 8) as u32,
            vec![
                state_in(self.state),
                ini("lines"),
                stride(),
                inb(&self.text_emb),
                inb(&self.audio_emb),
                outb("h.x"),
                ("out buffer<i32>", buf("rows")),
                i32a(DIM),
            ],
        );
        self.forward(g, "step", &rows, false);
        g.gemm_rows(
            "step.text_head",
            (buf("text_logits"), buf("h.xn"), buf(&self.text_linear)),
            count(&rows),
            (TEXT_VOCAB, DIM),
        );
        sample(
            g,
            "step.text",
            buf("text_logits"),
            TEXT_VOCAB,
            self.sampling.text_temperature,
            self.sampling.text_top_k,
            0,
        );
    }

    /// Every layer over `rows` rows of `h.x`; the output norm's rows end in `h.xn`.
    fn forward(&self, g: &mut Gen, label: &str, rows: &Value, prefill: bool) {
        let n = count(rows);
        let per_row = [rows.clone(), json!(1), json!(1)];
        let block = (DIM / 8) as u32;
        let (pos, slot, seq) = if prefill { ("t_pos", "t_slot", "t_seq") } else { ("pos", "slot", "pos") };
        g.launch(
            &format!("{label}.norm"),
            "lm_norm",
            per_row.clone(),
            block,
            vec![inb("h.x"), inb(&self.layers[0].norm1), outb("h.xn"), i32a(DIM), f32a(NORM_EPS)],
        );
        for (i, l) in self.layers.iter().enumerate() {
            let at = |s: &str| format!("{label}.l{i}.{s}");
            let kv = json!({"state": format!("kv{i}")});
            g.gemm_rows(&at("qkv"), (buf("h.qkv"), buf("h.xn"), buf(&l.qkv)), n.clone(), (3 * DIM, DIM));
            g.launch(
                &at("rope"),
                "lm_rope",
                [rows.clone(), json!((3 * HEADS).div_ceil(8)), json!(1)],
                256,
                vec![
                    io("h.qkv"),
                    ini(pos),
                    ini(slot),
                    ("inout state", kv.clone()),
                    i32a(HEADS),
                    f32a(rope_coef(HEAD_DIM)),
                ],
            );
            g.launch(
                &at("attn"),
                "lm_attend",
                [rows.clone(), json!(HEADS), json!(1)],
                128,
                vec![
                    inb("h.qkv"),
                    ("in state", kv),
                    ini(pos),
                    ini(seq),
                    i32a(prefill as usize),
                    ini("kv_indptr"),
                    ini("kv_pages"),
                    i32a(PAGE),
                    outb("h.attn"),
                    i32a(HEADS),
                    f32a(1.0 / (HEAD_DIM as f32).sqrt()),
                    i32a(CONTEXT),
                ],
            );
            g.gemm_rows(&at("o"), (buf("h.y"), buf("h.attn"), buf(&l.o)), n.clone(), (DIM, DIM));
            add_norm(g, &at("norm2"), rows, &l.norm2);
            g.gemm_rows(&at("gate_up"), (buf("h.gu"), buf("h.xn"), buf(&l.gate_up)), n.clone(), (2 * HIDDEN, DIM));
            g.each8(&at("silu_mul"), "lm_silu_mul", rows, HIDDEN, vec![inb("h.gu"), outb("h.act"), i32a(HIDDEN)]);
            g.gemm_rows(&at("down"), (buf("h.y"), buf("h.act"), buf(&l.down)), n.clone(), (DIM, HIDDEN));
            let next = self.layers.get(i + 1).map_or(&self.out_norm, |l| &l.norm1);
            add_norm(g, &at("next_norm"), rows, next);
        }
    }
}

/// `h.x += h.y`, then `h.xn = norm(h.x)`.
fn add_norm(g: &mut Gen, label: &str, rows: &Value, alpha: &str) {
    g.launch(
        label,
        "lm_add_norm",
        [rows.clone(), json!(1), json!(1)],
        (DIM / 8) as u32,
        vec![inb("h.y"), io("h.x"), inb(alpha), outb("h.xn"), i32a(DIM), f32a(NORM_EPS)],
    );
}

/// One draw per session from `logits` (rows of `vocab`) into column `col` of `drawn`.
pub fn sample(g: &mut Gen, label: &str, logits: Value, vocab: usize, temperature: f32, top_k: usize, col: usize) {
    g.launch(
        label,
        "lm_sample",
        [json!("seqs"), json!(1), json!(1)],
        SAMPLE_THREADS,
        vec![
            ("in buffer<bf16>", logits),
            i32a(vocab),
            ("i64", json!({"i64": vocab})),
            f32a(temperature),
            i32a(top_k),
            ini("force"),
            ini("seeds"),
            i32a(col),
            ("out buffer<i32>", buf("drawn")),
        ],
    );
}
