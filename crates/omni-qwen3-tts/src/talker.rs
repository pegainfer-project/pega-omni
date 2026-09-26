//! The talker and its code predictor as manifest segments: text in, one
//! 16-codebook frame per sequence per call out.
//!
//! - [`Talker::init`]: the text track's `<tts_pad>` embedding, once.
//! - [`Talker::prefill`]: the prompts of a ragged batch (`tokens` rows, row
//!   `n` of sequence `t_seq[n]`) through the talker, each sequence's last
//!   hidden state into `hidden`.
//! - [`Talker::advance`]: one row per sequence, its last frame's embedding
//!   read from its slot, through the talker into `hidden`.
//! - [`Talker::tail`]: from `hidden` to a frame: the codec head and the
//!   codebook-0 draw, then fifteen code-predictor passes, one codebook each.
//!   The predictor sees each frame on its own, so its KV is a workspace of
//!   [`SPAN`] slots per sequence, rewritten every call.
//!
//! The talker's region of a sequence's slot of the `seq` state holds its last
//! frame (sixteen i32) and the bitmap of codebook-0 codes under the repetition
//! penalty; the draws write both, so nothing but the frame itself goes back to
//! the host.

use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

use crate::config;
use crate::stack::HEAD_DIM;
use crate::stack::Keep;
use crate::stack::Kv;
use crate::stack::Pass;
use crate::stack::Scratch;
use crate::stack::Stack;
use omni_kern::Gen;
use omni_kern::buf;
use omni_kern::buf_at;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::inf;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::state_in;
use omni_kern::state_io;
use omni_kern::stride;
use omni_kern::weights::File;

pub const GROUPS: usize = 16;
/// Code-predictor KV slots per sequence: one per position of a frame.
pub const SPAN: usize = GROUPS;
/// Frames before the end token may be drawn.
const MIN_FRAMES: usize = 2;
/// The top of the codec vocabulary is control tokens, never drawn but the end.
const CONTROL: usize = 1024;
const SAMPLE_THREADS: u32 = 1024;

const T: Scratch = Scratch { x: "t.x", res: "t.res", qkv: "t.qkv", attn: "t.attn", gate_up: "t.gate_up", act: "t.act" };
const P: Scratch = Scratch { x: "p.x", res: "p.res", qkv: "p.qkv", attn: "p.attn", gate_up: "p.gate_up", act: "p.act" };

pub struct Talker {
    cfg: config::Model,
    /// Offset of the talker's region of a sequence's slot.
    state: u64,
    sampling: config::Generation,
    talker: Stack,
    predictor: Stack,
    text_emb: String,
    codec_emb: String,
    fc1: (String, String),
    fc2: (String, String),
    codec_head: String,
    mtp: (String, String),
    p_emb: String,
    lm_heads: String,
}

impl Talker {
    pub fn load(g: &mut Gen, file: &File, cfg: &config::Model, sampling: &config::Generation) -> Result<Self> {
        let t = &cfg.talker_config;
        let (h, p) = (t.stack.hidden_size, t.code_predictor_config.hidden_size);
        let (vocab, p_vocab) = (t.stack.vocab_size, t.code_predictor_config.vocab_size);
        ensure!(t.num_code_groups == GROUPS, "{} code groups, the engine is built for {GROUPS}", t.num_code_groups);
        ensure!(vocab <= 4096 && p_vocab <= 4096, "the sampler is built for vocabularies up to 4096");
        let w = |n: &str, shape: &[usize]| file.expect(n, shape);
        let text_vocab = file.shape("talker.model.text_embedding.weight")?[0];
        let stacked = |n: &str, rows: usize, cols: usize| -> Result<Vec<f32>> {
            Ok((0..GROUPS - 1)
                .map(|g| w(&n.replace("{g}", &g.to_string()), &[rows, cols]))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(|t| t.data)
                .collect())
        };
        let mut pair = |name: &str, src: &str, n: usize, k: usize| -> Result<(String, String)> {
            Ok((
                g.weight(&format!("{name}.w"), &[n, k], &w(&format!("{src}.weight"), &[n, k])?.data),
                g.weight(&format!("{name}.b"), &[n], &w(&format!("{src}.bias"), &[n])?.data),
            ))
        };
        let fc1 = pair("talker.fc1", "talker.text_projection.linear_fc1", h, h)?;
        let fc2 = pair("talker.fc2", "talker.text_projection.linear_fc2", h, h)?;
        let mtp = pair("talker.mtp", "talker.code_predictor.small_to_mtp_projection", p, h)?;
        Ok(Self {
            state: g.region(4 * (GROUPS + vocab.div_ceil(32))),
            text_emb: g.weight(
                "talker.text_emb",
                &[text_vocab, h],
                &w("talker.model.text_embedding.weight", &[text_vocab, h])?.data,
            ),
            codec_emb: g.weight(
                "talker.codec_emb",
                &[vocab, h],
                &w("talker.model.codec_embedding.weight", &[vocab, h])?.data,
            ),
            codec_head: g.weight("talker.codec_head", &[vocab, h], &w("talker.codec_head.weight", &[vocab, h])?.data),
            p_emb: g.weight(
                "predictor.emb",
                &[(GROUPS - 1) * p_vocab, h],
                &stacked("talker.code_predictor.model.codec_embedding.{g}.weight", p_vocab, h)?,
            ),
            lm_heads: g.weight(
                "predictor.heads",
                &[(GROUPS - 1) * p_vocab, p],
                &stacked("talker.code_predictor.lm_head.{g}.weight", p_vocab, p)?,
            ),
            talker: Stack::load(g, file, "talker.model", "talker", &t.stack)?,
            predictor: Stack::load(g, file, "talker.code_predictor.model", "predictor", &t.code_predictor_config)?,
            fc1,
            fc2,
            mtp,
            cfg: cfg.clone(),
            sampling: *sampling,
        })
    }

    pub fn text_embedding(&self) -> &str {
        &self.text_emb
    }

    fn vocab(&self) -> usize {
        self.cfg.talker_config.stack.vocab_size
    }

    fn p_vocab(&self) -> usize {
        self.cfg.talker_config.code_predictor_config.vocab_size
    }

    fn hidden(&self) -> usize {
        self.talker.cfg.hidden_size
    }

    /// The `["tokens", width]` workspaces, by name.
    pub fn token_workspaces(&self) -> Vec<(&'static str, usize)> {
        let c = &self.talker.cfg;
        let h = c.hidden_size;
        vec![
            ("t.e", h),
            ("t.p", h),
            (T.x, h),
            (T.res, h),
            (T.qkv, self.talker.qkv_width()),
            (T.attn, c.num_attention_heads * HEAD_DIM),
            (T.gate_up, 2 * c.intermediate_size),
            (T.act, c.intermediate_size),
        ]
    }

    /// Writes `pad_embed`.
    pub fn init(&self, g: &mut Gen) {
        let h = self.hidden();
        let one = json!(1);
        g.launch(
            "init.embed",
            "talker_embed_id",
            [json!(1), json!(1), json!(1)],
            (h / 8) as u32,
            vec![inb(&self.text_emb), i32a(self.cfg.tts_pad_token_id), outb("t.e"), i32a(h)],
        );
        g.gemm_rows("init.fc1", (buf("t.p"), buf("t.e"), buf(&self.fc1.0)), json!({"i32": 1}), (h, h));
        g.each8("init.fc1.bias", "talker_bias_silu", &one, h, vec![io("t.p"), inb(&self.fc1.1), i32a(h)]);
        g.gemm_rows("init.fc2", (buf("pad_embed"), buf("t.p"), buf(&self.fc2.0)), json!({"i32": 1}), (h, h));
        g.each8("init.fc2.bias", "talker_bias", &one, h, vec![io("pad_embed"), inb(&self.fc2.1), i32a(h)]);
    }

    /// Prompts in, `embeds` and each sequence's last hidden state out.
    pub fn prefill(&self, g: &mut Gen) {
        let h = self.hidden();
        let rows = json!("tokens");
        g.launch(
            "prefill.embed",
            "talker_gather",
            [rows.clone(), json!(1), json!(1)],
            (h / 8) as u32,
            vec![ini("t_ids"), i32a(1), inb(&self.text_emb), outb("t.e"), i32a(h)],
        );
        g.gemm_rows("prefill.fc1", (buf("t.p"), buf("t.e"), buf(&self.fc1.0)), count(&rows), (h, h));
        g.each8("prefill.fc1.bias", "talker_bias_silu", &rows, h, vec![io("t.p"), inb(&self.fc1.1), i32a(h)]);
        g.gemm_rows("prefill.fc2", (buf("embeds"), buf("t.p"), buf(&self.fc2.0)), count(&rows), (h, h));
        g.launch(
            "prefill.codec",
            "talker_prompt_codec",
            [rows.clone(), json!(1), json!(1)],
            (h / 8) as u32,
            vec![io("embeds"), inb(&self.fc2.1), inb(&self.codec_emb), ini("t_codec"), i32a(h)],
        );
        let pass = Pass {
            label: "prefill",
            rows,
            input: "embeds",
            kv: Kv::Paged { prefix: "kv", ragged: true },
            keep: Keep { out: T.x, every: 1, which: 0 },
        };
        self.talker.forward(g, pass, &T);
        g.launch(
            "prefill.last",
            "talker_gather",
            [json!("seqs"), json!(1), json!(1)],
            (h / 8) as u32,
            vec![ini("last_row"), i32a(1), inb(T.x), outb("hidden"), i32a(h)],
        );
    }

    /// Every sequence's last frame in, its next hidden state out.
    pub fn advance(&self, g: &mut Gen) {
        let h = self.hidden();
        g.launch(
            "decode.embed",
            "talker_frame_embed",
            [json!("seqs"), json!(1), json!(1)],
            (h / 8) as u32,
            vec![
                state_in(self.state),
                ini("lines"),
                stride(),
                inb("pad_embed"),
                inb(&self.codec_emb),
                inb(&self.p_emb),
                outb("t.e"),
                i32a(h),
                i32a(self.p_vocab()),
            ],
        );
        let pass = Pass {
            label: "decode",
            rows: json!("seqs"),
            input: "t.e",
            kv: Kv::Paged { prefix: "kv", ragged: false },
            keep: Keep { out: "hidden", every: 1, which: 0 },
        };
        self.talker.forward(g, pass, &T);
    }

    /// One draw per sequence of codebook `group` from `logits`.
    fn sample(&self, g: &mut Gen, group: usize, logits: Value) {
        let s = &self.sampling;
        let eos = self.cfg.talker_config.codec_eos_token_id;
        let (vocab, temperature, top_k, penalty, suppress, exempt, min_frames) = match group {
            0 => {
                let v = self.vocab();
                (v, s.temperature, s.top_k, s.repetition_penalty, (v - CONTROL, v), eos, MIN_FRAMES)
            }
            _ => (self.p_vocab(), s.subtalker_temperature, s.subtalker_top_k, 1.0, (0, 0), -1, 0),
        };
        g.launch(
            &format!("sample{group}"),
            "talker_sample",
            [json!("seqs"), json!(1), json!(1)],
            SAMPLE_THREADS,
            vec![
                ("in buffer<bf16>", logits),
                i32a(vocab),
                f32a(temperature),
                i32a(top_k),
                f32a(penalty),
                i32a(suppress.0),
                i32a(suppress.1),
                i32a(exempt),
                i32a(min_frames),
                ini("pos"),
                inf("uniforms"),
                ini("force"),
                state_io(self.state),
                ini("lines"),
                stride(),
                i32a((group == 0) as usize),
                ("out buffer<i32>", buf("codes")),
                i32a(group),
            ],
        );
    }

    /// `hidden` in, `codes` (and `logits`, `p_logits`) out, the frame in each slot.
    pub fn tail(&self, g: &mut Gen, max_seqs: usize) {
        let (h, p, p_vocab) = (self.hidden(), self.predictor.cfg.hidden_size, self.p_vocab());
        let pc = &self.predictor.cfg;
        let seqs = json!("seqs");
        g.gemm_rows("head", (buf("logits"), buf("hidden"), buf(&self.codec_head)), count(&seqs), (self.vocab(), h));
        self.sample(g, 0, buf("logits"));
        g.launch(
            "pred.input",
            "talker_pred_input",
            [seqs.clone(), json!(2), json!(1)],
            (h / 8) as u32,
            vec![inb("hidden"), inb(&self.codec_emb), ini("codes"), outb("p.in"), i32a(h)],
        );
        for (name, w) in [
            ("p.in", 2 * h),
            ("p.h", 2 * p),
            ("p.last", p),
            (P.x, 2 * p),
            (P.res, 2 * p),
            (P.qkv, 2 * self.predictor.qkv_width()),
            (P.attn, 2 * pc.num_attention_heads * HEAD_DIM),
            (P.gate_up, 4 * pc.intermediate_size),
            (P.act, 2 * pc.intermediate_size),
        ] {
            g.need(name, w);
        }
        for i in 0..pc.num_hidden_layers {
            g.need(&format!("pkv{i}"), SPAN * Stack::kv_bytes(pc) / 2);
        }
        for group in 1..GROUPS {
            let label = format!("p{group}");
            let (rows, per, base, keep) = match group {
                1 => (json!({"mul": ["seqs", 2]}), 2, 0, Keep { out: "p.last", every: 2, which: 1 }),
                _ => (seqs.clone(), 1, group, Keep { out: "p.last", every: 1, which: 0 }),
            };
            if group > 1 {
                g.launch(
                    &format!("{label}.embed"),
                    "talker_gather",
                    [seqs.clone(), json!(1), json!(1)],
                    (h / 8) as u32,
                    vec![
                        ("in buffer<i32>", buf_at("codes", (group - 1) * 4)),
                        i32a(GROUPS),
                        ("in buffer<bf16>", buf_at(&self.p_emb, (group - 2) * p_vocab * h * 2)),
                        outb("p.in"),
                        i32a(h),
                    ],
                );
            }
            g.gemm_rows(&format!("{label}.mtp"), (buf("p.h"), buf("p.in"), buf(&self.mtp.0)), count(&rows), (p, h));
            g.each8(&format!("{label}.mtp.bias"), "talker_bias", &rows, p, vec![io("p.h"), inb(&self.mtp.1), i32a(p)]);
            let kv = Kv::Dense { prefix: "pkv", per, base, span: SPAN };
            self.predictor.forward(g, Pass { label: &label, rows: rows.clone(), input: "p.h", kv, keep }, &P);
            let logits = buf_at("p_logits", (group - 1) * max_seqs * p_vocab * 2);
            g.gemm_rows(
                &format!("{label}.head"),
                (logits.clone(), buf("p.last"), buf_at(&self.lm_heads, (group - 1) * p_vocab * p * 2)),
                count(&seqs),
                (p_vocab, p),
            );
            self.sample(g, group, logits);
        }
    }
}
