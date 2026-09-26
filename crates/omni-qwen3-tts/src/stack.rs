//! A Qwen3 decoder stack as manifest calls: the talker and the code predictor
//! are both one of these.
//!
//! [`Stack::forward`] emits one pass over `rows` token rows: from an input
//! embedding buffer to the final norm's output, the residual stream alongside.
//! Where a row's K and V go, and which keys it attends to, is its [`Kv`]: the
//! talker's paged state, or the code predictor's per-call workspace.

use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

use crate::config;
use omni_kern::Gen;
use omni_kern::buf;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::weights::File;
use omni_kern::weights::concat_rows;

pub const HEAD_DIM: usize = 128;
/// Tokens per page of the talker's paged KV.
pub const PAGE: usize = 16;

struct Layer {
    ln1: String,
    qkv: String,
    q_norm: String,
    k_norm: String,
    o: String,
    ln2: String,
    gate_up: String,
    down: String,
}

pub struct Stack {
    pub cfg: config::Stack,
    layers: Vec<Layer>,
    norm: String,
}

/// Where a pass's K and V live.
#[derive(Clone, Copy)]
pub enum Kv<'a> {
    /// Layer `i` in state `{prefix}{i}` at each row's `t_slot`, the row at
    /// `t_pos`, its sequence `t_seq[n]` when `ragged` (else row `n` is
    /// sequence `n`) with its pages in `kv_indptr` / `kv_pages`.
    Paged { prefix: &'a str, ragged: bool },
    /// Layer `i` in workspace `{prefix}{i}`: `per` rows per sequence at
    /// positions `base..base + per`, `span` slots per sequence.
    Dense { prefix: &'a str, per: usize, base: usize, span: usize },
}

/// The rows a pass keeps: the final norm's rows `n % every == which`, compacted into `out`.
pub struct Keep<'a> {
    pub out: &'a str,
    pub every: usize,
    pub which: usize,
}

/// One pass: its calls labelled `{label}.*`, over `rows` rows (a var
/// expression, e.g. `"seqs"` or `{"mul": ["seqs", 2]}`) of the embeddings in
/// `input`.
pub struct Pass<'a> {
    pub label: &'a str,
    pub rows: Value,
    pub input: &'a str,
    pub kv: Kv<'a>,
    pub keep: Keep<'a>,
}

/// Workspace names of a stack's activations, each `rows` wide at most.
pub struct Scratch {
    pub x: &'static str,
    pub res: &'static str,
    pub qkv: &'static str,
    pub attn: &'static str,
    pub gate_up: &'static str,
    pub act: &'static str,
}

impl Stack {
    /// Registers `{prefix}.layers.*` and `{prefix}.norm` as weights named `{name}.*`,
    /// Q|K|V and gate|up fused.
    pub fn load(g: &mut Gen, file: &File, prefix: &str, name: &str, cfg: &config::Stack) -> Result<Self> {
        ensure!(
            cfg.head_dim == HEAD_DIM,
            "{prefix}: head_dim {} unsupported (attention is built for 128)",
            cfg.head_dim
        );
        ensure!(cfg.hidden_size.is_multiple_of(256) && cfg.hidden_size <= 8192, "{prefix}: hidden {}", cfg.hidden_size);
        let (h, d) = (cfg.hidden_size, cfg.head_dim);
        let (q, kv, inter) = (cfg.num_attention_heads * d, cfg.num_key_value_heads * d, cfg.intermediate_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                let w = |n: &str, shape: &[usize]| file.expect(&format!("{p}.{n}"), shape);
                let n = |s: &str| format!("{name}.l{i}.{s}");
                Ok(Layer {
                    ln1: g.weight(&n("ln1"), &[h], &w("input_layernorm.weight", &[h])?.data),
                    qkv: g.weight(
                        &n("qkv"),
                        &[q + 2 * kv, h],
                        &concat_rows(&[
                            w("self_attn.q_proj.weight", &[q, h])?,
                            w("self_attn.k_proj.weight", &[kv, h])?,
                            w("self_attn.v_proj.weight", &[kv, h])?,
                        ]),
                    ),
                    q_norm: g.weight(&n("q_norm"), &[d], &w("self_attn.q_norm.weight", &[d])?.data),
                    k_norm: g.weight(&n("k_norm"), &[d], &w("self_attn.k_norm.weight", &[d])?.data),
                    o: g.weight(&n("o"), &[h, q], &w("self_attn.o_proj.weight", &[h, q])?.data),
                    ln2: g.weight(&n("ln2"), &[h], &w("post_attention_layernorm.weight", &[h])?.data),
                    gate_up: g.weight(
                        &n("gate_up"),
                        &[2 * inter, h],
                        &concat_rows(&[w("mlp.gate_proj.weight", &[inter, h])?, w("mlp.up_proj.weight", &[inter, h])?]),
                    ),
                    down: g.weight(&n("down"), &[h, inter], &w("mlp.down_proj.weight", &[h, inter])?.data),
                })
            })
            .collect::<Result<_>>()?;
        let norm = g.weight(&format!("{name}.norm"), &[h], &file.expect(&format!("{prefix}.norm.weight"), &[h])?.data);
        Ok(Self { cfg: cfg.clone(), layers, norm })
    }

    pub fn qkv_width(&self) -> usize {
        (self.cfg.num_attention_heads + 2 * self.cfg.num_key_value_heads) * self.cfg.head_dim
    }

    /// Bytes of K and V one token takes in one layer.
    pub fn kv_bytes(cfg: &config::Stack) -> usize {
        2 * cfg.num_key_value_heads * cfg.head_dim * 2
    }

    pub fn forward(&self, g: &mut Gen, pass: Pass, s: &Scratch) {
        let Pass { label, rows, input, kv, keep } = pass;
        let rows = &rows;
        let c = &self.cfg;
        let (h, hq, inter) = (c.hidden_size, c.num_attention_heads, c.intermediate_size);
        let n_arg = count(rows);
        let norm_block = (h / 8) as u32;
        let per_row = [rows.clone(), json!(1), json!(1)];
        g.launch(
            &format!("{label}.embed_norm"),
            "talker_norm_copy",
            per_row.clone(),
            norm_block,
            vec![inb(input), inb(&self.layers[0].ln1), outb(s.x), outb(s.res), i32a(h), f32a(c.rms_norm_eps)],
        );
        for (i, l) in self.layers.iter().enumerate() {
            let at = |s: &str| format!("{label}.l{i}.{s}");
            g.gemm_rows(&at("qkv"), (buf(s.qkv), buf(s.x), buf(&l.qkv)), n_arg.clone(), (self.qkv_width(), h));
            self.attention(g, &format!("{label}.l{i}"), rows, (i, l), kv, s);
            g.gemm_rows(&at("o"), (buf(s.x), buf(s.attn), buf(&l.o)), n_arg.clone(), (h, hq * HEAD_DIM));
            g.launch(
                &at("post_attn_norm"),
                "talker_add_norm",
                per_row.clone(),
                norm_block,
                vec![inb(s.x), io(s.res), inb(&l.ln2), outb(s.x), i32a(h), f32a(c.rms_norm_eps), i32a(1), i32a(0)],
            );
            g.gemm_rows(&at("gate_up"), (buf(s.gate_up), buf(s.x), buf(&l.gate_up)), n_arg.clone(), (2 * inter, h));
            g.each8(&at("silu_mul"), "talker_silu_mul", rows, inter, vec![inb(s.gate_up), outb(s.act), i32a(inter)]);
            g.gemm_rows(&at("down"), (buf(s.x), buf(s.act), buf(&l.down)), n_arg.clone(), (h, inter));
            let (next, dst, every, which) = match self.layers.get(i + 1) {
                Some(n) => (&n.ln1, s.x, 1, 0),
                None => (&self.norm, keep.out, keep.every, keep.which),
            };
            g.launch(
                &at("next_norm"),
                "talker_add_norm",
                per_row.clone(),
                norm_block,
                vec![
                    inb(s.x),
                    io(s.res),
                    inb(next),
                    outb(dst),
                    i32a(h),
                    f32a(c.rms_norm_eps),
                    i32a(every),
                    i32a(which),
                ],
            );
        }
    }

    /// Layer `i`'s Q/K norm and rotary embedding, K and V into `kv`, and
    /// attention from `s.qkv` into `s.attn`.
    fn attention(&self, g: &mut Gen, label: &str, rows: &Value, (i, l): (usize, &Layer), kv: Kv, s: &Scratch) {
        let c = &self.cfg;
        let (hq, hk) = (c.num_attention_heads, c.num_key_value_heads);
        let (rope, rope_kv, attend, attend_kv) = match kv {
            Kv::Paged { prefix, ragged } => {
                let state = json!({"state": format!("{prefix}{i}")});
                let paged = vec![
                    ("in state", state.clone()),
                    ini("t_pos"),
                    ini("t_seq"),
                    i32a(ragged as usize),
                    ini("kv_indptr"),
                    ini("kv_pages"),
                    i32a(PAGE),
                ];
                ("talker_rope", vec![ini("t_pos"), ini("t_slot"), ("inout state", state)], "talker_attend", paged)
            }
            Kv::Dense { prefix, per, base, span } => {
                let ws = format!("{prefix}{i}");
                let layout = [i32a(per), i32a(base), i32a(span)];
                let rope_kv = [&[outb(&ws)][..], &layout].concat();
                ("talker_rope_dense", rope_kv, "talker_attend_dense", [&[inb(&ws)][..], &layout].concat())
            }
        };
        let heads = json!((hq + 2 * hk).div_ceil(8));
        let rope_args = [
            &[io(s.qkv), inb(&l.q_norm), inb(&l.k_norm)][..],
            &rope_kv,
            &[i32a(hq), i32a(hk), f32a(c.rms_norm_eps), f32a(c.rope_theta)],
        ];
        g.launch(&format!("{label}.rope"), rope, [rows.clone(), heads, json!(1)], 256, rope_args.concat());
        let scale = f32a(1.0 / (HEAD_DIM as f32).sqrt());
        let attend_args = [&[inb(s.qkv)][..], &attend_kv, &[outb(s.attn), i32a(hq), i32a(hk), scale]];
        g.launch(&format!("{label}.attn"), attend, [rows.clone(), json!(hk), json!(1)], 128, attend_args.concat());
    }
}
