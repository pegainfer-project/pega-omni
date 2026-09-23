//! A Qwen3 decoder stack over a paged KV cache: the talker and the code
//! predictor are both one of these.
//!
//! A forward pass takes a ragged batch, rows of any query length appended to
//! KV they already own, so one call serves admissions and decode rows alike.
//! [`Batch`] is the host-side description (pure, built from each row's pages
//! and lengths); [`Meta`] holds its device copy.

use anyhow::Result;
use anyhow::ensure;
use omni_cuda::Buf;
use omni_cuda::Gpu;
use omni_cuda::PagedKv;
use omni_cuda::PlanBufs;
use omni_cuda::PrefillPlan;
use omni_cuda::bf16;

use crate::config;
use crate::weights::File;
use crate::weights::concat_rows;
use crate::weights::upload;

pub struct Layer {
    ln1: Buf<bf16>,
    qkv: Buf<bf16>,
    q_norm: Buf<bf16>,
    k_norm: Buf<bf16>,
    o: Buf<bf16>,
    ln2: Buf<bf16>,
    gate_up: Buf<bf16>,
    down: Buf<bf16>,
}

pub struct Stack {
    pub cfg: config::Stack,
    layers: Vec<Layer>,
    norm: Buf<bf16>,
}

impl Stack {
    /// Loads `{prefix}.layers.*` and `{prefix}.norm`, fusing Q|K|V and gate|up.
    pub fn load(gpu: &Gpu, file: &File, prefix: &str, cfg: &config::Stack) -> Result<Self> {
        ensure!(cfg.head_dim == 128, "{prefix}: head_dim {} unsupported (attention is built for 128)", cfg.head_dim);
        let (h, d) = (cfg.hidden_size, cfg.head_dim);
        let (q, kv, inter) = (cfg.num_attention_heads * d, cfg.num_key_value_heads * d, cfg.intermediate_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                let w = |n: &str, shape: &[usize]| file.expect(&format!("{p}.{n}"), shape);
                Ok(Layer {
                    ln1: upload(gpu, &w("input_layernorm.weight", &[h])?.data)?,
                    qkv: upload(
                        gpu,
                        &concat_rows(&[
                            w("self_attn.q_proj.weight", &[q, h])?,
                            w("self_attn.k_proj.weight", &[kv, h])?,
                            w("self_attn.v_proj.weight", &[kv, h])?,
                        ]),
                    )?,
                    q_norm: upload(gpu, &w("self_attn.q_norm.weight", &[d])?.data)?,
                    k_norm: upload(gpu, &w("self_attn.k_norm.weight", &[d])?.data)?,
                    o: upload(gpu, &w("self_attn.o_proj.weight", &[h, q])?.data)?,
                    ln2: upload(gpu, &w("post_attention_layernorm.weight", &[h])?.data)?,
                    gate_up: upload(
                        gpu,
                        &concat_rows(&[w("mlp.gate_proj.weight", &[inter, h])?, w("mlp.up_proj.weight", &[inter, h])?]),
                    )?,
                    down: upload(gpu, &w("mlp.down_proj.weight", &[h, inter])?.data)?,
                })
            })
            .collect::<Result<_>>()?;
        let norm = upload(gpu, &file.expect(&format!("{prefix}.norm.weight"), &[h])?.data)?;
        Ok(Self { cfg: cfg.clone(), layers, norm })
    }

    fn qkv_width(&self) -> usize {
        (self.cfg.num_attention_heads + 2 * self.cfg.num_key_value_heads) * self.cfg.head_dim
    }

    /// Runs `rows` tokens: embeddings in `s.h`, final normed hidden states out in `s.h`.
    pub fn forward(&self, gpu: &Gpu, s: &Scratch, kv: &KvPool, meta: &Meta) -> Result<()> {
        let c = &self.cfg;
        let (rows, batch) = (meta.rows, meta.batch);
        let (h, d, hq, hk, inter) =
            (c.hidden_size, c.head_dim, c.num_attention_heads, c.num_key_value_heads, c.intermediate_size);
        let (qkv_w, eps) = (self.qkv_width(), c.rms_norm_eps);
        gpu.copy(s.residual.ptr(), s.h.ptr(), rows * h * 2)?;
        gpu.rms_norm(s.h.ptr(), self.layers[0].ln1.ptr(), s.h.ptr(), rows, h, eps)?;
        for (i, l) in self.layers.iter().enumerate() {
            gpu.linear(s.qkv.ptr(), s.h.ptr(), l.qkv.ptr(), rows, qkv_w, h)?;
            gpu.qk_norm_rope(
                s.qkv.ptr(),
                qkv_w,
                rows,
                (hq, hk, d),
                (l.q_norm.ptr(), l.k_norm.ptr(), eps),
                meta.positions.ptr(),
                (meta.slots.ptr(), kv.k[i].ptr(), kv.v[i].ptr()),
                c.rope_theta,
            )?;
            let paged = PagedKv {
                k_pool: kv.k[i].ptr(),
                v_pool: kv.v[i].ptr(),
                page_size: kv.page_size as u32,
                page_indices: meta.page_indices.ptr(),
                page_indptr: meta.page_indptr.ptr(),
                last_page_len: meta.last_page_len.ptr(),
            };
            gpu.paged_prefill((s.qkv.ptr(), qkv_w), s.attn.ptr(), &paged, &meta.plan, (rows, batch), (hq, hk))?;
            gpu.linear(s.h.ptr(), s.attn.ptr(), l.o.ptr(), rows, h, hq * d)?;
            gpu.add_rms_norm(s.h.ptr(), s.residual.ptr(), l.ln2.ptr(), rows, h, eps)?;
            gpu.linear(s.gate_up.ptr(), s.h.ptr(), l.gate_up.ptr(), rows, 2 * inter, h)?;
            gpu.silu_mul(s.gate_up.ptr(), s.act.ptr(), rows, inter)?;
            gpu.linear(s.h.ptr(), s.act.ptr(), l.down.ptr(), rows, h, inter)?;
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.ln1);
            gpu.add_rms_norm(s.h.ptr(), s.residual.ptr(), next.ptr(), rows, h, eps)?;
        }
        Ok(())
    }
}

/// Paged K and V for every layer: `pages` pages of `page_size` tokens.
pub struct KvPool {
    k: Vec<Buf<bf16>>,
    v: Vec<Buf<bf16>>,
    pub page_size: usize,
    pub pages: usize,
}

impl KvPool {
    pub fn new(gpu: &Gpu, cfg: &config::Stack, page_size: usize, pages: usize) -> Result<Self> {
        let len = pages * page_size * cfg.num_key_value_heads * cfg.head_dim;
        let pool = || (0..cfg.num_hidden_layers).map(|_| gpu.alloc::<bf16>(len)).collect::<Result<Vec<_>>>();
        Ok(Self { k: pool()?, v: pool()?, page_size, pages })
    }

    /// Bytes one page takes across all layers, K and V.
    pub fn page_bytes(cfg: &config::Stack, page_size: usize) -> usize {
        2 * cfg.num_hidden_layers * page_size * cfg.num_key_value_heads * cfg.head_dim * 2
    }
}

/// Activations for up to `rows` tokens of one stack.
pub struct Scratch {
    pub h: Buf<bf16>,
    residual: Buf<bf16>,
    qkv: Buf<bf16>,
    attn: Buf<bf16>,
    gate_up: Buf<bf16>,
    act: Buf<bf16>,
}

impl Scratch {
    pub fn new(gpu: &Gpu, cfg: &config::Stack, rows: usize) -> Result<Self> {
        let q = cfg.num_attention_heads * cfg.head_dim;
        let qkv = q + 2 * cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            h: gpu.alloc(rows * cfg.hidden_size)?,
            residual: gpu.alloc(rows * cfg.hidden_size)?,
            qkv: gpu.alloc(rows * qkv)?,
            attn: gpu.alloc(rows * q)?,
            gate_up: gpu.alloc(rows * 2 * cfg.intermediate_size)?,
            act: gpu.alloc(rows * cfg.intermediate_size)?,
        })
    }
}

/// One row of a forward: the pages it owns, how many tokens they already hold,
/// and how many it appends.
#[derive(Clone, Debug)]
pub struct RowKv<'a> {
    pub pages: &'a [i32],
    pub cached: usize,
    pub new: usize,
}

/// A ragged batch, host side: per token its position and KV slot, per row its
/// page table after the append.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Batch {
    pub positions: Vec<i32>,
    pub slots: Vec<i32>,
    pub q_indptr: Vec<i32>,
    pub page_indices: Vec<i32>,
    pub page_indptr: Vec<i32>,
    pub last_page_len: Vec<i32>,
    pub qo_lens: Vec<u32>,
}

impl Batch {
    pub fn new(rows: &[RowKv], page_size: usize) -> Self {
        let mut b = Batch { q_indptr: vec![0], page_indptr: vec![0], ..Default::default() };
        for r in rows {
            let total = r.cached + r.new;
            let used = total.div_ceil(page_size);
            assert!(used <= r.pages.len(), "row holds {} pages, needs {used}", r.pages.len());
            for pos in r.cached..total {
                b.positions.push(pos as i32);
                b.slots.push(r.pages[pos / page_size] * page_size as i32 + (pos % page_size) as i32);
            }
            b.q_indptr.push(b.positions.len() as i32);
            b.page_indices.extend_from_slice(&r.pages[..used]);
            b.page_indptr.push(b.page_indices.len() as i32);
            b.last_page_len.push((total - (used - 1) * page_size) as i32);
            b.qo_lens.push(r.new as u32);
        }
        b
    }
}

/// Device copies of a [`Batch`], sized for the largest batch a stack runs.
pub struct Meta {
    positions: Buf<i32>,
    slots: Buf<i32>,
    page_indices: Buf<i32>,
    page_indptr: Buf<i32>,
    last_page_len: Buf<i32>,
    plan: PlanBufs,
    rows: usize,
    batch: usize,
}

impl Meta {
    pub fn new(gpu: &Gpu, max_rows: usize, max_batch: usize, max_pages: usize, group: usize) -> Result<Self> {
        let max_tiles = max_rows * group + max_batch;
        Ok(Self {
            positions: gpu.alloc(max_rows)?,
            slots: gpu.alloc(max_rows)?,
            page_indices: gpu.alloc(max_pages)?,
            page_indptr: gpu.alloc(max_batch + 1)?,
            last_page_len: gpu.alloc(max_batch)?,
            plan: PlanBufs::new(gpu, max_batch, max_tiles)?,
            rows: 0,
            batch: 0,
        })
    }

    /// Uploads `b` and its tile plan.
    pub fn set(&mut self, gpu: &Gpu, b: &Batch, cfg: &config::Stack) -> Result<()> {
        let group = (cfg.num_attention_heads / cfg.num_key_value_heads) as u32;
        ensure!(
            b.positions.len() <= self.positions.len(),
            "batch of {} tokens exceeds the stack's scratch",
            b.positions.len()
        );
        self.plan.set(gpu, &PrefillPlan::new(&b.qo_lens, group, cfg.head_dim as u32), &b.q_indptr)?;
        gpu.write(&mut self.positions, 0, &b.positions)?;
        gpu.write(&mut self.slots, 0, &b.slots)?;
        gpu.write(&mut self.page_indices, 0, &b.page_indices)?;
        gpu.write(&mut self.page_indptr, 0, &b.page_indptr)?;
        gpu.write(&mut self.last_page_len, 0, &b.last_page_len)?;
        self.rows = b.positions.len();
        self.batch = b.qo_lens.len();
        Ok(())
    }
}
