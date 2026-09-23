//! The talker and its code predictor: text in, one 16-codebook codec frame per
//! row per step out.
//!
//! One [`Talker::step`] is one talker forward over a ragged batch (admitted
//! prompts first, then running rows with the frame they produced last step),
//! the codebook-0 draw, then fifteen code-predictor passes filling codebooks
//! 1–15 for every row. The code predictor sees each frame on its own: its KV is
//! one page per batch slot, rewritten every step.
//!
//! Every sampled code stays on the device until the frame is complete, so a
//! step synchronizes with the host once.

use anyhow::Result;
use anyhow::ensure;
use omni_cuda::Act;
use omni_cuda::Buf;
use omni_cuda::Gpu;
use omni_cuda::Ptr;
use omni_cuda::SampleArgs;
use omni_cuda::bf16;

use crate::config;
use crate::prompt::Prompt;
use crate::stack::Batch;
use crate::stack::KvPool;
use crate::stack::Meta;
use crate::stack::RowKv;
use crate::stack::Scratch;
use crate::stack::Stack;
use crate::weights::File;
use crate::weights::upload;

pub const GROUPS: usize = 16;

/// What a row feeds the talker this step.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    Prompt(&'a Prompt),
    Frame(&'a [i32; GROUPS]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub sub_temperature: f32,
    pub sub_top_k: i32,
}

/// One row of a step.
#[derive(Clone, Copy, Debug)]
pub struct Row<'a> {
    pub pages: &'a [i32],
    /// Tokens already in the row's KV.
    pub cached: usize,
    pub input: Input<'a>,
    pub sampling: Sampling,
    /// Codebook-0 tokens under the repetition penalty, as a bitmap.
    pub seen: &'a [u32],
    /// Frames generated so far; the end token is masked until two exist.
    pub generated: usize,
    /// One uniform draw per codebook.
    pub uniforms: [f32; GROUPS],
}

impl Row<'_> {
    fn len(&self) -> usize {
        match self.input {
            Input::Prompt(p) => p.len(),
            Input::Frame(_) => 1,
        }
    }
}

/// A row's step result: a frame, or the end of its speech.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    Codes([i32; GROUPS]),
    End,
}

/// Test instrumentation for [`Talker::step`]: forces each row's codes to the
/// given frame after every draw (teacher forcing) and collects the talker's
/// input embeddings and the raw logits the draws saw, row-major.
#[derive(Debug, Default)]
pub struct Probe {
    pub force: Vec<[i32; GROUPS]>,
    pub inputs: Vec<f32>,
    pub talker_logits: Vec<f32>,
    pub predictor_logits: Vec<Vec<f32>>,
}

/// Capacity of a talker: rows per step, tokens per step, KV pages.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_batch: usize,
    pub max_tokens: usize,
    pub pages: usize,
    pub page_size: usize,
}

fn table(gpu: &Gpu, ptrs: &[Ptr]) -> Result<Buf<u64>> {
    gpu.upload(ptrs)
}

pub struct Talker {
    pub cfg: config::Model,
    pub limits: Limits,
    talker: Stack,
    kv: KvPool,
    scratch: Scratch,
    meta: Meta,
    predictor: Stack,
    p_kv: KvPool,
    p_scratch: Scratch,
    p_meta: Meta,

    /// Embedding tables the device pointer tables index into.
    _embeddings: [Buf<bf16>; 3],
    fc1: (Buf<bf16>, Buf<bf16>),
    fc2: (Buf<bf16>, Buf<bf16>),
    codec_head: Buf<bf16>,
    pad_embed: Buf<bf16>,
    mtp: (Buf<bf16>, Buf<bf16>),
    lm_heads: Buf<bf16>,

    t_text: Buf<u64>,
    t_codec: Buf<u64>,
    t_frame: Buf<u64>,
    t_last: Buf<u64>,
    t_rows: Buf<u64>,
    t_p_embed: Vec<Buf<u64>>,
    t_p_last: Buf<u64>,

    proj_a: Buf<bf16>,
    proj_b: Buf<bf16>,
    ids_text: Buf<i32>,
    ids_codec: Buf<i32>,
    ids_frame: Buf<i32>,
    last_idx: Buf<i32>,
    arange: Buf<i32>,
    odd: Buf<i32>,
    last: Buf<bf16>,
    logits: Buf<bf16>,
    p_in: Buf<bf16>,
    p_last: Buf<bf16>,
    p_logits: Buf<bf16>,

    temperature: Buf<f32>,
    top_k: Buf<i32>,
    penalty: Buf<f32>,
    seen: Buf<u32>,
    block_end: Buf<u8>,
    sub_temperature: Buf<f32>,
    sub_top_k: Buf<i32>,
    uniforms: Buf<f32>,
    codes: Buf<i32>,
}

impl Talker {
    pub fn load(gpu: &Gpu, file: &File, cfg: &config::Model, limits: Limits) -> Result<Self> {
        let t = &cfg.talker_config;
        let (h, p) = (t.stack.hidden_size, t.code_predictor_config.hidden_size);
        let (codec_vocab, p_vocab) = (t.stack.vocab_size, t.code_predictor_config.vocab_size);
        ensure!(t.num_code_groups == GROUPS, "{} code groups, the engine is built for {GROUPS}", t.num_code_groups);
        let w = |n: &str, shape: &[usize]| file.expect(n, shape);
        let text_vocab = file.shape("talker.model.text_embedding.weight")?[0];
        let talker = Stack::load(gpu, file, "talker.model", &t.stack)?;
        let predictor = Stack::load(gpu, file, "talker.code_predictor.model", &t.code_predictor_config)?;
        let p_embeddings: Vec<f32> = (0..GROUPS - 1)
            .map(|g| w(&format!("talker.code_predictor.model.codec_embedding.{g}.weight"), &[p_vocab, h]))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|t| t.data)
            .collect();
        let lm_heads: Vec<f32> = (0..GROUPS - 1)
            .map(|g| w(&format!("talker.code_predictor.lm_head.{g}.weight"), &[p_vocab, p]))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|t| t.data)
            .collect();

        let (b, n) = (limits.max_batch, limits.max_tokens);
        let group = t.stack.num_attention_heads / t.stack.num_key_value_heads;
        let p_group = t.code_predictor_config.num_attention_heads / t.code_predictor_config.num_key_value_heads;
        let scratch = Scratch::new(gpu, &t.stack, n)?;
        let p_scratch = Scratch::new(gpu, &t.code_predictor_config, 2 * b)?;
        let text_embedding = upload(gpu, &w("talker.model.text_embedding.weight", &[text_vocab, h])?.data)?;
        let codec_embedding = upload(gpu, &w("talker.model.codec_embedding.weight", &[codec_vocab, h])?.data)?;
        let p_embeddings = upload(gpu, &p_embeddings)?;
        let p_table = |g: usize| p_embeddings.at(g * p_vocab * h);
        let t_frame = table(gpu, &[vec![codec_embedding.ptr()], (0..GROUPS - 1).map(p_table).collect()].concat())?;
        let t_p_embed = (0..GROUPS - 1).map(|g| table(gpu, &[p_table(g)])).collect::<Result<_>>()?;

        let mut talker_ = Self {
            cfg: cfg.clone(),
            limits,
            kv: KvPool::new(gpu, &t.stack, limits.page_size, limits.pages)?,
            meta: Meta::new(gpu, n, b, limits.pages, group)?,
            p_kv: KvPool::new(gpu, &t.code_predictor_config, GROUPS, b)?,
            p_meta: Meta::new(gpu, 2 * b, b, b, p_group)?,
            t_text: table(gpu, &[text_embedding.ptr()])?,
            t_codec: table(gpu, &[codec_embedding.ptr()])?,
            t_frame,
            t_last: gpu.upload(&[0u64])?,
            t_rows: gpu.upload(&[0u64])?,
            t_p_embed,
            t_p_last: table(gpu, &[p_scratch.h.ptr()])?,
            fc1: (
                upload(gpu, &w("talker.text_projection.linear_fc1.weight", &[h, h])?.data)?,
                upload(gpu, &w("talker.text_projection.linear_fc1.bias", &[h])?.data)?,
            ),
            fc2: (
                upload(gpu, &w("talker.text_projection.linear_fc2.weight", &[h, h])?.data)?,
                upload(gpu, &w("talker.text_projection.linear_fc2.bias", &[h])?.data)?,
            ),
            codec_head: upload(gpu, &w("talker.codec_head.weight", &[codec_vocab, h])?.data)?,
            pad_embed: gpu.alloc(h)?,
            mtp: (
                upload(gpu, &w("talker.code_predictor.small_to_mtp_projection.weight", &[p, h])?.data)?,
                upload(gpu, &w("talker.code_predictor.small_to_mtp_projection.bias", &[p])?.data)?,
            ),
            lm_heads: upload(gpu, &lm_heads)?,
            _embeddings: [text_embedding, codec_embedding, p_embeddings],
            talker,
            predictor,
            scratch,
            p_scratch,
            proj_a: gpu.alloc(n * h)?,
            proj_b: gpu.alloc(n * h)?,
            ids_text: gpu.alloc(n)?,
            ids_codec: gpu.alloc(n)?,
            ids_frame: gpu.alloc(b * GROUPS)?,
            last_idx: gpu.alloc(b)?,
            arange: gpu.upload(&(0..b as i32).collect::<Vec<_>>())?,
            odd: gpu.upload(&(0..b as i32).map(|i| 2 * i + 1).collect::<Vec<_>>())?,
            last: gpu.alloc(b * h)?,
            logits: gpu.alloc(b * codec_vocab)?,
            p_in: gpu.alloc(2 * b * h)?,
            p_last: gpu.alloc(b * p)?,
            p_logits: gpu.alloc(b * p_vocab)?,
            temperature: gpu.alloc(b)?,
            top_k: gpu.alloc(b)?,
            penalty: gpu.alloc(b)?,
            seen: gpu.alloc(b * codec_vocab.div_ceil(32))?,
            block_end: gpu.alloc(b)?,
            sub_temperature: gpu.alloc(b)?,
            sub_top_k: gpu.alloc(b)?,
            uniforms: gpu.alloc(b * GROUPS)?,
            codes: gpu.alloc(b * GROUPS)?,
        };
        talker_.t_last = table(gpu, &[talker_.scratch.h.ptr()])?;
        talker_.t_rows = table(gpu, &[talker_.last.ptr()])?;
        let pad = cfg.tts_pad_token_id;
        talker_.project_text(gpu, &[pad], talker_.pad_embed.ptr())?;
        Ok(talker_)
    }

    /// Words of the codebook-0 bitmap a [`Row::seen`] holds.
    pub fn seen_words(&self) -> usize {
        self.cfg.talker_config.stack.vocab_size.div_ceil(32)
    }

    /// `out[i] = fc2(silu(fc1(text_embedding[ids[i]])))` without fc2's bias.
    fn project_text_raw(&mut self, gpu: &Gpu, ids: &[i32], out: Ptr) -> Result<()> {
        let h = self.cfg.talker_config.stack.hidden_size;
        let n = ids.len();
        gpu.write(&mut self.ids_text, 0, ids)?;
        gpu.gather_sum((self.proj_a.ptr(), h), false, 0, self.t_text.ptr(), (self.ids_text.ptr(), 1, 1), n, h)?;
        gpu.linear(self.proj_b.ptr(), self.proj_a.ptr(), self.fc1.0.ptr(), n, h, h)?;
        gpu.bias_act(self.proj_b.ptr(), self.fc1.1.ptr(), 0, self.proj_b.ptr(), Act::Silu, n, h)?;
        gpu.linear(out, self.proj_b.ptr(), self.fc2.0.ptr(), n, h, h)
    }

    fn project_text(&mut self, gpu: &Gpu, ids: &[i32], out: Ptr) -> Result<()> {
        let h = self.cfg.talker_config.stack.hidden_size;
        self.project_text_raw(gpu, ids, out)?;
        gpu.bias_act(out, self.fc2.1.ptr(), 0, out, Act::None, ids.len(), h)
    }

    fn force(&mut self, gpu: &Gpu, probe: &Probe, group: usize, b: usize) -> Result<()> {
        if probe.force.is_empty() {
            return Ok(());
        }
        let codes: Vec<i32> = probe.force.iter().map(|f| f[group]).collect();
        ensure!(codes.len() == b, "probe forces {} rows, the step has {b}", codes.len());
        gpu.write(&mut self.codes, group * b, &codes)
    }

    /// Advances every row by one frame. Prompt rows must come first.
    pub fn step(&mut self, gpu: &Gpu, rows: &[Row], mut probe: Option<&mut Probe>) -> Result<Vec<Frame>> {
        let t = self.cfg.talker_config.clone();
        let (h, p) = (t.stack.hidden_size, t.code_predictor_config.hidden_size);
        let (codec_vocab, p_vocab) = (t.stack.vocab_size, t.code_predictor_config.vocab_size);
        let b = rows.len();
        ensure!(b <= self.limits.max_batch, "{b} rows exceed max_batch {}", self.limits.max_batch);
        let prompts = rows.iter().take_while(|r| matches!(r.input, Input::Prompt(_))).count();
        ensure!(rows[prompts..].iter().all(|r| matches!(r.input, Input::Frame(_))), "prompt rows must lead the batch");

        // Talker input embeddings: prompts, then frames.
        let text: Vec<i32> = rows[..prompts]
            .iter()
            .flat_map(|r| match r.input {
                Input::Prompt(p) => p.text.clone(),
                Input::Frame(_) => vec![],
            })
            .collect();
        let codec: Vec<i32> = rows[..prompts]
            .iter()
            .flat_map(|r| match r.input {
                Input::Prompt(p) => p.codec.clone(),
                Input::Frame(_) => vec![],
            })
            .collect();
        let frames: Vec<i32> = rows[prompts..]
            .iter()
            .flat_map(|r| match r.input {
                Input::Frame(f) => f.to_vec(),
                Input::Prompt(_) => vec![],
            })
            .collect();
        let n_prompt = text.len();
        if n_prompt > 0 {
            let h_ptr = self.scratch.h.ptr();
            self.project_text_raw(gpu, &text, h_ptr)?;
            gpu.write(&mut self.ids_codec, 0, &codec)?;
            gpu.gather_sum(
                (h_ptr, h),
                true,
                self.fc2.1.ptr(),
                self.t_codec.ptr(),
                (self.ids_codec.ptr(), 1, 1),
                n_prompt,
                h,
            )?;
        }
        if !frames.is_empty() {
            gpu.write(&mut self.ids_frame, 0, &frames)?;
            gpu.gather_sum(
                (self.scratch.h.at(n_prompt * h), h),
                false,
                self.pad_embed.ptr(),
                self.t_frame.ptr(),
                (self.ids_frame.ptr(), GROUPS, GROUPS),
                b - prompts,
                h,
            )?;
        }

        if let Some(p) = probe.as_deref_mut() {
            p.inputs = read_f32(gpu, self.scratch.h.ptr(), rows.iter().map(Row::len).sum::<usize>() * h)?;
        }
        let row_kv: Vec<RowKv> =
            rows.iter().map(|r| RowKv { pages: r.pages, cached: r.cached, new: r.len() }).collect();
        let batch = Batch::new(&row_kv, self.kv.page_size);
        let plan = self.meta.set(gpu, &batch, &t.stack)?;
        self.talker.forward(gpu, &self.scratch, &self.kv, &self.meta, &plan)?;

        let last: Vec<i32> = batch.q_indptr[1..].iter().map(|&e| e - 1).collect();
        gpu.write(&mut self.last_idx, 0, &last)?;
        gpu.gather_sum((self.last.ptr(), h), false, 0, self.t_last.ptr(), (self.last_idx.ptr(), 1, 1), b, h)?;
        gpu.linear(self.logits.ptr(), self.last.ptr(), self.codec_head.ptr(), b, codec_vocab, h)?;

        let col = |f: fn(&Row) -> f32| rows.iter().map(f).collect::<Vec<f32>>();
        gpu.write(&mut self.temperature, 0, &col(|r| r.sampling.temperature))?;
        gpu.write(&mut self.penalty, 0, &col(|r| r.sampling.repetition_penalty))?;
        gpu.write(&mut self.sub_temperature, 0, &col(|r| r.sampling.sub_temperature))?;
        gpu.write(&mut self.top_k, 0, &rows.iter().map(|r| r.sampling.top_k).collect::<Vec<_>>())?;
        gpu.write(&mut self.sub_top_k, 0, &rows.iter().map(|r| r.sampling.sub_top_k).collect::<Vec<_>>())?;
        gpu.write(&mut self.seen, 0, &rows.iter().flat_map(|r| r.seen.iter().copied()).collect::<Vec<_>>())?;
        gpu.write(&mut self.block_end, 0, &rows.iter().map(|r| (r.generated < 2) as u8).collect::<Vec<_>>())?;
        let uniforms: Vec<f32> = (0..GROUPS).flat_map(|g| rows.iter().map(move |r| r.uniforms[g])).collect();
        gpu.write(&mut self.uniforms, 0, &uniforms)?;

        let eos = t.codec_eos_token_id;
        let special = (codec_vocab - 1024) as u32;
        let args = SampleArgs {
            temperature: self.temperature.ptr(),
            top_k: self.top_k.ptr(),
            penalty: self.penalty.ptr(),
            seen: self.seen.ptr(),
            suppress: (special, codec_vocab as u32),
            exempt: eos,
            block_exempt: self.block_end.ptr(),
            uniform: self.uniforms.ptr(),
        };
        gpu.sample((self.logits.ptr(), codec_vocab), b, codec_vocab, &args, self.codes.ptr())?;
        if let Some(p) = probe.as_deref_mut() {
            p.talker_logits = read_f32(gpu, self.logits.ptr(), b * codec_vocab)?;
            self.force(gpu, p, 0, b)?;
        }

        // Code predictor: [talker hidden, codebook-0 embedding] per row, then one code per pass.
        gpu.gather_sum((self.p_in.ptr(), 2 * h), false, 0, self.t_rows.ptr(), (self.arange.ptr(), 1, 1), b, h)?;
        gpu.gather_sum((self.p_in.at(h), 2 * h), false, 0, self.t_codec.ptr(), (self.codes.ptr(), 1, 1), b, h)?;
        let (sub_temperature, sub_top_k, uniforms) =
            (self.sub_temperature.ptr(), self.sub_top_k.ptr(), self.uniforms.ptr());
        let sub_args = |g: usize| SampleArgs {
            temperature: sub_temperature,
            top_k: sub_top_k,
            penalty: 0,
            seen: 0,
            suppress: (0, 0),
            exempt: -1,
            block_exempt: 0,
            uniform: uniforms + (g * b * 4) as u64,
        };
        for g in 1..GROUPS {
            let (tokens, cached) = if g == 1 { (2, 0) } else { (1, g) };
            if g > 1 {
                gpu.gather_sum(
                    (self.p_in.ptr(), h),
                    false,
                    0,
                    self.t_p_embed[g - 2].ptr(),
                    (self.codes.at((g - 1) * b), 1, 1),
                    b,
                    h,
                )?;
            }
            let n = b * tokens;
            gpu.linear(self.p_scratch.h.ptr(), self.p_in.ptr(), self.mtp.0.ptr(), n, p, h)?;
            gpu.bias_act(self.p_scratch.h.ptr(), self.mtp.1.ptr(), 0, self.p_scratch.h.ptr(), Act::None, n, p)?;
            let pages: Vec<[i32; 1]> = (0..b as i32).map(|i| [i]).collect();
            let row_kv: Vec<RowKv> = pages.iter().map(|pg| RowKv { pages: pg, cached, new: tokens }).collect();
            let plan = self.p_meta.set(gpu, &Batch::new(&row_kv, GROUPS), &t.code_predictor_config)?;
            self.predictor.forward(gpu, &self.p_scratch, &self.p_kv, &self.p_meta, &plan)?;
            let head = self.lm_heads.at((g - 1) * p_vocab * p);
            if tokens == 2 {
                gpu.gather_sum((self.p_last.ptr(), p), false, 0, self.t_p_last.ptr(), (self.odd.ptr(), 1, 1), b, p)?;
                gpu.linear(self.p_logits.ptr(), self.p_last.ptr(), head, b, p_vocab, p)?;
            } else {
                gpu.linear(self.p_logits.ptr(), self.p_scratch.h.ptr(), head, b, p_vocab, p)?;
            }
            gpu.sample((self.p_logits.ptr(), p_vocab), b, p_vocab, &sub_args(g), self.codes.at(g * b))?;
            if let Some(p) = probe.as_deref_mut() {
                p.predictor_logits.push(read_f32(gpu, self.p_logits.ptr(), b * p_vocab)?);
                self.force(gpu, p, g, b)?;
            }
        }

        let codes: Vec<i32> = gpu.read(self.codes.ptr(), GROUPS * b)?;
        Ok((0..b)
            .map(|r| match codes[r] {
                c if c == eos => Frame::End,
                _ => Frame::Codes(std::array::from_fn(|g| codes[g * b + r])),
            })
            .collect())
    }
}

fn read_f32(gpu: &Gpu, ptr: Ptr, len: usize) -> Result<Vec<f32>> {
    Ok(gpu.read::<bf16>(ptr, len)?.into_iter().map(bf16::to_f32).collect())
}
