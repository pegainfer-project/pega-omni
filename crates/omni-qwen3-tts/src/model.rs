//! The whole synthesis path as one kern manifest on one runtime: prompts in,
//! PCM out.
//!
//! Four programs over two vars, `tokens` (prompt rows of a call) and `seqs`
//! (sequences of a call):
//!
//! - `init` (once): the `<tts_pad>` embedding every decode row adds.
//! - `prefill` (eager, ragged): new prompts into their KV, each one's last
//!   hidden state into `hidden`.
//! - `first` (graph): `hidden` → frame → PCM, the frame the prompt produced.
//! - `decode` (graph): last frame → talker → frame → PCM, one per sequence.
//!
//! A request is `prefill`, `first`, then `decode` until it draws the end
//! token. Graph calls pad `seqs` up to a [`bucket`]; padding rows run on a
//! lease of their own. A [`Seq`] is a lease: the talker's KV pages for its
//! prompt and frame cap, and a slot of the `seq` state holding its last
//! frame, its repetition bitmap and its codec decoder state.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use kern_pool::Denied;
use kern_pool::Lease;
use kern_runtime::Capacity;
use kern_runtime::Runtime;
use omni_kern::Gen;
use omni_kern::HostTensors;
use omni_kern::bf16s;
use omni_kern::bucket;
use omni_kern::hex;
use omni_kern::ints;
use omni_kern::kernels_dir;
use omni_kern::vars;
use omni_kern::weights::File;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;

use crate::codec;
use crate::config::Config;
use crate::prompt::NO_CODEC;
use crate::prompt::Prompt;
use crate::prompt::Tokenizer;
use crate::stack::PAGE;
use crate::stack::Stack;
use crate::talker::Talker;

pub use crate::talker::GROUPS;

const CODEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/codec.cubin"));
const TALKER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/talker.cubin"));

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Sequences running at once.
    pub max_batch: usize,
    /// Prompt tokens per `prefill` call.
    pub max_tokens: usize,
    /// Talker KV cache size.
    pub kv_gib: f64,
}

/// One sequence's KV and state.
pub struct Seq {
    lease: Lease,
    prompt: usize,
    frames: usize,
}

/// A sequence's draws for one frame: a uniform per codebook, or the frame to
/// take instead (teacher forcing).
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub uniforms: [f32; GROUPS],
    pub force: Option<[i32; GROUPS]>,
}

/// A sequence's row of a graph call.
struct Frame<'a> {
    lease: &'a Lease,
    /// The talker position its last frame goes in.
    position: usize,
    /// The index of the frame this call draws.
    frame: usize,
    draw: Draw,
}

/// A call's frames and their PCM, `samples_per_frame` per sequence, in order.
#[derive(Debug)]
pub struct Out {
    pub codes: Vec<[i32; GROUPS]>,
    pub wav: Vec<f32>,
}

pub struct Model {
    pub config: Config,
    pub tokenizer: Tokenizer,
    rt: Runtime,
    pad: Lease,
    limits: Limits,
    max_seqs: usize,
    /// The most pages the running sequences may hold.
    pages_max: usize,
}

impl Model {
    /// Loads the checkpoint at `dir` onto `device`.
    pub fn load(device: usize, dir: &Path, limits: Limits) -> Result<Self> {
        let config = Config::load(dir)?;
        let tokenizer = Tokenizer::load(dir)?;
        let max_seqs = bucket(limits.max_batch);
        let kv_tokens = (limits.kv_gib * (1u64 << 30) as f64) as usize / kv_bytes_per_token(&config);
        let pages_max = kv_tokens.div_ceil(PAGE) + 1;
        let cubins = [("codec", CODEC), ("talker", TALKER)].map(|(n, b)| (n, hex(&sha2::Sha256::digest(b)), b));
        // Each padding row of a graph call names the pad lease's page once more.
        let page_table = pages_max + max_seqs;
        let (manifest, tensors) = generate(dir, &config, &cubins, max_seqs, limits.max_tokens, page_table)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("manifest: {e}"))?;
        let dir = kernels_dir(&cubins.each_ref().map(|(n, sha, b)| (*n, sha.as_str(), *b)))?;
        let capacity = Capacity { tokens: Some(kv_tokens as u64), seqs: limits.max_batch as u64 + 1 };
        let mut rt = Runtime::load(&verified, Some(&dir), device, Some(capacity), None)?;
        rt.load_weights(&tensors)?;
        rt.issue("init", &BTreeMap::new())?;
        let pad = rt.lease(1)?;
        Ok(Self { config, tokenizer, rt, pad, limits, max_seqs, pages_max })
    }

    /// A sequence of a `prompt`-token prompt and up to `frames` frames, or
    /// why not: `Busy` and `Remapping` pass, the rest never will.
    pub fn open(&mut self, prompt: usize, frames: usize) -> Result<Result<Seq, Denied>> {
        let lease = match self.rt.lease(prompt + frames) {
            Ok(l) => l,
            Err(kern_runtime::Error::Denied(d)) => return Ok(Err(d)),
            Err(e) => return Err(e.into()),
        };
        if self.rt.pages_used() > self.pages_max {
            let alone = lease.page_ids().len() + self.pad.page_ids().len();
            return Ok(Err(if alone > self.pages_max { Denied::ExceedsPool } else { Denied::Busy }));
        }
        Ok(Ok(Seq { lease, prompt, frames: 0 }))
    }

    pub fn is_end(&self, frame: &[i32; GROUPS]) -> bool {
        frame[0] == self.config.model.talker_config.codec_eos_token_id
    }

    /// Prefills each sequence's prompt, then draws and decodes its first frame.
    pub fn start(&mut self, rows: &mut [(&mut Seq, &Prompt, Draw)]) -> Result<Out> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.limits.max_batch, "{n} sequences in one call");
        let tokens: usize = rows.iter().map(|(_, p, _)| p.len()).sum();
        ensure!(tokens <= self.limits.max_tokens, "{tokens} prompt tokens exceed {}", self.limits.max_tokens);
        let mut t = Rows::default();
        let mut last_row = Vec::with_capacity(n);
        for (s, (seq, prompt, _)) in rows.iter().enumerate() {
            ensure!(seq.frames == 0 && prompt.len() == seq.prompt, "sequence {s} is not fresh");
            t.ids.extend(&prompt.text);
            t.codec.extend(&prompt.codec);
            t.push_seq(&seq.lease, 0..prompt.len(), s);
            last_row.push(t.pos.len() as i32 - 1);
        }
        let v = vars(tokens, n);
        t.write(&mut self.rt, &v, true)?;
        self.rt.write_input_at("last_row", &ints(&last_row), &v)?;
        self.rt.issue("prefill", &v)?;
        let steps: Vec<Frame> =
            rows.iter().map(|(s, _, d)| Frame { lease: &s.lease, position: 0, frame: 0, draw: *d }).collect();
        self.graph_inputs(&steps, false)?;
        let out = self.run("first", n)?;
        rows.iter_mut().for_each(|(s, _, _)| s.frames = 1);
        Ok(out)
    }

    /// Advances every sequence by one frame.
    pub fn step(&mut self, rows: &mut [(&mut Seq, Draw)]) -> Result<Out> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.limits.max_batch, "{n} sequences in one call");
        ensure!(rows.iter().all(|(s, _)| s.frames > 0), "a sequence that has not started");
        let steps: Vec<Frame> = rows
            .iter()
            .map(|(s, d)| Frame { lease: &s.lease, position: s.prompt + s.frames - 1, frame: s.frames, draw: *d })
            .collect();
        self.graph_inputs(&steps, true)?;
        let out = self.run("decode", n)?;
        rows.iter_mut().for_each(|(s, _)| s.frames += 1);
        Ok(out)
    }

    /// The per-sequence inputs of a graph call, padded to its bucket: slot
    /// lines, frame indices, draws and, with `talker`, each sequence's
    /// talker row at the position its last frame goes in.
    fn graph_inputs(&mut self, frames: &[Frame], talker: bool) -> Result<()> {
        let b = bucket(frames.len());
        let draw = Draw { uniforms: [0.5; GROUPS], force: None };
        let pad = Frame { lease: &self.pad, position: 0, frame: 0, draw };
        let padded: Vec<&Frame> = frames.iter().chain(std::iter::repeat(&pad)).take(b).collect();
        let lines = padded.iter().map(|f| f.lease.seq_line("lines", 0)).collect::<Result<Vec<_>, _>>()?;
        let pos: Vec<i32> = padded.iter().map(|f| f.frame as i32).collect();
        let uniforms: Vec<u8> = padded.iter().flat_map(|f| f.draw.uniforms).flat_map(f32::to_le_bytes).collect();
        let force: Vec<i32> = padded.iter().flat_map(|f| f.draw.force.unwrap_or([-1; GROUPS])).collect();
        let v = vars(b, b);
        let rt = &mut self.rt;
        rt.write_input_at("lines", &ints(&lines), &v)?;
        rt.write_input_at("pos", &ints(&pos), &v)?;
        rt.write_input_at("uniforms", &uniforms, &v)?;
        rt.write_input_at("force", &ints(&force), &v)?;
        if talker {
            let mut t = Rows::default();
            padded.iter().enumerate().for_each(|(s, f)| t.push_seq(f.lease, f.position..f.position + 1, s));
            t.write(rt, &v, false)?;
        }
        Ok(())
    }

    fn run(&mut self, program: &str, n: usize) -> Result<Out> {
        let b = bucket(n);
        self.rt.issue(program, &vars(b, b))?;
        let spf = self.config.samples_per_frame;
        let codes = self.rt.read_buffer_prefix("codes", n * GROUPS * 4)?;
        let wav = self.rt.read_buffer_prefix("wav", n * spf * 2)?;
        Ok(Out {
            codes: codes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| i32::from_le_bytes(b))
                .collect::<Vec<_>>()
                .as_chunks::<GROUPS>()
                .0
                .to_vec(),
            wav: bf16s(&wav),
        })
    }

    /// The last `prefill`'s input embeddings, `[tokens, hidden]`.
    pub fn embeds(&self, tokens: usize) -> Result<Vec<f32>> {
        let h = self.config.model.talker_config.stack.hidden_size;
        Ok(bf16s(&self.rt.read_buffer_prefix("embeds", tokens * h * 2)?))
    }

    /// The last graph call's codebook-0 logits of sequence `s`.
    pub fn talker_logits(&self, s: usize) -> Result<Vec<f32>> {
        let v = self.config.model.talker_config.stack.vocab_size;
        Ok(bf16s(&self.rt.read_buffer_prefix("logits", (s + 1) * v * 2)?[s * v * 2..]))
    }

    /// The last graph call's logits of codebooks 1–15 of sequence `s`, `[15, p_vocab]`.
    pub fn predictor_logits(&self, s: usize) -> Result<Vec<f32>> {
        let v = self.config.model.talker_config.code_predictor_config.vocab_size;
        let all = self.rt.read_output("p_logits")?;
        Ok((0..GROUPS - 1)
            .flat_map(|g| {
                let at = ((g * self.max_seqs + s) * v) * 2;
                bf16s(&all[at..at + v * 2])
            })
            .collect())
    }
}

/// The talker rows of a call and the KV they read.
#[derive(Default)]
struct Rows {
    ids: Vec<i32>,
    codec: Vec<i32>,
    pos: Vec<i32>,
    slot: Vec<i32>,
    seq: Vec<i32>,
    indptr: Vec<i32>,
    pages: Vec<i32>,
}

impl Rows {
    fn push_seq(&mut self, lease: &Lease, positions: std::ops::Range<usize>, s: usize) {
        if self.indptr.is_empty() {
            self.indptr.push(0);
        }
        for p in positions {
            self.pos.push(p as i32);
            self.slot.push(lease.slot(p) as i32);
            self.seq.push(s as i32);
        }
        self.pages.extend(lease.page_ids());
        self.indptr.push(self.pages.len() as i32);
    }

    fn write(&self, rt: &mut Runtime, v: &BTreeMap<String, u64>, prompt: bool) -> Result<()> {
        if prompt {
            rt.write_input_at("t_ids", &ints(&self.ids), v)?;
            rt.write_input_at("t_codec", &ints(&self.codec), v)?;
        }
        rt.write_input_at("t_pos", &ints(&self.pos), v)?;
        rt.write_input_at("t_slot", &ints(&self.slot), v)?;
        rt.write_input_at("t_seq", &ints(&self.seq), v)?;
        rt.write_input_at("kv_indptr", &ints(&self.indptr), v)?;
        rt.write_input_at("kv_pages", &ints(&self.pages), v).context("kv_pages")?;
        Ok(())
    }
}

/// Talker KV bytes per token: every layer's K and V.
fn kv_bytes_per_token(config: &Config) -> usize {
    let t = &config.model.talker_config.stack;
    t.num_hidden_layers * Stack::kv_bytes(t)
}

/// The manifest (JSON) and the tensors its weight buffers bind.
fn generate(
    dir: &Path,
    config: &Config,
    cubins: &[(&str, String, &[u8]); 2],
    max_seqs: usize,
    max_tokens: usize,
    page_table: usize,
) -> Result<(Value, HostTensors)> {
    let t = &config.model.talker_config;
    let mut g = Gen::with_pdl(&["codec"]);
    let talker = Talker::load(&mut g, &File::open(&dir.join("model.safetensors"))?, &config.model, &config.generation)?;
    talker.init(&mut g);
    let init = g.take();
    talker.prefill(&mut g);
    let prefill = g.take();
    talker.advance(&mut g);
    let advance = g.take();
    talker.tail(&mut g, max_seqs);
    let tail = g.take();
    let codec_file = File::open(&dir.join("speech_tokenizer/model.safetensors"))?;
    codec::build(&mut g, &codec_file, &config.codec, config.samples_per_frame)?;
    let codec = g.take();

    let mut programs = serde_json::Map::new();
    let batch = json!({"groups": max_seqs, "rows": 1});
    programs.insert("init".into(), json!({"once": true, "calls": init}));
    programs.insert("prefill".into(), json!({"calls": prefill}));
    let first = [&tail[..], &codec[..]].concat();
    let decode = [&advance[..], &tail[..], &codec[..]].concat();
    programs.insert("first".into(), json!({"batch": batch, "graph": true, "calls": first}));
    programs.insert("decode".into(), json!({"batch": batch, "graph": true, "calls": decode}));
    let s = g.finish(&mut programs);

    let (h, vocab) = (t.stack.hidden_size, t.stack.vocab_size);
    let p_vocab = t.code_predictor_config.vocab_size;
    let min = |lo: i64| json!({"min": lo});
    g.input("lines", json!([1, "seqs"]), json!({"index_into": "seq", "stride": s}));
    g.input("pos", json!(["seqs"]), min(0));
    g.input("force", json!(["seqs", GROUPS]), json!({"min": -1, "max": vocab - 1}));
    g.buffer("uniforms", "f32", json!(["seqs", GROUPS]), "input");
    g.input("t_ids", json!(["tokens"]), json!({"index_into": talker.text_embedding()}));
    g.input("t_codec", json!(["tokens"]), json!({"min": NO_CODEC, "max": vocab - 1}));
    g.input("t_pos", json!(["tokens"]), min(0));
    g.input("t_slot", json!(["tokens"]), json!({"index_into": "kv0"}));
    g.input("t_seq", json!(["tokens"]), min(0));
    g.input("last_row", json!(["seqs"]), min(0));
    g.input("kv_indptr", json!([max_seqs + 1]), json!({"min": 0, "monotone": true}));
    g.input("kv_pages", json!([page_table]), json!({"index_into": "kv0", "stride": PAGE}));
    g.buffer("codes", "i32", json!(["seqs", GROUPS]), "output");
    g.buffer("wav", "bf16", json!(["seqs", config.samples_per_frame]), "output");
    g.buffer("logits", "bf16", json!(["seqs", vocab]), "output");
    g.buffer("p_logits", "bf16", json!([(GROUPS - 1) * max_seqs, p_vocab]), "output");
    g.buffer("embeds", "bf16", json!(["tokens", h]), "output");
    g.buffer("hidden", "bf16", json!([max_seqs, h]), "carry");
    g.buffer("pad_embed", "bf16", json!([h]), "carry");
    for (name, w) in talker.token_workspaces() {
        g.buffer(name, "bf16", json!(["tokens", w]), "workspace");
    }
    let mut states = serde_json::Map::new();
    for i in 0..t.stack.num_hidden_layers {
        states.insert(format!("kv{i}"), json!({"bytes_per_token": Stack::kv_bytes(&t.stack)}));
    }
    states.insert("seq".into(), json!({"bytes_per_seq": s}));
    let (buffers, ops, tensors) = g.into_parts();
    let modules: serde_json::Map<String, Value> = cubins
        .iter()
        .map(|(n, sha, _)| ((*n).into(), json!({"source": format!("{n}-{}.cubin", &sha[..12]), "sha256": sha})))
        .collect();
    let manifest = json!({
        "schema_version": 5,
        "model": "qwen3-tts-12hz",
        "vars": {"seqs": {"max": max_seqs}, "tokens": {"max": max_tokens.max(max_seqs)}},
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    });
    Ok((manifest, tensors))
}
