//! The generation path as one kern manifest on one runtime: prompts in, delayed code rows out.
//!
//! Three programs over two vars, `tokens` (prompt rows of a call) and `seqs` (sequences of a call):
//!
//! - `prefill` (eager, ragged): new prompts into their KV, each one's last hidden state into `hidden`.
//! - `first` (graph): `hidden` → logits → the first row of codes.
//! - `decode` (graph): last row of codes → body → logits → the next row, one per sequence.
//!
//! A request is `prefill`, `first`, then `decode` until its [`Delay`] says it has ended. Graph calls
//! pad `seqs` up to a [`bucket`]; padding rows run on a lease of their own. A [`Seq`] is a lease:
//! the KV pages for its prompt and row cap, and a slot of the `seq` state holding its last row.

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
use omni_kern::buf;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::hex;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::inf;
use omni_kern::ini;
use omni_kern::ints;
use omni_kern::kernels_dir;
use omni_kern::outb;
use omni_kern::state_in;
use omni_kern::state_io;
use omni_kern::stride;
use omni_kern::vars;
use omni_kern::weights::File;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;

use crate::codec;
use crate::config;
use crate::config::CODEBOOKS;
use crate::config::SAMPLES_PER_FRAME;
use crate::config::VOCAB;
use crate::delay::BOC;
use crate::delay::Delay;
use crate::delay::Row;
use crate::prompt::Tokenizer;
use omni_qwen3::HEAD_DIM;
use omni_qwen3::Keep;
use omni_qwen3::Kv;
use omni_qwen3::PAGE;
use omni_qwen3::Pass;
use omni_qwen3::Scratch;
use omni_qwen3::Stack;

const HIGGS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/higgs.cubin"));
const CODEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/codec.cubin"));
const T: Scratch = Scratch { x: "t.x", res: "t.res", qkv: "t.qkv", attn: "t.attn", gate_up: "t.gate_up", act: "t.act" };
const SAMPLE_THREADS: u32 = 1024;
/// The fused head's width: a codebook's logits after another's.
const HEAD: usize = CODEBOOKS * VOCAB;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Sequences running at once.
    pub max_batch: usize,
    /// Prompt tokens per `prefill` call.
    pub max_tokens: usize,
    /// KV cache size.
    pub kv_gib: f64,
    pub sampling: Sampling,
}

/// How every codebook is drawn: a temperature of zero is the argmax, a top-k of zero keeps all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: i32,
}

pub struct Seq {
    lease: Lease,
    prompt: usize,
    rows: usize,
    delay: Delay,
}

/// A sequence's draws for one row: a uniform per codebook, or the row to take instead (teacher
/// forcing). The delay pattern's [`BOC`] codes are forced either way.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub uniforms: [f32; CODEBOOKS],
    pub force: Option<Row>,
}

struct Step<'a> {
    lease: &'a Lease,
    /// The position the sequence's last row goes in.
    position: usize,
    draw: Draw,
    force: Row,
}

/// A call's rows, one per sequence in order, and whether each sequence's run has ended.
#[derive(Debug)]
pub struct Out {
    pub rows: Vec<Row>,
    pub ended: Vec<bool>,
}

pub struct Model {
    pub tokenizer: Tokenizer,
    rt: Runtime,
    pad: Lease,
    limits: Limits,
    /// The most pages the running sequences may hold.
    pages_max: usize,
}

impl Model {
    pub fn load(device: usize, dir: &Path, limits: Limits) -> Result<Self> {
        let stack = config::load(dir)?;
        let tokenizer = Tokenizer::load(dir)?;
        let max_seqs = bucket(limits.max_batch);
        let kv_tokens = (limits.kv_gib * (1u64 << 30) as f64) as usize / kv_bytes_per_token(&stack);
        let pages_max = kv_tokens.div_ceil(PAGE) + 1;
        let cubins = [("codec", CODEC), ("higgs", HIGGS), ("qwen3", omni_qwen3::CUBIN)]
            .map(|(n, b)| (n, hex(&sha2::Sha256::digest(b)), b));
        // Each padding row of a graph call names the pad lease's page once more.
        let page_table = pages_max + max_seqs;
        let (manifest, tensors) = generate(dir, &stack, &cubins, limits, max_seqs, page_table)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("manifest: {e}"))?;
        let dir = kernels_dir(&cubins.each_ref().map(|(n, sha, b)| (*n, sha.as_str(), *b)))?;
        let capacity = Capacity { tokens: Some(kv_tokens as u64), seqs: limits.max_batch as u64 + 1 };
        let mut rt = Runtime::load(&verified, Some(&dir), device, Some(capacity), None)?;
        rt.load_weights(&tensors)?;
        let pad = rt.lease(1)?;
        Ok(Self { tokenizer, rt, pad, limits, pages_max })
    }

    /// A sequence of a `prompt`-token prompt and up to `rows` rows, or why not: `Busy` and
    /// `Remapping` pass, the rest never will.
    pub fn open(&mut self, prompt: usize, rows: usize) -> Result<Result<Seq, Denied>> {
        let lease = match self.rt.lease(prompt + rows) {
            Ok(l) => l,
            Err(kern_runtime::Error::Denied(d)) => return Ok(Err(d)),
            Err(e) => return Err(e.into()),
        };
        if self.rt.pages_used() > self.pages_max {
            let alone = lease.page_ids().len() + self.pad.page_ids().len();
            return Ok(Err(if alone > self.pages_max { Denied::ExceedsPool } else { Denied::Busy }));
        }
        Ok(Ok(Seq { lease, prompt, rows: 0, delay: Delay::default() }))
    }

    /// Prefills each sequence's prompt (`prompts[i]` token ids), then draws its first row.
    pub fn start(&mut self, rows: &mut [(&mut Seq, &[i32], Draw)]) -> Result<Out> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.limits.max_batch, "{n} sequences in one call");
        let tokens: usize = rows.iter().map(|(_, p, _)| p.len()).sum();
        ensure!(tokens <= self.limits.max_tokens, "{tokens} prompt tokens exceed {}", self.limits.max_tokens);
        let mut t = Rows::default();
        let mut last_row = Vec::with_capacity(n);
        for (s, (seq, prompt, _)) in rows.iter().enumerate() {
            ensure!(seq.rows == 0 && prompt.len() == seq.prompt, "sequence {s} is not fresh");
            t.ids.extend(*prompt);
            t.push_seq(&seq.lease, 0..prompt.len(), s);
            last_row.push(t.pos.len() as i32 - 1);
        }
        let v = vars(tokens, n);
        t.write(&mut self.rt, &v, true)?;
        self.rt.write_input_at("last_row", &ints(&last_row), &v)?;
        self.rt.issue("prefill", &v)?;
        let steps: Vec<Step> = rows.iter().map(|(s, _, d)| step(s, 0, *d)).collect();
        self.graph_inputs(&steps, false)?;
        let mut out = self.run("first", n)?;
        finish(rows.iter_mut().map(|(s, _, _)| &mut **s), &mut out);
        Ok(out)
    }

    pub fn step(&mut self, rows: &mut [(&mut Seq, Draw)]) -> Result<Out> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.limits.max_batch, "{n} sequences in one call");
        ensure!(rows.iter().all(|(s, _)| s.rows > 0), "a sequence that has not started");
        let steps: Vec<Step> = rows.iter().map(|(s, d)| step(s, s.prompt + s.rows - 1, *d)).collect();
        self.graph_inputs(&steps, true)?;
        let mut out = self.run("decode", n)?;
        finish(rows.iter_mut().map(|(s, _)| &mut **s), &mut out);
        Ok(out)
    }

    /// The per-sequence inputs of a graph call, padded to its bucket: slot lines, draws and, with
    /// `body`, each sequence's row at the position its last row goes in.
    fn graph_inputs(&mut self, steps: &[Step], body: bool) -> Result<()> {
        let b = bucket(steps.len());
        let draw = Draw { uniforms: [0.5; CODEBOOKS], force: None };
        let pad = Step { lease: &self.pad, position: 0, draw, force: [-1; CODEBOOKS] };
        let padded: Vec<&Step> = steps.iter().chain(std::iter::repeat(&pad)).take(b).collect();
        let lines = padded.iter().map(|f| f.lease.seq_line("lines", 0)).collect::<Result<Vec<_>, _>>()?;
        let uniforms: Vec<u8> = padded.iter().flat_map(|f| f.draw.uniforms).flat_map(f32::to_le_bytes).collect();
        let force: Vec<i32> = padded.iter().flat_map(|f| f.force).collect();
        let v = vars(b, b);
        let rt = &mut self.rt;
        rt.write_input_at("lines", &ints(&lines), &v)?;
        rt.write_input_at("uniforms", &uniforms, &v)?;
        rt.write_input_at("force", &ints(&force), &v)?;
        if body {
            let mut t = Rows::default();
            padded.iter().enumerate().for_each(|(s, f)| t.push_seq(f.lease, f.position..f.position + 1, s));
            t.write(rt, &v, false)?;
        }
        Ok(())
    }

    fn run(&mut self, program: &str, n: usize) -> Result<Out> {
        let b = bucket(n);
        self.rt.issue(program, &vars(b, b))?;
        let codes = self.rt.read_buffer_prefix("codes", n * CODEBOOKS * 4)?;
        let rows = codes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b))
            .collect::<Vec<_>>()
            .as_chunks::<CODEBOOKS>()
            .0
            .to_vec();
        Ok(Out { ended: vec![false; rows.len()], rows })
    }

    /// PCM for a window of frames (codec codes, at most [`codec::MAX_FRAMES`]), `SAMPLES_PER_FRAME`
    /// per frame. The edges see zeros where a longer window would see neighbours.
    pub fn decode(&mut self, frames: &[Row]) -> Result<Vec<f32>> {
        let n = frames.len();
        ensure!((1..=codec::MAX_FRAMES).contains(&n), "{n} frames in one window");
        ensure!(
            frames.iter().flatten().all(|&c| (0..codec::CODES as i32).contains(&c)),
            "a frame holds a code outside the codec's range"
        );
        let mut v = vars(1, 1);
        v.insert("frames".into(), n as u64);
        self.rt.write_input_at("frame_codes", &ints(&frames.concat()), &v)?;
        self.rt.issue("codec", &v)?;
        let pcm = self.rt.read_buffer_prefix("pcm", n * SAMPLES_PER_FRAME * 4)?;
        Ok(pcm.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
    }

    /// The last graph call's logits of sequence `s`, `[CODEBOOKS, VOCAB]`.
    pub fn logits(&self, s: usize) -> Result<Vec<f32>> {
        Ok(bf16s(&self.rt.read_buffer_prefix("logits", (s + 1) * HEAD * 2)?[s * HEAD * 2..]))
    }
}

/// A sequence's row of a graph call: the delay pattern's forced codes, or the caller's.
fn step(seq: &Seq, position: usize, draw: Draw) -> Step<'_> {
    let ramp = std::array::from_fn(|b| if b >= seq.delay.forced_from() { BOC } else { -1 });
    Step { lease: &seq.lease, position, draw, force: draw.force.unwrap_or(ramp) }
}

/// Counts a call's rows against each sequence's delay pattern.
fn finish<'a>(seqs: impl Iterator<Item = &'a mut Seq>, out: &mut Out) {
    for ((seq, row), ended) in seqs.zip(&out.rows).zip(&mut out.ended) {
        seq.rows += 1;
        *ended = seq.delay.advance(row[0]);
    }
}

/// The rows of a call and the KV they read.
#[derive(Default)]
struct Rows {
    ids: Vec<i32>,
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
        }
        rt.write_input_at("t_pos", &ints(&self.pos), v)?;
        rt.write_input_at("t_slot", &ints(&self.slot), v)?;
        rt.write_input_at("t_seq", &ints(&self.seq), v)?;
        rt.write_input_at("kv_indptr", &ints(&self.indptr), v)?;
        rt.write_input_at("kv_pages", &ints(&self.pages), v).context("kv_pages")?;
        Ok(())
    }
}

/// KV bytes per token: every layer's K and V.
fn kv_bytes_per_token(c: &config::Stack) -> usize {
    c.num_hidden_layers * Stack::kv_bytes(c)
}

/// The manifest (JSON) and the tensors its weight buffers bind.
fn generate(
    dir: &Path,
    c: &config::Stack,
    cubins: &[(&str, String, &[u8]); 3],
    limits: Limits,
    max_seqs: usize,
    page_table: usize,
) -> Result<(Value, HostTensors)> {
    let h = c.hidden_size;
    let file = File::open(&dir.join("model.safetensors"))?;
    let mut g = Gen::default();
    let stack = Stack::load(&mut g, &file, "body", "body", c)?;
    let text_vocab = c.vocab_size;
    let text = g.weight(
        "text_emb",
        &[text_vocab, h],
        &file.expect("tied.embedding.text_embedding.weight", &[text_vocab, h])?.data,
    );
    let fused_name = "tied.embedding.modality_embeddings.0.embedding.weight";
    let fused = g.weight("fused", &[HEAD, h], &file.expect(fused_name, &[HEAD, h])?.data);
    let state = g.region(4 * CODEBOOKS);

    let seqs = json!("seqs");
    let rows = json!("tokens");
    g.launch(
        "prefill.embed",
        "higgs_gather",
        [rows.clone(), json!(1), json!(1)],
        (h / 8) as u32,
        vec![ini("t_ids"), i32a(1), inb(&text), outb("embeds"), i32a(h)],
    );
    let pass = Pass {
        label: "prefill",
        rows: rows.clone(),
        input: "embeds",
        kv: Kv::Paged { prefix: "kv", ragged: true },
        keep: Keep { out: T.x, every: 1, which: 0 },
    };
    stack.forward(&mut g, pass, &T);
    g.launch(
        "prefill.last",
        "higgs_gather",
        [seqs.clone(), json!(1), json!(1)],
        (h / 8) as u32,
        vec![ini("last_row"), i32a(1), inb(T.x), outb("hidden"), i32a(h)],
    );
    let prefill = g.take();

    g.launch(
        "decode.embed",
        "higgs_frame_embed",
        [seqs.clone(), json!(1), json!(1)],
        (h / 8) as u32,
        vec![state_in(state), ini("lines"), stride(), inb(&fused), outb("t.e"), i32a(h), i32a(VOCAB)],
    );
    let pass = Pass {
        label: "decode",
        rows: seqs.clone(),
        input: "t.e",
        kv: Kv::Paged { prefix: "kv", ragged: false },
        keep: Keep { out: "hidden", every: 1, which: 0 },
    };
    stack.forward(&mut g, pass, &T);
    let advance = g.take();

    g.gemm_rows("head", (buf("logits"), buf("hidden"), buf(&fused)), count(&seqs), (HEAD, h));
    let Sampling { temperature, top_k } = limits.sampling;
    g.launch(
        "sample",
        "higgs_sample",
        [seqs.clone(), json!(CODEBOOKS), json!(1)],
        SAMPLE_THREADS,
        vec![
            ("in buffer<bf16>", buf("logits")),
            i32a(VOCAB),
            f32a(temperature),
            i32a(top_k),
            inf("uniforms"),
            ini("force"),
            state_io(state),
            ini("lines"),
            stride(),
            ("out buffer<i32>", buf("codes")),
        ],
    );
    let tail = g.take();
    codec::build(&mut g, &file)?;
    let decoding = g.take();

    let mut programs = serde_json::Map::new();
    let batch = json!({"groups": max_seqs, "rows": 1});
    programs.insert("prefill".into(), json!({"calls": prefill}));
    programs.insert("codec".into(), json!({"calls": decoding}));
    let decode = [&advance[..], &tail[..]].concat();
    programs.insert("first".into(), json!({"batch": batch, "graph": true, "calls": tail}));
    programs.insert("decode".into(), json!({"batch": batch, "graph": true, "calls": decode}));
    let s = g.finish(&mut programs);

    let min = |lo: i64| json!({"min": lo});
    g.input("lines", json!([1, "seqs"]), json!({"index_into": "seq", "stride": s}));
    g.input("force", json!(["seqs", CODEBOOKS]), json!({"min": -1, "max": VOCAB - 1}));
    g.buffer("uniforms", "f32", json!(["seqs", CODEBOOKS]), "input");
    g.input("t_ids", json!(["tokens"]), json!({"index_into": &text}));
    g.input("t_pos", json!(["tokens"]), min(0));
    g.input("t_slot", json!(["tokens"]), json!({"index_into": "kv0"}));
    g.input("t_seq", json!(["tokens"]), min(0));
    g.input("last_row", json!(["seqs"]), min(0));
    g.input("kv_indptr", json!([max_seqs + 1]), json!({"min": 0, "monotone": true}));
    g.input("kv_pages", json!([page_table]), json!({"index_into": "kv0", "stride": PAGE}));
    g.input("frame_codes", json!(["frames", CODEBOOKS]), json!({"min": 0, "max": codec::CODES - 1}));
    g.buffer("pcm", "f32", json!([codec::MAX_FRAMES * SAMPLES_PER_FRAME]), "output");
    for (name, elements) in codec::workspaces() {
        g.buffer(name, "bf16", json!([elements]), "workspace");
    }
    g.buffer("codes", "i32", json!(["seqs", CODEBOOKS]), "output");
    g.buffer("logits", "bf16", json!(["seqs", HEAD]), "output");
    g.buffer("embeds", "bf16", json!(["tokens", h]), "output");
    g.buffer("hidden", "bf16", json!([max_seqs, h]), "carry");
    g.buffer("t.e", "bf16", json!(["tokens", h]), "workspace");
    for (name, w) in [
        (T.x, h),
        (T.res, h),
        (T.qkv, stack.qkv_width()),
        (T.attn, c.num_attention_heads * HEAD_DIM),
        (T.gate_up, 2 * c.intermediate_size),
        (T.act, c.intermediate_size),
    ] {
        g.buffer(name, "bf16", json!(["tokens", w]), "workspace");
    }
    let mut states = serde_json::Map::new();
    for i in 0..c.num_hidden_layers {
        states.insert(format!("kv{i}"), json!({"bytes_per_token": Stack::kv_bytes(c)}));
    }
    states.insert("seq".into(), json!({"bytes_per_seq": s}));
    let (buffers, ops, tensors) = g.into_parts();
    let modules: serde_json::Map<String, Value> = cubins
        .iter()
        .map(|(n, sha, _)| ((*n).into(), json!({"source": format!("{n}-{}.cubin", &sha[..12]), "sha256": sha})))
        .collect();
    let manifest = json!({
        "schema_version": 5,
        "model": "higgs-tts-3",
        "vars": {
            "seqs": {"max": max_seqs},
            "tokens": {"max": limits.max_tokens.max(max_seqs)},
            "frames": {"max": codec::MAX_FRAMES},
        },
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    });
    Ok((manifest, tensors))
}
