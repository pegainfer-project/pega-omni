//! The whole model as one kern manifest on one runtime: prompts in, a frame
//! of audio in and out per session per tick.
//!
//! Two programs over two vars, `tokens` (prompt rows of a call) and `seqs`
//! (sessions of a call):
//!
//! - `prefill` (eager, ragged): new sessions' prompts into their KV rings,
//!   and their token state set to where the prompt leaves it.
//! - `tick` (graph): per session, Mimi encodes the caller's frame, Helium
//!   and the depformer draw the agent's text token and codebooks, the token
//!   state advances, and Mimi decodes the agent's frame.
//!
//! Graph calls pad `seqs` up to a [`bucket`]; padding rows run on a lease of
//! their own. A [`Session`] is a lease: [`CONTEXT`] KV positions (a ring,
//! position `p` in slot `p % CONTEXT`, so its memory never grows) and a slot
//! of the `seq` state holding its token state and Mimi's streaming state.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
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

use crate::config::CARD;
use crate::config::CODEBOOKS;
use crate::config::CONTEXT;
use crate::config::DIM;
use crate::config::DRAWN;
use crate::config::FRAME;
use crate::config::LAYERS;
use crate::config::MIMI_DIM;
use crate::config::SILENCE_STEPS;
use crate::config::STREAMS;
use crate::config::Sampling;
use crate::config::TEXT_VOCAB;
use crate::depformer::Depformer;
use crate::helium;
use crate::helium::Helium;
use crate::helium::KV_BYTES;
use crate::helium::PAGE;
use crate::mimi;
use crate::prompt;
use crate::prompt::Prompt;
use crate::tokenizer::Tokenizer;
use crate::voice::Voice;

const LM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lm.cubin"));
const MIMI: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mimi.cubin"));

const INIT: usize = STREAMS + CODEBOOKS - 1;
const PAGES: usize = CONTEXT.div_ceil(PAGE);

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Sessions running at once.
    pub max_sessions: usize,
    /// Prompt rows per `prefill` call.
    pub max_prefill: usize,
}

/// One session's KV ring and state.
pub struct Session {
    lease: Lease,
    /// Helium position of the next step.
    pos: usize,
    frame: u64,
    first_force: [i32; DRAWN],
}

impl Session {
    /// Frames run so far.
    pub fn frame(&self) -> u64 {
        self.frame
    }
}

/// A session's inputs to one tick.
#[derive(Clone, Debug)]
pub struct Step<'a> {
    /// The caller's frame, `FRAME` samples in [-1, 1].
    pub pcm: &'a [f32],
    /// Tokens to take instead of drawing (text, then the agent's codebooks); -1 draws.
    pub force: [i32; DRAWN],
    /// The caller's codes to hear instead of the encoder's; -1 hears the encoder.
    pub force_caller: [i32; CODEBOOKS],
    pub seed: u64,
}

/// A tick's results, one entry per session in call order.
#[derive(Debug)]
pub struct Out {
    /// The text token spoken now, and the agent frame decoded into `pcm`.
    pub emitted: Vec<[i32; DRAWN]>,
    /// `FRAME` samples per session.
    pub pcm: Vec<f32>,
}

/// A prompt this model can prefill: [`Prompt`] and where its voice's embeddings are.
pub struct Prefix {
    prompt: Prompt,
    voice_row: usize,
}

impl Prefix {
    /// Helium positions the prefix fills.
    pub fn len(&self) -> usize {
        self.prompt.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompt.is_empty()
    }
}

pub struct Model {
    pub tokenizer: Tokenizer,
    rt: Runtime,
    pad: Lease,
    limits: Limits,
    max_seqs: usize,
    /// Each voice's first row in the `voices` weight, and the voice.
    voices: BTreeMap<String, (usize, Voice)>,
}

fn rows_of<const N: usize>(bytes: &[u8]) -> Vec<[i32; N]> {
    bytes.as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect::<Vec<_>>().as_chunks::<N>().0.to_vec()
}

/// Characters of role prompt that always fit a prefix of `rows` rows after a
/// voice of `voice_steps`: a character is at most four byte-fallback tokens,
/// and the tokenizer prepends one `▁`.
pub fn max_instructions_chars(rows: usize, voice_steps: usize) -> usize {
    let text = rows.saturating_sub(voice_steps + 2 * SILENCE_STEPS);
    (text.saturating_sub(1) / 4).saturating_sub(prompt::wrap("").chars().count())
}

impl Model {
    /// Loads the checkpoint at `dir` (`model.safetensors`, Mimi, the tokenizer, `voices/`) onto `device`.
    pub fn load(device: usize, dir: &Path, limits: Limits, sampling: Sampling) -> Result<Self> {
        let tokenizer = Tokenizer::load(&dir.join("tokenizer_spm_32k_3.model"))?;
        let voices = load_voices(&dir.join("voices"))?;
        let max_seqs = bucket(limits.max_sessions);
        let cubins = [("lm", LM), ("mimi", MIMI)].map(|(n, b)| (n, hex(&sha2::Sha256::digest(b)), b));
        let (manifest, tensors) = generate(dir, &voices, &cubins, max_seqs, limits.max_prefill, sampling)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("manifest: {e}"))?;
        let kernels = kernels_dir(&cubins.each_ref().map(|(n, sha, b)| (*n, sha.as_str(), *b)))?;
        let tokens = ((limits.max_sessions * PAGES + 1) * PAGE) as u64;
        let capacity = Capacity { tokens: Some(tokens), seqs: limits.max_sessions as u64 + 1 };
        let mut rt = Runtime::load(&verified, Some(&kernels), device, Some(capacity), None)?;
        rt.load_weights(&tensors)?;
        let pad = rt.lease(1)?;
        Ok(Self { tokenizer, rt, pad, limits, max_seqs, voices })
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    /// The longest role prompt, in characters, whose prefix always fits.
    pub fn max_instructions_chars(&self) -> usize {
        let longest = self.voices.values().map(|(_, v)| v.steps()).max().unwrap_or(0);
        max_instructions_chars(self.limits.max_prefill.min(CONTEXT - 1), longest)
    }

    /// The prefix of `voice` with the role prompt `instructions`.
    pub fn prompt(&self, voice: &str, instructions: &str) -> Result<Prefix> {
        let (voice_row, v) = self.voices.get(voice).with_context(|| format!("unknown voice `{voice}`"))?;
        let ids = self.tokenizer.encode(&prompt::wrap(instructions))?;
        let prompt = prompt::build(v, &ids)?;
        ensure!(prompt.len() < CONTEXT, "a prompt of {} steps does not fit the {CONTEXT}-step context", prompt.len());
        ensure!(
            prompt.len() <= self.limits.max_prefill,
            "a prompt of {} steps exceeds {} per prefill",
            prompt.len(),
            self.limits.max_prefill
        );
        Ok(Prefix { prompt, voice_row: *voice_row })
    }

    /// A fresh session's lease.
    pub fn open(&mut self) -> Result<Session> {
        let lease = self.rt.lease(CONTEXT)?;
        Ok(Session { lease, pos: 0, frame: 0, first_force: [-1; DRAWN] })
    }

    /// Prefills each fresh session's prefix; the sessions' first tick follows.
    pub fn start(&mut self, rows: &mut [(&mut Session, &Prefix)]) -> Result<()> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.max_seqs, "{n} sessions in one prefill");
        let tokens: usize = rows.iter().map(|(_, p)| p.len()).sum();
        ensure!(tokens <= self.limits.max_prefill, "{tokens} prompt rows exceed {}", self.limits.max_prefill);
        let mut t = Rows::default();
        let (mut ids, mut voice, mut init, mut lines) = (vec![], vec![], vec![], vec![]);
        for (s, (session, prefix)) in rows.iter_mut().enumerate() {
            ensure!(session.pos == 0, "session {s} was started before");
            let (p, first) = (&prefix.prompt, prefix.voice_row);
            voice.extend((0..p.voice_steps).map(|i| (first + i) as i32));
            voice.extend(std::iter::repeat_n(-1, p.rows.len()));
            ids.extend(std::iter::repeat_n(-1, p.voice_steps * STREAMS));
            ids.extend(p.rows.iter().flatten());
            init.extend(p.first.iter().chain(&p.pending));
            t.push(&session.lease, 0..p.len(), s);
            lines.push(session.lease.seq_line("lines", 0)?);
            session.pos = p.len();
            session.first_force = p.first_force;
        }
        let v = vars(tokens, n);
        let rt = &mut self.rt;
        rt.write_input_at("p_ids", &ints(&ids), &v)?;
        rt.write_input_at("p_voice", &ints(&voice), &v)?;
        rt.write_input_at("init", &ints(&init), &v)?;
        rt.write_input_at("lines", &ints(&lines), &v)?;
        rt.write_input_at("t_pos", &ints(&t.pos), &v)?;
        rt.write_input_at("t_slot", &ints(&t.slot), &v)?;
        rt.write_input_at("t_seq", &ints(&t.seq), &v)?;
        t.write_pages(rt, &v)?;
        rt.issue("prefill", &v)?;
        Ok(())
    }

    /// Advances every session by one frame.
    pub fn tick(&mut self, rows: &mut [(&mut Session, Step)]) -> Result<Out> {
        let n = rows.len();
        ensure!(n > 0 && n <= self.limits.max_sessions, "{n} sessions in one tick");
        ensure!(rows.iter().all(|(s, _)| s.pos > 0), "a session that was not started");
        ensure!(rows.iter().all(|(_, x)| x.pcm.len() == FRAME), "a caller frame is not {FRAME} samples");
        let b = bucket(n);
        let silence = [0.0; FRAME];
        let pad = Step { pcm: &silence, force: [-1; DRAWN], force_caller: [-1; CODEBOOKS], seed: 0 };
        let padded: Vec<(&Lease, usize, u64, Step)> = rows
            .iter()
            .map(|(s, x)| {
                let mut x = x.clone();
                if s.frame == 0 {
                    x.force = std::array::from_fn(|k| if x.force[k] >= 0 { x.force[k] } else { s.first_force[k] });
                }
                (&s.lease, s.pos, s.frame, x)
            })
            .chain(std::iter::repeat_with(|| (&self.pad, 0, 0, pad.clone())))
            .take(b)
            .collect();
        let mut t = Rows::default();
        padded.iter().enumerate().for_each(|(s, (lease, pos, _, _))| t.push(lease, *pos..*pos + 1, s));
        let lines = padded.iter().map(|(l, ..)| l.seq_line("lines", 0)).collect::<Result<Vec<_>, _>>()?;
        let frame: Vec<i32> = padded.iter().map(|(_, _, f, _)| *f as i32).collect();
        let pcm: Vec<u8> = padded.iter().flat_map(|(.., x)| x.pcm.iter().flat_map(|v| v.to_le_bytes())).collect();
        let force: Vec<i32> = padded.iter().flat_map(|(.., x)| x.force).collect();
        let force_caller: Vec<i32> = padded.iter().flat_map(|(.., x)| x.force_caller).collect();
        let seeds: Vec<i32> = padded.iter().flat_map(|(.., x)| [x.seed as i32, (x.seed >> 32) as i32]).collect();
        let v = vars(b, b);
        let rt = &mut self.rt;
        rt.write_input_at("lines", &ints(&lines), &v)?;
        rt.write_input_at("pos", &ints(&t.pos), &v)?;
        rt.write_input_at("slot", &ints(&t.slot), &v)?;
        rt.write_input_at("frame", &ints(&frame), &v)?;
        rt.write_input_at("caller", &pcm, &v)?;
        rt.write_input_at("force", &ints(&force), &v)?;
        rt.write_input_at("force_caller", &ints(&force_caller), &v)?;
        rt.write_input_at("seeds", &ints(&seeds), &v)?;
        t.write_pages(rt, &v)?;
        rt.issue("tick", &v)?;
        rows.iter_mut().for_each(|(s, _)| {
            s.pos += 1;
            s.frame += 1;
        });
        let read = |name: &str, width: usize| rt.read_buffer_prefix(name, n * width * 4);
        Ok(Out {
            emitted: rows_of(&read("emitted", DRAWN)?),
            pcm: read("pcm", FRAME)?.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        })
    }

    /// The input row Helium read in the last tick for session `s`.
    pub fn input_row(&self, s: usize) -> Result<[i32; STREAMS]> {
        Ok(rows_of(&self.rt.read_buffer_prefix("rows", (s + 1) * STREAMS * 4)?)[s])
    }

    /// The encoder's codes of session `s`'s caller frame in the last tick.
    pub fn caller_codes(&self, s: usize) -> Result<[i32; CODEBOOKS]> {
        Ok(rows_of(&self.rt.read_buffer_prefix("caller_codes", (s + 1) * CODEBOOKS * 4)?)[s])
    }

    /// The last tick's text logits of session `s`.
    pub fn text_logits(&self, s: usize) -> Result<Vec<f32>> {
        Ok(bf16s(&self.rt.read_buffer_prefix("text_logits", (s + 1) * TEXT_VOCAB * 2)?[s * TEXT_VOCAB * 2..]))
    }

    /// The last tick's unquantized encoding of session `s`'s caller frame.
    pub fn caller_latent(&self, s: usize) -> Result<Vec<f32>> {
        Ok(bf16s(&self.rt.read_buffer_prefix("caller_latent", (s + 1) * MIMI_DIM * 2)?[s * MIMI_DIM * 2..]))
    }

    /// The last tick's logits of the agent's codebooks of session `s`, `[8, 2048]`.
    pub fn audio_logits(&self, s: usize) -> Result<Vec<f32>> {
        let all = self.rt.read_output("audio_logits")?;
        Ok((0..CODEBOOKS)
            .flat_map(|k| {
                let at = (k * self.max_seqs + s) * CARD * 2;
                bf16s(&all[at..at + CARD * 2])
            })
            .collect())
    }
}

/// The Helium rows of a call and the KV they read.
#[derive(Default)]
struct Rows {
    pos: Vec<i32>,
    slot: Vec<i32>,
    seq: Vec<i32>,
    indptr: Vec<i32>,
    pages: Vec<i32>,
}

impl Rows {
    fn push(&mut self, lease: &Lease, positions: std::ops::Range<usize>, s: usize) {
        if self.indptr.is_empty() {
            self.indptr.push(0);
        }
        for p in positions {
            self.pos.push(p as i32);
            self.slot.push(lease.slot(p % CONTEXT) as i32);
            self.seq.push(s as i32);
        }
        self.pages.extend(lease.page_ids());
        self.indptr.push(self.pages.len() as i32);
    }

    fn write_pages(&self, rt: &mut Runtime, v: &BTreeMap<String, u64>) -> Result<()> {
        rt.write_input_at("kv_indptr", &ints(&self.indptr), v)?;
        rt.write_input_at("kv_pages", &ints(&self.pages), v).context("kv_pages")?;
        Ok(())
    }
}

fn load_voices(dir: &Path) -> Result<BTreeMap<String, (usize, Voice)>> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "pt"))
        .collect();
    names.sort();
    let mut first = 0;
    names
        .into_iter()
        .map(|path| {
            let voice = Voice::load(&path)?;
            let name = path.file_stem().context("a voice file name")?.to_string_lossy().into_owned();
            let at = first;
            first += voice.steps();
            Ok((name, (at, voice)))
        })
        .collect()
}

/// The manifest (JSON) and the tensors its weight buffers bind.
fn generate(
    dir: &Path,
    voices: &BTreeMap<String, (usize, Voice)>,
    cubins: &[(&str, String, &[u8]); 2],
    max_seqs: usize,
    max_prefill: usize,
    sampling: Sampling,
) -> Result<(Value, HostTensors)> {
    let mut g = Gen::default();
    let lm_file = File::open(&dir.join("model.safetensors"))?;
    let helium = Helium::load(&mut g, &lm_file, sampling)?;
    let depformer = Depformer::load(&mut g, &lm_file, sampling)?;
    drop(lm_file);
    let voice_rows: usize = voices.values().map(|(_, v)| v.steps()).sum();
    let voice_data: Vec<f32> = voices.values().flat_map(|(_, v)| bf16s(&v.embeddings)).collect();
    g.weight("voices", &[voice_rows, DIM], &voice_data);

    helium.prefill(&mut g);
    let prefill = g.take();
    let mimi_file = File::open(&dir.join("tokenizer-e351c8d8-checkpoint125.safetensors"))?;
    mimi::encode(&mut g, &mimi_file)?;
    helium.step(&mut g);
    depformer.step(&mut g, max_seqs);
    g.launch(
        "advance",
        "lm_advance",
        [json!({"ceil_div": ["seqs", 128]}), json!(1), json!(1)],
        128,
        vec![
            ("inout state", json!({"state": "seq", "offset": helium.state})),
            omni_kern::ini("lines"),
            omni_kern::stride(),
            omni_kern::ini("drawn"),
            omni_kern::ini("caller_codes"),
            omni_kern::ini("force_caller"),
            ("out buffer<i32>", omni_kern::buf("emitted")),
            ("i32", json!({"var": "seqs"})),
        ],
    );
    mimi::decode(&mut g, &mimi_file)?;
    let tick = g.take();

    let mut programs = serde_json::Map::new();
    programs.insert("prefill".into(), json!({"calls": prefill}));
    programs.insert("tick".into(), json!({"batch": {"groups": max_seqs, "rows": 1}, "graph": true, "calls": tick}));
    let s = g.finish(&mut programs);

    let min = |lo: i64| json!({"min": lo});
    let page_table = max_seqs * PAGES;
    g.input("lines", json!([1, "seqs"]), json!({"index_into": "seq", "stride": s}));
    g.input("pos", json!(["seqs"]), min(0));
    g.input("slot", json!(["seqs"]), json!({"index_into": "kv0"}));
    g.input("frame", json!(["seqs"]), min(0));
    g.input("force", json!(["seqs", DRAWN]), json!({"min": -1, "max": TEXT_VOCAB - 1}));
    g.input("force_caller", json!(["seqs", CODEBOOKS]), json!({"min": -1, "max": CARD - 1}));
    g.input("seeds", json!(["seqs", 2]), json!({"min": i32::MIN}));
    g.buffer("caller", "f32", json!(["seqs", FRAME]), "input");
    g.input("p_ids", json!(["tokens", STREAMS]), json!({"min": -1, "max": TEXT_VOCAB}));
    g.input("p_voice", json!(["tokens"]), json!({"min": -1, "max": voice_rows - 1}));
    g.input("t_pos", json!(["tokens"]), min(0));
    g.input("t_slot", json!(["tokens"]), json!({"index_into": "kv0"}));
    g.input("t_seq", json!(["tokens"]), min(0));
    g.input("init", json!(["seqs", INIT]), json!({"min": 0, "max": TEXT_VOCAB}));
    g.input("kv_indptr", json!([max_seqs + 1]), json!({"min": 0, "monotone": true}));
    g.input("kv_pages", json!([page_table]), json!({"index_into": "kv0", "stride": PAGE}));
    g.buffer("drawn", "i32", json!(["seqs", DRAWN]), "output");
    g.buffer("emitted", "i32", json!(["seqs", DRAWN]), "output");
    g.buffer("rows", "i32", json!(["seqs", STREAMS]), "output");
    g.buffer("caller_codes", "i32", json!(["seqs", CODEBOOKS]), "output");
    g.buffer("pcm", "f32", json!(["seqs", FRAME]), "output");
    g.buffer("caller_latent", "bf16", json!(["seqs", MIMI_DIM]), "output");
    g.buffer("text_logits", "bf16", json!(["seqs", TEXT_VOCAB]), "output");
    g.buffer("audio_logits", "bf16", json!([CODEBOOKS * max_seqs, CARD]), "output");
    for (name, w) in helium::SCRATCH {
        g.buffer(name, "bf16", json!(["tokens", w]), "workspace");
    }
    let mut states = serde_json::Map::new();
    for i in 0..LAYERS {
        states.insert(format!("kv{i}"), json!({"bytes_per_token": KV_BYTES}));
    }
    states.insert("seq".into(), json!({"bytes_per_seq": s}));
    let (buffers, ops, tensors) = g.into_parts();
    let modules: serde_json::Map<String, Value> = cubins
        .iter()
        .map(|(n, sha, _)| ((*n).into(), json!({"source": format!("{n}-{}.cubin", &sha[..12]), "sha256": sha})))
        .collect();
    let manifest = json!({
        "schema_version": 5,
        "model": "personaplex-7b",
        "vars": {"seqs": {"max": max_seqs}, "tokens": {"max": max_prefill.max(max_seqs)}},
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    });
    Ok((manifest, tensors))
}
