//! Serving: requests in through an [`Inbox`], PCM chunks out through each
//! request's sink.
//!
//! One engine thread runs steps back to back. A step admits waiting requests
//! (FIFO, while batch slots, the per-step token budget and KV pages allow),
//! advances every running request by one frame through the [`Talker`], decodes
//! every new frame to audio in one batched [`Codec`] call, and sends whichever
//! requests have a chunk due. The codec streams (each request keeps its
//! decoder state), so a chunk's size decides only when audio leaves, not what
//! it costs. Serial on purpose; overlapping talker and codec is a measured
//! change for later.
//!
//! Admission reserves a request's whole KV and its codec state up front, the
//! KV sized by its frame cap (proportional to the input length), so a running
//! request never waits for memory and nothing is ever preempted.
//!
//! The decisions are pure functions ([`chunk_due`], [`frame_cap`],
//! [`pages_for`]); [`Engine`] is the shell around them.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;

use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use omni_cuda::Gpu;
use omni_engine::Done;
use omni_engine::EngineInfo;
use omni_engine::Event;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::Handle;
use omni_engine::Inbox;
use omni_engine::Speech;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

use crate::codec::Codec;
use crate::codec::Stream;
use crate::config::Config;
use crate::prompt;
use crate::prompt::Prompt;
use crate::prompt::Tokenizer;
use crate::stack::KvPool;
use crate::talker::Frame;
use crate::talker::GROUPS;
use crate::talker::Input;
use crate::talker::Limits;
use crate::talker::Row;
use crate::talker::Talker;
use crate::weights::File;

#[derive(Clone, Debug)]
pub struct Options {
    pub max_batch: usize,
    /// Talker tokens per step, prompts included.
    pub max_step_tokens: usize,
    pub kv_gib: f64,
    pub page_size: usize,
    pub first_chunk_frames: usize,
    pub chunk_frames: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_batch: 64,
            max_step_tokens: 8192,
            kv_gib: 16.0,
            page_size: 16,
            first_chunk_frames: 2,
            chunk_frames: 8,
        }
    }
}

/// Frames to decode now for a request with `generated` frames of which
/// `emitted` are already audio: the first chunk once `first` frames exist,
/// then every `steady`, and the remainder when it has `finished`.
pub fn chunk_due(generated: usize, emitted: usize, finished: bool, first: usize, steady: usize) -> Option<usize> {
    let pending = generated - emitted;
    let target = if emitted == 0 { first } else { steady };
    (pending > 0 && (pending >= target || finished)).then_some(pending.min(target))
}

/// The most frames a request may generate: generous for slow speech (Chinese
/// runs ~3 frames per character, English under 1), capped by the model.
pub fn frame_cap(input_chars: usize, model_max: usize) -> usize {
    (50 + 6 * input_chars).min(model_max)
}

pub fn pages_for(tokens: usize, page_size: usize) -> usize {
    tokens.div_ceil(page_size)
}

/// Everything loaded from a checkpoint directory.
pub struct Model {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub talker: Talker,
    pub codec: Codec,
}

impl Model {
    pub fn load(gpu: &Gpu, dir: &Path, opts: &Options) -> Result<Self> {
        let config = Config::load(dir)?;
        let tokenizer = Tokenizer::load(dir)?;
        let t = &config.model.talker_config;
        let page_bytes = KvPool::page_bytes(&t.stack, opts.page_size);
        let limits = Limits {
            max_batch: opts.max_batch,
            max_tokens: opts.max_step_tokens,
            pages: (opts.kv_gib * (1u64 << 30) as f64) as usize / page_bytes,
            page_size: opts.page_size,
        };
        let weights = File::open(&dir.join("model.safetensors"))?;
        let talker = Talker::load(gpu, &weights, &config.model, &config.generation, limits)?;
        let codec = Codec::load(
            gpu.ordinal(),
            &File::open(&dir.join("speech_tokenizer/model.safetensors"))?,
            &config.codec,
            config.samples_per_frame,
            opts.max_batch,
        )?;
        Ok(Self { config, tokenizer, talker, codec })
    }

    /// The front end's view: CustomVoice speakers, languages, and a sampling seed.
    pub fn info(&self, name: &str, max_input_chars: usize) -> EngineInfo {
        let t = &self.config.model.talker_config;
        let languages = std::iter::once("auto".to_string()).chain(t.codec_language_id.keys().cloned()).collect();
        EngineInfo {
            model: name.into(),
            sample_rate: self.config.sample_rate,
            voices: t.spk_id.keys().cloned().collect(),
            extra: BTreeMap::from([
                ("language".to_string(), Extra::OneOf(languages)),
                ("seed".to_string(), Extra::Integer(0..=i64::MAX)),
            ]),
            speeds: 1.0..=1.0,
            max_input_chars,
        }
    }
}

struct Job {
    id: u64,
    sink: UnboundedSender<Event>,
    prompt: Prompt,
    pages: Vec<i32>,
    cached: usize,
    codec: Option<Stream>,
    last: Option<[i32; GROUPS]>,
    frames: usize,
    /// Decoded audio not yet sent, s16le.
    pcm: Vec<u8>,
    emitted: usize,
    seen: Vec<u32>,
    rng: StdRng,
    cap: usize,
    ended: bool,
}

impl Job {
    fn prefilled(&self) -> bool {
        self.cached > 0
    }

    fn done(&self) -> bool {
        self.ended || self.frames >= self.cap
    }
}

pub struct Engine {
    gpu: Gpu,
    model: Model,
    opts: Options,
    free: Vec<i32>,
    waiting: VecDeque<Job>,
    running: Vec<Job>,
}

impl Engine {
    pub fn new(gpu: Gpu, model: Model, opts: Options) -> Self {
        let free = (0..model.talker.limits.pages as i32).rev().collect();
        Self { gpu, model, opts, free, waiting: VecDeque::new(), running: Vec::new() }
    }

    fn job(&self, speech: &Speech, sink: UnboundedSender<Event>) -> Result<Job, String> {
        let m = &self.model;
        let language = speech.extra.get("language").and_then(|v| v.as_str()).unwrap_or("auto");
        let voice = prompt::voice(&m.config.model, &speech.voice, language)?;
        let instruct = speech.instructions.as_deref().filter(|s| !s.is_empty()).map(|s| m.tokenizer.instruct(s));
        let prompt =
            prompt::assemble(&m.config.model, voice, &m.tokenizer.assistant(&speech.input), instruct.as_deref());
        let seed = speech.extra.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(rand::random);
        Ok(Job {
            id: speech.id,
            sink,
            prompt,
            pages: vec![],
            cached: 0,
            codec: None,
            last: None,
            frames: 0,
            pcm: vec![],
            emitted: 0,
            seen: vec![0; m.talker.seen_words()],
            rng: StdRng::seed_from_u64(seed),
            cap: frame_cap(speech.input.chars().count(), m.config.generation.max_new_tokens),
            ended: false,
        })
    }

    fn take(&mut self, speech: Speech, sink: UnboundedSender<Event>) {
        match self.job(&speech, sink.clone()) {
            Ok(job) => self.waiting.push_back(job),
            Err(e) => {
                tracing::warn!(id = speech.id, "rejected: {e}");
                let _ = sink.send(Event::Done(Done { finish: Finish::Aborted, input_units: 0, frames: 0 }));
            }
        }
    }

    fn admit(&mut self) -> Result<()> {
        let mut tokens: usize = self.running.len();
        while let Some(job) = self.waiting.front() {
            let need = pages_for(job.prompt.len() + job.cap, self.opts.page_size);
            let fits = self.running.len() < self.opts.max_batch
                && tokens + job.prompt.len() <= self.opts.max_step_tokens
                && need <= self.free.len();
            if !fits {
                break;
            }
            let mut job = self.waiting.pop_front().expect("front exists");
            tokens += job.prompt.len();
            job.pages = self.free.split_off(self.free.len() - need);
            job.codec = Some(self.model.codec.open()?);
            self.running.push(job);
        }
        Ok(())
    }

    /// One step: admit, one frame for every running request, due chunks out.
    pub fn step(&mut self) -> Result<()> {
        self.admit()?;
        if self.running.is_empty() {
            return Ok(());
        }
        let order: Vec<usize> = (0..self.running.len())
            .filter(|&i| !self.running[i].prefilled())
            .chain((0..self.running.len()).filter(|&i| self.running[i].prefilled()))
            .collect();
        let uniforms: Vec<[f32; GROUPS]> =
            order.iter().map(|&i| std::array::from_fn(|_| self.running[i].rng.random::<f32>())).collect();
        let rows: Vec<Row> = order
            .iter()
            .zip(&uniforms)
            .map(|(&i, &u)| {
                let j = &self.running[i];
                Row {
                    pages: &j.pages,
                    cached: j.cached,
                    input: match &j.last {
                        Some(f) if j.prefilled() => Input::Frame(f),
                        _ => Input::Prompt(&j.prompt),
                    },
                    seen: &j.seen,
                    generated: j.frames,
                    uniforms: u,
                }
            })
            .collect();
        let frames = self.model.talker.step(&self.gpu, &rows, None)?;

        let mut fresh = Vec::with_capacity(order.len());

        for (&i, frame) in order.iter().zip(frames) {
            let j = &mut self.running[i];
            j.cached += if j.prefilled() { 1 } else { j.prompt.len() };
            match frame {
                Frame::End => j.ended = true,
                Frame::Codes(c) => {
                    j.seen[c[0] as usize / 32] |= 1 << (c[0] % 32);
                    j.frames += 1;
                    j.last = Some(c);
                    fresh.push(i);
                }
            }
        }
        fresh.sort_unstable();
        self.decode(&fresh)?;
        self.emit()
    }

    /// The frame each of `fresh` (ascending) produced this step, decoded in one call.
    fn decode(&mut self, fresh: &[usize]) -> Result<()> {
        let mut batch: Vec<(&mut Stream, [i32; GROUPS])> = self
            .running
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| fresh.binary_search(i).is_ok())
            .map(|(_, j)| (j.codec.as_mut().expect("admitted"), j.last.expect("a frame")))
            .collect();
        let wav = self.model.codec.decode(&mut batch)?;
        let spf = self.model.config.samples_per_frame;
        for (i, samples) in fresh.iter().zip(wav.chunks(spf)) {
            let pcm = samples.iter().flat_map(|&x| ((x * 32767.0).round() as i16).to_le_bytes());
            self.running[*i].pcm.extend(pcm);
        }
        Ok(())
    }

    fn emit(&mut self) -> Result<()> {
        let (first, steady, spf) =
            (self.opts.first_chunk_frames, self.opts.chunk_frames, self.model.config.samples_per_frame);
        let mut keep = Vec::with_capacity(self.running.len());
        for mut j in std::mem::take(&mut self.running) {
            let mut alive = true;
            while let Some(n) = chunk_due(j.frames, j.emitted, j.done(), first, steady) {
                let rest = j.pcm.split_off(n * spf * 2);
                let pcm = std::mem::replace(&mut j.pcm, rest);
                j.emitted += n;
                if j.sink.send(Event::Audio(Bytes::from(pcm))).is_err() {
                    tracing::debug!(id = j.id, "client gone");
                    alive = false;
                    break;
                }
            }
            if alive && j.done() {
                let done =
                    Done { finish: Finish::Complete, input_units: j.prompt.len() as u32, frames: j.frames as u32 };
                let _ = j.sink.send(Event::Done(done));
                alive = false;
            }
            if alive {
                keep.push(j);
            } else {
                self.free.append(&mut j.pages);
            }
        }
        self.running = keep;
        Ok(())
    }

    fn idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
}

/// Loads the checkpoint at `dir` on a thread of its own and serves from it
/// until every [`omni_engine::Handle`] is dropped and the admitted work is
/// done. The model is loaded where it runs: the codec's kern runtime is bound
/// to the thread that made it.
pub fn start(
    device: usize,
    dir: PathBuf,
    opts: Options,
    name: String,
    max_input_chars: usize,
    queue: usize,
) -> Result<(Handle, JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-qwen3-tts".into()).spawn(move || {
        let loaded = (|| {
            let gpu = Gpu::new(device)?;
            gpu.bind()?;
            let model = Model::load(&gpu, &dir, &opts).with_context(|| format!("load {}", dir.display()))?;
            let (handle, inbox) = omni_engine::channel(model.info(&name, max_input_chars), queue);
            anyhow::Ok((handle, inbox, Engine::new(gpu, model, opts)))
        })();
        match loaded {
            Ok((handle, inbox, engine)) => {
                let _ = tx.send(Ok(handle));
                if let Err(e) = run(inbox, engine) {
                    tracing::error!("engine stopped: {e:#}");
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e));
            }
        }
    })?;
    let handle = rx.recv().context("the engine thread died while loading")??;
    Ok((handle, thread))
}

fn run(inbox: Inbox, mut engine: Engine) -> Result<()> {
    engine.gpu.bind()?;
    let mut open = true;
    loop {
        if engine.idle() {
            if !open {
                return Ok(());
            }
            match inbox.rx.recv() {
                Ok(s) => engine.take(s.speech, s.sink),
                Err(_) => return Ok(()),
            }
        }
        loop {
            match inbox.rx.try_recv() {
                Ok(s) => engine.take(s.speech, s.sink),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        engine.step()?;
        inbox.load.waiting.store(engine.waiting.len(), Ordering::Relaxed);
        inbox.load.running.store(engine.running.len(), Ordering::Relaxed);
    }
}
