//! Serving: requests in through an [`Inbox`], PCM chunks out through each
//! request's sink.
//!
//! One engine thread runs steps back to back. A step admits waiting requests
//! (FIFO, while batch slots, the per-step token budget and KV pages allow),
//! advances every running request by one frame through the [`Talker`], then
//! decodes whichever requests have a chunk due. Serial on purpose: talker and
//! codec share the stream; overlapping them is a measured change for later.
//!
//! Admission reserves a request's whole KV up front, sized by its frame cap
//! (proportional to the input length), so a running request never waits for
//! pages and nothing is ever preempted.
//!
//! The decisions are pure functions ([`chunk_due`], [`frame_cap`],
//! [`pages_for`]); [`Engine`] is the shell around them.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;

use anyhow::Result;
use bytes::Bytes;
use omni_cuda::Gpu;
use omni_engine::Done;
use omni_engine::EngineInfo;
use omni_engine::Event;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::Inbox;
use omni_engine::Speech;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

use crate::codec::Codec;
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
use crate::talker::Sampling;
use crate::talker::Talker;
use crate::weights::File;

#[derive(Clone, Debug)]
pub struct Options {
    pub device: usize,
    pub max_batch: usize,
    /// Talker tokens per step, prompts included.
    pub max_step_tokens: usize,
    pub kv_gib: f64,
    pub page_size: usize,
    pub first_chunk_frames: usize,
    pub chunk_frames: usize,
    /// Frames of left context each chunk is decoded with. The decoder's
    /// attention window (72) keeps streamed audio within bf16 noise of a
    /// whole-utterance decode (37 dB against the official run); the official
    /// `chunked_decode`'s 25 drops that to 22 dB over 8-frame chunks.
    pub context_frames: usize,
    pub max_input_chars: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_batch: 64,
            max_step_tokens: 8192,
            kv_gib: 16.0,
            page_size: 16,
            first_chunk_frames: 2,
            chunk_frames: 8,
            context_frames: 72,
            max_input_chars: 4096,
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
        let talker = Talker::load(gpu, &File::open(&dir.join("model.safetensors"))?, &config.model, limits)?;
        let codec_frames = opts.context_frames + opts.first_chunk_frames.max(opts.chunk_frames);
        let codec = Codec::load(
            gpu,
            &File::open(&dir.join("speech_tokenizer/model.safetensors"))?,
            &config.codec,
            config.samples_per_frame,
            codec_frames,
        )?;
        Ok(Self { config, tokenizer, talker, codec })
    }

    /// The front end's view: CustomVoice speakers, languages and sampling knobs.
    pub fn info(&self, name: &str, max_input_chars: usize) -> EngineInfo {
        let t = &self.config.model.talker_config;
        let languages = std::iter::once("auto".to_string()).chain(t.codec_language_id.keys().cloned()).collect();
        EngineInfo {
            model: name.into(),
            sample_rate: self.config.sample_rate,
            voices: t.spk_id.keys().cloned().collect(),
            extra: BTreeMap::from([
                ("language".to_string(), Extra::OneOf(languages)),
                ("temperature".to_string(), Extra::Number(0.0..=2.0)),
                ("top_k".to_string(), Extra::Integer(0..=t.stack.vocab_size as i64)),
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
    frames: Vec<[i32; GROUPS]>,
    emitted: usize,
    seen: Vec<u32>,
    rng: StdRng,
    sampling: Sampling,
    cap: usize,
    ended: bool,
}

impl Job {
    fn prefilled(&self) -> bool {
        self.cached > 0
    }

    fn done(&self) -> bool {
        self.ended || self.frames.len() >= self.cap
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
        let g = &m.config.generation;
        let temperature = speech.extra.get("temperature").and_then(|v| v.as_f64()).map_or(g.temperature, |x| x as f32);
        let top_k = speech.extra.get("top_k").and_then(|v| v.as_i64()).map_or(g.top_k, |x| x as i32);
        let seed = speech.extra.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(rand::random);
        Ok(Job {
            id: speech.id,
            sink,
            prompt,
            pages: vec![],
            cached: 0,
            frames: vec![],
            emitted: 0,
            seen: vec![0; m.talker.seen_words()],
            rng: StdRng::seed_from_u64(seed),
            sampling: Sampling {
                temperature,
                top_k,
                repetition_penalty: g.repetition_penalty,
                sub_temperature: g.subtalker_temperature,
                sub_top_k: g.subtalker_top_k,
            },
            cap: frame_cap(speech.input.chars().count(), g.max_new_tokens),
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

    fn admit(&mut self) {
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
            self.running.push(job);
        }
    }

    /// One step: admit, one frame for every running request, due chunks out.
    pub fn step(&mut self) -> Result<()> {
        self.admit();
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
                    input: match j.frames.last() {
                        Some(f) if j.prefilled() => Input::Frame(f),
                        _ => Input::Prompt(&j.prompt),
                    },
                    sampling: j.sampling,
                    seen: &j.seen,
                    generated: j.frames.len(),
                    uniforms: u,
                }
            })
            .collect();
        let frames = self.model.talker.step(&self.gpu, &rows, None)?;

        for (&i, frame) in order.iter().zip(frames) {
            let j = &mut self.running[i];
            j.cached += if j.prefilled() { 1 } else { j.prompt.len() };
            match frame {
                Frame::End => j.ended = true,
                Frame::Codes(c) => {
                    j.seen[c[0] as usize / 32] |= 1 << (c[0] % 32);
                    j.frames.push(c);
                }
            }
        }
        self.emit()
    }

    fn emit(&mut self) -> Result<()> {
        let (first, steady, ctx, spf) = (
            self.opts.first_chunk_frames,
            self.opts.chunk_frames,
            self.opts.context_frames,
            self.model.config.samples_per_frame,
        );
        let mut keep = Vec::with_capacity(self.running.len());
        for mut j in std::mem::take(&mut self.running) {
            let mut alive = true;
            while let Some(n) = chunk_due(j.frames.len(), j.emitted, j.done(), first, steady) {
                let start = j.emitted.saturating_sub(ctx);
                let codes: Vec<i32> = j.frames[start..j.emitted + n].iter().flatten().copied().collect();
                let wav = self.model.codec.decode(&self.gpu, &codes)?;
                let pcm: Vec<u8> = wav[(j.emitted - start) * spf..]
                    .iter()
                    .flat_map(|&x| ((x * 32767.0).round() as i16).to_le_bytes())
                    .collect();
                j.emitted += n;
                if j.sink.send(Event::Audio(Bytes::from(pcm))).is_err() {
                    tracing::debug!(id = j.id, "client gone");
                    alive = false;
                    break;
                }
            }
            if alive && j.done() {
                let done = Done {
                    finish: Finish::Complete,
                    input_units: j.prompt.len() as u32,
                    frames: j.frames.len() as u32,
                };
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

/// Runs `engine` on its own thread until every [`omni_engine::Handle`] is
/// dropped and the admitted work is done.
pub fn spawn(inbox: Inbox, engine: Engine) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("omni-qwen3-tts".into())
        .spawn(move || {
            if let Err(e) = run(inbox, engine) {
                tracing::error!("engine stopped: {e:#}");
            }
        })
        .expect("spawn omni-qwen3-tts")
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
