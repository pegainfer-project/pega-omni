//! Serving: requests in through an [`Inbox`], PCM chunks out through each request's sink.
//!
//! One engine thread runs steps back to back. A step admits waiting requests (FIFO, while batch
//! slots, the per-step prompt budget and KV allow), starts them (`prefill` and their first row of
//! codes), advances every other running request by one row, and sends the chunks that are due.
//!
//! The codec is not causal, so a chunk is decoded from a window that reaches [`OVERLAP`] frames
//! back into audio already sent and [`HOLDBACK`] frames past the chunk, and only the chunk's own
//! samples leave. Until a request ends, its last [`HOLDBACK`] frames wait for that context.
//!
//! Admission leases a request's whole KV and its state slot up front, the KV sized by its row cap
//! (proportional to the input length), so a running request never waits for memory and nothing is
//! ever preempted.
//!
//! The decisions are pure functions ([`window`], [`row_cap`]); [`Engine`] is the shell around them.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;

use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use kern_pool::Denied;
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

pub use crate::codec::MAX_FRAMES;
use crate::config::SAMPLE_RATE;
use crate::config::SAMPLES_PER_FRAME;
use crate::delay::Row;
use crate::delay::undelay;
use crate::model::Draw;
use crate::model::Limits;
use crate::model::Model;
use crate::model::Sampling;
use crate::model::Seq;

/// Frames of already-sent audio a chunk's window starts with, and frames past a chunk it ends with.
pub const OVERLAP: usize = 8;
pub const HOLDBACK: usize = 4;
/// The most rows a request draws: the model's training length.
pub const MAX_ROWS: usize = 1024;

#[derive(Clone, Debug)]
pub struct Options {
    pub max_batch: usize,
    /// Prompt tokens per step.
    pub max_step_tokens: usize,
    pub kv_gib: f64,
    pub first_chunk_frames: usize,
    pub chunk_frames: usize,
    pub sampling: Sampling,
}

/// One chunk's decode: the frames to decode, and the samples of the result that are the chunk's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub frames: Range<usize>,
    pub keep: Range<usize>,
}

/// The chunk to decode for a request with `frames` frames of which `emitted` are already audio:
/// the first chunk once `first` frames can be sent, then every `steady`, and the remainder once
/// the request has `finished`. The window is clipped to the codec's longest.
pub fn window(frames: usize, emitted: usize, finished: bool, first: usize, steady: usize) -> Option<Window> {
    let held = if finished { 0 } else { HOLDBACK };
    let pending = frames.saturating_sub(held).saturating_sub(emitted);
    let target = if emitted == 0 { first } else { steady };
    if pending == 0 || (pending < target && !finished) {
        return None;
    }
    let end = emitted + pending.min(target);
    let to = (end + held).min(frames);
    let from = emitted.saturating_sub(OVERLAP).max(to.saturating_sub(MAX_FRAMES));
    let at = |f: usize| (f - from) * SAMPLES_PER_FRAME;
    Some(Window { frames: from..to, keep: at(emitted)..at(end) })
}

/// The most rows a request may draw: generous for slow speech, capped by the model.
pub fn row_cap(input_chars: usize) -> usize {
    (50 + 8 * input_chars).min(MAX_ROWS)
}

/// The front end's view: one voice (the model's own), and a sampling seed.
pub fn info(name: &str, max_input_chars: usize) -> EngineInfo {
    EngineInfo {
        model: name.into(),
        sample_rate: SAMPLE_RATE,
        voices: BTreeSet::from(["default".to_string()]),
        extra: BTreeMap::from([("seed".to_string(), Extra::Integer(0..=i64::MAX))]),
        speeds: 1.0..=1.0,
        max_input_chars,
    }
}

struct Job {
    id: u64,
    sink: UnboundedSender<Event>,
    prompt: Vec<i32>,
    seq: Option<Seq>,
    /// Rows drawn, delayed.
    rows: Vec<Row>,
    ended: bool,
    emitted: usize,
    rng: StdRng,
    cap: usize,
}

impl Job {
    fn done(&self) -> bool {
        self.ended || self.rows.len() >= self.cap
    }

    fn draw(&mut self) -> Draw {
        Draw { uniforms: std::array::from_fn(|_| self.rng.random::<f32>()), force: None }
    }
}

pub struct Engine {
    model: Model,
    opts: Options,
    waiting: VecDeque<Job>,
    running: Vec<Job>,
}

fn abort(sink: &UnboundedSender<Event>) {
    let _ = sink.send(Event::Done(Done { finish: Finish::Aborted, input_units: 0, frames: 0 }));
}

impl Engine {
    pub fn new(model: Model, opts: Options) -> Self {
        Self { model, opts, waiting: VecDeque::new(), running: Vec::new() }
    }

    fn job(&self, speech: &Speech, sink: UnboundedSender<Event>) -> Result<Job, String> {
        let prompt = self.model.tokenizer.prompt(&speech.input);
        if prompt.len() > self.opts.max_step_tokens {
            return Err(format!("{} prompt tokens exceed the step's {}", prompt.len(), self.opts.max_step_tokens));
        }
        let seed = speech.extra.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(rand::random);
        Ok(Job {
            id: speech.id,
            sink,
            prompt,
            seq: None,
            rows: Vec::new(),
            ended: false,
            emitted: 0,
            rng: StdRng::seed_from_u64(seed),
            cap: row_cap(speech.input.chars().count()),
        })
    }

    fn enqueue(&mut self, speech: Speech, sink: UnboundedSender<Event>) {
        match self.job(&speech, sink.clone()) {
            Ok(job) => self.waiting.push_back(job),
            Err(e) => {
                tracing::warn!(id = speech.id, "rejected: {e}");
                abort(&sink);
            }
        }
    }

    /// Leases waiting requests in order; returns how many joined `running`.
    fn admit(&mut self) -> Result<usize> {
        let mut tokens = 0;
        let mut admitted = 0;
        while let Some(job) = self.waiting.front() {
            if self.running.len() >= self.opts.max_batch || tokens + job.prompt.len() > self.opts.max_step_tokens {
                break;
            }
            match self.model.open(job.prompt.len(), job.cap)? {
                Ok(seq) => {
                    let mut job = self.waiting.pop_front().expect("front exists");
                    tokens += job.prompt.len();
                    job.seq = Some(seq);
                    self.running.push(job);
                    admitted += 1;
                }
                Err(Denied::Busy | Denied::Remapping) => break,
                Err(d) => {
                    let job = self.waiting.pop_front().expect("front exists");
                    tracing::warn!(
                        id = job.id,
                        "rejected: {} prompt tokens and {} rows: {d}",
                        job.prompt.len(),
                        job.cap
                    );
                    abort(&job.sink);
                }
            }
        }
        Ok(admitted)
    }

    /// One step: admit and start, one row for every other running request, due chunks out.
    pub fn step(&mut self) -> Result<()> {
        let fresh = self.admit()?;
        let split = self.running.len() - fresh;
        let (old, new) = self.running.split_at_mut(split);
        if !new.is_empty() {
            let draws: Vec<Draw> = new.iter_mut().map(Job::draw).collect();
            let mut rows: Vec<_> =
                new.iter_mut().zip(draws).map(|(j, d)| (j.seq.as_mut().expect("admitted"), &j.prompt[..], d)).collect();
            let out = self.model.start(&mut rows)?;
            take(new, out.rows, out.ended);
        }
        if !old.is_empty() {
            let draws: Vec<Draw> = old.iter_mut().map(Job::draw).collect();
            let mut rows: Vec<_> =
                old.iter_mut().zip(draws).map(|(j, d)| (j.seq.as_mut().expect("admitted"), d)).collect();
            let out = self.model.step(&mut rows)?;
            take(old, out.rows, out.ended);
        }
        self.emit();
        Ok(())
    }

    fn emit(&mut self) {
        let (first, steady) = (self.opts.first_chunk_frames, self.opts.chunk_frames);
        let mut kept = Vec::with_capacity(self.running.len());
        for mut job in std::mem::take(&mut self.running) {
            let frames = undelay(&job.rows);
            let mut alive = true;
            while let Some(w) = window(frames.len(), job.emitted, job.done(), first, steady) {
                let pcm = match self.model.decode(&frames[w.frames.clone()]) {
                    Ok(pcm) => pcm,
                    Err(e) => {
                        tracing::warn!(id = job.id, "aborted: {e:#}");
                        abort(&job.sink);
                        alive = false;
                        break;
                    }
                };
                job.emitted = w.frames.start + w.keep.end / SAMPLES_PER_FRAME;
                if job.sink.send(Event::Audio(Bytes::from(s16le(&pcm[w.keep])))).is_err() {
                    tracing::debug!(id = job.id, "client gone");
                    alive = false;
                    break;
                }
            }
            if alive && job.done() && job.emitted == frames.len() {
                let done = Done {
                    finish: Finish::Complete,
                    input_units: job.prompt.len() as u32,
                    frames: frames.len() as u32,
                };
                let _ = job.sink.send(Event::Done(done));
                alive = false;
            }
            if alive {
                kept.push(job);
            }
        }
        self.running = kept;
    }

    fn idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
}

fn take(jobs: &mut [Job], rows: Vec<Row>, ended: Vec<bool>) {
    for ((job, row), end) in jobs.iter_mut().zip(rows).zip(ended) {
        job.rows.push(row);
        job.ended |= end;
    }
}

/// Mono signed 16-bit little-endian PCM from samples in `[-1, 1]` (clipped).
fn s16le(pcm: &[f32]) -> Vec<u8> {
    pcm.iter().flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect()
}

/// Loads the checkpoint at `dir` on a thread of its own and serves from it until every
/// [`omni_engine::Handle`] is dropped and the admitted work is done. The model is loaded where it
/// runs: its kern runtime is bound to the thread that made it.
pub fn start(
    device: usize,
    dir: PathBuf,
    opts: Options,
    name: String,
    max_input_chars: usize,
    queue: usize,
) -> Result<(Handle, JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-higgs-audio".into()).spawn(move || {
        let loaded = (|| {
            let limits = Limits {
                max_batch: opts.max_batch,
                max_tokens: opts.max_step_tokens,
                kv_gib: opts.kv_gib,
                sampling: opts.sampling,
            };
            let model = Model::load(device, &dir, limits).with_context(|| format!("load {}", dir.display()))?;
            let (handle, inbox) = omni_engine::channel(info(&name, max_input_chars), queue);
            anyhow::Ok((handle, inbox, Engine::new(model, opts)))
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
    let mut open = true;
    loop {
        if engine.idle() {
            if !open {
                return Ok(());
            }
            match inbox.rx.recv() {
                Ok(s) => engine.enqueue(s.speech, s.sink),
                Err(_) => return Ok(()),
            }
        }
        loop {
            match inbox.rx.try_recv() {
                Ok(s) => engine.enqueue(s.speech, s.sink),
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
