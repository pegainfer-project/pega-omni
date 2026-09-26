//! The clock around [`Model`]: one thread that owns the GPU and runs every
//! open session one frame per 80 ms tick, as a [`Ticker`] under
//! [`omni_engine::live::drive`].
//!
//! A session that arrives builds its prefix (voice and role prompt) and
//! leases its KV ring at once; the next tick prefills every waiting prefix,
//! in as few calls as `max_prefill` allows ([`batches`]), before it runs the
//! frame, so a session's frame 0 is in the tick that prefilled it. Each tick
//! drains every session's audio into its [`Jitter`] buffer, takes one frame
//! each (silence on an underrun), runs one [`Model::tick`] over all of them,
//! and sends each its agent frame as s16le and the text its token completed.
//!
//! A session ends when its caller hangs up, at `max_frames`, when its sink is
//! gone, or, for every session, on a model fault, which also stops the
//! engine.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use omni_engine::live::CloseReason;
use omni_engine::live::Jitter;
use omni_engine::live::Line;
use omni_engine::live::LiveHandle;
use omni_engine::live::LiveInbox;
use omni_engine::live::LiveInfo;
use omni_engine::live::LiveSubmission;
use omni_engine::live::Ticker;
use omni_engine::live::frames_in;
use omni_engine::live::s16le;

use crate::config::CODEBOOKS;
use crate::config::DRAWN;
use crate::config::FRAME;
use crate::config::SAMPLE_RATE;
use crate::config::Sampling;
use crate::model;
use crate::model::Limits;
use crate::model::Model;
use crate::model::Prefix;
use crate::model::Step;
use crate::tokenizer::Detok;

pub const DEFAULT_VOICE: &str = "NATF2";
pub const DEFAULT_INSTRUCTIONS: &str = "You are a helpful and friendly assistant. Answer the user's questions and help with their requests in a clear, concise and natural way.";

#[derive(Clone, Debug)]
pub struct Options {
    pub max_sessions: usize,
    /// Prompt rows per prefill call.
    pub max_prefill: usize,
    /// Session length before the engine closes it as expired.
    pub max_session: Duration,
    /// Caller audio buffered, in frames, before the oldest is dropped.
    pub jitter_frames: usize,
    /// Caller audio buffered, in frames, before playout starts.
    pub prebuffer_frames: usize,
    pub sampling: Sampling,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_sessions: 16,
            max_prefill: 4096,
            max_session: Duration::from_secs(240),
            jitter_frames: 4,
            prebuffer_frames: 2,
            sampling: Sampling::default(),
        }
    }
}

impl Options {
    pub fn check(self) -> Result<Self> {
        ensure!(self.max_sessions > 0, "max_sessions must be positive");
        ensure!(self.max_frames() > 0, "max_session is shorter than a frame");
        ensure!(
            (1..=self.jitter_frames).contains(&self.prebuffer_frames),
            "prebuffer_frames must be between 1 and jitter_frames ({}), got {}",
            self.jitter_frames,
            self.prebuffer_frames
        );
        Ok(self)
    }

    fn max_frames(&self) -> u64 {
        frames_in(self.max_session, SAMPLE_RATE, FRAME)
    }
}

fn info(model: &Model, name: &str, opts: &Options) -> LiveInfo {
    LiveInfo {
        model: name.into(),
        sample_rate: SAMPLE_RATE,
        frame_samples: FRAME as u32,
        voices: model.voices().map(String::from).collect(),
        default_voice: DEFAULT_VOICE.into(),
        default_instructions: DEFAULT_INSTRUCTIONS.into(),
        max_instructions_chars: model.max_instructions_chars(),
        max_sessions: opts.max_sessions,
        max_frames: opts.max_frames(),
    }
}

/// Loads the checkpoint at `dir` onto `device` and serves it on a thread of its own.
pub fn start(
    device: usize,
    dir: PathBuf,
    opts: Options,
    name: String,
    queue: usize,
) -> Result<(LiveHandle, JoinHandle<()>)> {
    let opts = opts.check()?;
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-personaplex".into()).spawn(move || {
        let loaded = (|| {
            let limits = Limits { max_sessions: opts.max_sessions, max_prefill: opts.max_prefill };
            let model =
                Model::load(device, &dir, limits, opts.sampling).with_context(|| format!("load {}", dir.display()))?;
            ensure!(
                model.voices().any(|v| v == DEFAULT_VOICE),
                "{} has no voice {DEFAULT_VOICE}, the default",
                dir.display()
            );
            let (handle, inbox) = omni_engine::live::live_channel(info(&model, &name, &opts), queue);
            anyhow::Ok((handle, inbox, model))
        })();
        match loaded {
            Ok((handle, inbox, model)) => {
                let _ = tx.send(Ok(handle));
                if let Err(e) = run(&inbox, model, opts) {
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

/// How many prefixes of `lens` rows, in order, each prefill call takes: as
/// many as fit `max` rows, and at least one.
pub fn batches(lens: &[usize], max: usize) -> Vec<usize> {
    let calls = lens.iter().fold(Vec::<(usize, usize)>::new(), |mut calls, &len| {
        match calls.last_mut() {
            Some((n, rows)) if *rows + len <= max => {
                *n += 1;
                *rows += len;
            }
            _ => calls.push((1, len)),
        }
        calls
    });
    calls.into_iter().map(|(n, _)| n).collect()
}

fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

struct Call {
    line: Line,
    session: model::Session,
    detok: Detok,
    seed: u64,
}

/// A session taken and leased, its prefix not yet prefilled.
struct Admitted {
    submission: LiveSubmission,
    session: model::Session,
    prefix: Prefix,
}

struct Engine {
    model: Model,
    opts: Options,
    calls: BTreeMap<u64, Call>,
    admitted: Vec<Admitted>,
}

impl Engine {
    /// Prefills every admitted session and starts its line.
    fn prefill(&mut self) -> Result<()> {
        let lens: Vec<usize> = self.admitted.iter().map(|a| a.prefix.len()).collect();
        for n in batches(&lens, self.opts.max_prefill) {
            let mut batch: Vec<Admitted> = self.admitted.drain(..n).collect();
            let mut rows: Vec<_> = batch.iter_mut().map(|a| (&mut a.session, &a.prefix)).collect();
            if let Err(e) = self.model.start(&mut rows) {
                batch.into_iter().for_each(|a| a.submission.refuse(CloseReason::Aborted));
                return Err(e);
            }
            for a in batch {
                let (id, o) = (a.submission.id, &self.opts);
                let jitter = Jitter::new(FRAME, o.prebuffer_frames, o.jitter_frames);
                if let Some(line) = Line::start(a.submission, jitter, o.max_frames()) {
                    let call = Call { line, session: a.session, detok: Detok::default(), seed: rand::random() };
                    self.calls.insert(id, call);
                }
            }
        }
        Ok(())
    }

    fn abort(&mut self) {
        std::mem::take(&mut self.calls).into_values().for_each(|c| c.line.close(CloseReason::Aborted));
        self.admitted.drain(..).for_each(|a| a.submission.refuse(CloseReason::Aborted));
    }
}

impl Ticker for Engine {
    type Error = anyhow::Error;

    fn sessions(&self) -> usize {
        self.calls.len() + self.admitted.len()
    }

    fn take(&mut self, s: LiveSubmission) -> Result<()> {
        match self.model.prompt(&s.session.voice, &s.session.instructions) {
            Ok(prefix) => {
                let session = self.model.open()?;
                self.admitted.push(Admitted { submission: s, session, prefix });
            }
            Err(e) => {
                tracing::error!("session {}: {e:#}", s.id);
                s.refuse(CloseReason::Aborted);
            }
        }
        Ok(())
    }

    fn tick(&mut self, skipped: u64) -> Result<()> {
        self.prefill()?;
        let ended: Vec<_> = self.calls.iter_mut().filter_map(|(&id, c)| Some((id, c.line.listen()?))).collect();
        ended.into_iter().for_each(|(id, reason)| self.calls.remove(&id).expect("a call").line.close(reason));
        if self.calls.is_empty() {
            return Ok(());
        }
        let frames: Vec<Vec<f32>> = self.calls.values_mut().map(|c| c.line.hear(skipped)).collect();
        let mut rows: Vec<(&mut model::Session, Step)> = self
            .calls
            .values_mut()
            .zip(&frames)
            .map(|(c, pcm)| {
                let seed = splitmix(c.seed ^ c.line.frames());
                (&mut c.session, Step { pcm, force: [-1; DRAWN], force_caller: [-1; CODEBOOKS], seed })
            })
            .collect();
        let out = self.model.tick(&mut rows)?;
        let tokenizer = &self.model.tokenizer;
        let gone: Vec<u64> = self
            .calls
            .iter_mut()
            .enumerate()
            .filter_map(|(n, (&id, c))| {
                let pcm = s16le(&out.pcm[n * FRAME..(n + 1) * FRAME]);
                let text = u32::try_from(out.emitted[n][0]).ok().and_then(|t| c.detok.push(tokenizer, t));
                (!c.line.say(pcm, text)).then_some(id)
            })
            .collect();
        gone.iter().for_each(|id| drop(self.calls.remove(id)));
        Ok(())
    }
}

fn run(inbox: &LiveInbox, model: Model, opts: Options) -> Result<()> {
    let mut engine = Engine { model, opts, calls: BTreeMap::new(), admitted: Vec::new() };
    let result = omni_engine::live::drive(inbox, &mut engine);
    engine.abort();
    inbox.pulse.sessions.store(0, Ordering::Relaxed);
    result
}
