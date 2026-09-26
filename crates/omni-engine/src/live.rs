//! The contract between the front end and a live (full-duplex) engine.
//!
//! A live engine runs on a clock, not on requests. Every tick it advances every
//! open session by one frame of `frame_samples` samples: it takes a frame of
//! the caller's audio (silence if none arrived) and emits a frame of the
//! agent's, whether anyone is speaking or not. Turn taking is the model's
//! business; there is no response boundary, no commit and no VAD in the
//! contract.
//!
//! A session is a [`LiveSubmission`]: the checked [`Session`], a receiver of
//! the caller's audio (s16le mono at `sample_rate`, chunks of any size) and a
//! sink for [`Output`]. Frame `n` of a session is the `n`th frame the engine
//! ran for it, so the agent's audio is one contiguous stream from
//! [`Output::Started`], frame `n` covering `n * frame_samples ..` samples of
//! it. A tick the engine overran is not run for anyone: the caller's audio of
//! it is discarded ([`Jitter::skip`]) so the caller's latency stays bounded,
//! and the agent's stream arrives one frame later than the wall clock, which
//! the client's playout buffer absorbs or does not.
//!
//! Ending is ownership, both ways: the front end drops the audio sender when
//! the caller leaves, and the engine answers with [`CloseReason::Hangup`] (its
//! final counts) if anyone still listens; the engine drops the sink after
//! [`Output::Closed`] when it ends the session itself. Either side notices the
//! other on its next receive or send.
//!
//! [`drive`] is the engine's shell, shared by every live engine: the clock
//! (tick `k` is due at `k * period` after the epoch; a tick that runs late
//! skips the ticks it overran instead of catching up, [`Clock`]), admission up
//! to `max_sessions`, and the [`Pulse`] counters. An engine is a [`Ticker`];
//! a session inside it is a [`Line`].

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::error::TryRecvError as InputError;
use tokio::sync::mpsc::unbounded_channel;

use crate::Invalid;
use crate::Rejected;

/// What a live engine serves, fixed at launch.
#[derive(Clone, Debug)]
pub struct LiveInfo {
    pub model: String,
    pub sample_rate: u32,
    /// Samples per tick, in and out.
    pub frame_samples: u32,
    pub voices: BTreeSet<String>,
    /// The voice of a session that names none.
    pub default_voice: String,
    /// The instructions of a session that gives none.
    pub default_instructions: String,
    pub max_instructions_chars: usize,
    /// Sessions the engine runs at once; the next one is refused.
    pub max_sessions: usize,
    /// Frames a session runs before the engine closes it as expired.
    pub max_frames: u64,
}

impl LiveInfo {
    /// Milliseconds from the session start to the start of frame `frame`.
    pub fn frame_ms(&self, frame: u64) -> u64 {
        frame * self.frame_samples as u64 * 1000 / self.sample_rate as u64
    }

    /// The time between ticks.
    pub fn period(&self) -> Duration {
        period(self.sample_rate, self.frame_samples as usize)
    }

    /// Accepts a session this engine can run, or names the first field it cannot.
    pub fn check(&self, draft: SessionDraft) -> Result<Session, Invalid> {
        let voice = draft.voice.unwrap_or_else(|| self.default_voice.clone());
        if !self.voices.contains(&voice) {
            return Err(Invalid {
                param: "audio.output.voice",
                message: format!("unknown voice `{voice}`; expected one of {:?}", self.voices),
            });
        }
        let instructions = draft.instructions.unwrap_or_else(|| self.default_instructions.clone());
        let chars = instructions.chars().count();
        if chars > self.max_instructions_chars {
            return Err(Invalid {
                param: "instructions",
                message: format!("{chars} characters, the limit is {}", self.max_instructions_chars),
            });
        }
        Ok(Session { voice, instructions })
    }
}

/// The time `frame` samples take at `sample_rate`.
pub fn period(sample_rate: u32, frame: usize) -> Duration {
    Duration::from_nanos(frame as u64 * 1_000_000_000 / sample_rate as u64)
}

/// Whole frames of `frame` samples at `sample_rate` in `length`.
pub fn frames_in(length: Duration, sample_rate: u32, frame: usize) -> u64 {
    (length.as_nanos() / period(sample_rate, frame).as_nanos()) as u64
}

/// The unchecked session fields, as the protocol layer parsed them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionDraft {
    pub voice: Option<String>,
    pub instructions: Option<String>,
}

/// A session the engine can run as is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Session {
    pub voice: String,
    pub instructions: String,
}

/// What a session's sink receives: `Started`, then frames in order, then at
/// most one `Closed`.
#[derive(Clone, Debug)]
pub enum Output {
    /// The prompt is in; frames follow.
    Started,
    /// Frame `frame` of the agent's audio, `frame_samples` samples of s16le.
    Audio {
        frame: u64,
        pcm: Bytes,
    },
    /// Text the agent spoke with frame `frame`, never empty.
    Text {
        frame: u64,
        delta: String,
    },
    Closed(Closed),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Closed {
    pub reason: CloseReason,
    /// Frames the session ran.
    pub frames: u64,
    /// Times the caller's audio ran dry mid-playout ([`Jitter::underruns`]).
    pub underruns: u64,
    /// Caller samples dropped for arriving too late ([`Jitter::dropped`]).
    pub dropped: u64,
}

impl Closed {
    /// A session refused before it ran.
    pub fn refused(reason: CloseReason) -> Self {
        Self { reason, frames: 0, underruns: 0, dropped: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseReason {
    /// The session reached [`LiveInfo::max_frames`].
    Expired,
    /// [`LiveInfo::max_sessions`] were already running.
    Busy,
    /// The engine gave up on the session (shutdown, internal fault).
    Aborted,
    /// The front end dropped the audio sender; the engine retired the session.
    Hangup,
}

pub struct LiveSubmission {
    pub id: u64,
    pub session: Session,
    pub input: UnboundedReceiver<Bytes>,
    pub sink: UnboundedSender<Output>,
}

impl LiveSubmission {
    /// Answers with [`Closed::refused`]; the session never runs.
    pub fn refuse(self, reason: CloseReason) {
        let _ = self.sink.send(Output::Closed(Closed::refused(reason)));
    }
}

/// A live engine's counters, written by [`drive`] and read by the front end.
///
/// `ticks` and `busy_us` are cumulative, so a reader diffing two samples gets
/// the mean tick and the engine's duty cycle between them.
#[derive(Debug, Default)]
pub struct Pulse {
    pub sessions: AtomicUsize,
    pub ticks: AtomicU64,
    /// Ticks the engine overran and skipped.
    pub late_ticks: AtomicU64,
    pub busy_us: AtomicU64,
    pub last_tick_us: AtomicU64,
}

impl Pulse {
    fn record(&self, took: Duration, skipped: u64) {
        let us = took.as_micros() as u64;
        self.ticks.fetch_add(1, Ordering::Relaxed);
        self.late_ticks.fetch_add(skipped, Ordering::Relaxed);
        self.busy_us.fetch_add(us, Ordering::Relaxed);
        self.last_tick_us.store(us, Ordering::Relaxed);
    }
}

/// The front end's side of a live engine's channel.
#[derive(Clone)]
pub struct LiveHandle {
    pub info: Arc<LiveInfo>,
    pub pulse: Arc<Pulse>,
    tx: mpsc::SyncSender<LiveSubmission>,
    next_id: Arc<AtomicU64>,
}

/// The engine's side.
pub struct LiveInbox {
    pub info: Arc<LiveInfo>,
    pub pulse: Arc<Pulse>,
    pub rx: mpsc::Receiver<LiveSubmission>,
}

/// A connected handle and inbox; `queue` bounds sessions not yet taken by the engine.
pub fn live_channel(info: LiveInfo, queue: usize) -> (LiveHandle, LiveInbox) {
    let info = Arc::new(info);
    let pulse = Arc::new(Pulse::default());
    let (tx, rx) = mpsc::sync_channel(queue);
    let handle = LiveHandle { info: info.clone(), pulse: pulse.clone(), tx, next_id: Arc::new(AtomicU64::new(1)) };
    (handle, LiveInbox { info, pulse, rx })
}

/// A session the engine took: its id, where the caller's audio goes, and where the engine's output comes from.
pub struct Opened {
    pub id: u64,
    pub session: Session,
    pub audio: UnboundedSender<Bytes>,
    pub output: UnboundedReceiver<Output>,
}

impl LiveHandle {
    /// Opens a session without blocking.
    pub fn start(&self, session: Session) -> Result<Opened, Rejected> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (audio, input) = unbounded_channel();
        let (sink, output) = unbounded_channel();
        match self.tx.try_send(LiveSubmission { id, session: session.clone(), input, sink }) {
            Ok(()) => Ok(Opened { id, session, audio, output }),
            Err(mpsc::TrySendError::Full(_)) => Err(Rejected::Full),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(Rejected::Stopped),
        }
    }
}

/// The caller's audio between the network and the tick.
///
/// The network delivers in bursts, the tick takes exactly one frame. Playout
/// starts once `prebuffer` frames are buffered, so arrival jitter up to that
/// much is absorbed instead of cutting silence into the caller's speech. A
/// tick that finds less than a frame while playing gets silence and leaves the
/// partial frame (an underrun), and playout waits to refill the prebuffer.
/// Audio beyond `max_frames` frames is late by more than the buffer is meant
/// to absorb, so the oldest is dropped to keep the caller's latency bounded.
#[derive(Clone, Debug)]
pub struct Jitter {
    samples: VecDeque<i16>,
    /// A trailing odd byte, completed by the next push.
    odd: Option<u8>,
    frame: usize,
    prebuffer: usize,
    max_frames: usize,
    playing: bool,
    underruns: u64,
    dropped: u64,
}

impl Jitter {
    pub fn new(frame: usize, prebuffer: usize, max_frames: usize) -> Self {
        assert!(frame > 0 && prebuffer > 0, "a jitter buffer plays whole, non-empty frames");
        assert!(prebuffer <= max_frames, "the prebuffer must fit the buffer");
        Self {
            samples: VecDeque::new(),
            odd: None,
            frame,
            prebuffer,
            max_frames,
            playing: false,
            underruns: 0,
            dropped: 0,
        }
    }

    /// Appends s16le audio.
    pub fn push(&mut self, pcm: &[u8]) {
        let bytes: Vec<u8> = self.odd.take().into_iter().chain(pcm.iter().copied()).collect();
        let (pairs, rest) = bytes.as_chunks::<2>();
        self.odd = rest.first().copied();
        self.samples.extend(pairs.iter().map(|&p| i16::from_le_bytes(p)));
        let over = self.samples.len().saturating_sub(self.frame * self.max_frames);
        self.samples.drain(..over);
        self.dropped += over as u64;
    }

    /// The next frame as floats in [-1, 1); silence while prebuffering.
    pub fn pop(&mut self) -> Vec<f32> {
        self.playing |= self.samples.len() >= self.prebuffer * self.frame;
        if !self.playing || self.samples.len() < self.frame {
            self.underruns += self.playing as u64;
            self.playing = false;
            return vec![0.0; self.frame];
        }
        self.samples.drain(..self.frame).map(|s| s as f32 / 32768.0).collect()
    }

    /// Discards up to `frames` frames of buffered audio, the caller's share of
    /// ticks nobody ran; it is not playout, so it is never an underrun.
    pub fn skip(&mut self, frames: u64) {
        let n = self.samples.len().min((frames as usize).saturating_mul(self.frame));
        self.samples.drain(..n);
    }

    /// Samples buffered.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Times playout ran dry.
    pub fn underruns(&self) -> u64 {
        self.underruns
    }

    /// Samples dropped for arriving too late.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// Floats in [-1, 1] as s16le, the inverse of [`Jitter::pop`].
pub fn s16le(frame: &[f32]) -> Bytes {
    frame.iter().flat_map(|&x| ((x * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes()).collect()
}

/// A running session as an engine holds it: the caller's audio through a
/// [`Jitter`] buffer in, [`Output`] out, and the frames it ran.
pub struct Line {
    input: UnboundedReceiver<Bytes>,
    sink: UnboundedSender<Output>,
    jitter: Jitter,
    frames: u64,
    max_frames: u64,
}

impl Line {
    /// Starts `s`'s line; `None` if the front end already left.
    pub fn start(s: LiveSubmission, jitter: Jitter, max_frames: u64) -> Option<Self> {
        s.sink.send(Output::Started).ok()?;
        Some(Self { input: s.input, sink: s.sink, jitter, frames: 0, max_frames })
    }

    /// Frames run so far; the next one is this.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Drains the caller's audio; why the session is over, if it is: the
    /// caller hung up, or it ran `max_frames`.
    pub fn listen(&mut self) -> Option<CloseReason> {
        loop {
            match self.input.try_recv() {
                Ok(pcm) => self.jitter.push(&pcm),
                Err(InputError::Disconnected) => return Some(CloseReason::Hangup),
                Err(InputError::Empty) => return (self.frames >= self.max_frames).then_some(CloseReason::Expired),
            }
        }
    }

    /// The caller's frame for this tick, after discarding the `skipped` frames of ticks nobody ran.
    pub fn hear(&mut self, skipped: u64) -> Vec<f32> {
        self.jitter.skip(skipped);
        self.jitter.pop()
    }

    /// Sends the agent's frame and the text it spoke; false once nobody listens.
    pub fn say(&mut self, pcm: Bytes, text: Option<String>) -> bool {
        let frame = self.frames;
        self.frames += 1;
        self.sink.send(Output::Audio { frame, pcm }).is_ok()
            && text.filter(|t| !t.is_empty()).is_none_or(|delta| self.sink.send(Output::Text { frame, delta }).is_ok())
    }

    /// Ends the session with its final counts.
    pub fn close(self, reason: CloseReason) {
        let closed =
            Closed { reason, frames: self.frames, underruns: self.jitter.underruns(), dropped: self.jitter.dropped() };
        let _ = self.sink.send(Output::Closed(closed));
    }
}

/// What the clock says when tick `next` is the next one to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    /// Tick `next` starts after this much more time.
    Wait(Duration),
    /// Run tick `tick`, the latest one due; `skipped` ticks before it were overrun and are not run.
    Run { tick: u64, skipped: u64 },
}

/// The clock's decision `elapsed` after the epoch, ticks `period` apart.
pub fn due(next: u64, elapsed: Duration, period: Duration) -> Due {
    let now = (elapsed.as_nanos() / period.as_nanos()) as u64;
    if now < next {
        Due::Wait(Duration::from_nanos(period.as_nanos() as u64 * next) - elapsed)
    } else {
        Due::Run { tick: now, skipped: now - next }
    }
}

/// Ticks `period` apart from an epoch, without drift or bursts.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    period: Duration,
    next: u64,
}

impl Clock {
    pub fn new(period: Duration) -> Self {
        Self { period, next: 0 }
    }

    /// What to do `elapsed` after the epoch; a [`Due::Run`] is taken, so the next poll looks past it.
    pub fn poll(&mut self, elapsed: Duration) -> Due {
        let due = due(self.next, elapsed, self.period);
        if let Due::Run { tick, .. } = due {
            self.next = tick + 1;
        }
        due
    }

    /// Forgets the ticks that passed while nothing ran, so the next poll is not late.
    pub fn resume(&mut self, elapsed: Duration) {
        self.next = self.next.max((elapsed.as_nanos() / self.period.as_nanos()) as u64);
    }
}

/// A live engine as [`drive`] runs it.
pub trait Ticker {
    type Error;

    /// Sessions taken and not yet ended.
    fn sessions(&self) -> usize;

    /// Takes a session; [`drive`] has made sure there is room for it.
    fn take(&mut self, s: LiveSubmission) -> Result<(), Self::Error>;

    /// Runs one tick; the `skipped` ticks before it were overrun.
    fn tick(&mut self, skipped: u64) -> Result<(), Self::Error>;
}

/// Runs `engine` on the clock until the handles are gone and its last
/// session ended, or it fails.
pub fn drive<T: Ticker>(inbox: &LiveInbox, engine: &mut T) -> Result<(), T::Error> {
    let admit = |engine: &mut T, s: LiveSubmission| {
        let taken = if engine.sessions() >= inbox.info.max_sessions {
            s.refuse(CloseReason::Busy);
            Ok(())
        } else {
            engine.take(s)
        };
        inbox.pulse.sessions.store(engine.sessions(), Ordering::Relaxed);
        taken
    };
    let epoch = Instant::now();
    let mut clock = Clock::new(inbox.info.period());
    loop {
        if engine.sessions() == 0 {
            match inbox.rx.recv() {
                Ok(s) => admit(engine, s)?,
                Err(_) => return Ok(()),
            }
            clock.resume(epoch.elapsed());
            continue;
        }
        match clock.poll(epoch.elapsed()) {
            Due::Wait(d) => match inbox.rx.recv_timeout(d) {
                Ok(s) => admit(engine, s)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => std::thread::sleep(d),
            },
            Due::Run { skipped, .. } => {
                while let Ok(s) = inbox.rx.try_recv() {
                    admit(engine, s)?;
                }
                let began = Instant::now();
                engine.tick(skipped)?;
                inbox.pulse.record(began.elapsed(), skipped);
                inbox.pulse.sessions.store(engine.sessions(), Ordering::Relaxed);
            }
        }
    }
}
