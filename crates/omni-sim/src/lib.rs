//! A CPU-only speech engine with the timing shape of a codec-frame TTS model;
//! [`image`] is the image counterpart.
//!
//! The sim exists to load the front end without a GPU: it admits requests in
//! FIFO order into a bounded batch, generates one codec frame per running
//! request per step, and releases audio in chunks (a short first chunk for time
//! to first packet, then steady chunks). The audio is a fixed tone sliced out
//! of one shared buffer, so emitting a chunk costs a reference count, not a
//! synthesis.
//!
//! [`Sim`] is the decision core: a pure state machine with no clock and no
//! channels. [`Sim::step`] returns what happened in one step and what that step
//! costs under the [`Profile`]. [`spawn`] is the shell: a thread that feeds the
//! core from an [`Inbox`], sleeps out each step's cost, then delivers its
//! emissions, the way a GPU step's output exists only once the step is done.
//! Requests that arrive during a step join the next one. A zero-cost profile
//! runs steps back to back, which is how the front end's own ceiling is
//! measured.
//!
//! [`live`] is the same idea for full-duplex sessions.

pub mod live;

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use omni_engine::Done;
use omni_engine::EngineInfo;
use omni_engine::Event;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::Inbox;
use omni_engine::Speech;
use tokio::sync::mpsc::UnboundedSender;

pub mod image;

/// The sim's timing and chunking model.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub sample_rate: u32,
    /// Codec frames per second of audio.
    pub frame_rate: f64,
    /// Frames generated per input character at speed 1.0.
    pub frames_per_char: f64,
    pub max_frames: u32,
    pub first_chunk_frames: u32,
    pub chunk_frames: u32,
    pub max_batch: usize,
    /// Step cost: `step_base + step_per_row * rows + prefill_per_char * admitted chars`.
    pub step_base: Duration,
    pub step_per_row: Duration,
    pub prefill_per_char: Duration,
}

impl Default for Profile {
    /// Qwen3-TTS-12Hz output shape (24 kHz, 12.5 frames/s, ~15 characters of
    /// English per second of speech) with zero cost.
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            frame_rate: 12.5,
            frames_per_char: 0.8,
            max_frames: 4096,
            first_chunk_frames: 1,
            chunk_frames: 4,
            max_batch: 256,
            step_base: Duration::ZERO,
            step_per_row: Duration::ZERO,
            prefill_per_char: Duration::ZERO,
        }
    }
}

impl Profile {
    /// Checks the profile's arithmetic holds: whole samples per frame, non-empty chunks and batch.
    pub fn check(self) -> Result<Self, String> {
        let per_frame = self.sample_rate as f64 / self.frame_rate;
        if !(per_frame >= 1.0 && per_frame.fract() == 0.0) {
            return Err(format!(
                "{} Hz / {} frames/s is not a whole number of samples",
                self.sample_rate, self.frame_rate
            ));
        }
        if self.first_chunk_frames == 0 || self.chunk_frames == 0 || self.max_batch == 0 || self.max_frames == 0 {
            return Err("chunk sizes, max_batch and max_frames must be positive".into());
        }
        if self.frames_per_char.is_nan() || self.frames_per_char <= 0.0 {
            return Err("frames_per_char must be positive".into());
        }
        Ok(self)
    }

    pub fn samples_per_frame(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate) as usize
    }

    pub fn bytes_per_frame(&self) -> usize {
        self.samples_per_frame() * 2
    }

    /// Frames the sim generates for a request: `extra.frames` if given, else scaled by its length and speed.
    pub fn planned_frames(&self, speech: &Speech) -> u32 {
        let fixed = speech.extra.get("frames").and_then(|v| v.as_u64());
        let frames = fixed.unwrap_or_else(|| {
            let chars = speech.input.chars().count() as f64;
            (chars * self.frames_per_char / speech.speed as f64).ceil() as u64
        });
        frames.clamp(1, self.max_frames as u64) as u32
    }

    /// The front end's view of this sim.
    pub fn info(&self, model: &str) -> EngineInfo {
        EngineInfo {
            model: model.into(),
            sample_rate: self.sample_rate,
            voices: VOICES.iter().map(|v| v.to_string()).collect(),
            extra: [("frames".to_string(), Extra::Integer(1..=self.max_frames as i64))].into(),
            speeds: 0.25..=4.0,
            max_input_chars: 4096,
        }
    }
}

/// OpenAI's voice names, so stock clients work against the sim unchanged.
pub const VOICES: [&str; 11] =
    ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emission {
    Chunk { id: u64, frames: u32 },
    Done { id: u64, frames: u32, input_units: u32 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Step {
    pub emissions: Vec<Emission>,
    /// Rows that generated a frame this step.
    pub rows: usize,
    pub admitted_chars: u64,
}

#[derive(Clone, Copy, Debug)]
struct Job {
    id: u64,
    planned: u32,
    generated: u32,
    pending: u32,
    chunks: u32,
    input_units: u32,
}

/// The decision core: FIFO admission into a bounded batch, one frame per row per step.
#[derive(Debug)]
pub struct Sim {
    profile: Profile,
    waiting: VecDeque<Job>,
    running: Vec<Job>,
}

impl Sim {
    pub fn new(profile: Profile) -> Self {
        Self { profile, waiting: VecDeque::new(), running: Vec::new() }
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn admit(&mut self, speech: &Speech) {
        let input_units = speech.input.chars().count() as u32;
        let planned = self.profile.planned_frames(speech);
        self.waiting.push_back(Job { id: speech.id, planned, generated: 0, pending: 0, chunks: 0, input_units });
    }

    /// Drops a request wherever it is; unknown ids are ignored.
    pub fn retire(&mut self, id: u64) {
        self.waiting.retain(|j| j.id != id);
        self.running.retain(|j| j.id != id);
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn running(&self) -> usize {
        self.running.len()
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Advances one step. Emission order within a step follows admission order.
    pub fn step(&mut self) -> Step {
        let room = self.profile.max_batch - self.running.len();
        let admitted: Vec<Job> = self.waiting.drain(..room.min(self.waiting.len())).collect();
        let admitted_chars = admitted.iter().map(|j| j.input_units as u64).sum();
        self.running.extend(admitted);

        let (first, steady) = (self.profile.first_chunk_frames, self.profile.chunk_frames);
        let rows = self.running.len();
        let mut emissions = Vec::with_capacity(rows);
        self.running.retain_mut(|j| {
            j.generated += 1;
            j.pending += 1;
            let finished = j.generated == j.planned;
            let target = if j.chunks == 0 { first } else { steady };
            if j.pending == target || finished {
                emissions.push(Emission::Chunk { id: j.id, frames: j.pending });
                j.pending = 0;
                j.chunks += 1;
            }
            if finished {
                emissions.push(Emission::Done { id: j.id, frames: j.generated, input_units: j.input_units });
            }
            !finished
        });
        Step { emissions, rows, admitted_chars }
    }

    pub fn cost(&self, step: &Step) -> Duration {
        let p = &self.profile;
        p.step_base + p.step_per_row * step.rows as u32 + p.prefill_per_char * step.admitted_chars as u32
    }
}

/// A tone long enough for the largest chunk, sliced per emission.
fn tone(profile: &Profile) -> Bytes {
    let frames = profile.first_chunk_frames.max(profile.chunk_frames) as usize;
    let samples = frames * profile.samples_per_frame();
    let step = std::f64::consts::TAU * 220.0 / profile.sample_rate as f64;
    let pcm: Vec<u8> = (0..samples).flat_map(|i| (((i as f64 * step).sin() * 8_000.0) as i16).to_le_bytes()).collect();
    Bytes::from(pcm)
}

/// Runs the sim on its own thread until every [`omni_engine::Handle`] is dropped.
pub fn spawn(inbox: Inbox, profile: Profile) -> JoinHandle<()> {
    std::thread::Builder::new().name("omni-sim".into()).spawn(move || run(inbox, profile)).expect("spawn omni-sim")
}

type Sinks = BTreeMap<u64, UnboundedSender<Event>>;

fn take(sim: &mut Sim, sinks: &mut Sinks, s: omni_engine::Submission) {
    sim.admit(&s.speech);
    sinks.insert(s.speech.id, s.sink);
}

fn run(inbox: Inbox, profile: Profile) {
    let pcm = tone(&profile);
    let bytes_per_frame = profile.bytes_per_frame();
    let mut sim = Sim::new(profile);
    let mut sinks = Sinks::new();

    loop {
        if sim.is_idle() {
            match inbox.rx.recv() {
                Ok(s) => take(&mut sim, &mut sinks, s),
                Err(_) => return,
            }
        }
        loop {
            match inbox.rx.try_recv() {
                Ok(s) => take(&mut sim, &mut sinks, s),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if sim.is_idle() => return,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        let started = Instant::now();
        let step = sim.step();
        let cost = sim.cost(&step);
        let deadline = started + cost;
        while let Some(left) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
            match inbox.rx.recv_timeout(left) {
                Ok(s) => take(&mut sim, &mut sinks, s),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    std::thread::sleep(left);
                    break;
                }
            }
        }
        for e in step.emissions {
            let (id, event) = match e {
                Emission::Chunk { id, frames } => (id, Event::Audio(pcm.slice(..frames as usize * bytes_per_frame))),
                Emission::Done { id, frames, input_units } => {
                    (id, Event::Done(Done { finish: Finish::Complete, input_units, frames }))
                }
            };
            let done = matches!(event, Event::Done(_));
            let delivered = sinks.get(&id).is_some_and(|s| s.send(event).is_ok());
            if !delivered {
                sim.retire(id);
            }
            if !delivered || done {
                sinks.remove(&id);
            }
        }
        inbox.load.waiting.store(sim.waiting(), Ordering::Relaxed);
        inbox.load.running.store(sim.running(), Ordering::Relaxed);
    }
}
