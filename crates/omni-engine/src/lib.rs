//! The contract between the HTTP front end and a speech engine.
//!
//! An engine is whatever sits on the far side of an [`Inbox`]: a thread (or a
//! GPU step loop) that takes [`Submission`]s, generates audio, and pushes
//! [`Event`]s into each submission's sink. There is no engine trait: the front
//! end holds a [`Handle`], the engine holds the matching [`Inbox`], and the
//! channel between them is the whole interface.
//!
//! Audio crosses the boundary as mono signed 16-bit little-endian PCM at
//! [`EngineInfo::sample_rate`]. Container formats (wav, SSE framing) are the
//! front end's business; the engine never encodes.
//!
//! Cancellation is ownership: when the client goes away the front end drops the
//! event receiver, the engine's next send fails, and the engine retires the
//! request. No cancel message exists.
//!
//! A [`Speech`] only exists after [`EngineInfo::check`] accepted it, so an
//! engine never re-validates the voice, the speed or the extension values.
//!
//! Full-duplex engines speak the [`live`] contract instead: sessions on a
//! clock rather than requests.

pub mod live;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

use bytes::Bytes;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;

/// What an engine serves, fixed at launch.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub model: String,
    pub sample_rate: u32,
    pub voices: BTreeSet<String>,
    /// Keys the engine accepts inside a request's `extra` object, and their values.
    pub extra: BTreeMap<String, Extra>,
    pub speeds: std::ops::RangeInclusive<f32>,
    pub max_input_chars: usize,
}

/// The values an `extra` key accepts.
#[derive(Clone, Debug, PartialEq)]
pub enum Extra {
    Integer(std::ops::RangeInclusive<i64>),
    OneOf(BTreeSet<String>),
}

impl Extra {
    fn check(&self, value: &Value) -> Result<(), String> {
        match self {
            Self::Integer(range) => match value.as_i64() {
                Some(x) if range.contains(&x) => Ok(()),
                _ => Err(format!("expected an integer in {}..={}, got {value}", range.start(), range.end())),
            },
            Self::OneOf(names) => match value.as_str() {
                Some(x) if names.contains(x) => Ok(()),
                _ => Err(format!("expected one of {names:?}, got {value}")),
            },
        }
    }
}

/// A request the engine can run as is.
#[derive(Clone, Debug)]
pub struct Speech {
    pub id: u64,
    pub input: String,
    pub voice: String,
    pub instructions: Option<String>,
    pub speed: f32,
    pub extra: BTreeMap<String, Value>,
}

/// The unchecked fields of a speech request, as the protocol layer parsed them.
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub input: String,
    pub voice: String,
    pub instructions: Option<String>,
    pub speed: Option<f32>,
    pub extra: BTreeMap<String, Value>,
}

/// Why a draft was refused: the request field at fault and what was wrong.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{param}`: {message}")]
pub struct Invalid {
    pub param: &'static str,
    pub message: String,
}

impl EngineInfo {
    /// Accepts a draft this engine can serve, or names the first field it cannot.
    pub fn check(&self, draft: Draft) -> Result<Speech, Invalid> {
        let chars = draft.input.chars().count();
        let invalid = |param, message: String| Err(Invalid { param, message });
        if chars == 0 {
            return invalid("input", "must not be empty".into());
        }
        if chars > self.max_input_chars {
            return invalid("input", format!("{chars} characters, the limit is {}", self.max_input_chars));
        }
        if !self.voices.contains(&draft.voice) {
            return invalid("voice", format!("unknown voice `{}`; see GET /v1/audio/voices", draft.voice));
        }
        let speed = draft.speed.unwrap_or(1.0);
        if !self.speeds.contains(&speed) {
            return invalid("speed", format!("{speed} is outside {}..={}", self.speeds.start(), self.speeds.end()));
        }
        for (key, value) in &draft.extra {
            match self.extra.get(key) {
                None => return invalid("extra", format!("`{key}` is not an extension of model `{}`", self.model)),
                Some(spec) => {
                    spec.check(value).map_err(|m| Invalid { param: "extra", message: format!("`{key}`: {m}") })?
                }
            }
        }
        Ok(Speech {
            id: 0,
            input: draft.input,
            voice: draft.voice,
            instructions: draft.instructions,
            speed,
            extra: draft.extra,
        })
    }
}

/// One step of a request's output stream.
#[derive(Clone, Debug)]
pub enum Event {
    /// PCM s16le mono; never empty.
    Audio(Bytes),
    Done(Done),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Done {
    pub finish: Finish,
    /// Input units the engine consumed (tokens for a model, characters for the sim).
    pub input_units: u32,
    /// Codec frames generated.
    pub frames: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Complete,
    /// The engine gave up on the request (shutdown, internal fault).
    Aborted,
}

pub struct Submission {
    pub speech: Speech,
    pub sink: UnboundedSender<Event>,
}

/// Engine occupancy, written by the engine and read by the front end's metrics.
#[derive(Debug, Default)]
pub struct Load {
    pub waiting: AtomicUsize,
    pub running: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Rejected {
    #[error("the engine queue is full")]
    Full,
    #[error("the engine has stopped")]
    Stopped,
}

/// The front end's side of the channel.
#[derive(Clone)]
pub struct Handle {
    pub info: Arc<EngineInfo>,
    pub load: Arc<Load>,
    tx: mpsc::SyncSender<Submission>,
    next_id: Arc<AtomicU64>,
}

/// The engine's side of the channel.
pub struct Inbox {
    pub info: Arc<EngineInfo>,
    pub load: Arc<Load>,
    pub rx: mpsc::Receiver<Submission>,
}

/// A connected handle and inbox; `queue` bounds submissions not yet taken by the engine.
pub fn channel(info: EngineInfo, queue: usize) -> (Handle, Inbox) {
    let info = Arc::new(info);
    let load = Arc::new(Load::default());
    let (tx, rx) = mpsc::sync_channel(queue);
    let handle = Handle { info: info.clone(), load: load.clone(), tx, next_id: Arc::new(AtomicU64::new(1)) };
    (handle, Inbox { info, load, rx })
}

impl Handle {
    /// Queues a request without blocking; the receiver yields its audio then exactly one `Done`.
    pub fn submit(&self, mut speech: Speech) -> Result<UnboundedReceiver<Event>, Rejected> {
        speech.id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sink, events) = unbounded_channel();
        match self.tx.try_send(Submission { speech, sink }) {
            Ok(()) => Ok(events),
            Err(mpsc::TrySendError::Full(_)) => Err(Rejected::Full),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(Rejected::Stopped),
        }
    }
}
