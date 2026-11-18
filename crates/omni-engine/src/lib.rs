//! The contract between the HTTP front end and a speech engine. The image
//! contract, the same shape with a prompt in and pictures out, is [`image`].
//!
//! An engine is whatever sits on the far side of an [`Inbox`]: a thread (or a
//! GPU step loop) that takes [`Submission`]s, generates audio, and pushes
//! [`Event`]s into each submission's sink. There is no engine trait: the front
//! end holds a [`Handle`], the engine holds the matching [`Inbox`], and the
//! channel between them is the whole interface.
//!
//! Audio crosses the boundary as mono signed 16-bit little-endian PCM.
//! Output uses [`EngineInfo::sample_rate`]; reference input has its own rate.
//! Containers and wire framing are the front end's business.
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
use std::ops::RangeInclusive;
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

pub mod image;

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
    /// Reference conditions accepted in place of a preset voice.
    pub reference_audio: Option<ReferenceAudioInfo>,
}

#[derive(Clone, Debug)]
pub struct ReferenceAudioInfo {
    pub sample_rate: u32,
    pub duration_seconds: RangeInclusive<u32>,
    pub audio_only: bool,
    pub with_transcript: bool,
}

impl ReferenceAudioInfo {
    /// Checks whole PCM frames at a source or normalized sample rate.
    pub fn check_duration(&self, frames: usize, sample_rate: u32) -> Result<(), Invalid> {
        let bound = |seconds: u32| u64::from(seconds) * u64::from(sample_rate);
        let (min, max) = (*self.duration_seconds.start(), *self.duration_seconds.end());
        if sample_rate == 0 || !(bound(min)..=bound(max)).contains(&(frames as u64)) {
            return Err(Invalid {
                param: "ref_audio",
                message: format!("expected {min}..={max} seconds, got {frames} samples at {sample_rate} Hz"),
            });
        }
        Ok(())
    }
}

/// Mono signed 16-bit little-endian reference audio as decoded by the front end.
#[derive(Clone, Debug)]
pub struct ReferencePcm {
    pub pcm: Bytes,
    pub sample_rate: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReferenceMode {
    AudioOnly,
    WithTranscript(String),
}

/// A reference whose PCM and conditioning mode have been checked for this engine.
#[derive(Clone, Debug)]
pub struct Reference {
    pub pcm: Bytes,
    pub sample_rate: u32,
    pub mode: ReferenceMode,
}

/// The values an `extra` key accepts.
#[derive(Clone, Debug, PartialEq)]
pub enum Extra {
    Integer(std::ops::RangeInclusive<i64>),
    OneOf(BTreeSet<String>),
}

impl Extra {
    pub(crate) fn check(&self, value: &Value) -> Result<(), String> {
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
    pub reference: Option<Reference>,
}

/// The unchecked fields of a speech request, as the protocol layer parsed them.
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub input: String,
    pub voice: String,
    pub instructions: Option<String>,
    pub speed: Option<f32>,
    pub extra: BTreeMap<String, Value>,
    pub ref_audio: Option<ReferencePcm>,
    pub ref_text: Option<String>,
    pub x_vector_only_mode: Option<bool>,
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
        let reference_voice = self.reference_audio.is_some() && self.voices.is_empty();
        if !reference_voice && !self.voices.contains(&draft.voice) {
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
        let reference = self.check_reference(draft.ref_audio, draft.ref_text, draft.x_vector_only_mode)?;
        Ok(Speech {
            id: 0,
            input: draft.input,
            voice: draft.voice,
            instructions: draft.instructions,
            speed,
            extra: draft.extra,
            reference,
        })
    }

    fn check_reference(
        &self,
        audio: Option<ReferencePcm>,
        text: Option<String>,
        audio_only: Option<bool>,
    ) -> Result<Option<Reference>, Invalid> {
        let Some(info) = &self.reference_audio else {
            for (present, param) in [
                (audio.is_some(), "ref_audio"),
                (text.is_some(), "ref_text"),
                (audio_only.is_some(), "x_vector_only_mode"),
            ] {
                if present {
                    return Err(Invalid { param, message: "this model does not accept reference audio".into() });
                }
            }
            return Ok(None);
        };
        let ReferencePcm { pcm, sample_rate } =
            audio.ok_or_else(|| Invalid { param: "ref_audio", message: "reference audio is required".into() })?;
        if sample_rate != info.sample_rate || !pcm.len().is_multiple_of(2) {
            return Err(Invalid {
                param: "ref_audio",
                message: format!("expected whole mono PCM16 samples at {} Hz", info.sample_rate),
            });
        }
        info.check_duration(pcm.len() / 2, sample_rate)?;

        let audio_only = audio_only.unwrap_or(false);
        if !(if audio_only { info.audio_only } else { info.with_transcript }) {
            return Err(Invalid {
                param: "x_vector_only_mode",
                message: format!("this model does not support x_vector_only_mode={audio_only}"),
            });
        }
        let mode = if audio_only {
            if text.is_some() {
                return Err(Invalid { param: "ref_text", message: "must be omitted in audio-only mode".into() });
            }
            ReferenceMode::AudioOnly
        } else {
            let text = text.filter(|text| !text.trim().is_empty()).ok_or_else(|| Invalid {
                param: "ref_text",
                message: "a nonblank transcript is required in audio-and-transcript mode".into(),
            })?;
            ReferenceMode::WithTranscript(text)
        };
        Ok(Some(Reference { pcm, sample_rate, mode }))
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
