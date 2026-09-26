//! OpenAI's GPT-Live session events, the subset a full-duplex engine serves.
//!
//! A socket carries one session. The client opens it with `session.start`,
//! streams the caller's audio with `session.input_audio.append` (base64 s16le
//! mono at the engine's rate, any chunk size, never acknowledged), may mute and
//! unmute, and ends it with `session.close`. The server answers
//! `session.started` once the engine has taken the prompt, then streams
//! `session.output_audio.delta` and `session.output_transcript.delta` on the
//! agent's timeline (`start_ms` / `end_ms` into the agent's audio, which is
//! one contiguous stream from `session.started`; there is no done event),
//! reports `session.usage.updated`, and ends with `session.closed`.
//!
//! What the engine cannot honour is refused by name, never ignored: the
//! context-injection events (`session.instructions.append`, `.thinking.`,
//! `.commentary.`), responses and delegation need a model that takes text
//! mid-session, and `session.update` one whose prompt can change after it
//! started. Unknown fields are errors, so a typo never becomes a default.
//!
//! [`parse`] turns a client frame into a [`Received`]; the `*_event`
//! functions build server events without their `event_id`, which the socket
//! stamps in order.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use omni_engine::Invalid;
use omni_engine::live::LiveInfo;
use omni_engine::live::Session;
use omni_engine::live::SessionDraft;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    Start { model: Option<String>, draft: SessionDraft },
    Append(Bytes),
    Mute,
    Unmute,
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Received {
    pub client_event_id: Option<String>,
    pub event: ClientEvent,
}

/// An `error` event's payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveError {
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub client_event_id: Option<String>,
}

impl LiveError {
    pub fn invalid(code: &'static str, param: Option<&str>, message: String) -> Self {
        Self { kind: "invalid_request_error", code, message, param: param.map(String::from), client_event_id: None }
    }

    pub fn server(code: &'static str, message: String) -> Self {
        Self { kind: "server_error", code, message, param: None, client_event_id: None }
    }

    pub fn answering(self, id: Option<String>) -> Self {
        Self { client_event_id: id, ..self }
    }

    /// A session field the engine refused, named from the event's root.
    pub fn session(e: Invalid) -> Self {
        Self::invalid("invalid_value", Some(&format!("session.{}", e.param)), e.message)
    }
}

const UNSUPPORTED: [&str; 7] = [
    "session.update",
    "session.instructions.append",
    "session.thinking.append",
    "session.commentary.append",
    "response.item.create",
    "response.create",
    "session.delegation.append",
];

type Object = Map<String, Value>;

/// Errors unless `obj` has only `allowed` keys; `at` is its path.
fn only(obj: &Object, allowed: &[&str], at: &str) -> Result<(), LiveError> {
    match obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        None => Ok(()),
        Some(k) => {
            let param = format!("{at}{k}");
            Err(LiveError::invalid("unknown_parameter", Some(&param), format!("unknown parameter `{param}`")))
        }
    }
}

fn object<'a>(v: &'a Value, at: &str) -> Result<&'a Object, LiveError> {
    v.as_object().ok_or_else(|| LiveError::invalid("invalid_type", Some(at), format!("`{at}` must be an object")))
}

fn string(obj: &Object, key: &str, at: &str) -> Result<Option<String>, LiveError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => {
            let param = format!("{at}{key}");
            Err(LiveError::invalid("invalid_type", Some(&param), format!("`{param}` must be a string")))
        }
    }
}

/// `{"type": "audio/pcm", "rate": R}` with the engine's rate, if present.
fn format(obj: &Object, at: &str, rate: u32) -> Result<(), LiveError> {
    let Some(f) = obj.get("format") else { return Ok(()) };
    let param = format!("{at}format");
    let f = object(f, &param)?;
    only(f, &["type", "rate"], &format!("{param}."))?;
    let kind = f.get("type").and_then(Value::as_str);
    let got_rate = f.get("rate").map(|r| r.as_u64());
    if kind != Some("audio/pcm") || got_rate.is_some_and(|r| r != Some(rate as u64)) {
        let message =
            format!("only {{\"type\": \"audio/pcm\", \"rate\": {rate}}} is served, got {}", Value::Object(f.clone()));
        return Err(LiveError::invalid("invalid_value", Some(&param), message));
    }
    Ok(())
}

fn start(event: &Object, rate: u32) -> Result<ClientEvent, LiveError> {
    only(event, &["type", "client_event_id", "session"], "")?;
    let empty = Value::Object(Object::new());
    let session = object(event.get("session").unwrap_or(&empty), "session")?;
    if session.contains_key("delegation") {
        let message = "delegation is not supported by this server".to_string();
        return Err(LiveError::invalid("unsupported", Some("session.delegation"), message));
    }
    only(session, &["model", "instructions", "audio"], "session.")?;
    let mut voice = None;
    if let Some(audio) = session.get("audio") {
        let audio = object(audio, "session.audio")?;
        only(audio, &["input", "output"], "session.audio.")?;
        if let Some(input) = audio.get("input") {
            let input = object(input, "session.audio.input")?;
            only(input, &["format"], "session.audio.input.")?;
            format(input, "session.audio.input.", rate)?;
        }
        if let Some(output) = audio.get("output") {
            let output = object(output, "session.audio.output")?;
            only(output, &["format", "voice"], "session.audio.output.")?;
            format(output, "session.audio.output.", rate)?;
            voice = string(output, "voice", "session.audio.output.")?;
        }
    }
    Ok(ClientEvent::Start {
        model: string(session, "model", "session.")?,
        draft: SessionDraft { voice, instructions: string(session, "instructions", "session.")? },
    })
}

/// Parses one text frame; `rate` is the engine's sample rate.
pub fn parse(text: &str, rate: u32) -> Result<Received, LiveError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| LiveError::invalid("invalid_json", None, format!("the event is not JSON: {e}")))?;
    let event = object(&v, "event")?;
    let id = event.get("client_event_id").and_then(Value::as_str).map(String::from);
    let parsed = (|| {
        let kind = event.get("type").and_then(Value::as_str).ok_or_else(|| {
            LiveError::invalid("missing_required_parameter", Some("type"), "`type` is required".into())
        })?;
        let bare = |e| only(event, &["type", "client_event_id"], "").map(|()| e);
        match kind {
            "session.start" => start(event, rate),
            "session.input_audio.append" => {
                only(event, &["type", "client_event_id", "audio"], "")?;
                let b64 = string(event, "audio", "")?.ok_or_else(|| {
                    LiveError::invalid("missing_required_parameter", Some("audio"), "`audio` is required".into())
                })?;
                let pcm = STANDARD.decode(b64.as_bytes()).map_err(|e| {
                    LiveError::invalid("invalid_value", Some("audio"), format!("`audio` is not base64: {e}"))
                })?;
                Ok(ClientEvent::Append(Bytes::from(pcm)))
            }
            "session.input_audio.mute" => bare(ClientEvent::Mute),
            "session.input_audio.unmute" => bare(ClientEvent::Unmute),
            "session.close" => bare(ClientEvent::Close),
            k if UNSUPPORTED.contains(&k) => {
                Err(LiveError::invalid("unsupported", Some("type"), format!("`{k}` is not supported by this server")))
            }
            k => Err(LiveError::invalid("unknown_event", Some("type"), format!("unknown event type `{k}`"))),
        }
    })();
    parsed.map(|event| Received { client_event_id: id.clone(), event }).map_err(|e| e.answering(id))
}

fn session_id(id: u64) -> String {
    format!("sess_{id}")
}

fn pcm_format(rate: u32) -> Value {
    json!({"type": "audio/pcm", "rate": rate})
}

pub(crate) fn started_event(info: &LiveInfo, id: u64, session: &Session) -> Value {
    json!({
        "type": "session.started",
        "session": {
            "id": session_id(id),
            "object": "live.session",
            "model": info.model,
            "instructions": session.instructions,
            "audio": {
                "input": {"format": pcm_format(info.sample_rate)},
                "output": {"format": pcm_format(info.sample_rate), "voice": session.voice},
            },
        },
    })
}

pub(crate) fn audio_event(pcm: &[u8], start_ms: u64, end_ms: u64) -> Value {
    json!({"type": "session.output_audio.delta", "delta": STANDARD.encode(pcm), "start_ms": start_ms, "end_ms": end_ms})
}

pub(crate) fn transcript_event(delta: &str, start_ms: u64, end_ms: u64) -> Value {
    json!({"type": "session.output_transcript.delta", "delta": delta, "start_ms": start_ms, "end_ms": end_ms})
}

pub(crate) fn mute_event(muted: bool, id: Option<String>) -> Value {
    let kind = if muted { "session.input_audio.muted" } else { "session.input_audio.unmuted" };
    json!({"type": kind, "client_event_id": id})
}

fn usage(seconds: f64) -> Value {
    json!({"seconds": (seconds * 100.0).round() / 100.0})
}

pub(crate) fn usage_event(seconds: f64) -> Value {
    json!({"type": "session.usage.updated", "usage": usage(seconds)})
}

/// Why a `session.closed` was sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosedReason {
    CloseRequested,
    Expired,
    ConnectionLost,
}

impl ClosedReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CloseRequested => "close_requested",
            Self::Expired => "expired",
            Self::ConnectionLost => "connection_lost",
        }
    }
}

pub(crate) fn closed_event(reason: ClosedReason, seconds: f64) -> Value {
    json!({"type": "session.closed", "reason": reason.as_str(), "usage": usage(seconds)})
}

pub(crate) fn error_event(e: &LiveError) -> Value {
    json!({
        "type": "error",
        "error": {
            "type": e.kind,
            "code": e.code,
            "message": e.message,
            "param": e.param,
            "client_event_id": e.client_event_id,
        },
    })
}

/// `event` with its `event_id`, the `n`th the socket sent.
pub(crate) fn stamp(mut event: Value, n: u64) -> Value {
    event["event_id"] = Value::String(format!("event_{n}"));
    event
}
