//! OpenAI's GPT-Live session events on the primary WebSocket, the subset a
//! full-duplex engine serves; the official `openai` SDK's `types.live` is the
//! schema.
//!
//! A socket carries one session. The client opens it with `session.start`
//! (`model` required; `audio.format` picks the wire audio, see
//! [`crate::live_audio`]), streams the caller's audio with
//! `session.input_audio.append` (base64, never empty, any chunk size, never
//! acknowledged), may mute and unmute, and ends it with `session.close`. The
//! server answers `session.started` with the resolved session once the engine
//! has taken the prompt, then streams `session.output_audio.delta` and
//! `session.output_transcript.delta` on the agent's timeline (`start_ms` /
//! `end_ms` into the agent's audio, which is one contiguous stream from
//! `session.started`; there is no done event), reports
//! `session.usage.updated`, and ends with `session.closed`, which repeats the
//! session snapshot.
//!
//! Client events may carry `event_id`. The server event a command causes
//! echoes it as `client_event_id` (`session.started`, the mute
//! acknowledgements, a requested `session.closed`), and an `error` as
//! `error.client_event_id`.
//!
//! What the engine cannot honour is refused by name, never ignored: the
//! context-injection events (`session.instructions.append`, `.thinking.`,
//! `.commentary.`), responses, Responses delegation and initial `input` items
//! need a model that takes text, `session.update` one whose prompt can change
//! after it started, `store` a server that keeps sessions, and `client` a
//! WebRTC transport. Client delegation is accepted: the model never delegates,
//! so there is nothing for the client to handle. Unknown fields are errors, so
//! a typo never becomes a default.
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

use crate::live_audio::Format;
use crate::live_audio::G711_RATE;

#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    Start { model: String, draft: SessionDraft, format: Format },
    Append(Bytes),
    Mute,
    Unmute,
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Received {
    /// The client's `event_id`, echoed as `client_event_id`.
    pub event_id: Option<String>,
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

    fn unsupported(param: &str, message: &str) -> Self {
        Self::invalid("unsupported", Some(param), message.to_string())
    }
}

const UNSUPPORTED: [&str; 6] = [
    "session.update",
    "session.instructions.append",
    "session.thinking.append",
    "session.commentary.append",
    "response.item.create",
    "response.create",
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

/// `obj[key]`, with `null` read as absent.
fn field<'a>(obj: &'a Object, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| !v.is_null())
}

fn required<'a>(obj: &'a Object, key: &str, at: &str) -> Result<&'a Value, LiveError> {
    field(obj, key).ok_or_else(|| {
        let param = format!("{at}{key}");
        LiveError::invalid("missing_required_parameter", Some(&param), format!("`{param}` is required"))
    })
}

fn invalid_type(param: &str, expected: &str, got: &Value) -> LiveError {
    LiveError::invalid("invalid_type", Some(param), format!("`{param}` must be {expected}, got {got}"))
}

fn object<'a>(v: &'a Value, at: &str) -> Result<&'a Object, LiveError> {
    v.as_object().ok_or_else(|| invalid_type(at, "an object", v))
}

fn string(obj: &Object, key: &str, at: &str) -> Result<Option<String>, LiveError> {
    match field(obj, key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(invalid_type(&format!("{at}{key}"), "a string", v)),
    }
}

fn audio_format(v: &Value) -> Result<Format, LiveError> {
    let f = object(v, "session.audio.format")?;
    only(f, &["type", "rate"], "session.audio.format.")?;
    let kind = required(f, "type", "session.audio.format.")?;
    let rate = required(f, "rate", "session.audio.format.")?;
    let rate = rate.as_u64().ok_or_else(|| invalid_type("session.audio.format.rate", "an integer", rate))?;
    let bad_rate = |expected: &str| {
        let message = format!("{kind} is served at {expected} Hz, got {rate}");
        LiveError::invalid("invalid_value", Some("session.audio.format.rate"), message)
    };
    match (kind.as_str(), rate) {
        (Some("audio/pcm"), 16_000 | 24_000) => Ok(Format::Pcm(rate as u32)),
        (Some("audio/pcm"), _) => Err(bad_rate("16000 or 24000")),
        (Some("audio/pcmu" | "audio/pcma"), r) if r != G711_RATE as u64 => Err(bad_rate("8000")),
        (Some("audio/pcmu"), _) => Ok(Format::Pcmu),
        (Some("audio/pcma"), _) => Ok(Format::Pcma),
        _ => {
            let message = format!("expected \"audio/pcm\", \"audio/pcmu\" or \"audio/pcma\", got {kind}");
            Err(LiveError::invalid("invalid_value", Some("session.audio.format.type"), message))
        }
    }
}

/// `session.audio`: the wire format and the voice.
fn audio(v: &Value) -> Result<(Format, Option<String>), LiveError> {
    let audio = object(v, "session.audio")?;
    only(audio, &["format", "output"], "session.audio.")?;
    let format = field(audio, "format").map(audio_format).transpose()?.unwrap_or(Format::DEFAULT);
    let Some(output) = field(audio, "output") else { return Ok((format, None)) };
    let output = object(output, "session.audio.output")?;
    only(output, &["voice"], "session.audio.output.")?;
    let voice = match field(output, "voice") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(custom)) => {
            only(custom, &["id"], "session.audio.output.voice.")?;
            let id = required(custom, "id", "session.audio.output.voice.")?;
            Some(id.as_str().ok_or_else(|| invalid_type("session.audio.output.voice.id", "a string", id))?.to_string())
        }
        Some(v) => return Err(invalid_type("session.audio.output.voice", "a voice name or {\"id\": ...}", v)),
    };
    Ok((format, voice))
}

/// `session.delegation`: `{"type": "client"}` (like `null`) leaves delegation
/// to the client, which never hears of one because the model never delegates.
fn delegation(v: &Value) -> Result<(), LiveError> {
    let d = object(v, "session.delegation")?;
    match field(d, "type").and_then(Value::as_str) {
        Some("client") => only(d, &["type"], "session.delegation."),
        Some("responses") => Err(LiveError::unsupported(
            "session.delegation",
            "Responses delegation is not supported: the model cannot delegate",
        )),
        _ => {
            let got = field(d, "type").unwrap_or(&Value::Null);
            let message = format!("expected `type` \"client\" or \"responses\", got {got}");
            Err(LiveError::invalid("invalid_value", Some("session.delegation.type"), message))
        }
    }
}

fn start(event: &Object) -> Result<ClientEvent, LiveError> {
    only(event, &["type", "event_id", "session"], "")?;
    let session = object(required(event, "session", "")?, "session")?;
    only(session, &["model", "instructions", "audio", "delegation", "input", "store", "client"], "session.")?;
    let model = string(session, "model", "session.")?.ok_or_else(|| {
        LiveError::invalid("missing_required_parameter", Some("session.model"), "`session.model` is required".into())
    })?;
    if let Some(d) = field(session, "delegation") {
        delegation(d)?;
    }
    match field(session, "input") {
        None => {}
        Some(Value::Array(items)) if items.is_empty() => {}
        Some(Value::Array(_)) => {
            return Err(LiveError::unsupported(
                "session.input",
                "initial `input` items are not supported: the model takes no text history",
            ));
        }
        Some(v) => return Err(invalid_type("session.input", "an array", v)),
    }
    match field(session, "store") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            return Err(LiveError::unsupported(
                "session.store",
                "stored sessions are not supported: nothing is kept after a session closes",
            ));
        }
        Some(v) => return Err(invalid_type("session.store", "a boolean", v)),
    }
    if field(session, "client").is_some() {
        return Err(LiveError::unsupported(
            "session.client",
            "`session.client` configures a WebRTC data channel; this server serves the WebSocket",
        ));
    }
    let (format, voice) = field(session, "audio").map(audio).transpose()?.unwrap_or((Format::DEFAULT, None));
    let instructions = string(session, "instructions", "session.")?.filter(|s| !s.trim().is_empty());
    Ok(ClientEvent::Start { model, draft: SessionDraft { voice, instructions }, format })
}

fn append(event: &Object) -> Result<ClientEvent, LiveError> {
    only(event, &["type", "event_id", "audio"], "")?;
    let b64 = string(event, "audio", "")?
        .ok_or_else(|| LiveError::invalid("missing_required_parameter", Some("audio"), "`audio` is required".into()))?;
    let audio = STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| LiveError::invalid("invalid_audio", Some("audio"), format!("`audio` is not base64: {e}")))?;
    if audio.is_empty() {
        return Err(LiveError::invalid("invalid_audio", Some("audio"), "`audio` must not be empty".into()));
    }
    Ok(ClientEvent::Append(Bytes::from(audio)))
}

/// Parses one text frame.
pub fn parse(text: &str) -> Result<Received, LiveError> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| LiveError::invalid("invalid_json", None, format!("the event is not JSON: {e}")))?;
    let event = object(&v, "event")?;
    let id = event.get("event_id").and_then(Value::as_str).map(String::from);
    let parsed = (|| {
        string(event, "event_id", "")?;
        let kind = string(event, "type", "")?.ok_or_else(|| {
            LiveError::invalid("missing_required_parameter", Some("type"), "`type` is required".into())
        })?;
        let bare = |e| only(event, &["type", "event_id"], "").map(|()| e);
        match kind.as_str() {
            "session.start" => start(event),
            "session.input_audio.append" => append(event),
            "session.input_audio.mute" => bare(ClientEvent::Mute),
            "session.input_audio.unmute" => bare(ClientEvent::Unmute),
            "session.close" => bare(ClientEvent::Close),
            k if UNSUPPORTED.contains(&k) => {
                Err(LiveError::unsupported("type", &format!("`{k}` is not supported by this server")))
            }
            k => Err(LiveError::invalid("unknown_event", Some("type"), format!("unknown event type `{k}`"))),
        }
    })();
    parsed.map(|event| Received { event_id: id.clone(), event }).map_err(|e| e.answering(id))
}

/// `event` with `client_event_id`, when the command that caused it had an `event_id`.
fn answering(mut event: Value, id: Option<String>) -> Value {
    if let Some(id) = id {
        event["client_event_id"] = Value::String(id);
    }
    event
}

/// The session as `session.started` and `session.closed` report it (the
/// SDK's `SessionResource`); it is fixed at startup, so both carry the same.
pub fn resource(info: &LiveInfo, id: u64, session: &Session, format: Format, expires_at: u64) -> Value {
    json!({
        "id": format!("sess_{id}"),
        "status": "active",
        "expires_at": expires_at,
        "model": info.model,
        "instructions": session.instructions,
        "audio": {"format": format.to_json(), "output": {"voice": session.voice}},
        "delegation": {"type": "client"},
    })
}

pub fn started_event(resource: &Value, id: Option<String>) -> Value {
    answering(json!({"type": "session.started", "session": resource}), id)
}

pub fn audio_event(audio: &[u8], start_ms: u64, end_ms: u64) -> Value {
    json!({"type": "session.output_audio.delta", "delta": STANDARD.encode(audio), "start_ms": start_ms, "end_ms": end_ms})
}

pub fn transcript_event(delta: &str, start_ms: u64, end_ms: u64) -> Value {
    json!({"type": "session.output_transcript.delta", "delta": delta, "start_ms": start_ms, "end_ms": end_ms})
}

pub fn mute_event(muted: bool, id: Option<String>) -> Value {
    let kind = if muted { "session.input_audio.muted" } else { "session.input_audio.unmuted" };
    answering(json!({"type": kind}), id)
}

fn usage(seconds: f64) -> Value {
    json!({"seconds": (seconds * 100.0).round() / 100.0})
}

/// Without `context_window`: the engines' context is a ring that never fills,
/// so no usage ratio would mean what the field promises.
pub fn usage_event(seconds: f64) -> Value {
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

pub fn closed_event(resource: &Value, reason: ClosedReason, seconds: f64, id: Option<String>) -> Value {
    let event =
        json!({"type": "session.closed", "session": resource, "reason": reason.as_str(), "usage": usage(seconds)});
    answering(event, id)
}

pub fn error_event(e: &LiveError) -> Value {
    let mut error = json!({"type": e.kind, "code": e.code, "message": e.message});
    if let Some(p) = &e.param {
        error["param"] = json!(p);
    }
    if let Some(id) = &e.client_event_id {
        error["client_event_id"] = json!(id);
    }
    json!({"type": "error", "error": error})
}

/// `event` with its `event_id`, the `n`th the socket sent.
pub fn stamp(mut event: Value, n: u64) -> Value {
    event["event_id"] = Value::String(format!("event_{n}"));
    event
}
