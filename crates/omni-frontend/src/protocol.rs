//! OpenAI's `POST /v1/audio/speech` request, parsed into a [`Speech`] plus how to deliver it.
//!
//! The wire format follows OpenAI's API reference. Two deliberate differences:
//! `response_format` defaults to `wav` (not `mp3`) and only `wav` / `pcm` are
//! encoded today; the others are refused by name rather than silently
//! substituted. Engines can opt into the shared reference audio fields.
//! Model-specific options live in `extra`. Unknown top-level fields are an error.

use std::borrow::Cow;
use std::collections::BTreeMap;

use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use omni_engine::Draft;
use omni_engine::EngineInfo;
use omni_engine::Invalid;
use omni_engine::Speech;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;

use crate::reference_audio;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Pcm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The encoded audio as the response body, streamed as it is generated.
    Audio,
    /// `speech.audio.delta` / `speech.audio.done` server-sent events.
    Sse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delivery {
    pub format: Format,
    pub framing: Framing,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Body<'a> {
    model: String,
    input: String,
    voice: Voice,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    stream_format: Option<String>,
    /// Not OpenAI's: vLLM's speech clients (`vllm bench serve`) always send it.
    /// Every response streams, and a streamed body read whole is the same audio,
    /// so either value is accepted.
    #[serde(default, rename = "stream")]
    _stream: Option<bool>,
    #[serde(default)]
    extra: BTreeMap<String, Value>,
    #[serde(default, borrow, deserialize_with = "present")]
    ref_audio: Option<&'a RawValue>,
    #[serde(default, borrow, deserialize_with = "present")]
    ref_text: Option<&'a RawValue>,
    #[serde(default, borrow, deserialize_with = "present")]
    x_vector_only_mode: Option<&'a RawValue>,
}

/// Keep an explicit JSON null distinct from omission so type errors name the field.
fn present<'de, D: Deserializer<'de>>(de: D) -> Result<Option<&'de RawValue>, D::Error> {
    <&RawValue>::deserialize(de).map(Some)
}

/// A raw token identifies the JSON type while keeping container contents in the request buffer.
fn reference_field<'de, T: Deserialize<'de>>(
    value: Option<&'de RawValue>,
    field: &'static str,
    expected: &str,
) -> Result<Option<T>, ApiError> {
    let Some(value) = value else { return Ok(None) };
    let kind = match value.get().as_bytes()[0] {
        b'n' => "null",
        b't' | b'f' => "a boolean",
        b'"' => "a string",
        b'[' => "an array",
        b'{' => "an object",
        _ => "a number",
    };
    if kind != expected {
        return Err(ApiError::invalid(Some(field), format!("expected {expected}, got {kind}")));
    }
    T::deserialize(value).map(Some).map_err(|e| ApiError::invalid(Some(field), e.to_string()))
}

/// Ordinary data URIs borrow the request bytes. JSON escapes need an owned string.
#[derive(Deserialize)]
struct AudioUri<'a>(#[serde(borrow)] Cow<'a, str>);

#[derive(Deserialize)]
#[serde(untagged)]
enum Voice {
    Name(String),
    Object { id: String },
}

/// An OpenAI-shaped error response.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub param: Option<&'static str>,
    pub message: String,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl ApiError {
    pub fn invalid(param: Option<&'static str>, message: String) -> Self {
        Self { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", code: None, param, message }
    }

    pub fn overloaded(message: String) -> Self {
        let status = StatusCode::TOO_MANY_REQUESTS;
        Self { status, kind: "server_error", code: Some("engine_overloaded"), param: None, message }
    }

    pub fn unavailable(message: String) -> Self {
        Self { status: StatusCode::SERVICE_UNAVAILABLE, kind: "server_error", code: None, param: None, message }
    }

    pub fn internal(message: String) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, kind: "server_error", code: None, param: None, message }
    }

    /// A request for a model this server does not serve.
    pub fn model_not_found(asked: &str, served: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: Some("model_not_found"),
            param: Some("model"),
            message: format!("model `{asked}` is not served here; this server serves `{served}`"),
        }
    }
}

impl From<Invalid> for ApiError {
    fn from(e: Invalid) -> Self {
        Self::invalid(Some(e.param), e.message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let detail = ErrorDetail { message: &self.message, kind: self.kind, param: self.param, code: self.code };
        (self.status, Json(ErrorBody { error: detail })).into_response()
    }
}

fn format(name: Option<&str>) -> Result<Format, ApiError> {
    match name.unwrap_or("wav") {
        "wav" => Ok(Format::Wav),
        "pcm" => Ok(Format::Pcm),
        f @ ("mp3" | "opus" | "aac" | "flac") => Err(ApiError::invalid(
            Some("response_format"),
            format!("`{f}` is not encoded by this server; use wav or pcm"),
        )),
        f => Err(ApiError::invalid(Some("response_format"), format!("unknown format `{f}`"))),
    }
}

fn framing(name: Option<&str>) -> Result<Framing, ApiError> {
    match name.unwrap_or("audio") {
        "audio" => Ok(Framing::Audio),
        "sse" => Ok(Framing::Sse),
        f => Err(ApiError::invalid(Some("stream_format"), format!("unknown stream format `{f}`; use audio or sse"))),
    }
}

/// Parses a request body against what the engine serves.
pub fn parse(bytes: &[u8], info: &EngineInfo) -> Result<(Speech, Delivery), ApiError> {
    let body: Body = serde_json::from_slice(bytes).map_err(|e| ApiError::invalid(None, e.to_string()))?;
    if body.model != info.model {
        return Err(ApiError::model_not_found(&body.model, &info.model));
    }
    let delivery =
        Delivery { format: format(body.response_format.as_deref())?, framing: framing(body.stream_format.as_deref())? };
    let voice = match body.voice {
        Voice::Name(v) | Voice::Object { id: v } => v,
    };
    let uri = reference_field::<AudioUri>(body.ref_audio, "ref_audio", "a string")?.map(|uri| uri.0);
    let ref_text = reference_field(body.ref_text, "ref_text", "a string")?;
    let x_vector_only_mode = reference_field(body.x_vector_only_mode, "x_vector_only_mode", "a boolean")?;
    let draft = Draft {
        input: body.input,
        voice,
        instructions: body.instructions,
        speed: body.speed,
        extra: body.extra,
        ref_audio: uri.as_deref(),
        ref_text,
        x_vector_only_mode,
    };
    let speech = info.check_with_audio(draft, reference_audio::decode)?;
    Ok((speech, delivery))
}
