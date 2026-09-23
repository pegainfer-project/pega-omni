//! OpenAI's `POST /v1/images/generations` over an image engine.
//!
//! The response is OpenAI's non-streaming one: every picture as base64 PNG in
//! `data[].b64_json`, sent once the last picture is done. What this server does
//! not produce is refused by name rather than substituted: `response_format:
//! url` (nothing is stored to link to), `output_format` other than `png`, and
//! `stream: true`. Model-specific options live in one `extra` object that the
//! engine declares; any other unknown field is an error.

use std::collections::BTreeMap;
use std::time::Instant;
use std::time::SystemTime;

use base64::Engine as _;
use omni_engine::Finish;
use omni_engine::Rejected;
use omni_engine::image::Draft;
use omni_engine::image::Event;
use omni_engine::image::Generation;
use omni_engine::image::Handle;
use omni_engine::image::ImageInfo;
use omni_engine::image::Rgb;
use omni_engine::image::Size;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;

use crate::protocol::ApiError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    model: String,
    prompt: String,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    /// OpenAI's end-user identifier; accepted and not used.
    #[serde(default, rename = "user")]
    _user: Option<String>,
    #[serde(default)]
    extra: BTreeMap<String, Value>,
}

/// Parses a request body against what the engine serves.
pub fn parse(bytes: &[u8], info: &ImageInfo) -> Result<Generation, ApiError> {
    let body: Body = serde_json::from_slice(bytes).map_err(|e| ApiError::invalid(None, e.to_string()))?;
    if body.model != info.model {
        return Err(ApiError::model_not_found(&body.model, &info.model));
    }
    match body.response_format.as_deref() {
        None | Some("b64_json") => {}
        Some("url") => {
            return Err(ApiError::invalid(
                Some("response_format"),
                "`url` is not served: nothing is stored to link to; use b64_json".into(),
            ));
        }
        Some(f) => return Err(ApiError::invalid(Some("response_format"), format!("unknown format `{f}`"))),
    }
    match body.output_format.as_deref() {
        None | Some("png") => {}
        Some(f) => {
            return Err(ApiError::invalid(
                Some("output_format"),
                format!("`{f}` is not encoded by this server; use png"),
            ));
        }
    }
    if body.stream == Some(true) {
        return Err(ApiError::invalid(Some("stream"), "streaming image generation is not served".into()));
    }
    let size = match body.size.as_deref() {
        None | Some("auto") => None,
        Some(s) => Some(s.parse::<Size>().map_err(|m| ApiError::invalid(Some("size"), m))?),
    };
    let draft = Draft { prompt: body.prompt, size, n: body.n, extra: body.extra };
    Ok(info.check(draft)?)
}

/// Runs one request to completion and answers with every picture.
pub async fn generate(engine: &Handle, body: &[u8]) -> Result<axum::Json<Value>, ApiError> {
    let arrived = Instant::now();
    let generation = parse(body, &engine.info).inspect_err(|_| count("invalid"))?;
    let size = generation.size;
    let mut events = engine.submit(generation).map_err(|e| {
        count("rejected");
        match e {
            Rejected::Full => ApiError::overloaded(e.to_string()),
            Rejected::Stopped => ApiError::unavailable(e.to_string()),
        }
    })?;
    let mut pictures = Vec::new();
    let finish = loop {
        match events.recv().await {
            Some(Event::Image(rgb)) => pictures.push(rgb),
            Some(Event::Done(finish)) => break finish,
            None => break Finish::Aborted,
        }
    };
    if finish != Finish::Complete {
        count("aborted");
        return Err(ApiError::internal("the engine gave up on the request".into()));
    }
    let data = tokio::task::spawn_blocking(move || pictures.iter().map(encode).collect::<Result<Vec<_>, _>>())
        .await
        .map_err(|e| ApiError::internal(format!("png encoding panicked: {e}")))?
        .map_err(|e| ApiError::internal(format!("png encoding failed: {e}")))?;
    count("complete");
    metrics::histogram!("omni_e2e_seconds").record(arrived.elapsed().as_secs_f64());
    metrics::counter!("omni_images_total").increment(data.len() as u64);
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let data: Vec<Value> = data.into_iter().map(|b64| json!({ "b64_json": b64 })).collect();
    Ok(axum::Json(json!({ "created": created, "data": data, "output_format": "png", "size": size.to_string() })))
}

fn count(outcome: &'static str) {
    metrics::counter!("omni_requests_total", "outcome" => outcome).increment(1);
}

/// Base64 of the picture as an 8-bit RGB PNG.
fn encode(rgb: &Rgb) -> Result<String, png::EncodingError> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, rgb.size.width, rgb.size.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.write_header()?.write_image_data(&rgb.pixels)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(out))
}
