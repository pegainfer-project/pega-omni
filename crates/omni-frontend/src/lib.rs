//! The OpenAI-compatible HTTP surface of pega-omni.
//!
//! The front end knows no model. It parses a request against the engine's
//! [`EngineInfo`](omni_engine::EngineInfo), hands a checked
//! [`Speech`](omni_engine::Speech) to the engine through a
//! [`Handle`], and streams whatever audio comes back. Admission never blocks:
//! a full engine queue is an immediate `429`, so the only waiting a request
//! does is inside the engine, where it is visible as `omni_engine_waiting`.
//!
//! A server fronts a speech engine or a live (full-duplex) engine
//! ([`Engines`]). Routes: `POST /v1/audio/speech` (speech), the GPT-Live
//! WebSocket `GET /v1/live/sessions` and the browser demo at `/` (live; see
//! [`live`]), `GET /v1/models`, `GET /v1/audio/voices`, `GET /health`, and
//! `GET /metrics` (Prometheus text) when a recorder is installed.
//!
//! Serve through [`serve`], not `axum::serve` directly: audio leaves in small
//! writes, and without `TCP_NODELAY` Nagle holds each one until the client's
//! delayed ACK, which adds a flat ~40 ms to every packet.

pub mod audio;
mod live;
pub mod live_audio;
pub mod live_protocol;
pub mod protocol;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::StatusCode;
use axum::response::Html;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use axum::serve::ListenerExt;
use metrics_exporter_prometheus::PrometheusHandle;
use omni_engine::Handle;
use omni_engine::Rejected;
use omni_engine::live::LiveHandle;
use serde_json::json;

use crate::audio::Outcome;
use crate::protocol::ApiError;

/// The engine a server fronts.
#[derive(Clone)]
pub enum Engines {
    Speech(Handle),
    Live(LiveHandle),
}

impl From<Handle> for Engines {
    fn from(h: Handle) -> Self {
        Self::Speech(h)
    }
}

impl From<LiveHandle> for Engines {
    fn from(h: LiveHandle) -> Self {
        Self::Live(h)
    }
}

impl Engines {
    fn speech(&self) -> Option<&Handle> {
        match self {
            Self::Speech(h) => Some(h),
            Self::Live(_) => None,
        }
    }

    fn live(&self) -> Option<&LiveHandle> {
        match self {
            Self::Live(h) => Some(h),
            Self::Speech(_) => None,
        }
    }

    fn model(&self) -> &str {
        match self {
            Self::Speech(h) => &h.info.model,
            Self::Live(h) => &h.info.model,
        }
    }

    fn voices(&self) -> &std::collections::BTreeSet<String> {
        match self {
            Self::Speech(h) => &h.info.voices,
            Self::Live(h) => &h.info.voices,
        }
    }
}

struct Shared {
    engines: Engines,
    prometheus: Option<PrometheusHandle>,
}

const DEMO: &str = include_str!("live.html");

/// The full route table over `engines`; `prometheus` enables `/metrics`.
pub fn router(engines: impl Into<Engines>, prometheus: Option<PrometheusHandle>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .route("/v1/live/sessions", get(live_socket))
        .route("/", get(demo))
        .route("/live/config", get(live_config))
        .route("/live/stats", get(live_stats))
        .route("/v1/models", get(models))
        .route("/v1/audio/voices", get(voices))
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(scrape))
        .with_state(Arc::new(Shared { engines: engines.into(), prometheus }))
}

fn not_here(what: &str) -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        kind: "invalid_request_error",
        code: Some("unsupported_endpoint"),
        param: None,
        message: format!("this server runs no {what} model"),
    }
}

fn live(s: &Shared) -> Result<&LiveHandle, ApiError> {
    s.engines.live().ok_or_else(|| not_here("live"))
}

async fn live_socket(State(s): State<Arc<Shared>>, ws: WebSocketUpgrade) -> Result<Response, ApiError> {
    let handle = live(&s)?.clone();
    Ok(ws.on_upgrade(move |socket| live::serve(socket, handle)))
}

async fn demo(State(s): State<Arc<Shared>>) -> Result<Html<&'static str>, ApiError> {
    live(&s).map(|_| Html(DEMO))
}

/// What the demo page offers: the session defaults and the voices.
async fn live_config(State(s): State<Arc<Shared>>) -> Result<Json<serde_json::Value>, ApiError> {
    let info = &live(&s)?.info;
    Ok(Json(json!({
        "model": info.model,
        "sample_rate": info.sample_rate,
        "frame_ms": info.frame_ms(1),
        "voices": info.voices,
        "voice": info.default_voice,
        "instructions": info.default_instructions,
        "max_instructions_chars": info.max_instructions_chars,
        "max_seconds": info.frame_ms(info.max_frames) / 1000,
    })))
}

/// The live engine's [`Pulse`](omni_engine::live::Pulse): a reader diffing
/// two samples gets the mean tick and the engine's duty cycle between them.
async fn live_stats(State(s): State<Arc<Shared>>) -> Result<Json<serde_json::Value>, ApiError> {
    let live = live(&s)?;
    let p = &live.pulse;
    Ok(Json(json!({
        "sessions": p.sessions.load(Ordering::Relaxed),
        "max_sessions": live.info.max_sessions,
        "frame_ms": live.info.frame_ms(1),
        "ticks": p.ticks.load(Ordering::Relaxed),
        "tick_busy_us": p.busy_us.load(Ordering::Relaxed),
        "last_tick_us": p.last_tick_us.load(Ordering::Relaxed),
        "late_ticks": p.late_ticks.load(Ordering::Relaxed),
    })))
}

/// Serves `app` on `listener` with `TCP_NODELAY` on every accepted connection until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let listener = listener.tap_io(|tcp| {
        let _ = tcp.set_nodelay(true);
    });
    axum::serve(listener, app).with_graceful_shutdown(shutdown).await
}

async fn speech(State(s): State<Arc<Shared>>, body: Bytes) -> Result<Response, ApiError> {
    let arrived = Instant::now();
    let engine = s.engines.speech().ok_or_else(|| not_here("speech"))?;
    let info = &engine.info;
    let (speech, delivery) = protocol::parse(&body, info).inspect_err(|_| {
        metrics::counter!("omni_requests_total", "outcome" => "invalid").increment(1);
    })?;
    let events = engine.submit(speech).map_err(|e| {
        metrics::counter!("omni_requests_total", "outcome" => "rejected").increment(1);
        match e {
            Rejected::Full => ApiError::overloaded(e.to_string()),
            Rejected::Stopped => ApiError::unavailable(e.to_string()),
        }
    })?;
    let report = move |o: Outcome| record(arrived, o);
    Ok(audio::respond(events, delivery, info.sample_rate, Box::new(report)))
}

fn record(arrived: Instant, o: Outcome) {
    let outcome = match o.done {
        Some(d) if d.finish == omni_engine::Finish::Complete => "complete",
        Some(_) => "aborted",
        None => "cancelled",
    };
    metrics::counter!("omni_requests_total", "outcome" => outcome).increment(1);
    if let Some(t) = o.first_packet {
        metrics::histogram!("omni_ttfp_seconds").record(t.duration_since(arrived).as_secs_f64());
    }
    if o.done.is_some() {
        metrics::histogram!("omni_e2e_seconds").record(arrived.elapsed().as_secs_f64());
    }
    metrics::counter!("omni_audio_samples_total").increment(o.samples);
}

async fn models(State(s): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let model = json!({ "id": s.engines.model(), "object": "model", "created": 0, "owned_by": "pega-omni" });
    Json(json!({ "object": "list", "data": [model] }))
}

async fn voices(State(s): State<Arc<Shared>>) -> Json<serde_json::Value> {
    Json(json!({ "voices": s.engines.voices() }))
}

async fn scrape(State(s): State<Arc<Shared>>) -> Response {
    let Some(p) = &s.prometheus else {
        return ApiError::unavailable("metrics are disabled".into()).into_response();
    };
    match &s.engines {
        Engines::Speech(h) => {
            metrics::gauge!("omni_engine_waiting").set(h.load.waiting.load(Ordering::Relaxed) as f64);
            metrics::gauge!("omni_engine_running").set(h.load.running.load(Ordering::Relaxed) as f64);
        }
        Engines::Live(h) => {
            let pulse = &h.pulse;
            metrics::gauge!("omni_engine_running").set(pulse.sessions.load(Ordering::Relaxed) as f64);
            metrics::counter!("omni_engine_ticks_total").absolute(pulse.ticks.load(Ordering::Relaxed));
            metrics::counter!("omni_engine_late_ticks_total").absolute(pulse.late_ticks.load(Ordering::Relaxed));
            metrics::counter!("omni_engine_tick_busy_microseconds_total")
                .absolute(pulse.busy_us.load(Ordering::Relaxed));
        }
    }
    p.render().into_response()
}
