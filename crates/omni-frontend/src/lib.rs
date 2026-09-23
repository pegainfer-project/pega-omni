//! The OpenAI-compatible HTTP surface of pega-omni.
//!
//! The front end knows no model. It parses a request against the engine's
//! [`EngineInfo`](omni_engine::EngineInfo), hands a checked
//! [`Speech`](omni_engine::Speech) to the engine through a
//! [`Handle`], and streams whatever audio comes back. Admission never blocks:
//! a full engine queue is an immediate `429`, so the only waiting a request
//! does is inside the engine, where it is visible as `omni_engine_waiting`.
//!
//! Routes: `POST /v1/audio/speech`, `GET /v1/models`, `GET /v1/audio/voices`,
//! `GET /health`, and `GET /metrics` (Prometheus text) when a recorder is
//! installed.
//!
//! Serve through [`serve`], not `axum::serve` directly: audio leaves in small
//! writes, and without `TCP_NODELAY` Nagle holds each one until the client's
//! delayed ACK, which adds a flat ~40 ms to every packet.

pub mod audio;
pub mod protocol;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use axum::serve::ListenerExt;
use metrics_exporter_prometheus::PrometheusHandle;
use omni_engine::Handle;
use omni_engine::Rejected;
use serde_json::json;

use crate::audio::Outcome;
use crate::protocol::ApiError;

struct Shared {
    engine: Handle,
    prometheus: Option<PrometheusHandle>,
}

/// The full route table over one engine; `prometheus` enables `/metrics`.
pub fn router(engine: Handle, prometheus: Option<PrometheusHandle>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .route("/v1/models", get(models))
        .route("/v1/audio/voices", get(voices))
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(scrape))
        .with_state(Arc::new(Shared { engine, prometheus }))
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
    let info = &s.engine.info;
    let (speech, delivery) = protocol::parse(&body, info).inspect_err(|_| {
        metrics::counter!("omni_requests_total", "outcome" => "invalid").increment(1);
    })?;
    let events = s.engine.submit(speech).map_err(|e| {
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
    Json(json!({
        "object": "list",
        "data": [{ "id": s.engine.info.model, "object": "model", "created": 0, "owned_by": "pega-omni" }],
    }))
}

async fn voices(State(s): State<Arc<Shared>>) -> Json<serde_json::Value> {
    Json(json!({ "voices": s.engine.info.voices }))
}

async fn scrape(State(s): State<Arc<Shared>>) -> Response {
    let Some(p) = &s.prometheus else {
        return ApiError::unavailable("metrics are disabled".into()).into_response();
    };
    metrics::gauge!("omni_engine_waiting").set(s.engine.load.waiting.load(Ordering::Relaxed) as f64);
    metrics::gauge!("omni_engine_running").set(s.engine.load.running.load(Ordering::Relaxed) as f64);
    p.render().into_response()
}
