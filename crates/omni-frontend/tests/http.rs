use axum::Router;
use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use base64::Engine as _;
use http_body_util::BodyExt;
use omni_frontend::audio::WAV_HEADER_LEN;
use omni_sim::Profile;
use serde_json::Value;
use serde_json::json;
use tower::ServiceExt;

fn app() -> Router {
    let profile = Profile::default();
    let (handle, inbox) = omni_engine::channel(profile.info("sim"), 64);
    omni_sim::spawn(inbox, profile);
    omni_frontend::router(handle, None)
}

async fn post(app: &Router, body: Value) -> (StatusCode, Option<String>, Vec<u8>) {
    let req = Request::post("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let mime = resp.headers().get("content-type").map(|v| v.to_str().unwrap().to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, mime, bytes)
}

fn error(bytes: &[u8]) -> (String, Option<String>) {
    let v: Value = serde_json::from_slice(bytes).unwrap();
    (v["error"]["type"].as_str().unwrap().to_string(), v["error"]["param"].as_str().map(String::from))
}

#[tokio::test]
async fn wav_is_a_streaming_header_then_exactly_the_planned_audio() {
    let app = app();
    let body = json!({ "model": "sim", "input": "hello", "voice": "alloy", "extra": { "frames": 7 } });
    let (status, mime, bytes) = post(&app, body).await;
    assert_eq!((status, mime.as_deref(), bytes.len()), (StatusCode::OK, Some("audio/wav"), WAV_HEADER_LEN + 7 * 3840));
    let header = &bytes[..WAV_HEADER_LEN];
    let u32_at = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    assert_eq!((&header[0..4], &header[8..12], u32_at(24), u32_at(40)), (&b"RIFF"[..], &b"WAVE"[..], 24_000, u32::MAX));
}

#[tokio::test]
async fn pcm_accepts_a_voice_object_and_scales_with_speed() {
    let app = app();
    let body = json!({ "model": "sim", "input": "a".repeat(10), "voice": { "id": "nova" }, "response_format": "pcm", "speed": 2.0 });
    let (status, mime, bytes) = post(&app, body).await;
    assert_eq!((status, mime.as_deref(), bytes.len()), (StatusCode::OK, Some("audio/pcm"), 4 * 3840));
}

#[tokio::test]
async fn sse_carries_base64_deltas_then_usage() {
    let app = app();
    let body = json!({ "model": "sim", "input": "hello", "voice": "alloy", "response_format": "pcm",
                       "stream_format": "sse", "extra": { "frames": 6 } });
    let (status, mime, bytes) = post(&app, body).await;
    assert_eq!((status, mime.as_deref()), (StatusCode::OK, Some("text/event-stream")));
    let events: Vec<Value> = String::from_utf8(bytes)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let deltas: Vec<usize> = events
        .iter()
        .filter(|e| e["type"] == "speech.audio.delta")
        .map(|e| base64::engine::general_purpose::STANDARD.decode(e["audio"].as_str().unwrap()).unwrap().len())
        .collect();
    let last = events.last().unwrap();
    assert_eq!(deltas, vec![3840, 4 * 3840, 3840]);
    assert_eq!((last["type"].as_str(), last["usage"]["output_tokens"].as_u64()), (Some("speech.audio.done"), Some(6)));
}

#[tokio::test]
async fn bad_requests_name_the_field_at_fault() {
    let app = app();
    let base = json!({ "model": "sim", "input": "hi", "voice": "alloy" });
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    let cases = [
        (with("voice", json!("nobody")), StatusCode::BAD_REQUEST, Some("voice")),
        (with("speed", json!(9.0)), StatusCode::BAD_REQUEST, Some("speed")),
        (with("input", json!("")), StatusCode::BAD_REQUEST, Some("input")),
        (with("input", json!("x".repeat(4097))), StatusCode::BAD_REQUEST, Some("input")),
        (with("response_format", json!("mp3")), StatusCode::BAD_REQUEST, Some("response_format")),
        (with("stream_format", json!("chunks")), StatusCode::BAD_REQUEST, Some("stream_format")),
        (with("extra", json!({ "seed": 1 })), StatusCode::BAD_REQUEST, Some("extra")),
        (with("temperature", json!(0.5)), StatusCode::BAD_REQUEST, None),
        (with("model", json!("other")), StatusCode::NOT_FOUND, Some("model")),
    ];
    for (body, status, param) in cases {
        let (got, _, bytes) = post(&app, body.clone()).await;
        assert_eq!(
            (got, error(&bytes)),
            (status, ("invalid_request_error".to_string(), param.map(String::from))),
            "{body}"
        );
    }
}

#[tokio::test]
async fn a_full_queue_is_429_not_a_wait() {
    let profile = Profile::default();
    let (handle, _inbox) = omni_engine::channel(profile.info("sim"), 1);
    let app = omni_frontend::router(handle, None);
    let body = json!({ "model": "sim", "input": "hi", "voice": "alloy" });
    let req = || {
        Request::post("/v1/audio/speech").header("content-type", "application/json").body(Body::from(body.to_string()))
    };
    let first = app.clone().oneshot(req().unwrap()).await.unwrap();
    let second = app.clone().oneshot(req().unwrap()).await.unwrap();
    assert_eq!((first.status(), second.status()), (StatusCode::OK, StatusCode::TOO_MANY_REQUESTS));
}

#[tokio::test]
async fn models_and_voices_describe_the_engine() {
    let app = app();
    let get = |uri: &'static str| {
        let app = app.clone();
        async move {
            let resp = app.oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
            serde_json::from_slice::<Value>(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
        }
    };
    let (models, voices) = (get("/v1/models").await, get("/v1/audio/voices").await);
    assert_eq!((models["data"][0]["id"].as_str(), voices["voices"].as_array().map(Vec::len)), (Some("sim"), Some(11)));
}
