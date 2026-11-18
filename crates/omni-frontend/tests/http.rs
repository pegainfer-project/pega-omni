use std::convert::Infallible;
use std::sync::mpsc::TryRecvError;
use std::task::Poll;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use axum::response::Response;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::stream;
use http_body_util::BodyExt;
use omni_engine::Inbox;
use omni_engine::Reference;
use omni_engine::ReferenceAudioInfo;
use omni_engine::ReferenceMode;
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

fn speech_request(body: Body) -> Request<Body> {
    Request::post("/v1/audio/speech").header("content-type", "application/json").body(body).unwrap()
}

async fn post_response(app: &Router, body: Value) -> Response {
    app.clone().oneshot(speech_request(Body::from(body.to_string()))).await.unwrap()
}

async fn post(app: &Router, body: Value) -> (StatusCode, Option<String>, Vec<u8>) {
    let resp = post_response(app, body).await;
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
async fn vllm_bench_requests_are_served() {
    let app = app();
    let body = json!({ "model": "sim", "input": "hello", "voice": "alloy", "stream": true, "stream_format": "audio",
                       "response_format": "pcm", "extra": { "frames": 3 } });
    let (status, mime, bytes) = post(&app, body).await;
    assert_eq!((status, mime.as_deref(), bytes.len()), (StatusCode::OK, Some("audio/pcm"), 3 * 3840));
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
        .map(|e| STANDARD.decode(e["audio"].as_str().unwrap()).unwrap().len())
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
    let reference_audio = wav_uri(&reference_wav(REFERENCE_RATE, 1, &vec![0; 2 * REFERENCE_RATE as usize]));
    let cases = [
        (with("voice", json!("nobody")), StatusCode::BAD_REQUEST, Some("voice")),
        (with("speed", json!(9.0)), StatusCode::BAD_REQUEST, Some("speed")),
        (with("input", json!("")), StatusCode::BAD_REQUEST, Some("input")),
        (with("input", json!("x".repeat(4097))), StatusCode::BAD_REQUEST, Some("input")),
        (with("response_format", json!("mp3")), StatusCode::BAD_REQUEST, Some("response_format")),
        (with("stream_format", json!("chunks")), StatusCode::BAD_REQUEST, Some("stream_format")),
        (with("extra", json!({ "seed": 1 })), StatusCode::BAD_REQUEST, Some("extra")),
        (with("ref_audio", json!(reference_audio)), StatusCode::BAD_REQUEST, Some("ref_audio")),
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
    let first = post_response(&app, body.clone()).await;
    let second = post_response(&app, body).await;
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

const REFERENCE_RATE: u32 = 24_000;
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(5);

fn reference_app(audio_only: bool, with_transcript: bool) -> (Router, Inbox) {
    let mut info = Profile::default().info("sim");
    info.voices.clear();
    info.sample_rate = 16_000;
    info.reference_audio =
        Some(ReferenceAudioInfo { sample_rate: REFERENCE_RATE, duration_seconds: 1..=30, audio_only, with_transcript });
    let (handle, inbox) = omni_engine::channel(info, 1);
    (omni_frontend::router(handle, None), inbox)
}

fn pcm(samples: impl IntoIterator<Item = i16>) -> Vec<u8> {
    samples.into_iter().flat_map(i16::to_le_bytes).collect()
}

fn reference_wav(rate: u32, channels: u16, pcm: &[u8]) -> Vec<u8> {
    let frame_size = channels * 2;
    let data_len = u32::try_from(pcm.len()).unwrap();
    let mut header = omni_frontend::audio::wav_header(rate);
    header[4..8].copy_from_slice(&(36 + data_len).to_le_bytes());
    header[22..24].copy_from_slice(&channels.to_le_bytes());
    header[28..32].copy_from_slice(&(rate * u32::from(frame_size)).to_le_bytes());
    header[32..34].copy_from_slice(&frame_size.to_le_bytes());
    header[40..44].copy_from_slice(&data_len.to_le_bytes());
    [header.as_slice(), pcm].concat()
}

fn wav_with_metadata(wav: &[u8], chunk: &[u8]) -> Vec<u8> {
    let (format, data) = wav.split_at(WAV_HEADER_LEN - 8);
    let mut with_chunk = [format, chunk, data].concat();
    let riff_len = u32::try_from(with_chunk.len() - 8).unwrap();
    with_chunk[4..8].copy_from_slice(&riff_len.to_le_bytes());
    with_chunk
}

fn wav_uri(wav: &[u8]) -> String {
    format!("data:audio/wav;base64,{}", STANDARD.encode(wav))
}

fn reference_body(fields: Value) -> Value {
    let mut body = json!({ "model": "sim", "input": "hi", "voice": "reference", "response_format": "pcm" });
    let Value::Object(fields) = fields else { panic!("expected request fields") };
    body.as_object_mut().unwrap().extend(fields);
    body
}

fn audio_only_body(wav: &[u8]) -> Value {
    reference_body(json!({ "ref_audio": wav_uri(wav), "x_vector_only_mode": true }))
}

async fn submitted_reference(app: &Router, inbox: &Inbox, body: Value) -> Reference {
    let response = post_response(app, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    inbox.rx.try_recv().unwrap().speech.reference.expect("normalized reference")
}

async fn assert_reference_error(app: &Router, inbox: &Inbox, label: &str, body: Value, param: &str) {
    let (status, mime, bytes) = post(app, body).await;
    assert_eq!((status, mime.as_deref()), (StatusCode::BAD_REQUEST, Some("application/json")), "{label}");
    assert_eq!(error(&bytes), ("invalid_request_error".into(), Some(param.into())), "{label}");
    assert_eq!(inbox.rx.try_recv().err(), Some(TryRecvError::Empty), "{label}");
}

#[tokio::test]
async fn reference_at_the_engine_rate_preserves_signed_samples_and_transcript() {
    let (app, inbox) = reference_app(true, true);
    let samples = pcm([i16::MIN, -271, 0, i16::MAX].into_iter().cycle().take(REFERENCE_RATE as usize));
    let wav = reference_wav(REFERENCE_RATE, 1, &samples);
    let padded = wav_with_metadata(&wav, b"JUNK\x01\0\0\0x\0");
    let audio = wav_uri(&padded);
    let body = reference_body(json!({ "ref_audio": audio, "ref_text": "These words are spoken." }));
    let reference = submitted_reference(&app, &inbox, body).await;

    assert_eq!((reference.sample_rate, reference.pcm.as_ref()), (REFERENCE_RATE, samples.as_slice()));
    let ReferenceMode::WithTranscript(text) = reference.mode else { panic!("default reference mode") };
    assert_eq!(text, "These words are spoken.");
}

#[tokio::test]
async fn stereo_reference_downmixes_both_channels_without_overflow() {
    let (app, inbox) = reference_app(true, false);
    let samples = [i16::MAX, i16::MAX, i16::MIN, i16::MIN, 2000, -4000, -231, 231];
    let stereo = pcm(samples.into_iter().cycle().take(2 * REFERENCE_RATE as usize));
    let mono = pcm([i16::MAX, i16::MIN, -1000, 0].into_iter().cycle().take(REFERENCE_RATE as usize));
    let body = audio_only_body(&reference_wav(REFERENCE_RATE, 2, &stereo));
    let reference = submitted_reference(&app, &inbox, body).await;

    assert!(matches!(reference.mode, ReferenceMode::AudioOnly));
    assert_eq!((reference.sample_rate, reference.pcm.as_ref()), (REFERENCE_RATE, mono.as_slice()));
}

#[tokio::test]
async fn reference_resampling_rounds_up_and_keeps_the_tail() {
    let (app, inbox) = reference_app(true, false);
    let samples = pcm(std::iter::repeat_n(0, 43_662).chain(std::iter::repeat_n(12_000, 440)));
    let body = audio_only_body(&reference_wav(44_100, 1, &samples));
    let reference = submitted_reference(&app, &inbox, body).await;

    // A 44,102-frame clip at 44.1 kHz is 24,001.088... frames at 24 kHz.
    assert_eq!((reference.sample_rate, reference.pcm.len()), (REFERENCE_RATE, 2 * 24_002));
    let tail = i16::from_le_bytes(reference.pcm.last_chunk().copied().unwrap());
    assert!(tail > 1_000, "the final output frame was {tail}");
}

#[tokio::test]
async fn reference_duration_limits_use_source_frames_and_accept_thirty_seconds_of_stereo() {
    let (app, inbox) = reference_app(true, false);
    let max = 30 * REFERENCE_RATE as usize;
    for (label, rate, frames) in [("one frame short", 48_000, 47_999), ("one frame long", REFERENCE_RATE, max + 1)] {
        let body = audio_only_body(&reference_wav(rate, 1, &vec![0; 2 * frames]));
        assert_reference_error(&app, &inbox, label, body, "ref_audio").await;
    }

    let stereo = vec![0; 4 * 48_000 * 30];
    let body = audio_only_body(&reference_wav(48_000, 2, &stereo));
    let reference = submitted_reference(&app, &inbox, body).await;
    assert_eq!((reference.sample_rate, reference.pcm.len()), (REFERENCE_RATE, 2 * max));
}

#[tokio::test]
async fn invalid_references_name_the_field_and_never_reach_the_engine() {
    let (app, inbox) = reference_app(true, true);
    let wav = reference_wav(REFERENCE_RATE, 1, &vec![0; 2 * REFERENCE_RATE as usize]);
    let audio = wav_uri(&wav);
    let mut not_pcm = wav.clone();
    not_pcm[20..22].copy_from_slice(&3u16.to_le_bytes());
    let missing_pad = wav_with_metadata(&wav, b"JUNK\x01\0\0\0x");
    let cases = [
        ("missing audio", json!({ "ref_text": "hello" }), "ref_audio"),
        ("numeric audio", json!({ "ref_audio": 4 }), "ref_audio"),
        ("bare base64", json!({ "ref_audio": STANDARD.encode(&wav), "ref_text": "hello" }), "ref_audio"),
        ("bad base64", json!({ "ref_audio": "data:audio/wav;base64,!!!", "ref_text": "hello" }), "ref_audio"),
        ("truncated data", json!({ "ref_audio": wav_uri(&wav[..wav.len() - 1]), "ref_text": "hello" }), "ref_audio"),
        ("not PCM", json!({ "ref_audio": wav_uri(&not_pcm), "ref_text": "hello" }), "ref_audio"),
        ("missing chunk pad", json!({ "ref_audio": wav_uri(&missing_pad), "ref_text": "hello" }), "ref_audio"),
        ("default transcript missing", json!({ "ref_audio": audio }), "ref_text"),
        ("blank transcript", json!({ "ref_audio": audio, "ref_text": " \n\t" }), "ref_text"),
        (
            "audio-only with text",
            json!({ "ref_audio": audio, "ref_text": "hello", "x_vector_only_mode": true }),
            "ref_text",
        ),
        ("null mode", json!({ "ref_audio": audio, "x_vector_only_mode": null }), "x_vector_only_mode"),
    ];
    for (label, fields, param) in cases {
        assert_reference_error(&app, &inbox, label, reference_body(fields), param).await;
    }

    let (audio_only_app, inbox) = reference_app(true, false);
    let body = reference_body(json!({ "ref_audio": audio, "ref_text": "hello" }));
    assert_reference_error(&audio_only_app, &inbox, "ICL disabled", body, "x_vector_only_mode").await;
}

#[tokio::test]
async fn stalled_uploads_take_permits_before_body_read_and_abort_releases_a_permit() {
    let (app, inbox) = reference_app(true, false);
    let mut uploads = Vec::new();
    for _ in 0..4 {
        let (started, polled) = tokio::sync::oneshot::channel();
        let partial = stream::once(async move {
            started.send(()).unwrap();
            Ok::<_, Infallible>(Bytes::from_static(b"{"))
        });
        let body = Body::from_stream(partial.chain(stream::pending()));
        uploads.push(tokio::spawn(app.clone().oneshot(speech_request(body))));
        tokio::time::timeout(UPLOAD_TIMEOUT, polled).await.expect("body poll").expect("upload still running");
    }

    let unread = stream::poll_fn(|_| -> Poll<Option<Result<Bytes, Infallible>>> {
        panic!("overload must be detected before polling the body");
    });
    let future = app.clone().oneshot(speech_request(Body::from_stream(unread)));
    let rejected = tokio::time::timeout(UPLOAD_TIMEOUT, future).await.expect("immediate rejection").unwrap();
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(inbox.rx.try_recv().err(), Some(TryRecvError::Empty));

    let aborted = uploads.pop().unwrap();
    aborted.abort();
    assert!(aborted.await.unwrap_err().is_cancelled());
    let wav = reference_wav(REFERENCE_RATE, 1, &vec![0; 2 * REFERENCE_RATE as usize]);
    let recovered = submitted_reference(&app, &inbox, audio_only_body(&wav)).await;
    assert_eq!(recovered.pcm.len(), 2 * REFERENCE_RATE as usize);
    for upload in uploads {
        upload.abort();
    }
}

#[tokio::test]
async fn streaming_uploads_obey_the_route_body_limit_without_content_length() {
    let oversized = |mib| {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let chunks = std::iter::repeat_n(Ok::<_, Infallible>(chunk), mib).chain([Ok(Bytes::from_static(b" "))]);
        speech_request(Body::from_stream(stream::iter(chunks)))
    };
    let (reference_router, inbox) = reference_app(true, false);
    let req = oversized(8);
    assert!(req.headers().get("content-length").is_none());
    let response = reference_router.clone().oneshot(req).await.unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(response.headers()["content-type"], "application/json");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(error(&bytes), ("invalid_request_error".into(), None));
    assert_eq!(inbox.rx.try_recv().err(), Some(TryRecvError::Empty));

    let preset = app().oneshot(oversized(2)).await.unwrap();
    assert_eq!(preset.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
