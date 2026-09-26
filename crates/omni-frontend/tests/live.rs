use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use futures_util::SinkExt;
use futures_util::StreamExt;
use omni_engine::live::CloseReason;
use omni_engine::live::Closed;
use omni_engine::live::LiveInbox;
use omni_engine::live::Output;
use omni_frontend::live_protocol::ClientEvent;
use omni_frontend::live_protocol::parse;
use omni_sim::live::LiveProfile;
use proptest::prelude::*;
use serde_json::Value;
use serde_json::json;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

async fn server(profile: LiveProfile) -> String {
    let (handle, inbox) = omni_engine::live::live_channel(profile.info("sim-live"), 16);
    omni_sim::live::spawn_live(inbox, profile);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(omni_frontend::serve(listener, omni_frontend::router(handle, None), std::future::pending()));
    format!("{addr}")
}

/// A server over an engine the test plays by hand.
async fn fake(queue: usize, engine: impl FnOnce(LiveInbox) + Send + 'static) -> (String, std::thread::JoinHandle<()>) {
    let (handle, inbox) = omni_engine::live::live_channel(LiveProfile::default().info("sim-live"), queue);
    let engine = std::thread::spawn(move || engine(inbox));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(omni_frontend::serve(listener, omni_frontend::router(handle, None), std::future::pending()));
    (format!("{addr}"), engine)
}

async fn connect(addr: &str) -> Ws {
    tokio_tungstenite::connect_async(format!("ws://{addr}/v1/live/sessions")).await.unwrap().0
}

async fn send(ws: &mut Ws, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap();
}

/// The next event, skipping usage updates; `None` once the socket closed.
async fn recv(ws: &mut Ws) -> Option<Value> {
    loop {
        match tokio::time::timeout(Duration::from_secs(5), ws.next()).await.expect("an event in time") {
            Some(Ok(Message::Text(t))) => {
                let v: Value = serde_json::from_str(&t).unwrap();
                if v["type"] != "session.usage.updated" {
                    return Some(v);
                }
            }
            Some(Ok(Message::Close(_))) | None => return None,
            Some(Ok(_)) => {}
            Some(Err(e)) => panic!("{e}"),
        }
    }
}

async fn until(ws: &mut Ws, kind: &str) -> Value {
    loop {
        let v = recv(ws).await.unwrap_or_else(|| panic!("closed before {kind}"));
        if v["type"] == kind {
            return v;
        }
    }
}

fn error_of(v: &Value) -> (&str, &str, Option<&str>) {
    assert_eq!(v["type"], "error", "{v}");
    let e = &v["error"];
    (e["type"].as_str().unwrap(), e["code"].as_str().unwrap(), e["param"].as_str())
}

#[tokio::test]
async fn a_session_streams_frames_on_the_timeline_and_closes_on_request() {
    let addr = server(LiveProfile::default()).await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.input_audio.append", "audio": "", "client_event_id": "early"})).await;
    let early = recv(&mut ws).await.unwrap();
    assert_eq!(
        (error_of(&early).1, early["error"]["client_event_id"].as_str()),
        ("session_not_started", Some("early"))
    );

    send(
        &mut ws,
        json!({"type": "session.start", "session": {"model": "sim-live", "audio": {"output": {"voice": "nova"}}}}),
    )
    .await;
    let started = recv(&mut ws).await.unwrap();
    assert_eq!(started["type"], "session.started", "{started}");
    assert_eq!(
        (started["session"]["model"].as_str(), started["session"]["audio"]["output"]["voice"].as_str()),
        (Some("sim-live"), Some("nova"))
    );
    assert!(started["event_id"].is_string());

    let tone: Vec<u8> = (0..4800).flat_map(|i| (((i % 40) as i16 - 20) * 800).to_le_bytes()).collect();
    for chunk in tone.chunks(960) {
        send(&mut ws, json!({"type": "session.input_audio.append", "audio": STANDARD.encode(chunk)})).await;
    }
    send(&mut ws, json!({"type": "session.input_audio.mute", "client_event_id": "m"})).await;
    let mut starts = Vec::new();
    let mut muted = false;
    while starts.len() < 5 || !muted {
        let v = recv(&mut ws).await.unwrap();
        match v["type"].as_str().unwrap() {
            "session.output_audio.delta" => {
                let (s, e) = (v["start_ms"].as_u64().unwrap(), v["end_ms"].as_u64().unwrap());
                assert_eq!((e - s, STANDARD.decode(v["delta"].as_str().unwrap()).unwrap().len()), (80, 3840));
                starts.push(s);
            }
            "session.input_audio.muted" => {
                assert_eq!(v["client_event_id"], "m");
                muted = true;
            }
            "session.output_transcript.delta" => {}
            t => panic!("unexpected {t}"),
        }
    }
    assert!(starts.windows(2).all(|w| w[1] > w[0]) && starts[0] == 0, "{starts:?}");

    send(&mut ws, json!({"type": "session.close"})).await;
    let closed = until(&mut ws, "session.closed").await;
    assert_eq!(closed["reason"], "close_requested");
    assert!(closed["usage"]["seconds"].as_f64().unwrap() > 0.3);
    assert!(recv(&mut ws).await.is_none());
}

#[tokio::test]
async fn refusals_name_what_is_wrong_and_keep_the_socket() {
    let addr = server(LiveProfile::default()).await;
    let mut ws = connect(&addr).await;
    let cases = [
        (
            json!({"type": "session.start", "session": {"temperature": 1}}),
            "unknown_parameter",
            Some("session.temperature"),
        ),
        (
            json!({"type": "session.start", "session": {"audio": {"output": {"voice": "nobody"}}}}),
            "invalid_value",
            Some("session.audio.output.voice"),
        ),
        (json!({"type": "session.start", "session": {"model": "other"}}), "model_not_found", Some("session.model")),
        (
            json!({"type": "session.start", "session": {"delegation": {"type": "client"}}}),
            "unsupported",
            Some("session.delegation"),
        ),
        (
            json!({"type": "session.start", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}),
            "invalid_value",
            Some("session.audio.input.format"),
        ),
        (json!({"type": "session.update", "session": {}}), "unsupported", Some("type")),
        (json!({"type": "session.thinking.append", "text": "hm"}), "unsupported", Some("type")),
        (json!({"type": "nonsense"}), "unknown_event", Some("type")),
    ];
    for (event, code, param) in cases {
        send(&mut ws, event.clone()).await;
        let v = recv(&mut ws).await.unwrap();
        let (kind, got, p) = error_of(&v);
        assert_eq!((kind, got, p), ("invalid_request_error", code, param), "{event}");
    }
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(error_of(&until(&mut ws, "error").await).1, "session_already_started");
}

#[tokio::test]
async fn a_full_engine_says_busy_and_keeps_the_socket_for_a_retry() {
    let addr = server(LiveProfile { max_sessions: 1, ..LiveProfile::default() }).await;
    let mut first = connect(&addr).await;
    send(&mut first, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut first).await.unwrap()["type"], "session.started");
    let mut second = connect(&addr).await;
    send(&mut second, json!({"type": "session.start"})).await;
    assert_eq!(error_of(&recv(&mut second).await.unwrap()), ("server_error", "server_busy", None));
    send(&mut first, json!({"type": "session.close"})).await;
    until(&mut first, "session.closed").await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    send(&mut second, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut second).await.unwrap()["type"], "session.started");
}

#[tokio::test]
async fn a_full_queue_says_busy_and_keeps_the_socket() {
    let (addr, _engine) = fake(1, |inbox| {
        std::thread::sleep(Duration::from_secs(5));
        drop(inbox);
    })
    .await;
    let mut first = connect(&addr).await;
    send(&mut first, json!({"type": "session.start"})).await;
    let mut second = connect(&addr).await;
    send(&mut second, json!({"type": "session.start", "client_event_id": "s"})).await;
    let v = recv(&mut second).await.unwrap();
    assert_eq!(
        (error_of(&v), v["error"]["client_event_id"].as_str()),
        (("server_error", "server_busy", None), Some("s"))
    );
    send(&mut second, json!({"type": "session.close"})).await;
    assert_eq!(error_of(&recv(&mut second).await.unwrap()).1, "session_not_started");
}

#[tokio::test]
async fn an_engine_abort_is_an_error_then_closed() {
    let (addr, _engine) = fake(4, |inbox| {
        let s = inbox.rx.recv().unwrap();
        s.sink.send(Output::Started).unwrap();
        s.sink.send(Output::Closed(Closed::refused(CloseReason::Aborted))).unwrap();
        std::thread::sleep(Duration::from_secs(1));
    })
    .await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    assert_eq!(error_of(&recv(&mut ws).await.unwrap()), ("server_error", "server_error", None));
    assert_eq!(recv(&mut ws).await.unwrap()["reason"], "connection_lost");
    assert!(recv(&mut ws).await.is_none());
}

#[tokio::test]
async fn an_engine_that_stops_ends_its_sessions() {
    let (addr, _engine) = fake(4, |inbox| {
        let s = inbox.rx.recv().unwrap();
        s.sink.send(Output::Started).unwrap();
    })
    .await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    let e = recv(&mut ws).await.unwrap();
    assert!(e["error"]["message"].as_str().unwrap().contains("stopped"), "{e}");
    assert_eq!(recv(&mut ws).await.unwrap()["reason"], "connection_lost");
}

#[tokio::test]
async fn close_before_started_is_answered_by_the_engine() {
    let (addr, _engine) = fake(4, |inbox| {
        let s = inbox.rx.recv().unwrap();
        let mut input = s.input;
        while input.blocking_recv().is_some() {}
        s.sink.send(Output::Closed(Closed::refused(CloseReason::Hangup))).unwrap();
        std::thread::sleep(Duration::from_secs(1));
    })
    .await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    send(&mut ws, json!({"type": "session.close"})).await;
    let closed = recv(&mut ws).await.unwrap();
    assert_eq!((closed["type"].as_str(), closed["reason"].as_str()), (Some("session.closed"), Some("close_requested")));
    assert_eq!(closed["usage"]["seconds"], 0.0);
}

#[tokio::test]
async fn close_gives_up_on_a_silent_engine_after_the_grace() {
    let (addr, _engine) = fake(4, |inbox| {
        let s = inbox.rx.recv().unwrap();
        s.sink.send(Output::Started).unwrap();
        std::thread::sleep(Duration::from_secs(4));
        drop(s);
    })
    .await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    let asked = std::time::Instant::now();
    send(&mut ws, json!({"type": "session.close"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["reason"], "close_requested");
    assert!(asked.elapsed() >= Duration::from_millis(1900), "{:?}", asked.elapsed());
    assert!(recv(&mut ws).await.is_none());
}

#[tokio::test]
async fn muting_forwards_silence_of_the_same_length() {
    let (heard_tx, heard) = std::sync::mpsc::channel::<Bytes>();
    let (addr, _engine) = fake(4, move |inbox| {
        let s = inbox.rx.recv().unwrap();
        s.sink.send(Output::Started).unwrap();
        let mut input = s.input;
        while let Some(pcm) = input.blocking_recv() {
            let _ = heard_tx.send(pcm);
        }
    })
    .await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    let loud = STANDARD.encode([7u8; 6]);
    for event in [
        json!({"type": "session.input_audio.append", "audio": loud}),
        json!({"type": "session.input_audio.mute"}),
        json!({"type": "session.input_audio.append", "audio": loud}),
        json!({"type": "session.input_audio.unmute"}),
        json!({"type": "session.input_audio.append", "audio": loud}),
    ] {
        send(&mut ws, event).await;
    }
    until(&mut ws, "session.input_audio.unmuted").await;
    let got: Vec<Vec<u8>> = (0..3).map(|_| heard.recv_timeout(Duration::from_secs(5)).unwrap().to_vec()).collect();
    assert_eq!(got, [vec![7; 6], vec![0; 6], vec![7; 6]]);
}

#[tokio::test]
async fn expiry_closes_the_session() {
    let addr = server(LiveProfile { max_frames: 3, ..LiveProfile::default() }).await;
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    let closed = until(&mut ws, "session.closed").await;
    assert_eq!(closed["reason"], "expired");
}

#[tokio::test]
async fn a_live_only_server_refuses_speech_and_serves_the_demo() {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let profile = LiveProfile::default();
    let (handle, _inbox) = omni_engine::live::live_channel(profile.info("sim-live"), 1);
    let app = omni_frontend::router(handle, None);
    let call = |req: Request<Body>| {
        let app = app.clone();
        async move {
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status().as_u16();
            (status, String::from_utf8(resp.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap())
        }
    };
    let get = |uri: &str| Request::get(uri).body(Body::empty()).unwrap();
    let (status, speech) = call(Request::post("/v1/audio/speech").body(Body::from("{}")).unwrap()).await;
    assert!(status == 404 && speech.contains("unsupported_endpoint"), "{speech}");
    let (status, demo) = call(get("/")).await;
    assert!(status == 200 && demo.contains("/v1/live/sessions"));
    let (_, config) = call(get("/live/config")).await;
    let config: Value = serde_json::from_str(&config).unwrap();
    assert_eq!((config["frame_ms"].as_u64(), config["voice"].as_str()), (Some(80), Some("alloy")));
    let (_, models) = call(get("/v1/models")).await;
    assert!(models.contains("sim-live"), "{models}");
    let (_, stats) = call(get("/live/stats")).await;
    let stats: Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        (stats["sessions"].as_u64(), stats["max_sessions"].as_u64(), stats["ticks"].as_u64()),
        (Some(0), Some(64), Some(0))
    );
}

proptest! {
    #[test]
    fn appends_decode_exactly_what_was_encoded(pcm in prop::collection::vec(any::<u8>(), 0..4000), id in "[a-z0-9]{0,8}") {
        let text = json!({"type": "session.input_audio.append", "audio": STANDARD.encode(&pcm), "client_event_id": id}).to_string();
        let r = parse(&text, 24_000).unwrap();
        prop_assert_eq!((r.event, r.client_event_id), (ClientEvent::Append(pcm.into()), Some(id)));
    }

    #[test]
    fn any_unknown_top_level_field_is_named(field in "[a-z_]{1,12}") {
        prop_assume!(!["type", "client_event_id", "session"].contains(&field.as_str()));
        let text = json!({"type": "session.start", field.clone(): 1}).to_string();
        let e = parse(&text, 24_000).unwrap_err();
        prop_assert_eq!((e.code, e.param), ("unknown_parameter", Some(field)));
    }
}

#[tokio::test]
async fn stats_count_the_running_session_and_its_ticks() {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let profile = LiveProfile::default();
    let (handle, inbox) = omni_engine::live::live_channel(profile.info("sim-live"), 4);
    omni_sim::live::spawn_live(inbox, profile);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = format!("{}", listener.local_addr().unwrap());
    tokio::spawn(omni_frontend::serve(listener, omni_frontend::router(handle.clone(), None), std::future::pending()));
    let stats = || async {
        let app = omni_frontend::router(handle.clone(), None);
        let resp = app.oneshot(Request::get("/live/stats").body(Body::empty()).unwrap()).await.unwrap();
        serde_json::from_slice::<Value>(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
    };
    let mut ws = connect(&addr).await;
    send(&mut ws, json!({"type": "session.start"})).await;
    for _ in 0..3 {
        until(&mut ws, "session.output_audio.delta").await;
    }
    let s = stats().await;
    assert_eq!(s["sessions"], 1, "{s}");
    assert!(s["ticks"].as_u64().unwrap() >= 3 && s["tick_busy_us"].as_u64().is_some(), "{s}");
    send(&mut ws, json!({"type": "session.close"})).await;
    until(&mut ws, "session.closed").await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(stats().await["sessions"], 0);
}
