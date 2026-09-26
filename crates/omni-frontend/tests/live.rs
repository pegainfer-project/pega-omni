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
use omni_engine::live::SessionDraft;
use omni_frontend::live_audio::Format;
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

fn start() -> Value {
    json!({"type": "session.start", "session": {"model": "sim-live"}})
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
    send(&mut ws, json!({"type": "session.input_audio.append", "audio": "AAA=", "event_id": "early"})).await;
    let early = recv(&mut ws).await.unwrap();
    assert_eq!(
        (error_of(&early).1, early["error"]["client_event_id"].as_str()),
        ("session_not_started", Some("early"))
    );

    let session =
        json!({"model": "sim-live", "audio": {"output": {"voice": "nova"}}, "delegation": {"type": "client"}});
    send(&mut ws, json!({"type": "session.start", "event_id": "go", "session": session})).await;
    let started = recv(&mut ws).await.unwrap();
    assert_eq!(started["type"], "session.started", "{started}");
    let s = &started["session"];
    assert_eq!(
        (s["model"].as_str(), s["audio"]["output"]["voice"].as_str(), s["status"].as_str(), &s["delegation"]),
        (Some("sim-live"), Some("nova"), Some("active"), &json!({"type": "client"}))
    );
    assert_eq!(s["audio"]["format"], json!({"type": "audio/pcm", "rate": 24000}));
    assert!(s["id"].is_string() && s["expires_at"].as_u64().is_some_and(|t| t > 1_700_000_000), "{s}");
    assert_eq!((started["event_id"].is_string(), started["client_event_id"].as_str()), (true, Some("go")));

    let tone: Vec<u8> = (0..4800).flat_map(|i| (((i % 40) as i16 - 20) * 800).to_le_bytes()).collect();
    for chunk in tone.chunks(960) {
        send(&mut ws, json!({"type": "session.input_audio.append", "audio": STANDARD.encode(chunk)})).await;
    }
    send(&mut ws, json!({"type": "session.input_audio.mute", "event_id": "m"})).await;
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

    send(&mut ws, json!({"type": "session.close", "event_id": "bye"})).await;
    let closed = until(&mut ws, "session.closed").await;
    assert_eq!((closed["reason"].as_str(), closed["client_event_id"].as_str()), (Some("close_requested"), Some("bye")));
    assert_eq!(closed["session"], started["session"]);
    assert!(closed["usage"]["seconds"].as_f64().unwrap() > 0.3);
    assert!(recv(&mut ws).await.is_none());
}

/// The sizes of a session's first output deltas, each 80 ms, when it picks `format`.
async fn delta_bytes(format: Value) -> Vec<usize> {
    let addr = server(LiveProfile::default()).await;
    let mut ws = connect(&addr).await;
    let session = json!({"model": "sim-live", "audio": {"format": format}});
    send(&mut ws, json!({"type": "session.start", "session": session})).await;
    let started = recv(&mut ws).await.unwrap();
    assert_eq!(started["session"]["audio"]["format"], format);
    let mut sizes = Vec::new();
    while sizes.len() < 4 {
        let v = until(&mut ws, "session.output_audio.delta").await;
        assert_eq!(v["end_ms"].as_u64().unwrap() - v["start_ms"].as_u64().unwrap(), 80);
        sizes.push(STANDARD.decode(v["delta"].as_str().unwrap()).unwrap().len());
    }
    sizes
}

#[tokio::test]
async fn the_wire_format_sets_the_rate_and_encoding_of_the_agents_audio() {
    assert_eq!(delta_bytes(json!({"type": "audio/pcm", "rate": 16000})).await, [2560; 4]);
    assert_eq!(delta_bytes(json!({"type": "audio/pcmu", "rate": 8000})).await, [640; 4]);
    assert_eq!(delta_bytes(json!({"type": "audio/pcma", "rate": 8000})).await, [640; 4]);
}

#[tokio::test]
async fn audio_that_is_not_whole_samples_is_refused_and_the_session_goes_on() {
    let addr = server(LiveProfile::default()).await;
    let mut ws = connect(&addr).await;
    send(&mut ws, start()).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    let odd = json!({"type": "session.input_audio.append", "audio": STANDARD.encode([1u8; 3]), "event_id": "odd"});
    send(&mut ws, odd).await;
    let e = until(&mut ws, "error").await;
    assert_eq!(
        (error_of(&e), e["error"]["client_event_id"].as_str()),
        (("invalid_request_error", "invalid_audio", Some("audio")), Some("odd"))
    );
    until(&mut ws, "session.output_audio.delta").await;
}

/// A `session.start` for sim-live with `extra` session fields.
fn start_with(extra: Value) -> Value {
    let mut session = json!({"model": "sim-live"});
    session.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    json!({"type": "session.start", "session": session})
}

#[tokio::test]
async fn refusals_name_what_is_wrong_and_keep_the_socket() {
    let addr = server(LiveProfile::default()).await;
    let mut ws = connect(&addr).await;
    let pcm = |kind: &str, rate: u32| json!({"audio": {"format": {"type": kind, "rate": rate}}});
    let cases = [
        (start_with(json!({"temperature": 1})), "unknown_parameter", "session.temperature"),
        (start_with(json!({"audio": {"output": {"voice": "nobody"}}})), "invalid_value", "session.audio.output.voice"),
        (
            start_with(json!({"audio": {"output": {"voice": {"id": "nobody"}}}})),
            "invalid_value",
            "session.audio.output.voice",
        ),
        (
            start_with(json!({"audio": {"output": {"voice": {"name": "nova"}}}})),
            "unknown_parameter",
            "session.audio.output.voice.name",
        ),
        (start_with(json!({"model": "other"})), "model_not_found", "session.model"),
        (json!({"type": "session.start", "session": {}}), "missing_required_parameter", "session.model"),
        (json!({"type": "session.start"}), "missing_required_parameter", "session"),
        (
            start_with(json!({"delegation": {"type": "responses", "responses": {"model": "gpt-5.5"}}})),
            "unsupported",
            "session.delegation",
        ),
        (start_with(json!({"delegation": {"type": "server"}})), "invalid_value", "session.delegation.type"),
        (start_with(json!({"input": [{"role": "user", "content": [{"text": "hi"}]}]})), "unsupported", "session.input"),
        (start_with(json!({"store": true})), "unsupported", "session.store"),
        (start_with(json!({"client": {"data_channel": {}}})), "unsupported", "session.client"),
        (start_with(json!({"audio": {"input": {}}})), "unknown_parameter", "session.audio.input"),
        (start_with(pcm("audio/pcm", 44_100)), "invalid_value", "session.audio.format.rate"),
        (start_with(pcm("audio/pcmu", 16_000)), "invalid_value", "session.audio.format.rate"),
        (start_with(pcm("audio/opus", 48_000)), "invalid_value", "session.audio.format.type"),
        (
            start_with(json!({"audio": {"format": {"type": "audio/pcm"}}})),
            "missing_required_parameter",
            "session.audio.format.rate",
        ),
        (json!({"type": "session.close", "client_event_id": "x"}), "unknown_parameter", "client_event_id"),
        (json!({"type": "session.input_audio.append", "audio": ""}), "invalid_audio", "audio"),
        (json!({"type": "session.update", "session": {}}), "unsupported", "type"),
        (json!({"type": "session.thinking.append", "content": "hm", "delegation_id": null}), "unsupported", "type"),
        (json!({"type": "nonsense"}), "unknown_event", "type"),
    ];
    for (event, code, param) in cases {
        send(&mut ws, event.clone()).await;
        let v = recv(&mut ws).await.unwrap();
        assert_eq!(error_of(&v), ("invalid_request_error", code, Some(param)), "{event}");
    }
    send(&mut ws, start()).await;
    assert_eq!(recv(&mut ws).await.unwrap()["type"], "session.started");
    send(&mut ws, start()).await;
    assert_eq!(error_of(&until(&mut ws, "error").await).1, "session_already_started");
}

#[tokio::test]
async fn a_full_engine_says_busy_and_keeps_the_socket_for_a_retry() {
    let addr = server(LiveProfile { max_sessions: 1, ..LiveProfile::default() }).await;
    let mut first = connect(&addr).await;
    send(&mut first, start()).await;
    assert_eq!(recv(&mut first).await.unwrap()["type"], "session.started");
    let mut second = connect(&addr).await;
    send(&mut second, start()).await;
    assert_eq!(error_of(&recv(&mut second).await.unwrap()), ("server_error", "server_busy", None));
    send(&mut first, json!({"type": "session.close"})).await;
    until(&mut first, "session.closed").await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    send(&mut second, start()).await;
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
    send(&mut first, start()).await;
    let mut second = connect(&addr).await;
    send(&mut second, json!({"type": "session.start", "event_id": "s", "session": {"model": "sim-live"}})).await;
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
    send(&mut ws, start()).await;
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
    send(&mut ws, start()).await;
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
    send(&mut ws, start()).await;
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
    send(&mut ws, start()).await;
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
    send(&mut ws, start()).await;
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
    send(&mut ws, start()).await;
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

/// `None` (absent), `null`, or one of `values`.
fn optional(values: Vec<Value>) -> impl Strategy<Value = Option<Value>> {
    prop_oneof![Just(None), Just(Some(Value::Null)), prop::sample::select(values).prop_map(Some)]
}

fn format_of(v: &Value) -> Format {
    match (v["type"].as_str(), v["rate"].as_u64()) {
        (Some("audio/pcmu"), _) => Format::Pcmu,
        (Some("audio/pcma"), _) => Format::Pcma,
        (_, rate) => Format::Pcm(rate.unwrap() as u32),
    }
}

proptest! {
    #[test]
    fn appends_decode_exactly_what_was_encoded(pcm in prop::collection::vec(any::<u8>(), 1..4000), id in "[a-z0-9]{0,8}") {
        let text = json!({"type": "session.input_audio.append", "audio": STANDARD.encode(&pcm), "event_id": id}).to_string();
        let r = parse(&text).unwrap();
        prop_assert_eq!((r.event, r.event_id), (ClientEvent::Append(pcm.into()), Some(id)));
    }

    #[test]
    fn any_unknown_top_level_field_is_named(field in "[a-z_]{1,12}") {
        prop_assume!(!["type", "event_id", "session"].contains(&field.as_str()));
        let text = json!({"type": "session.start", "session": {"model": "m"}, field.clone(): 1}).to_string();
        let e = parse(&text).unwrap_err();
        prop_assert_eq!((e.code, e.param), ("unknown_parameter", Some(field)));
    }

    /// Every mix of the optional startup fields the server accepts starts a
    /// session with exactly what they say; `null` and blank mean the default.
    #[test]
    fn accepted_startups_carry_their_format_voice_and_instructions(
        format in optional(vec![
            json!({"type": "audio/pcm", "rate": 16000}),
            json!({"type": "audio/pcm", "rate": 24000}),
            json!({"type": "audio/pcmu", "rate": 8000}),
            json!({"type": "audio/pcma", "rate": 8000}),
        ]),
        voice in optional(vec![json!("nova"), json!({"id": "nova"})]),
        instructions in optional(vec![json!("Be brief."), json!(""), json!("  ")]),
        delegation in optional(vec![json!({"type": "client"})]),
        input in optional(vec![json!([])]),
        store in optional(vec![json!(false)]),
    ) {
        let mut session = json!({"model": "m", "audio": {"output": {}}});
        if let Some(f) = &format {
            session["audio"]["format"] = f.clone();
        }
        if let Some(v) = &voice {
            session["audio"]["output"]["voice"] = v.clone();
        }
        for (key, v) in [("instructions", &instructions), ("delegation", &delegation), ("input", &input), ("store", &store)] {
            if let Some(v) = v {
                session[key] = v.clone();
            }
        }
        let r = parse(&json!({"type": "session.start", "session": session}).to_string()).unwrap();
        let text = |v: &Option<Value>| v.as_ref().and_then(Value::as_str).map(String::from);
        let voice = voice.map(|v| if v.is_object() { v["id"].clone() } else { v });
        let expected = ClientEvent::Start {
            model: "m".into(),
            draft: SessionDraft { voice: text(&voice), instructions: text(&instructions).filter(|s| !s.trim().is_empty()) },
            format: format.as_ref().filter(|f| !f.is_null()).map_or(Format::DEFAULT, format_of),
        };
        prop_assert_eq!(r.event, expected);
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
    send(&mut ws, start()).await;
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
