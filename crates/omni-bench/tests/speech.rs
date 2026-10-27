use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use serde_json::Value;
use serde_json::json;

async fn speech(State(requests): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>) -> impl IntoResponse {
    requests.lock().unwrap().push(body);
    let chunks = futures_util::stream::unfold(0, |part| async move {
        match part {
            0 => Some((Ok::<_, std::io::Error>(vec![0u8; 8]), 1)),
            1 => {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Some((Ok(vec![0u8; 8]), 2))
            }
            _ => None,
        }
    });
    ([(header::CONTENT_TYPE, "audio/pcm")], Body::from_stream(chunks))
}

#[test]
fn instructions_and_top_level_extensions_reach_the_server_and_report() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let addr = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/audio/speech", post(speech)).with_state(requests.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    });
    let dir = std::env::temp_dir().join(format!("omni-bench-speech-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prompts = dir.join("prompts.txt");
    let out = dir.join("report.json");
    std::fs::write(&prompts, "Hello from the first prompt.\n第二条测试文本。\n").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-bench"))
        .args([
            "--base-url",
            &format!("http://{addr}"),
            "--model",
            "voice-design",
            "--voice",
            "alloy",
            "--instructions",
            "A calm, warm voice.",
            "--extra",
            r#"{"seed":7}"#,
            "--extra-body",
            r#"{"task_type":"VoiceDesign","language":"English","stream":true}"#,
            "--num-requests",
            "2",
            "--warmup",
            "1",
            "--max-concurrency",
            "1",
            "--sample-rate",
            "1000",
        ])
        .arg("--prompts")
        .arg(&prompts)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let report: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["summary"]["requests"], 2);
    assert_eq!(report["summary"]["ok"], 2);
    assert_eq!(report["summary"]["failed"], 0);
    for record in report["records"].as_array().unwrap() {
        assert_eq!(record["ok"], true);
        assert!((record["audio_s"].as_f64().unwrap() - 0.008).abs() < 1e-9, "{record}");
        let ttfp = record["ttfp_ms"].as_f64().unwrap();
        let e2e = record["e2e_ms"].as_f64().unwrap();
        assert!(e2e - ttfp >= 40.0, "PCM must be observed before the delayed final chunk: {record}");
    }
    let template = &report["config"]["request_template"];
    assert_eq!(template["instructions"], "A calm, warm voice.");
    assert_eq!(template["task_type"], "VoiceDesign");
    assert_eq!(template["language"], "English");
    assert_eq!(template["stream"], true);
    assert_eq!(template["extra"], json!({"seed": 7}));
    assert!(template.get("input").is_none());
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3, "warmup plus two measured requests");
    assert_eq!(requests[0]["input"], "Hello from the first prompt.");
    let mut measured_inputs: Vec<_> = requests[1..].iter().map(|r| r["input"].as_str().unwrap()).collect();
    measured_inputs.sort();
    assert_eq!(measured_inputs, ["Hello from the first prompt.", "第二条测试文本。"]);
    for request in requests.iter() {
        let mut fields = request.as_object().unwrap().clone();
        fields.remove("input");
        assert_eq!(Value::Object(fields), *template);
        assert_eq!(request["model"], "voice-design");
        assert_eq!(request["voice"], "alloy");
        assert_eq!(request["response_format"], "pcm");
        assert_eq!(request["stream_format"], "audio");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn request_extensions_reject_nonobjects_and_reserved_fields_before_sending() {
    let reject = |flag: &str, body: &str, expected: &str| {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-bench"))
            .args(["--base-url", "http://127.0.0.1:9", "--num-requests", "1", flag, body])
            .output()
            .unwrap();
        assert!(!output.status.success(), "accepted {flag} {body}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{flag} {body}: {error}");
    };
    for flag in ["--extra-body", "--extra"] {
        for body in ["null", "[]", "true", "42", r#""text""#] {
            reject(flag, body, &format!("{flag} must be a JSON object"));
        }
        reject(flag, "{", &format!("{flag} is JSON"));
    }
    for name in ["model", "input", "voice", "response_format", "stream_format", "instructions", "extra"] {
        reject(
            "--extra-body",
            &json!({name: "override"}).to_string(),
            &format!("cannot override reserved field `{name}`"),
        );
    }
}
