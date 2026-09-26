use omni_sim::live::LiveProfile;
use serde_json::Value;

/// `omni-bench duplex` with `args` against a fresh sim-live server; its report.
fn duplex(args: &[&str]) -> Value {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let addr = rt.block_on(async {
        let profile = LiveProfile::default();
        let (handle, inbox) = omni_engine::live::live_channel(profile.info("sim-live"), 16);
        omni_sim::live::spawn_live(inbox, profile);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(omni_frontend::serve(listener, omni_frontend::router(handle, None), std::future::pending()));
        addr
    });
    let out = std::env::temp_dir().join(format!("omni-bench-duplex-{}-{}.json", std::process::id(), args.join("")));
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_omni-bench"))
        .args(["duplex", "--base-url", &format!("http://{addr}"), "--seconds", "1.5"])
        .args(args)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let report: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    report
}

#[test]
fn duplex_sessions_against_the_live_sim_run_clean() {
    for rate in ["24000", "16000"] {
        let report = duplex(&["--sessions", "4", "--rate", rate]);
        assert_eq!(
            (report["levels"][0]["sessions"].as_u64(), report["levels"][0]["failed"].as_u64()),
            (Some(4), Some(0))
        );
        assert_eq!(report["config"]["model"], "sim-live");
        let first = &report["records"][0][0];
        assert!(first["frames"].as_u64().unwrap() >= 15 && first["close_reason"] == "close_requested", "{first:#}");
        assert!(first["transcript"].as_str().unwrap().starts_with("I hear"), "{first:#}");
    }
}
