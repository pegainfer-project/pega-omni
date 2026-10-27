//! Exercise scheduling through the public CLI and a deliberately slow speech
//! endpoint: waiting for a client permit must remain visible in the report.

use std::time::Duration;

use omni_sim::Profile;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use serde_json::Value;

#[test]
fn scheduled_arrivals_include_waiting_for_a_client_slot() {
    for (rate, failures) in [("inf", false), ("1000", false), ("inf", true)] {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let addr = rt.block_on(async {
            let profile = Profile { step_base: Duration::from_millis(40), ..Profile::default() };
            let (handle, inbox) = omni_engine::channel(profile.info("slow-speech"), 16);
            omni_sim::spawn(inbox, profile);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(omni_frontend::serve(listener, omni_frontend::router(handle, None), std::future::pending()));
            addr
        });
        let out =
            std::env::temp_dir().join(format!("omni-bench-arrivals-{}-{rate}-{failures}.json", std::process::id()));
        let prompts = out.with_extension("txt");
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_omni-bench"));
        command
            .args([
                "--base-url",
                &format!("http://{addr}"),
                "--model",
                "slow-speech",
                "--num-requests",
                "8",
                "--max-concurrency",
                "1",
                "--request-rate",
                rate,
                "--seed",
                "42",
                "--extra",
                "{\"frames\":1}",
                "--warmup",
                "1",
            ])
            .arg("--out")
            .arg(&out);
        if failures {
            // Alternate a slow valid request and a prompt rejected by the
            // front end. Rejected requests still wait for the client slot.
            std::fs::write(&prompts, format!("hello\n{}\n", "x".repeat(4097))).unwrap();
            command.arg("--prompts").arg(&prompts);
        }
        let status = command.output().unwrap();
        assert_eq!(
            status.status.code(),
            Some(if failures { 2 } else { 0 }),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        let report: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        std::fs::remove_file(out).unwrap();
        if failures {
            std::fs::remove_file(prompts).unwrap();
        }
        assert_eq!(report["summary"]["ok"], if failures { 4 } else { 8 });
        assert_eq!(report["summary"]["failed"], if failures { 4 } else { 0 });
        let records = report["records"].as_array().unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let gaps = Exp::new(1000.0).unwrap();
        let mut scheduled = Duration::ZERO;
        for record in records {
            if rate != "inf" {
                scheduled += Duration::from_secs_f64(gaps.sample(&mut rng));
            }
            let get = |key: &str| record[key].as_f64().unwrap();
            assert!((get("scheduled_ms") - scheduled.as_secs_f64() * 1e3).abs() < 1e-6);
            assert!((get("dispatch_ms") - get("scheduled_ms") - get("client_wait_ms")).abs() < 1e-6);
            if record["ok"] == true {
                assert!((get("arrival_ttfp_ms") - get("ttfp_ms") - get("client_wait_ms")).abs() < 1e-6);
            } else {
                assert!(record["ttfp_ms"].is_null() && record["arrival_ttfp_ms"].is_null());
                assert!(record["error"].as_str().unwrap().contains("400"));
            }
            assert!((get("arrival_e2e_ms") - get("e2e_ms") - get("client_wait_ms")).abs() < 1e-6);
        }
        // Slow requests share one client slot. The last request's
        // arrival latency must expose the accumulated wait, even though its
        // dispatch-to-completion latency is still just one request's work.
        let last = records
            .iter()
            .max_by(|a, b| a["dispatch_ms"].as_f64().partial_cmp(&b["dispatch_ms"].as_f64()).unwrap())
            .unwrap();
        assert!(last["client_wait_ms"].as_f64().unwrap() > if failures { 100.0 } else { 150.0 }, "{report:#}");
        if !failures {
            assert!(report["summary"]["client_wait_ms"]["p50"].as_f64().unwrap() > 50.0, "{report:#}");
            assert!(
                report["summary"]["arrival_e2e_ms"]["p50"].as_f64().unwrap()
                    > report["summary"]["e2e_ms"]["p50"].as_f64().unwrap() + 50.0,
                "{report:#}"
            );
        }
        if failures {
            let successful: Vec<_> = records.iter().filter(|r| r["ok"] == true).collect();
            let mean = |rows: &[&Value], key: &str| {
                rows.iter().map(|r| r[key].as_f64().unwrap()).sum::<f64>() / rows.len() as f64
            };
            let all: Vec<_> = records.iter().collect();
            for key in ["arrival_ttfp_ms", "arrival_e2e_ms", "e2e_ms"] {
                let expected = mean(&successful, key);
                let measured = report["summary"][key]["mean"].as_f64().unwrap();
                assert!((measured - expected).abs() < expected * 0.002 + 0.001, "{report:#}");
            }
            let reported_wait = report["summary"]["client_wait_ms"]["mean"].as_f64().unwrap();
            let expected_wait = mean(&all, "client_wait_ms");
            assert!((reported_wait - expected_wait).abs() < expected_wait * 0.002 + 0.001, "{report:#}");
        }
    }
}
