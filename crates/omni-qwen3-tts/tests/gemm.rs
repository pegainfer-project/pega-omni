//! Pure profile and manifest tests through the same public API Model::load uses.

use omni_qwen3_tts::gemm::Algorithms;
use omni_qwen3_tts::gemm::Device;
use omni_qwen3_tts::gemm::Pins;
use serde_json::Value;
use serde_json::json;

fn device() -> Device {
    Device { gpu: "test GPU".into(), cublaslt: 123 }
}

fn profile() -> Value {
    json!({
        "device": {"gpu": "test GPU", "cublaslt": 123},
        "benchmark": {"workload_sha256": "example"},
        "gemms": [{
            "m": 1, "n": 13, "k": 17,
            "algo": {"id": 42, "tile": 0, "stages": 0, "split_k": 1, "reduction": 0,
                     "swizzle": 0, "custom": 9, "inner_shape": 0, "cluster_shape": 0},
            "selected_us": 1.0, "speedup": 0.5,
            "bitwise_equal_default_on_probe": false
        }]
    })
}

fn algorithms() -> Algorithms {
    serde_json::from_value::<Pins>(profile()).unwrap().for_device(&device()).unwrap()
}

fn call(op: &str, rows: Value, n: u64, k: u64) -> Value {
    json!({"op": op, "args": [{"buf": "x"}, {"buf": "w"}, {"buf": "y"},
                              rows, {"i32": n}, {"i32": k}]})
}

fn manifest(calls: Vec<Value>) -> Value {
    json!({
        "ops": {
            "gemm": {"params": ["original ABI"], "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}},
            "gemm_acc": {"params": ["accumulating ABI"]}
        },
        "programs": {"decode": {"calls": calls}}
    })
}

#[test]
fn sparse_profile_accepts_annotations_and_arbitrary_algorithms() {
    let pins = algorithms();
    let mut m = manifest(vec![call("gemm", json!({"var": "seqs"}), 13, 17)]);
    assert_eq!(pins.apply(&mut m).unwrap(), 1);
    let algo = &m["ops"]["qwen.m1.seqs.13x17"]["impl"]["launches"][0]["algo"];
    assert_eq!(algo["id"], 42);
    assert_eq!(algo["custom"], 9);
}

#[test]
fn refuses_other_gpu_or_blas_version() {
    for here in [Device { gpu: "another GPU".into(), ..device() }, Device { cublaslt: 124, ..device() }] {
        let pins: Pins = serde_json::from_value(profile()).unwrap();
        let error = pins.for_device(&here).unwrap_err().to_string();
        assert!(error.contains("device mismatch"), "{error}");
        assert!(error.contains("current"), "{error}");
    }
}

#[test]
fn refuses_ambiguous_or_unsupported_shapes() {
    let mut duplicate = profile();
    let row = duplicate["gemms"][0].clone();
    duplicate["gemms"].as_array_mut().unwrap().push(row);
    let mut batch_two = profile();
    batch_two["gemms"][0]["m"] = json!(2);
    let mut empty_dimension = profile();
    empty_dimension["gemms"][0]["k"] = json!(0);
    for (input, message) in [(duplicate, "duplicate"), (batch_two, "only m=1"), (empty_dimension, "positive")] {
        let error = serde_json::from_value::<Pins>(input).unwrap().for_device(&device()).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn routes_only_exact_rows_with_an_unpinned_batch_fallback() {
    let calls = vec![
        call("gemm", json!({"var": "seqs"}), 13, 17),
        call("gemm", json!({"var": "tokens"}), 13, 17),
        call("gemm", json!({"var": "seqs"}), 19, 17),
        call("gemm", json!({"expr": {"mul": ["seqs", 2]}}), 13, 17),
        call("gemm_acc", json!({"var": "seqs"}), 13, 17),
    ];
    let mut m = manifest(calls.clone());
    assert_eq!(algorithms().apply(&mut m).unwrap(), 2);
    let rewritten = m["programs"]["decode"]["calls"].as_array().unwrap();
    assert_eq!(&rewritten[2..], &calls[2..]);
    for (i, var) in ["seqs", "tokens"].into_iter().enumerate() {
        assert_eq!(rewritten[i]["args"], calls[i]["args"]);
        let op = &m["ops"][rewritten[i]["op"].as_str().unwrap()];
        assert_eq!(op["params"], m["ops"]["gemm"]["params"]);
        let launches = op["impl"]["launches"].as_array().unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0]["when"], json!({"var": var, "min": 1, "max": 1}));
        assert_eq!(launches[0]["algo"]["id"], 42);
        assert_eq!(launches[1], json!({"entry": "extern:cublaslt_bf16_tn", "when": {"var": var, "min": 2}}));
    }
}

#[test]
fn emits_only_referenced_ops() {
    let mut unused = manifest(vec![call("gemm", json!({"var": "seqs"}), 19, 17)]);
    let before = unused.clone();
    assert_eq!(algorithms().apply(&mut unused).unwrap(), 0);
    assert_eq!(unused, before);
    assert_eq!(Algorithms::default().apply(&mut unused).unwrap(), 0);
    assert_eq!(unused, before);

    let mut all_pinned = manifest(vec![call("gemm", json!({"var": "seqs"}), 13, 17)]);
    assert_eq!(algorithms().apply(&mut all_pinned).unwrap(), 1);
    assert!(all_pinned["ops"].get("gemm").is_none());
    assert!(all_pinned["ops"].get("qwen.m1.seqs.13x17").is_some());
}

#[test]
fn default_algorithms_keep_the_manifest_unchanged_without_cuda() {
    let mut m = manifest(vec![call("gemm", json!({"var": "seqs"}), 13, 17)]);
    let before = m.clone();
    assert_eq!(Algorithms::default().apply(&mut m).unwrap(), 0);
    assert_eq!(m, before);
}
