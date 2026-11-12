//! Env-gated: OMNI_GEMM_TEST=1. Runs one cuBLASLt GEMM (4096 rows) via kern and
//! compares against a host f32 reference. Isolates the GEMM from the DiT.
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn gemm_4096_rows() {
    if std::env::var("OMNI_GEMM_TEST").is_err() {
        return;
    }
    let dir = std::env::var("OMNI_GEMM_TEST_DIR")
        .expect("set OMNI_GEMM_TEST_DIR to the directory holding gemm_{x,w,ref}.bin");
    let x = std::fs::read(format!("{dir}/gemm_x.bin")).unwrap();
    let w = std::fs::read(format!("{dir}/gemm_w.bin")).unwrap();
    let reference = std::fs::read(format!("{dir}/gemm_ref.bin")).unwrap();
    let n = 4096usize;
    let k = 4096usize;
    let rows = 4096usize;

    let mut g = omni_kern::Gen::default();
    // weights as bf16 from the f32-as-bf16 bytes: reinterpret
    let w_bf16: Vec<half::bf16> = w.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c)).collect();
    let w_f32: Vec<f32> = w_bf16.iter().map(|b| b.to_f32()).collect();
    let wb = g.weight("w", &[n, k], &w_f32);
    g.buffer("x", "bf16", json!([rows, k]), "input");
    g.buffer("y", "bf16", json!([rows, n]), "output");
    g.gemm_rows(
        "g",
        (omni_kern::buf("y"), omni_kern::buf("x"), omni_kern::buf(&wb)),
        omni_kern::count(&json!(rows)),
        (n, k),
    );
    let calls = g.take();
    let mut programs = serde_json::Map::new();
    programs.insert("run".into(), json!({"calls": calls}));
    g.finish(&mut programs);
    let (buffers, ops, tensors) = g.into_parts();
    let manifest = json!({"schema_version":5,"model":"gemmtest","vars":{},"states":{},
        "buffers":buffers,"modules":{},"ops":ops,"programs":programs});
    let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string()).unwrap()).unwrap();
    let mut rt = kern_runtime::Runtime::load(
        &verified,
        None,
        0,
        Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }),
        None,
    )
    .unwrap();
    rt.load_weights(&tensors).unwrap();
    rt.write_input("x", &x).unwrap();
    rt.run("run", &BTreeMap::new()).unwrap();
    let y = rt.read_output("y").unwrap();

    let yb: Vec<half::bf16> = y.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c)).collect();
    let rb: Vec<half::bf16> = reference.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c)).collect();
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in yb.iter().zip(rb.iter()) {
        let a = a.to_f32() as f64;
        let b = b.to_f32() as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("GEMM 4096 rows cosine = {cos:.6}");
    assert!(cos > 0.999, "GEMM wrong: {cos}");
}
