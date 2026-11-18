//! Optional, device-bound cuBLASLt algorithms for ordinary M=1 GEMMs.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use kern_manifest::types::GemmAlgo;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

/// A sparse selection of M=1 algorithms; empty means the default heuristic.
#[derive(Clone, Debug, Default)]
pub struct Algorithms {
    pins: BTreeMap<(u64, u64, u64), GemmAlgo>,
}

/// Algorithms are specific to a GPU model and cuBLASLt version.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Device {
    pub gpu: String,
    pub cublaslt: usize,
}

impl Device {
    /// Identify the GPU and loaded cuBLASLt library.
    pub fn of(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).context("CUDA context for GEMM profile")?;
        Ok(Self { gpu: context.name()?, cublaslt: unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() } })
    }
}

/// One measured shape and its complete cuBLASLt algorithm configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pin {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub algo: GemmAlgo,
}

/// Benchmark annotations may accompany these fields. They do not determine
/// whether a pin is numerically suitable for a checkpoint or workload.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pins {
    pub device: Device,
    pub gemms: Vec<Pin>,
}

impl Pins {
    /// Load a profile measured on this GPU and the current cuBLASLt version.
    pub fn load(path: &Path, ordinal: usize) -> Result<Algorithms> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read GEMM profile {}", path.display()))?;
        let pins: Pins =
            serde_json::from_str(&text).with_context(|| format!("parse GEMM profile {}", path.display()))?;
        pins.for_device(&Device::of(ordinal)?).with_context(|| format!("GEMM profile {}", path.display()))
    }

    /// Validate an already parsed profile against a known device identity.
    /// This also checks shape uniqueness and M=1 compatibility without CUDA.
    pub fn for_device(self, here: &Device) -> Result<Algorithms> {
        ensure!(
            self.device == *here,
            "GEMM profile device mismatch: measured on {:?}, current {here:?}; measure a profile on this GPU and cuBLASLt version",
            self.device
        );
        let mut algorithms = BTreeMap::new();
        for pin in self.gemms {
            ensure!(pin.m == 1, "Qwen3-TTS GEMM profiles support only m=1, got m={}", pin.m);
            ensure!(pin.n > 0 && pin.k > 0, "GEMM profile dimensions must be positive: n={}, k={}", pin.n, pin.k);
            let shape = (pin.m, pin.n, pin.k);
            ensure!(algorithms.insert(shape, pin.algo).is_none(), "duplicate GEMM profile shape {shape:?}");
        }
        Ok(Algorithms { pins: algorithms })
    }
}

impl Algorithms {
    /// Rewrite only calls whose row count is exactly the tokens or seqs var,
    /// returning the number of routed calls. Factor-two predictor rows, codec
    /// im2col multiples and accumulating GEMMs retain their original operations.
    /// A sparse profile leaves other shapes alone; an empty selection is a no-op.
    pub fn apply(&self, manifest: &mut Value) -> Result<usize> {
        if self.pins.is_empty() {
            return Ok(0);
        }
        let params = manifest["ops"]["gemm"]["params"].clone();
        ensure!(params.is_array(), "manifest has no ordinary GEMM signature");
        ensure!(
            manifest["ops"]["gemm"]["impl"] == json!({"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}),
            "manifest ordinary GEMM implementation changed"
        );
        let mut additions = BTreeMap::new();
        let mut count = 0;
        let programs = manifest["programs"].as_object_mut().context("manifest programs are not an object")?;
        for program in programs.values_mut() {
            let calls = program["calls"].as_array_mut().context("manifest calls are not an array")?;
            for call in calls {
                if call["op"] != "gemm" {
                    continue;
                }
                let Some(args) = call["args"].as_array().filter(|args| args.len() == 6) else { continue };
                let Some(row_arg) = args[3].as_object().filter(|arg| arg.len() == 1) else { continue };
                let Some(var) = row_arg.get("var").and_then(Value::as_str).filter(|v| matches!(*v, "seqs" | "tokens"))
                else {
                    continue;
                };
                let Some(n) = args[4].get("i32").and_then(Value::as_u64) else { continue };
                let Some(k) = args[5].get("i32").and_then(Value::as_u64) else { continue };
                let Some(algo) = self.pins.get(&(1, n, k)) else { continue };
                let name = format!("qwen.m1.{var}.{n}x{k}");
                let op = json!({
                    "params": params,
                    "impl": {"launches": [
                        {"entry": "extern:cublaslt_bf16_tn", "algo": algo, "when": {"var": var, "min": 1, "max": 1}},
                        {"entry": "extern:cublaslt_bf16_tn", "when": {"var": var, "min": 2}}
                    ]}
                });
                additions.entry(name.clone()).or_insert(op);
                call["op"] = json!(name);
                count += 1;
            }
        }
        let ordinary_used =
            programs.values().flat_map(|p| p["calls"].as_array().into_iter().flatten()).any(|c| c["op"] == "gemm");
        let ops = manifest["ops"].as_object_mut().context("manifest ops are not an object")?;
        // kern checks every declared op, so do not declare pins that no call uses.
        for (name, op) in additions {
            ensure!(!ops.contains_key(&name), "GEMM profile op name collision: {name}");
            ops.insert(name, op);
        }
        if !ordinary_used {
            ops.remove("gemm");
        }
        Ok(count)
    }
}
