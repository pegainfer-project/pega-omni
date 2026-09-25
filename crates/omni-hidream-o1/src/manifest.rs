//! Building a kern manifest: weights, buffers, one op per kernel call, and the
//! call lists the programs are made of.
//!
//! A launch's geometry lives in its op, and almost every call here has its own
//! shape, so every kernel call is an op of its own, named by its label. GEMMs
//! call one of two built-in ops: `gemm` (cuBLASLt's own choice) or `gemm_wide`
//! (its 256x128 tile, for the step's decoder GEMMs of thousands of rows).
//! Calls accumulate until [`Gen::take`] cuts them into a program.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use half::bf16;
use kern_manifest::types::DType;
use kern_runtime::Blob;
use kern_runtime::Tensor;
use kern_runtime::Tensors;
use serde_json::Value;
use serde_json::json;

/// The manifest under construction.
#[derive(Default)]
pub struct Gen {
    buffers: serde_json::Map<String, Value>,
    ops: serde_json::Map<String, Value>,
    calls: Vec<Value>,
    tensors: BTreeMap<String, (Vec<u64>, Vec<u8>)>,
}

impl Gen {
    /// A bf16 weight of `shape` holding `data`.
    pub fn weight(&mut self, name: &str, shape: &[usize], data: &[f32]) -> String {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        let bytes = data.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        self.buffers.insert(
            name.into(),
            json!({"dtype": "bf16", "shape": shape, "kind": "weight", "bind": [{"tensor": name}]}),
        );
        self.tensors.insert(name.into(), (shape.iter().map(|&d| d as u64).collect(), bytes));
        name.into()
    }

    /// A buffer that is not a weight, e.g. `("h", "bf16", [4097, 4096], "workspace")`.
    pub fn buffer(&mut self, name: &str, dtype: &str, shape: Value, kind: &str) {
        self.buffers.insert(name.into(), json!({"dtype": dtype, "shape": shape, "kind": kind}));
    }

    /// An `i32` input with the values it may take, e.g. `{"index_into": "embed"}`.
    pub fn input(&mut self, name: &str, shape: Value, domain: Value) {
        self.buffers.insert(name.into(), json!({"dtype": "i32", "shape": shape, "kind": "input", "domain": domain}));
    }

    /// One kernel launch as its own op, `args` typed by param; `smem` bytes of dynamic shared memory.
    pub fn launch(
        &mut self,
        label: &str,
        entry: &str,
        (grid, block, smem): ([Value; 3], u32, usize),
        args: Vec<(&str, Value)>,
    ) {
        let params: Vec<&str> = args.iter().map(|(t, _)| *t).collect();
        let mut launch = json!({"module": "hidream", "entry": entry, "block": [block, 1, 1], "grid": grid});
        if smem > 0 {
            launch["shared_mem"] = json!(smem);
        }
        self.ops.insert(label.into(), json!({"params": params, "impl": {"launches": [launch]}}));
        let args: Vec<Value> = args.into_iter().map(|(_, v)| v).collect();
        self.calls.push(json!({"label": label, "op": label, "args": args}));
    }

    /// `y[rows, n] = x[rows, k] · w[n, k]ᵀ`, on the wide tile when `wide`.
    pub fn gemm(
        &mut self,
        label: &str,
        (y, x, w): (Value, Value, Value),
        rows: Value,
        (n, k): (usize, usize),
        wide: bool,
    ) {
        let op = if wide { "gemm_wide" } else { "gemm" };
        self.calls.push(json!({"label": label, "op": op, "args": [x, w, y, rows, {"i32": n}, {"i32": k}]}));
    }

    /// The calls emitted since the last cut.
    pub fn take(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.calls)
    }

    /// The buffers, ops (the two GEMM ops added) and weights.
    pub fn into_parts(mut self) -> (serde_json::Map<String, Value>, serde_json::Map<String, Value>, HostTensors) {
        let gemm = |entry: &str| {
            json!({"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                   "impl": {"launches": [{"entry": entry}]}})
        };
        self.ops.insert("gemm".into(), gemm("extern:cublaslt_bf16_tn"));
        self.ops.insert("gemm_wide".into(), gemm("extern:cublaslt_bf16_tn_wide"));
        (self.buffers, self.ops, HostTensors(self.tensors))
    }
}

/// A row count (a var name or a var expression) as a call argument.
pub fn count(rows: &Value) -> Value {
    match rows {
        Value::String(v) => json!({"var": v}),
        e => json!({"expr": e}),
    }
}

pub fn buf(name: &str) -> Value {
    json!({"buf": name})
}

/// `name` from byte `offset` on.
pub fn buf_at(name: &str, offset: usize) -> Value {
    json!({"buf": name, "offset": offset})
}

pub fn i32a(v: impl TryInto<i32, Error: std::fmt::Debug>) -> (&'static str, Value) {
    ("i32", json!({"i32": v.try_into().expect("an i32 argument")}))
}

pub fn f32a(v: f32) -> (&'static str, Value) {
    ("f32", json!({"f32": v}))
}

/// A row count as an `i32` argument.
pub fn rows_arg(rows: &Value) -> (&'static str, Value) {
    ("i32", count(rows))
}

/// An argument of `ty`, e.g. `("in buffer<bf16>", "h")`.
pub fn arg(ty: &'static str, name: &str) -> (&'static str, Value) {
    (ty, buf(name))
}

pub fn arg_at(ty: &'static str, name: &str, offset: usize) -> (&'static str, Value) {
    (ty, buf_at(name, offset))
}

/// Weights after the load-time transforms, by buffer name.
pub struct HostTensors(BTreeMap<String, (Vec<u64>, Vec<u8>)>);

impl Tensors for HostTensors {
    fn find(&self, name: &str) -> kern_runtime::Result<Tensor<'_>> {
        let (shape, data) =
            self.0.get(name).ok_or_else(|| kern_runtime::Error::WeightArtifact(format!("no tensor `{name}`")))?;
        Ok(Tensor { dtype: DType::Bf16, shape: shape.clone(), data: Blob::Host(data) })
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Where kern finds the cubin: a per-user cache directory holding it under
/// its hash, `hidream-<sha12>.cubin`.
pub fn kernels_dir(sha: &str, bytes: &[u8]) -> anyhow::Result<PathBuf> {
    let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let dir = var("XDG_CACHE_HOME")
        .or_else(|| var("HOME").map(|h| h.join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("pega-omni/kernels");
    let path = dir.join(format!("hidream-{}.cubin", &sha[..12]));
    if !path.exists() {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let tmp = dir.join(format!(".hidream-{}.{}", &sha[..12], std::process::id()));
        std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("placing {}", path.display()))?;
    }
    Ok(dir)
}
