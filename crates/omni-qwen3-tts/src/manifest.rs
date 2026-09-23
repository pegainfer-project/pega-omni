//! Building a kern manifest: buffers, one op per call, and the call lists the
//! programs are made of.
//!
//! [`Gen`] collects everything a generator emits. A launch's geometry lives in
//! its op, and almost every call here has its own shape, so every kernel call
//! is an op of its own, named by its label; every GEMM calls the one
//! `extern:cublaslt_bf16_tn` op (`..._acc` when it accumulates). Calls accumulate until [`Gen::take`] cuts them
//! into a segment; programs are concatenations of segments.
//!
//! Per-sequence state is one `seq` state; generators carve it into regions
//! ([`Gen::region`]) and pass [`stride`] wherever a kernel addresses a
//! sequence's slot, patched to the final slot size by [`Gen::finish`].

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

/// Batch sizes a graph is captured at; a call pads up to the next one.
pub const BUCKETS: [usize; 12] = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// The smallest bucket holding `n` rows.
pub fn bucket(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

pub const THREADS: u32 = 256;

/// The manifest under construction.
#[derive(Default)]
pub struct Gen {
    pub buffers: serde_json::Map<String, Value>,
    pub ops: serde_json::Map<String, Value>,
    pub calls: Vec<Value>,
    pub tensors: BTreeMap<String, (DType, Vec<u64>, Vec<u8>)>,
    pub state_bytes: u64,
    /// Per-sequence width of each `seqs`-shaped workspace: the widest thing written into it.
    pub widths: BTreeMap<String, usize>,
    /// The module `launch` takes its kernels from. The codec's kernels are
    /// written for programmatic dependent launch (each waits on the grid
    /// before touching what an earlier launch produced); the talker's are not.
    pub module: &'static str,
}

impl Gen {
    pub fn weight(&mut self, name: &str, shape: &[usize], data: &[f32]) -> String {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        let bytes = data.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        self.add_weight(name, DType::Bf16, shape, bytes)
    }

    pub fn weight_f32(&mut self, name: &str, data: Vec<f32>) -> String {
        let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.add_weight(name, DType::F32, &[data.len()], bytes)
    }

    fn add_weight(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: Vec<u8>) -> String {
        let dt = if dtype == DType::F32 { "f32" } else { "bf16" };
        self.buffers
            .insert(name.into(), json!({"dtype": dt, "shape": shape, "kind": "weight", "bind": [{"tensor": name}]}));
        self.tensors.insert(name.into(), (dtype, shape.iter().map(|&d| d as u64).collect(), bytes));
        name.into()
    }

    /// A buffer that is not a weight, e.g. `("hidden", "bf16", [64, 2048], "carry")`.
    pub fn buffer(&mut self, name: &str, dtype: &str, shape: Value, kind: &str) {
        self.buffers.insert(name.into(), json!({"dtype": dtype, "shape": shape, "kind": kind}));
    }

    /// An input a kernel indexes with, e.g. a page table (`{"index_into": "kv0", "stride": 16}`).
    pub fn input(&mut self, name: &str, shape: Value, domain: Value) {
        self.buffers.insert(name.into(), json!({"dtype": "i32", "shape": shape, "kind": "input", "domain": domain}));
    }

    /// A per-sequence state region of `bytes`, 256-aligned; returns its offset.
    pub fn region(&mut self, bytes: usize) -> u64 {
        let at = self.state_bytes;
        self.state_bytes += (bytes as u64).div_ceil(256) * 256;
        at
    }

    /// One kernel launch of `self.module` as its own op, `args` typed by param.
    pub fn launch(&mut self, label: &str, entry: &str, grid: [Value; 3], block: u32, args: Vec<(&str, Value)>) {
        let params: Vec<&str> = args.iter().map(|(t, _)| *t).collect();
        self.ops.insert(
            label.into(),
            json!({"params": params, "impl": {"launches": [
                {"module": self.module, "entry": entry, "block": [block, 1, 1], "grid": grid,
                 "pdl": self.module == "codec"}
            ]}}),
        );
        let args: Vec<Value> = args.into_iter().map(|(_, v)| v).collect();
        self.calls.push(json!({"label": label, "op": label, "args": args}));
    }

    /// Elementwise over `n` values per stream.
    pub fn each(&mut self, label: &str, entry: &str, n: usize, mut args: Vec<(&str, Value)>) {
        args.push(("i32", json!({"expr": per_seq(n)})));
        self.launch(label, entry, [blocks(n), json!(1), json!(1)], THREADS, args);
    }

    /// Elementwise over `n` values per stream, eight (16 bytes) per thread.
    pub fn each8(&mut self, label: &str, entry: &str, n: usize, args: Vec<(&str, Value)>) {
        assert_eq!(n % 8, 0, "{label}: {n} values do not split into 16-byte groups");
        self.each(label, entry, n / 8, args);
    }

    /// `y[rows, n] = x[rows, k] · w[n, k]ᵀ` over `t` rows per stream.
    pub fn gemm(&mut self, label: &str, y: &'static str, x: &str, w: &str, t: usize, (n, k): (usize, usize)) {
        self.need(y, t * n);
        self.gemm_rows(label, (buf(y), buf(x), buf(w)), rows_arg(t), (n, k));
    }

    /// `y[rows, n] += x[rows, k] · w[n, k]ᵀ` over `t` rows per stream.
    pub fn gemm_acc(&mut self, label: &str, y: &'static str, x: &str, w: &str, t: usize, (n, k): (usize, usize)) {
        self.need(y, t * n);
        self.calls.push(json!({"label": label, "op": "gemm_acc", "args": [
            buf(x), buf(w), buf(y), rows_arg(t), {"i32": n}, {"i32": k}
        ]}));
    }

    /// `y = x · wᵀ` over `rows` rows, every operand a call argument (a buffer, maybe at an offset).
    pub fn gemm_rows(&mut self, label: &str, (y, x, w): (Value, Value, Value), rows: Value, (n, k): (usize, usize)) {
        self.calls.push(json!({"label": label, "op": "gemm", "args": [x, w, y, rows, {"i32": n}, {"i32": k}]}));
    }

    pub fn need(&mut self, workspace: &str, width: usize) {
        let w = self.widths.entry(workspace.into()).or_default();
        *w = (*w).max(width);
    }

    /// The calls emitted since the last cut.
    pub fn take(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.calls)
    }

    /// The per-sequence slot size, with every [`stride`] placeholder in
    /// `programs` patched to it and the `seqs`-shaped workspaces declared.
    pub fn finish(&mut self, programs: &mut serde_json::Map<String, Value>) -> u64 {
        let s = self.state_bytes;
        for c in programs.values_mut().flat_map(|p| p["calls"].as_array_mut().into_iter().flatten()) {
            for a in c["args"].as_array_mut().into_iter().flatten() {
                if a == &json!({"i64": STRIDE}) {
                    *a = json!({"i64": s});
                }
            }
        }
        for (name, w) in &self.widths {
            self.buffers.insert(name.clone(), json!({"dtype": "bf16", "shape": ["seqs", w], "kind": "workspace"}));
        }
        let gemm = |entry: &str, y: &str| {
            json!({"params": ["in buffer<bf16>", "in buffer<bf16>", y, "i32", "i32", "i32"],
                   "impl": {"launches": [{"entry": entry}]}})
        };
        self.ops.insert("gemm".into(), gemm("extern:cublaslt_bf16_tn", "out buffer<bf16>"));
        self.ops.insert("gemm_acc".into(), gemm("extern:cublaslt_bf16_tn_acc", "inout buffer<bf16>"));
        s
    }
}

/// `seqs · n`, as a var expression.
pub fn per_seq(n: usize) -> Value {
    if n == 1 { json!("seqs") } else { json!({"mul": ["seqs", n]}) }
}

/// A row count (a var expression) as a call argument.
pub fn count(rows: &Value) -> Value {
    match rows {
        Value::String(v) => json!({"var": v}),
        e => json!({"expr": e}),
    }
}

pub fn rows_arg(n: usize) -> Value {
    if n == 1 { json!({"var": "seqs"}) } else { json!({"expr": {"mul": ["seqs", n]}}) }
}

pub fn blocks(n: usize) -> Value {
    json!({"ceil_div": [per_seq(n), THREADS]})
}

pub fn buf(name: &str) -> Value {
    json!({"buf": name})
}

pub fn buf_at(name: &str, offset: usize) -> Value {
    json!({"buf": name, "offset": offset})
}

pub fn i32a(v: usize) -> (&'static str, Value) {
    ("i32", json!({"i32": v as i32}))
}

pub fn f32a(v: f32) -> (&'static str, Value) {
    ("f32", json!({"f32": v}))
}

pub fn inb(name: &str) -> (&'static str, Value) {
    ("in buffer<bf16>", buf(name))
}

pub fn ini(name: &str) -> (&'static str, Value) {
    ("in buffer<i32>", buf(name))
}

pub fn inf(name: &str) -> (&'static str, Value) {
    ("in buffer<f32>", buf(name))
}

pub fn io(name: &str) -> (&'static str, Value) {
    ("inout buffer<bf16>", buf(name))
}

pub fn outb(name: &str) -> (&'static str, Value) {
    ("out buffer<bf16>", buf(name))
}

pub fn state_in(offset: u64) -> (&'static str, Value) {
    ("in state", json!({"state": "seq", "offset": offset}))
}

pub fn state_io(offset: u64) -> (&'static str, Value) {
    ("inout state", json!({"state": "seq", "offset": offset}))
}

/// Placeholder for the per-sequence state size, patched by [`Gen::finish`].
const STRIDE: i64 = -1;

pub fn stride() -> (&'static str, Value) {
    ("i64", json!({"i64": STRIDE}))
}

/// Weights after the load-time transforms, by buffer name.
pub struct HostTensors(pub BTreeMap<String, (DType, Vec<u64>, Vec<u8>)>);

impl Tensors for HostTensors {
    fn find(&self, name: &str) -> kern_runtime::Result<Tensor<'_>> {
        let (dtype, shape, data) =
            self.0.get(name).ok_or_else(|| kern_runtime::Error::WeightArtifact(format!("no tensor `{name}`")))?;
        Ok(Tensor { dtype: *dtype, shape: shape.clone(), data: Blob::Host(data) })
    }
}

pub fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only ever called on i32 and f32 slices, which have no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Where kern finds the cubins: a per-user cache directory holding each under
/// its hash, e.g. `codec-<sha12>.cubin`.
pub fn kernels_dir(cubins: &[(&str, &str, &[u8])]) -> anyhow::Result<PathBuf> {
    let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let dir = var("XDG_CACHE_HOME")
        .or_else(|| var("HOME").map(|h| h.join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("pega-omni/kernels");
    for (name, sha, bytes) in cubins {
        let path = dir.join(format!("{name}-{}.cubin", &sha[..12]));
        if !path.exists() {
            std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let tmp = dir.join(format!(".{name}-{}.{}", &sha[..12], std::process::id()));
            std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
            std::fs::rename(&tmp, &path).with_context(|| format!("placing {}", path.display()))?;
        }
    }
    Ok(dir)
}
