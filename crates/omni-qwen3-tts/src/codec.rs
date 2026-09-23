//! The codec decoder, streamed: every call turns one new frame of each of up
//! to `max_seqs` streams into 1920 samples, with nothing recomputed.
//!
//! RVQ lookup → causal conv → sliding-window transformer → two ×2 upsamplers
//! (transposed conv + ConvNeXt) → four SnakeBeta/transposed-conv blocks
//! (×8, ×5, ×4, ×3) with dilated residual units → SnakeBeta, conv, clamp.
//! Every layer is causal, so what a frame needs from the past is small and
//! fixed: each conv's last `(k - 1) · dilation` input rows, each overlapping
//! transposed conv's last GEMM row, each attention layer's last 72 frames of K
//! and V. A [`Stream`] owns a slot of kern per-sequence state holding exactly
//! that, and a call reads and advances it (`kernels/codec.cu` has the layout).
//! The output is the whole-utterance decode's, frame by frame.
//!
//! The decoder is a kern manifest this module generates from the checkpoint's
//! config: one `decode` program over a `seqs` var, captured as a CUDA graph per
//! batch bucket, executed by `kern-runtime` on its own stream. Activations are
//! `[rows, C]`; every dense conv is `im2col` and a GEMM, every transposed conv
//! a GEMM and an overlap-add; a residual branch's last GEMM accumulates into
//! its stream. Weight layout transforms and scale folding happen on the host at
//! load (LayerScale into the projections it scales, ConvNeXt's gamma into its
//! last linear, a bias into the linear op it feeds), then reach kern as named
//! tensors.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use kern_manifest::types::DType;
use kern_pool::Lease;
use kern_runtime::Blob;
use kern_runtime::Capacity;
use kern_runtime::Runtime;
use kern_runtime::Tensor;
use kern_runtime::Tensors;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;

use crate::config;
use crate::weights::File;
use crate::weights::Host;
use crate::weights::concat_rows;
use crate::weights::conv_taps;
use crate::weights::scale_rows;
use crate::weights::transposed_taps;

const GROUPS: usize = 16;
const KERNEL: usize = 7;
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/codec.cubin"));
/// Batch sizes a graph is captured at; a call pads up to the next one.
const BUCKETS: [usize; 12] = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// One stream's decoder state; dropping it frees the slot.
pub struct Stream {
    lease: Lease,
    frames: usize,
}

pub struct Codec {
    rt: Runtime,
    max_seqs: usize,
    samples_per_frame: usize,
    /// The slot padding rows decode into.
    pad: Lease,
}

impl Codec {
    /// Loads `speech_tokenizer/model.safetensors` onto `gpu` for up to `max_seqs` streams.
    pub fn load(
        gpu: usize,
        file: &File,
        cfg: &config::Codec,
        samples_per_frame: usize,
        max_seqs: usize,
    ) -> Result<Self> {
        let max_seqs = BUCKETS.iter().copied().find(|&b| b >= max_seqs).unwrap_or(max_seqs);
        let sha = hex(&sha2::Sha256::digest(CUBIN));
        let (manifest, tensors) = generate(file, cfg, samples_per_frame, max_seqs, &sha)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("codec manifest: {e}"))?;
        let dir = kernels_dir(&sha)?;
        let capacity = Capacity { tokens: Some(1), seqs: max_seqs as u64 + 1 };
        let mut rt = Runtime::load(&verified, Some(&dir), gpu, Some(capacity), None)?;
        rt.load_weights(&tensors)?;
        let pad = rt.lease_slot()?;
        Ok(Self { rt, max_seqs, samples_per_frame, pad })
    }

    /// A fresh stream: the utterance starts at its next frame.
    pub fn open(&mut self) -> Result<Stream> {
        Ok(Stream { lease: self.rt.lease_slot()?, frames: 0 })
    }

    /// Decodes the next frame of each stream: `samples_per_frame` samples in
    /// [-1, 1] per stream, concatenated in order.
    pub fn decode(&mut self, frames: &mut [(&mut Stream, [i32; GROUPS])]) -> Result<Vec<f32>> {
        let n = frames.len();
        if n == 0 {
            return Ok(vec![]);
        }
        ensure!(n <= self.max_seqs, "{n} streams exceed the decoder's {}", self.max_seqs);
        let seqs = BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(self.max_seqs);
        let pad_line = self.pad.seq_line("lines", 0)?;
        let mut codes = Vec::with_capacity(seqs * GROUPS);
        let mut pos = Vec::with_capacity(seqs);
        let mut lines = Vec::with_capacity(seqs);
        for (s, c) in frames.iter_mut() {
            codes.extend_from_slice(c);
            pos.push(s.frames as i32);
            lines.push(s.lease.seq_line("lines", 0)?);
            s.frames += 1;
        }
        codes.resize(seqs * GROUPS, 0);
        pos.resize(seqs, 0);
        lines.resize(seqs, pad_line);
        let vars = BTreeMap::from([("seqs".to_string(), seqs as u64)]);
        self.rt.write_input_at("codes", bytes_of(&codes), &vars)?;
        self.rt.write_input_at("pos", bytes_of(&pos), &vars)?;
        self.rt.write_input_at("lines", bytes_of(&lines), &vars)?;
        self.rt.issue("decode", &vars)?;
        let wav = self.rt.read_output("wav")?;
        Ok(wav[..n * self.samples_per_frame * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16::from_le_bytes(b).to_f32())
            .collect())
    }
}

fn bytes_of(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and u8 no alignment requirement.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Where kern finds the cubin: a directory holding it under its hash.
fn kernels_dir(sha: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("pega-omni-kernels");
    let path = dir.join(format!("codec-{}.cubin", &sha[..12]));
    if !path.exists() {
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!(".codec-{}.{}", &sha[..12], std::process::id()));
        std::fs::write(&tmp, CUBIN)?;
        std::fs::rename(&tmp, &path).with_context(|| format!("placing {}", path.display()))?;
    }
    Ok(dir)
}

/// The decoder's weights after the load-time transforms, by buffer name.
struct HostTensors(BTreeMap<String, (DType, Vec<u64>, Vec<u8>)>);

impl Tensors for HostTensors {
    fn find(&self, name: &str) -> kern_runtime::Result<Tensor<'_>> {
        let (dtype, shape, data) =
            self.0.get(name).ok_or_else(|| kern_runtime::Error::WeightArtifact(format!("no tensor `{name}`")))?;
        Ok(Tensor { dtype: *dtype, shape: shape.clone(), data: Blob::Host(data) })
    }
}

/// The manifest under construction: one op per call, since a launch's
/// geometry lives in its op and every call here has its own shape.
struct Gen {
    buffers: serde_json::Map<String, Value>,
    ops: serde_json::Map<String, Value>,
    calls: Vec<Value>,
    tensors: BTreeMap<String, (DType, Vec<u64>, Vec<u8>)>,
    state_bytes: u64,
    /// Per-stream width of each workspace: the widest thing written into it.
    widths: BTreeMap<&'static str, usize>,
}

const THREADS: u32 = 256;

/// `seqs · n`, as a var expression.
fn per_seq(n: usize) -> Value {
    if n == 1 { json!("seqs") } else { json!({"mul": ["seqs", n]}) }
}

fn rows_arg(n: usize) -> Value {
    if n == 1 { json!({"var": "seqs"}) } else { json!({"expr": {"mul": ["seqs", n]}}) }
}

fn blocks(n: usize) -> Value {
    json!({"ceil_div": [per_seq(n), THREADS]})
}

fn i32a(v: usize) -> (&'static str, Value) {
    ("i32", json!({"i32": v as i32}))
}

fn f32a(v: f32) -> (&'static str, Value) {
    ("f32", json!({"f32": v}))
}

impl Gen {
    fn weight(&mut self, name: &str, shape: &[usize], data: &[f32]) -> String {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        let bytes = data.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        self.add_weight(name, DType::Bf16, shape, bytes)
    }

    fn weight_f32(&mut self, name: &str, data: Vec<f32>) -> String {
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

    fn snake(&mut self, file: &File, prefix: &str, c: usize) -> Result<(String, String)> {
        let alpha = file.expect(&format!("{prefix}.alpha"), &[c])?;
        let beta = file.expect(&format!("{prefix}.beta"), &[c])?;
        Ok((
            self.weight_f32(&format!("{prefix}.a"), alpha.data.iter().map(|x| x.exp()).collect()),
            self.weight_f32(&format!("{prefix}.inv_b"), beta.data.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect()),
        ))
    }

    /// A per-stream state region of `bytes`, 256-aligned; returns its offset.
    fn region(&mut self, bytes: usize) -> u64 {
        let at = self.state_bytes;
        self.state_bytes += (bytes as u64).div_ceil(256) * 256;
        at
    }

    /// One kernel launch as its own op, `args` typed by param.
    fn launch(&mut self, label: &str, entry: &str, grid: [Value; 3], block: u32, args: Vec<(&str, Value)>) {
        let params: Vec<&str> = args.iter().map(|(t, _)| *t).collect();
        self.ops.insert(
            label.into(),
            json!({"params": params, "impl": {"launches": [
                {"module": "codec", "entry": entry, "block": [block, 1, 1], "grid": grid, "pdl": true}
            ]}}),
        );
        let args: Vec<Value> = args.into_iter().map(|(_, v)| v).collect();
        self.calls.push(json!({"label": label, "op": label, "args": args}));
    }

    /// Elementwise over `n` values per stream.
    fn each(&mut self, label: &str, entry: &str, n: usize, mut args: Vec<(&str, Value)>) {
        args.push(("i32", json!({"expr": per_seq(n)})));
        self.launch(label, entry, [blocks(n), json!(1), json!(1)], THREADS, args);
    }

    /// Elementwise over `n` values per stream, eight (16 bytes) per thread.
    fn each8(&mut self, label: &str, entry: &str, n: usize, args: Vec<(&str, Value)>) {
        assert_eq!(n % 8, 0, "{label}: {n} values do not split into 16-byte groups");
        self.each(label, entry, n / 8, args);
    }

    /// `y[rows, n] = x[rows, k] · w[n, k]ᵀ` over `t` rows per stream.
    fn gemm(&mut self, label: &str, y: &'static str, x: &str, w: &str, t: usize, (n, k): (usize, usize)) {
        self.gemm_op("gemm", label, y, x, w, t, (n, k));
    }

    /// `y[rows, n] += x[rows, k] · w[n, k]ᵀ` over `t` rows per stream.
    fn gemm_acc(&mut self, label: &str, y: &'static str, x: &str, w: &str, t: usize, (n, k): (usize, usize)) {
        self.gemm_op("gemm_acc", label, y, x, w, t, (n, k));
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_op(&mut self, op: &str, label: &str, y: &'static str, x: &str, w: &str, t: usize, (n, k): (usize, usize)) {
        self.need(y, t * n);
        self.calls.push(json!({"label": label, "op": op, "args": [
            {"buf": x}, {"buf": w}, {"buf": y}, rows_arg(t), {"i32": n}, {"i32": k}
        ]}));
    }

    fn need(&mut self, workspace: &'static str, width: usize) {
        let w = self.widths.entry(workspace).or_default();
        *w = (*w).max(width);
    }

    /// `out = conv(act(x + bias))` for a causal conv of `k` taps dilated by
    /// `d`, `act` SnakeBeta when given; the conv's own bias is left to its
    /// consumer. `act(x + bias)` of the last `(k - 1)·d` rows is its history.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        label: &str,
        (x, bias, act): (&str, &str, Option<&(String, String)>),
        (w, out): (&str, &'static str),
        t: usize,
        (cin, cout): (usize, usize),
        (k, d): (usize, usize),
    ) {
        let h = (k - 1) * d;
        let history = self.region(2 * h * cin * 2);
        self.need("col", t * k * cin);
        let mut args = vec![inb(x), inb(bias)];
        let entry = match act {
            Some((a, inv_b)) => {
                args.extend([inf(a), inf(inv_b)]);
                "codec_im2col_snake"
            }
            None => "codec_im2col",
        };
        args.extend([
            state_io(history),
            ini("pos"),
            ini("lines"),
            outb("col"),
            i32a(t),
            i32a(cin),
            i32a(k),
            i32a(d),
            stride(),
        ]);
        self.each8(&format!("{label}.im2col"), entry, (h + t) * cin, args);
        self.gemm(&format!("{label}.gemm"), out, "col", w, t, (cout, k * cin));
    }
}

fn inb(name: &str) -> (&'static str, Value) {
    ("in buffer<bf16>", json!({"buf": name}))
}

fn ini(name: &str) -> (&'static str, Value) {
    ("in buffer<i32>", json!({"buf": name}))
}

fn inf(name: &str) -> (&'static str, Value) {
    ("in buffer<f32>", json!({"buf": name}))
}

fn io(name: &str) -> (&'static str, Value) {
    ("inout buffer<bf16>", json!({"buf": name}))
}

fn outb(name: &str) -> (&'static str, Value) {
    ("out buffer<bf16>", json!({"buf": name}))
}

fn state_io(offset: u64) -> (&'static str, Value) {
    ("inout state", json!({"state": "codec", "offset": offset}))
}

/// Placeholder for the per-stream state size, patched once every region exists.
const STRIDE: i64 = -1;

fn stride() -> (&'static str, Value) {
    ("i64", json!({"i64": STRIDE}))
}

/// `w[n, k] · v[k]`.
fn matvec(w: &[f32], v: &[f32]) -> Vec<f32> {
    w.chunks(v.len()).map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

fn plus(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// A `[rows, cols]` matrix as `[cols, rows]`.
fn transpose(w: &[f32], rows: usize) -> Vec<f32> {
    let cols = w.len() / rows;
    (0..cols).flat_map(|j| (0..rows).map(move |r| w[r * cols + j])).collect()
}

/// The manifest (JSON) and the tensors its weight buffers bind.
fn generate(file: &File, cfg: &config::Codec, spf: usize, max_seqs: usize, sha: &str) -> Result<(Value, HostTensors)> {
    let (dim, cb, latent, hidden) = (cfg.codebook_dim, cfg.codebook_size, cfg.latent_dim, cfg.hidden_size);
    let half = dim / 2;
    ensure!(cfg.num_quantizers == GROUPS, "{} quantizers, the decoder is built for {GROUPS}", cfg.num_quantizers);
    ensure!(cfg.head_dim == 64, "codec head_dim {} unsupported (attention is built for 64)", cfg.head_dim);
    ensure!(cfg.sliding_window == 72, "codec window {} unsupported (the KV ring is built for 72)", cfg.sliding_window);
    ensure!(
        cfg.upsampling_ratios.iter().chain(&cfg.upsample_rates).product::<usize>() == spf,
        "the upsampling rates do not multiply to {spf} samples per frame"
    );
    ensure!(
        hidden.is_multiple_of(256) && hidden <= 1024,
        "codec hidden size {hidden} unsupported (the norms need 256 | h ≤ 1024)"
    );
    ensure!(latent.is_multiple_of(8) && latent / 8 <= 1024, "codec latent dim {latent} unsupported");
    let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
    ensure!(out_dim.is_multiple_of(8) && out_dim <= 128, "codec output conv width {out_dim} unsupported (at most 128)");
    let mut g = Gen {
        buffers: serde_json::Map::new(),
        ops: serde_json::Map::new(),
        calls: vec![],
        tensors: BTreeMap::new(),
        state_bytes: 0,
        widths: BTreeMap::new(),
    };

    // RVQ: [first | Σ rest] → one projection.
    let codebook = |p: &str| -> Result<Vec<f32>> {
        let sum = file.expect(&format!("{p}._codebook.embedding_sum"), &[cb, half])?;
        let usage = file.expect(&format!("{p}._codebook.cluster_usage"), &[cb])?;
        Ok(sum
            .data
            .chunks(half)
            .zip(&usage.data)
            .flat_map(|(row, &u)| row.iter().map(move |x| x / u.max(1e-5)))
            .collect())
    };
    let books: Vec<f32> = std::iter::once(codebook("decoder.quantizer.rvq_first.vq.layers.0"))
        .chain((0..GROUPS - 1).map(|i| codebook(&format!("decoder.quantizer.rvq_rest.vq.layers.{i}"))))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let books = g.weight("rvq.books", &[GROUPS * cb, half], &books);
    let proj = |p: &str| file.expect(&format!("decoder.quantizer.{p}.output_proj.weight"), &[dim, half, 1]);
    let (first, rest) = (proj("rvq_first")?, proj("rvq_rest")?);
    let rvq_out: Vec<f32> = (0..dim)
        .flat_map(|r| first.data[r * half..(r + 1) * half].iter().chain(&rest.data[r * half..(r + 1) * half]))
        .copied()
        .collect();
    let rvq_out = g.weight("rvq.out", &[dim, dim], &rvq_out);
    g.launch(
        "rvq",
        "codec_rvq",
        [json!("seqs"), json!(1), json!(1)],
        (half / 8) as u32,
        vec![ini("codes"), inb(&books), outb("col"), i32a(half), i32a(cb)],
    );
    g.need("col", dim);
    g.gemm("rvq.proj", "a", "col", &rvq_out, 1, (dim, dim));

    let conv_w = |g: &mut Gen, prefix: &str, name: &str, shape: [usize; 3]| -> Result<(String, Vec<f32>)> {
        let w = file.expect(&format!("{prefix}.weight"), &shape)?;
        let b = file.expect(&format!("{prefix}.bias"), &shape[..1])?;
        Ok((g.weight(&format!("{name}.w"), &[shape[0], shape[2] * shape[1]], &conv_taps(&w)), b.data))
    };
    let zeros = g.weight("zeros", &[dim.max(latent)], &vec![0.0; dim.max(latent)]);
    let (pre, pre_b) = conv_w(&mut g, "decoder.pre_conv.conv", "pre_conv", [latent, dim, 3])?;
    g.conv("pre_conv", ("a", &zeros, None), (&pre, "b"), 1, (dim, latent), (3, 1));

    // Transformer over the frames, residual stream in `res`; pre_conv's bias
    // goes through input_proj.
    let pt = "decoder.pre_transformer";
    let linear = |p: &str, shape: [usize; 2]| -> Result<(Host, Host)> {
        Ok((file.expect(&format!("{p}.weight"), &shape)?, file.expect(&format!("{p}.bias"), &shape[..1])?))
    };
    let (in_w, in_b) = linear(&format!("{pt}.input_proj"), [hidden, latent])?;
    let in_b = g.weight("input_proj.b", &[hidden], &plus(&in_b.data, &matvec(&in_w.data, &pre_b)));
    let in_w = g.weight("input_proj.w", &[hidden, latent], &in_w.data);
    g.gemm("input_proj", "res", "b", &in_w, 1, (hidden, latent));
    let (heads, hd, inter) = (cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size);
    let qd = heads * hd;
    let norm_w = |g: &mut Gen, name: &str, src: &str| -> Result<String> {
        Ok(g.weight(name, &[hidden], &file.expect(src, &[hidden])?.data))
    };
    let ln = norm_w(&mut g, "l0.ln1", &format!("{pt}.layers.0.input_layernorm.weight"))?;
    let eps = cfg.rms_norm_eps;
    let norm_grid = [json!({"ceil_div": ["seqs", 4]}), json!(1), json!(1)];
    g.launch(
        "l0.norm",
        "codec_bias_rms_norm",
        norm_grid.clone(),
        128,
        vec![io("res"), inb(&in_b), inb(&ln), outb("x"), i32a(hidden), f32a(eps), ("i32", json!({"var": "seqs"}))],
    );
    g.need("x", hidden);
    g.need("a", qd.max(inter));
    for i in 0..cfg.num_hidden_layers {
        let w = |n: &str, shape: &[usize]| file.expect(&format!("{pt}.layers.{i}.{n}"), shape);
        let attn_scale = w("self_attn_layer_scale.scale", &[hidden])?;
        let mlp_scale = w("mlp_layer_scale.scale", &[hidden])?;
        let qkv = concat_rows(&[
            w("self_attn.q_proj.weight", &[qd, hidden])?,
            w("self_attn.k_proj.weight", &[qd, hidden])?,
            w("self_attn.v_proj.weight", &[qd, hidden])?,
        ]);
        let qkv = g.weight(&format!("l{i}.qkv"), &[3 * qd, hidden], &qkv);
        let o = g.weight(
            &format!("l{i}.o"),
            &[hidden, qd],
            &scale_rows(&w("self_attn.o_proj.weight", &[hidden, qd])?.data, &attn_scale.data),
        );
        let ln2 = norm_w(&mut g, &format!("l{i}.ln2"), &format!("{pt}.layers.{i}.post_attention_layernorm.weight"))?;
        let gate_up =
            concat_rows(&[w("mlp.gate_proj.weight", &[inter, hidden])?, w("mlp.up_proj.weight", &[inter, hidden])?]);
        let gate_up = g.weight(&format!("l{i}.gate_up"), &[2 * inter, hidden], &gate_up);
        let down = g.weight(
            &format!("l{i}.down"),
            &[hidden, inter],
            &scale_rows(&w("mlp.down_proj.weight", &[hidden, inter])?.data, &mlp_scale.data),
        );
        let next = if i + 1 < cfg.num_hidden_layers {
            norm_w(&mut g, &format!("l{}.ln1", i + 1), &format!("{pt}.layers.{}.input_layernorm.weight", i + 1))?
        } else {
            norm_w(&mut g, "norm", &format!("{pt}.norm.weight"))?
        };
        let add_norm = |g: &mut Gen, label: String, w: &str| {
            g.launch(
                &label,
                "codec_add_rms_norm",
                norm_grid.clone(),
                128,
                vec![io("x"), io("res"), inb(w), i32a(hidden), f32a(eps), ("i32", json!({"var": "seqs"}))],
            );
        };

        let kv = g.region(2 * 72 * qd * 2);
        g.gemm(&format!("l{i}.qkv"), "col", "x", &qkv, 1, (3 * qd, hidden));
        g.launch(
            &format!("l{i}.attn"),
            "codec_attention",
            [json!("seqs"), json!(heads), json!(1)],
            128,
            vec![
                inb("col"),
                ini("pos"),
                ini("lines"),
                state_io(kv),
                outb("a"),
                stride(),
                i32a(heads),
                f32a(cfg.rope_theta),
            ],
        );
        g.gemm(&format!("l{i}.o"), "x", "a", &o, 1, (hidden, qd));
        add_norm(&mut g, format!("l{i}.post_attn_norm"), &ln2);
        g.gemm(&format!("l{i}.gate_up"), "col", "x", &gate_up, 1, (2 * inter, hidden));
        g.each8(&format!("l{i}.silu_mul"), "codec_silu_mul", inter, vec![inb("col"), outb("a"), i32a(inter)]);
        g.gemm(&format!("l{i}.down"), "x", "a", &down, 1, (hidden, inter));
        add_norm(&mut g, format!("l{i}.next_norm"), &next);
    }
    let (out_w, out_b) = linear(&format!("{pt}.output_proj"), [latent, hidden])?;
    let out_w = g.weight("output_proj.w", &[latent, hidden], &out_w.data);
    g.gemm("output_proj", "a", "x", &out_w, 1, (latent, hidden));

    // Upsamplers: transposed conv (kernel = stride, no overlap: its GEMM rows
    // are the output rows), then ConvNeXt, whose residual accumulates into its
    // input. `cur` holds the stage input and `pending` the bias it still
    // lacks, which the transposed conv folds into its own (per tap).
    let (mut cur, mut tmp) = ("a", "b");
    let mut pending = out_b.data;
    let mut t = 1;
    for (i, &r) in cfg.upsampling_ratios.iter().enumerate() {
        let p = format!("decoder.upsample.{i}");
        let up = transposed_taps(&file.expect(&format!("{p}.0.conv.weight"), &[latent, latent, r])?);
        let up_b = file.expect(&format!("{p}.0.conv.bias"), &[latent])?;
        let zb: Vec<f32> = matvec(&up, &pending).chunks(latent).flat_map(|tap| plus(tap, &up_b.data)).collect();
        let up_w = g.weight(&format!("up{i}.w"), &[r * latent, latent], &up);
        let zb = g.weight(&format!("up{i}.b"), &[r, latent], &zb);
        let dw = file.expect(&format!("{p}.1.dwconv.conv.weight"), &[latent, 1, KERNEL])?;
        let dw_w = g.weight(&format!("up{i}.dw.w"), &[KERNEL, latent], &transpose(&dw.data, latent));
        let dw_b = g.weight(
            &format!("up{i}.dw.b"),
            &[latent],
            &file.expect(&format!("{p}.1.dwconv.conv.bias"), &[latent])?.data,
        );
        let ln_w =
            g.weight(&format!("up{i}.ln.w"), &[latent], &file.expect(&format!("{p}.1.norm.weight"), &[latent])?.data);
        let ln_b =
            g.weight(&format!("up{i}.ln.b"), &[latent], &file.expect(&format!("{p}.1.norm.bias"), &[latent])?.data);
        let (pw1_w, pw1_b) = linear(&format!("{p}.1.pwconv1"), [4 * latent, latent])?;
        let pw1_w = g.weight(&format!("up{i}.pw1.w"), &[4 * latent, latent], &pw1_w.data);
        let pw1_b = g.weight(&format!("up{i}.pw1.b"), &[4 * latent], &pw1_b.data);
        let gamma = file.expect(&format!("{p}.1.gamma"), &[latent])?;
        let pw2: Host = file.expect(&format!("{p}.1.pwconv2.weight"), &[latent, 4 * latent])?;
        let pw2_b = file.expect(&format!("{p}.1.pwconv2.bias"), &[latent])?;
        let pw2_w = g.weight(&format!("up{i}.pw2.w"), &[latent, 4 * latent], &scale_rows(&pw2.data, &gamma.data));

        g.gemm(&format!("up{i}.gemm"), "col", cur, &up_w, t, (r * latent, latent));
        t *= r;
        let history = g.region(2 * (KERNEL - 1) * latent * 2);
        g.launch(
            &format!("up{i}.dwconv_ln"),
            "codec_dwconv_ln",
            [per_seq(t), json!(1), json!(1)],
            (latent / 8) as u32,
            vec![
                inb("col"),
                inb(&zb),
                inb(&dw_w),
                inb(&dw_b),
                inb(&ln_w),
                inb(&ln_b),
                state_io(history),
                ini("pos"),
                ini("lines"),
                outb(tmp),
                outb("x"),
                i32a(t),
                i32a(r),
                i32a(latent),
                i32a(KERNEL),
                stride(),
                f32a(1e-6),
            ],
        );
        g.gemm(&format!("up{i}.pw1"), "col", "x", &pw1_w, t, (4 * latent, latent));
        g.each8(
            &format!("up{i}.pw1.bias"),
            "codec_bias_gelu",
            t * 4 * latent,
            vec![io("col"), inb(&pw1_b), i32a(4 * latent)],
        );
        g.gemm_acc(&format!("up{i}.pw2"), tmp, "col", &pw2_w, t, (latent, 4 * latent));
        pending = scale_rows(&pw2_b.data, &gamma.data);
        g.need("x", t * latent);
        (cur, tmp) = (tmp, cur);
    }

    let pending_w = g.weight("conv_in.in_b", &[latent], &pending);
    let (conv_in, conv_in_b) = conv_w(&mut g, "decoder.decoder.0.conv", "conv_in", [cfg.decoder_dim, latent, KERNEL])?;
    g.conv("conv_in", (cur, &pending_w, None), (&conv_in, tmp), t, (latent, cfg.decoder_dim), (KERNEL, 1));
    (cur, tmp) = (tmp, cur);
    let mut pending = conv_in_b;

    // Decoder blocks: SnakeBeta, transposed conv (overlap-add into `cur`),
    // then residual units whose second conv accumulates into `cur`, its bias
    // carried to the next consumer.
    for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
        let p = format!("decoder.decoder.{}.block", i + 1);
        let (cin, cout) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
        let snake = g.snake(file, &format!("{p}.0"), cin)?;
        let in_b = g.weight(&format!("b{i}.in_b"), &[cin], &pending);
        let up = file.expect(&format!("{p}.1.conv.weight"), &[cin, cout, 2 * rate])?;
        let up_w = g.weight(&format!("b{i}.up.w"), &[2 * rate * cout, cin], &transposed_taps(&up));
        let up_b = g.weight(&format!("b{i}.up.b"), &[cout], &file.expect(&format!("{p}.1.conv.bias"), &[cout])?.data);

        g.each8(
            &format!("b{i}.snake"),
            "codec_bias_snake",
            t * cin,
            vec![inb(cur), inb(&in_b), inf(&snake.0), inf(&snake.1), outb(tmp), i32a(cin)],
        );
        g.gemm(&format!("b{i}.up"), "col", tmp, &up_w, t, (2 * rate * cout, cin));
        let prev = g.region(rate * cout * 2);
        g.each8(
            &format!("b{i}.col2im"),
            "codec_col2im",
            t * rate * cout,
            vec![
                inb("col"),
                inb(&up_b),
                state_io(prev),
                ini("lines"),
                outb(cur),
                i32a(t),
                i32a(rate),
                i32a(cout),
                stride(),
            ],
        );
        g.need(tmp, t * cin);
        t *= rate;
        g.need(cur, t * cout);
        g.need("x", t * cout);
        pending = vec![0.0; cout];
        for (u, dilation) in [1, 3, 9].into_iter().enumerate() {
            let q = format!("{p}.{}", u + 2);
            let label = format!("b{i}.u{u}");
            let s1 = g.snake(file, &format!("{q}.act1"), cout)?;
            let (c1, c1_b) =
                conv_w(&mut g, &format!("{q}.conv1.conv"), &format!("{label}.conv1"), [cout, cout, KERNEL])?;
            let c1_b = g.weight(&format!("{label}.conv1.b"), &[cout], &c1_b);
            let s2 = g.snake(file, &format!("{q}.act2"), cout)?;
            let (c2, c2_b) = conv_w(&mut g, &format!("{q}.conv2.conv"), &format!("{label}.conv2"), [cout, cout, 1])?;
            let in_b = g.weight(&format!("{label}.in_b"), &[cout], &pending);

            g.conv(&label, (cur, &in_b, Some(&s1)), (&c1, tmp), t, (cout, cout), (KERNEL, dilation));
            g.each8(
                &format!("{label}.snake2"),
                "codec_bias_snake",
                t * cout,
                vec![inb(tmp), inb(&c1_b), inf(&s2.0), inf(&s2.1), outb("x"), i32a(cout)],
            );
            g.gemm_acc(&format!("{label}.conv2"), cur, "x", &c2, t, (cout, cout));
            pending = plus(&pending, &c2_b);
        }
    }

    let n = cfg.upsample_rates.len() + 1;
    let snake_out = g.snake(file, &format!("decoder.decoder.{n}"), out_dim)?;
    let in_b = g.weight("conv_out.in_b", &[out_dim], &pending);
    let (conv_out, conv_out_b) =
        conv_w(&mut g, &format!("decoder.decoder.{}.conv", n + 1), "conv_out", [1, out_dim, KERNEL])?;
    let conv_out_b = g.weight("conv_out.b", &[1], &conv_out_b);
    ensure!(t >= KERNEL, "the output conv needs at least {KERNEL} rows per frame");
    let history = g.region(2 * (KERNEL - 1) * out_dim * 2);
    g.launch(
        "conv_out",
        "codec_conv_out",
        [json!(t.div_ceil(128)), json!("seqs"), json!(1)],
        128,
        vec![
            inb(cur),
            inb(&in_b),
            inf(&snake_out.0),
            inf(&snake_out.1),
            inb(&conv_out),
            inb(&conv_out_b),
            state_io(history),
            ini("pos"),
            ini("lines"),
            outb("wav"),
            i32a(t),
            i32a(out_dim),
            i32a(KERNEL),
            stride(),
        ],
    );
    debug_assert_eq!(t, spf);

    let s = g.state_bytes;
    let calls: Vec<Value> = g
        .calls
        .into_iter()
        .map(|mut c| {
            for a in c["args"].as_array_mut().into_iter().flatten() {
                if a == &json!({"i64": STRIDE}) {
                    *a = json!({"i64": s});
                }
            }
            c
        })
        .collect();
    let widths = g.widths;
    let mut buffers = g.buffers;
    let work = |w: usize| json!({"dtype": "bf16", "shape": ["seqs", w], "kind": "workspace"});
    buffers.insert(
        "codes".into(),
        json!({"dtype": "i32", "shape": ["seqs", GROUPS], "kind": "input", "domain": {"min": 0, "max": cb - 1}}),
    );
    buffers.insert("pos".into(), json!({"dtype": "i32", "shape": ["seqs"], "kind": "input", "domain": {"min": 0}}));
    buffers.insert(
        "lines".into(),
        json!({"dtype": "i32", "shape": [1, "seqs"], "kind": "input", "domain": {"index_into": "codec", "stride": s}}),
    );
    buffers.insert("wav".into(), json!({"dtype": "bf16", "shape": ["seqs", spf], "kind": "output"}));
    buffers.insert("res".into(), work(hidden));
    for (name, w) in widths {
        buffers.insert(name.into(), work(w));
    }
    let mut ops = g.ops;
    let gemm = |entry: &str, c: &str| {
        json!({"params": ["in buffer<bf16>", "in buffer<bf16>", c, "i32", "i32", "i32"],
               "impl": {"launches": [{"entry": entry}]}})
    };
    ops.insert("gemm".into(), gemm("extern:cublaslt_bf16_tn", "out buffer<bf16>"));
    ops.insert("gemm_acc".into(), gemm("extern:cublaslt_bf16_tn_acc", "inout buffer<bf16>"));
    let manifest = json!({
        "schema_version": 5,
        "model": "qwen3-tts-12hz-codec",
        "vars": {"seqs": {"max": max_seqs}},
        "states": {"codec": {"bytes_per_seq": s}},
        "buffers": buffers,
        "modules": {"codec": {"source": format!("codec-{}.cubin", &sha[..12]), "sha256": sha}},
        "ops": ops,
        "programs": {"decode": {"batch": {"groups": max_seqs, "rows": 1}, "graph": true, "calls": calls}},
    });
    Ok((manifest, HostTensors(g.tensors)))
}
