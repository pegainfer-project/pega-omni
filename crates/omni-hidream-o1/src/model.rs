//! HiDream-O1 as one kern manifest on one runtime: Qwen3-VL's text tower run
//! as a pixel-space diffusion transformer, and the sampler around it.
//!
//! A generation is one sequence (see [`crate::prompt`]). The causal text rows
//! never see the timestep or the patches, so their K/V is the same at every
//! denoising step: `prefill` runs them once per picture and leaves their K/V
//! in the caches. Each step, `predict` runs only the timestep slot and the
//! patches, which attend to the whole sequence, and leaves the patches' x0
//! prediction in `x0`. That is the reference's two-pass attention (causal over
//! text, full over everything, text rows replaced) without recomputing the
//! text.
//!
//! Programs, over three vars: `text` (the causal rows of a prompt), `patches`
//! (of the picture) and `rows` (`patches + 1`: the timestep slot and the
//! patches).
//!
//! - `prefill` (eager): the causal rows into the K/V caches.
//! - `start` (eager): the starting latent `z` from draw 0.
//! - `predict` (graph): `z` at the step's timestep to `x0`.
//! - `advance` (graph): fresh noise and the Euler step from `x0` into `z`.
//! - `rgb` (eager): `z` to 8-bit RGB.
//! - `set_z`, `advance_with` (eager): a latent and a draw from the host, for
//!   replaying a reference run.
//!
//! What varies within a grid size (the prompt's key count, the timestep, the
//! draw, sigma) is read from device inputs, so `predict` and `advance` are
//! captured once per size and replayed for every step of every request.
//!
//! Embeddings: text tokens from the table, the timestep slot from
//! `t_embedder1` (sinusoid, linear, SiLU, linear), patches through
//! `x_embedder` (a 1024-wide bottleneck). The vision tower and `lm_head` are
//! not loaded: text-to-image never runs them. The decoder's GEMMs in a step
//! run on the algorithms [`Gemms`] names, every other GEMM on cuBLASLt's own
//! choice.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use kern_runtime::Capacity;
use kern_runtime::Runtime;
use omni_kern::Gen;
use omni_kern::HostTensors;
use omni_kern::buf;
use omni_kern::buf_at;
use omni_kern::count;
use omni_kern::f32a;
use omni_kern::hex;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::inf;
use omni_kern::ini;
use omni_kern::ints;
use omni_kern::io;
use omni_kern::kernels_dir;
use omni_kern::outb;
use omni_kern::weights::concat_rows;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;

use crate::config::PATCH_DIM;
use crate::config::TIMESTEP_FREQUENCIES;
use crate::config::Text;
use crate::gemm::Gemms;
use crate::gemm::name;
use crate::gemm::step_shapes;
use crate::prompt;
use crate::weights::Checkpoint;

const HIDREAM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hidream.cubin"));

const THREADS: u32 = 256;

/// What one runtime is sized for.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Text tokens of a prompt, the timestep slot included.
    pub max_text: usize,
    /// Patches of a picture.
    pub max_patches: usize,
}

pub struct Model {
    pub cfg: Text,
    rt: Runtime,
    limits: Limits,
    /// The grid of the last `prefill`.
    grid: (usize, usize),
    /// The candidate each tuned shape runs, as `predict` vars (see [`Gemms::Candidates`]).
    picks: BTreeMap<String, u64>,
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn vars(pairs: &[(&str, usize)]) -> BTreeMap<String, u64> {
    pairs.iter().map(|&(k, v)| (k.to_string(), v as u64)).collect()
}

impl Model {
    pub fn load(device: usize, dir: &Path, limits: Limits, gemms: &Gemms) -> Result<Self> {
        let cfg = Text::load(dir)?;
        let sha = hex(&sha2::Sha256::digest(HIDREAM));
        let (manifest, tensors) = generate(dir, &cfg, &sha, limits, gemms)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("manifest: {e}"))?;
        let kernels = kernels_dir(&[("hidream", &sha, HIDREAM)])?;
        let mut rt =
            Runtime::load(&verified, Some(&kernels), device, Some(Capacity { tokens: Some(1), seqs: 1 }), None)?;
        rt.load_weights(&tensors)?;
        let picks = gemms.vars().into_keys().map(|v| (v, 1)).collect();
        Ok(Self { cfg, rt, limits, grid: (0, 0), picks })
    }

    fn patches(&self) -> usize {
        self.grid.0 * self.grid.1
    }

    /// Runs the prompt's causal rows (every text token but the timestep slot)
    /// into the K/V caches and sets the grid the following steps predict.
    pub fn prefill(&mut self, ids: &[i32], grid: (usize, usize)) -> Result<()> {
        let causal = ids.len().saturating_sub(1);
        let patches = grid.0 * grid.1;
        ensure!(
            causal >= 1 && ids.len() <= self.limits.max_text,
            "{} text tokens, the limit is {}",
            ids.len(),
            self.limits.max_text
        );
        ensure!(
            patches >= 1 && patches <= self.limits.max_patches,
            "{grid:?} patches exceed {}",
            self.limits.max_patches
        );
        let v = vars(&[("text", causal)]);
        let positions: Vec<i32> = (0..causal).flat_map(prompt::text_position).collect();
        let slots: Vec<i32> = (0..causal as i32).collect();
        let rt = &mut self.rt;
        rt.write_input_at("ids", &ints(&ids[..causal]), &v)?;
        rt.write_input_at("text_pos", &ints(&positions), &v)?;
        rt.write_input_at("text_slots", &ints(&slots), &v)?;
        rt.write_input("prefill_kv", &ints(&[causal as i32]))?;
        rt.issue("prefill", &v)?;

        // The timestep slot sits at the last text position, then the grid.
        let rows = 1 + patches;
        let v = vars(&[("rows", rows)]);
        let mut positions = prompt::text_position(causal).to_vec();
        positions.extend(prompt::patch_positions(grid.0, grid.1).into_iter().flatten());
        let slots: Vec<i32> = (causal..causal + rows).map(|s| s as i32).collect();
        rt.write_input_at("gen_pos", &ints(&positions), &v)?;
        rt.write_input_at("gen_slots", &ints(&slots), &v)?;
        rt.write_input("step_kv", &ints(&[(causal + rows) as i32]))?;
        rt.write_input("width", &ints(&[(grid.1 * crate::config::PATCH) as i32]))?;
        self.grid = grid;
        Ok(())
    }

    /// Sets `z` to the starting latent: draw 0 of `seed`, times `scale`.
    pub fn start(&mut self, seed: u64, scale: f32) -> Result<()> {
        self.draw(seed, 0)?;
        self.rt.write_input("step", &floats(&[1.0, scale, 0.0]))?;
        self.rt.issue("start", &vars(&[("patches", self.patches())]))?;
        Ok(())
    }

    /// One denoising forward at scheduler timestep `step_t` (0..1000): `z` to `x0`.
    pub fn predict(&mut self, step_t: f32) -> Result<()> {
        let freq: Vec<u8> =
            timestep_frequencies(step_t).iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        self.rt.write_input("freq", &freq)?;
        let p = self.patches();
        let mut v = vars(&[("patches", p), ("rows", p + 1)]);
        v.extend(self.picks.clone());
        self.rt.issue("predict", &v)?;
        Ok(())
    }

    /// Runs candidate `index` (from 1) of the shape `var` picks from the next `predict` on.
    pub fn pick(&mut self, var: &str, index: usize) {
        self.picks.insert(var.into(), index as u64);
    }

    /// Waits for everything issued.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.rt.synchronize()?)
    }

    /// `z = sigma_next * scale * clip(eps) + (1 - sigma_next) * x0` with `eps`
    /// draw `k` of `seed`, clipped to `clip_std` of its standard deviation.
    pub fn advance(&mut self, (seed, k): (u64, u32), sigma_next: f32, scale: f32, clip_std: f32) -> Result<()> {
        self.draw(seed, k)?;
        self.rt.write_input("step", &floats(&[sigma_next, scale, clip_std]))?;
        self.rt.issue("advance", &vars(&[("patches", self.patches())]))?;
        Ok(())
    }

    /// [`Model::advance`] with the normals given.
    pub fn advance_with(&mut self, noise: &[f32], sigma_next: f32, scale: f32, clip_std: f32) -> Result<()> {
        let v = vars(&[("patches", self.patches())]);
        self.rt.write_input_at("noise_in", &floats(noise), &v)?;
        self.rt.write_input("step", &floats(&[sigma_next, scale, clip_std]))?;
        self.rt.issue("advance_with", &v)?;
        Ok(())
    }

    /// Sets `z` to `latent`, `[patches, 3072]`.
    pub fn set_z(&mut self, latent: &[bf16]) -> Result<()> {
        let v = vars(&[("patches", self.patches())]);
        let bytes: Vec<u8> = latent.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.rt.write_input_at("z_in", &bytes, &v)?;
        self.rt.issue("set_z", &v)?;
        Ok(())
    }

    /// The last `predict`'s x0, `[patches, 3072]`.
    pub fn x0(&self) -> Result<Vec<bf16>> {
        let bytes = self.rt.read_buffer_prefix("x0", self.patches() * PATCH_DIM * 2)?;
        Ok(bytes.as_chunks::<2>().0.iter().map(|&b| bf16::from_le_bytes(b)).collect())
    }

    /// `z` as 8-bit HWC RGB.
    pub fn rgb(&mut self) -> Result<Vec<u8>> {
        let p = self.patches();
        self.rt.issue("rgb", &vars(&[("patches", p)]))?;
        Ok(self.rt.read_buffer_prefix("rgb", p * PATCH_DIM)?)
    }

    fn draw(&mut self, seed: u64, k: u32) -> Result<()> {
        let draw: Vec<u8> = [seed as u32, (seed >> 32) as u32, k].iter().flat_map(|x| x.to_le_bytes()).collect();
        self.rt.write_input("draw", &draw)?;
        Ok(())
    }
}

/// The sinusoid `t_embedder1` reads at scheduler timestep `step_t`: the model's
/// time is `1 - step_t / 1000`, embedded at 1000 times that, cosines then sines.
pub fn timestep_frequencies(step_t: f32) -> [f32; TIMESTEP_FREQUENCIES] {
    let half = TIMESTEP_FREQUENCIES / 2;
    let t = (1.0 - step_t / 1000.0) * 1000.0;
    let mut out = [0.0; TIMESTEP_FREQUENCIES];
    for i in 0..half {
        let freq = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
        let (s, c) = (t * freq).sin_cos();
        out[i] = c;
        out[half + i] = s;
    }
    out
}

struct Layer {
    ln1: String,
    qkv: String,
    q_norm: String,
    k_norm: String,
    o: String,
    ln2: String,
    gate_up: String,
    down: String,
}

/// The decoder's weights and shape, and how a pass over it is emitted.
struct Tower<'a> {
    cfg: &'a Text,
    layers: Vec<Layer>,
    norm: String,
}

/// One decoder pass over `rows` rows already embedded in `h`: where its
/// positions and cache slots are, how many keys it sees, whether it is causal.
struct Pass<'a> {
    label: &'a str,
    rows: Value,
    positions: &'a str,
    slots: &'a str,
    kv_len: &'a str,
    causal: bool,
    step: bool,
}

impl Tower<'_> {
    fn load(g: &mut Gen, ck: &Checkpoint, cfg: &Text) -> Result<Vec<Layer>> {
        let (h, d, inter) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
        let (q, kv) = (cfg.num_attention_heads * d, cfg.num_key_value_heads * d);
        (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("model.language_model.layers.{i}");
                let w = |n: &str, shape: &[usize]| ck.expect(&format!("{p}.{n}"), shape);
                let n = |s: &str| format!("l{i}.{s}");
                Ok(Layer {
                    ln1: g.weight(&n("ln1"), &[h], &w("input_layernorm.weight", &[h])?.data),
                    qkv: g.weight(
                        &n("qkv"),
                        &[q + 2 * kv, h],
                        &concat_rows(&[
                            w("self_attn.q_proj.weight", &[q, h])?,
                            w("self_attn.k_proj.weight", &[kv, h])?,
                            w("self_attn.v_proj.weight", &[kv, h])?,
                        ]),
                    ),
                    q_norm: g.weight(&n("q_norm"), &[d], &w("self_attn.q_norm.weight", &[d])?.data),
                    k_norm: g.weight(&n("k_norm"), &[d], &w("self_attn.k_norm.weight", &[d])?.data),
                    o: g.weight(&n("o"), &[h, q], &w("self_attn.o_proj.weight", &[h, q])?.data),
                    ln2: g.weight(&n("ln2"), &[h], &w("post_attention_layernorm.weight", &[h])?.data),
                    gate_up: g.weight(
                        &n("gate_up"),
                        &[2 * inter, h],
                        &concat_rows(&[w("mlp.gate_proj.weight", &[inter, h])?, w("mlp.up_proj.weight", &[inter, h])?]),
                    ),
                    down: g.weight(&n("down"), &[h, inter], &w("mlp.down_proj.weight", &[h, inter])?.data),
                })
            })
            .collect()
    }

    /// The decoder over the pass's rows of `h`; leaves the final-normed rows in `h`.
    fn forward(&self, g: &mut Gen, pass: &Pass) {
        let c = self.cfg;
        let (h, d, hq, hk, inter) =
            (c.hidden_size, c.head_dim, c.num_attention_heads, c.num_key_value_heads, c.intermediate_size);
        let (qkv_w, eps) = (c.qkv_width(), c.rms_norm_eps);
        let rows = &pass.rows;
        let n = count(rows);
        let label = pass.label;
        let norm = (h / 8) as u32;
        let per_row = [rows.clone(), json!(1), json!(1)];
        g.launch(
            &format!("{label}.norm0"),
            "hidream_norm_copy",
            per_row.clone(),
            norm,
            vec![inb("h"), inb(&self.layers[0].ln1), outb("h"), outb("res"), i32a(h), f32a(eps)],
        );
        for (i, l) in self.layers.iter().enumerate() {
            let at = |s: &str| format!("{label}.l{i}.{s}");
            gemm(g, &at("qkv"), (buf("qkv"), buf("h"), buf(&l.qkv)), n.clone(), (qkv_w, h), pass.step);
            g.launch(
                &at("rope"),
                "hidream_qk_rope",
                [rows.clone(), json!((hq + 2 * hk).div_ceil(8)), json!(1)],
                THREADS,
                vec![
                    io("qkv"),
                    i32a(qkv_w),
                    inb(&l.q_norm),
                    inb(&l.k_norm),
                    ini(pass.positions),
                    ini(pass.slots),
                    io(&format!("k{i}")),
                    io(&format!("v{i}")),
                    i32a(hq),
                    i32a(hk),
                    i32a(c.mrope_hw()),
                    f32a(eps),
                    f32a(c.rope_theta),
                ],
            );
            let blocks = json!({"ceil_div": [{"mul": [rows, 4]}, 128]});
            g.launch_shared(
                &at("attn"),
                "hidream_attend",
                [blocks, json!(hk), json!(1)],
                THREADS,
                4 * 64 * 128 * 2,
                vec![
                    inb("qkv"),
                    inb(&format!("k{i}")),
                    inb(&format!("v{i}")),
                    outb("attn"),
                    rows_arg(rows),
                    ini(pass.kv_len),
                    i32a(qkv_w),
                    i32a(pass.causal as i32),
                    f32a((d as f32).powf(-0.5) * std::f32::consts::LOG2_E),
                ],
            );
            gemm(g, &at("o"), (buf("h"), buf("attn"), buf(&l.o)), n.clone(), (h, hq * d), pass.step);
            g.launch(
                &at("post_attn_norm"),
                "hidream_add_norm",
                per_row.clone(),
                norm,
                vec![inb("h"), io("res"), inb(&l.ln2), outb("h"), i32a(h), f32a(eps)],
            );
            gemm(g, &at("gate_up"), (buf("gate_up"), buf("h"), buf(&l.gate_up)), n.clone(), (2 * inter, h), pass.step);
            let groups = json!({"mul": [rows, inter / 8]});
            g.launch(
                &at("silu_mul"),
                "hidream_silu_mul",
                [json!({"ceil_div": [groups.clone(), THREADS]}), json!(1), json!(1)],
                THREADS,
                vec![inb("gate_up"), outb("act"), i32a(inter), ("i32", json!({"expr": groups}))],
            );
            gemm(g, &at("down"), (buf("h"), buf("act"), buf(&l.down)), n.clone(), (h, inter), pass.step);
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.ln1);
            g.launch(
                &at("next_norm"),
                "hidream_add_norm",
                per_row.clone(),
                norm,
                vec![inb("h"), io("res"), inb(next), outb("h"), i32a(h), f32a(eps)],
            );
        }
    }
}

/// `x = act(x + bias)` over `rows` rows of `cols` from byte `offset` of `x`.
fn bias_act(g: &mut Gen, label: &str, (x, offset): (&str, usize), bias: &str, (rows, cols): (Value, usize), act: i32) {
    let total = match rows.as_u64() {
        Some(r) => json!(r * cols as u64),
        None => json!({"mul": [rows, cols]}),
    };
    g.launch(
        label,
        "hidream_bias_act",
        [json!({"ceil_div": [total.clone(), THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![
            arg_at("inout buffer<bf16>", x, offset),
            inb(bias),
            i32a(cols),
            i32a(act),
            ("i32", json!({"expr": total})),
        ],
    );
}

/// The manifest (JSON) and the tensors its weight buffers bind.
fn generate(dir: &Path, cfg: &Text, sha: &str, limits: Limits, gemms: &Gemms) -> Result<(Value, HostTensors)> {
    let ck = Checkpoint::open(dir)?;
    let mut g = Gen::default();
    let (h, inter, hk, d) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_key_value_heads, cfg.head_dim);
    let lm = "model.language_model";
    let one = |g: &mut Gen, name: &str, tensor: &str, shape: &[usize]| -> Result<String> {
        Ok(g.weight(name, shape, &ck.expect(tensor, shape)?.data))
    };
    let embed = one(&mut g, "embed", &format!("{lm}.embed_tokens.weight"), &[cfg.vocab_size, h])?;
    let layers = Tower::load(&mut g, &ck, cfg)?;
    let norm = one(&mut g, "norm", &format!("{lm}.norm.weight"), &[h])?;
    let bottleneck = h / 4;
    let x_proj1 = one(&mut g, "x_proj1", "model.x_embedder.proj1.weight", &[bottleneck, PATCH_DIM])?;
    let x_proj2 = one(&mut g, "x_proj2", "model.x_embedder.proj2.weight", &[h, bottleneck])?;
    let x_proj2_b = one(&mut g, "x_proj2.b", "model.x_embedder.proj2.bias", &[h])?;
    let t_fc1 = one(&mut g, "t_fc1", "model.t_embedder1.mlp.0.weight", &[h, TIMESTEP_FREQUENCIES])?;
    let t_fc1_b = one(&mut g, "t_fc1.b", "model.t_embedder1.mlp.0.bias", &[h])?;
    let t_fc2 = one(&mut g, "t_fc2", "model.t_embedder1.mlp.2.weight", &[h, h])?;
    let t_fc2_b = one(&mut g, "t_fc2.b", "model.t_embedder1.mlp.2.bias", &[h])?;
    let head = one(&mut g, "head", "model.final_layer2.linear.weight", &[PATCH_DIM, h])?;
    let head_b = one(&mut g, "head.b", "model.final_layer2.linear.bias", &[PATCH_DIM])?;
    let tower = Tower { cfg, layers, norm };

    // prefill: the causal text rows.
    g.launch(
        "prefill.embed",
        "hidream_embed",
        [json!("text"), json!(1), json!(1)],
        (h / 8) as u32,
        vec![ini("ids"), inb(&embed), outb("h"), i32a(h)],
    );
    let text_pass = Pass {
        label: "prefill",
        rows: json!("text"),
        positions: "text_pos",
        slots: "text_slots",
        kv_len: "prefill_kv",
        causal: true,
        step: false,
    };
    tower.forward(&mut g, &text_pass);
    let prefill = g.take();

    // predict: the timestep slot into row 0, the patches of `z` into rows 1.., the tower, x0.
    let n = json!({"mul": ["patches", PATCH_DIM]});
    g.gemm_rows(
        "predict.t_fc1",
        (buf("t_hidden"), buf("freq"), buf(&t_fc1)),
        json!({"i32": 1}),
        (h, TIMESTEP_FREQUENCIES),
    );
    bias_act(&mut g, "predict.t_fc1.bias", ("t_hidden", 0), &t_fc1_b, (json!(1), h), 1);
    g.gemm_rows("predict.t_fc2", (buf("h"), buf("t_hidden"), buf(&t_fc2)), json!({"i32": 1}), (h, h));
    bias_act(&mut g, "predict.t_fc2.bias", ("h", 0), &t_fc2_b, (json!(1), h), 0);
    let patches = json!("patches");
    let np = count(&patches);
    g.gemm_rows("predict.x_proj1", (buf("bottleneck"), buf("z"), buf(&x_proj1)), np.clone(), (bottleneck, PATCH_DIM));
    g.gemm_rows("predict.x_proj2", (buf_at("h", h * 2), buf("bottleneck"), buf(&x_proj2)), np.clone(), (h, bottleneck));
    bias_act(&mut g, "predict.x_proj2.bias", ("h", h * 2), &x_proj2_b, (patches.clone(), h), 0);
    let step_pass = Pass {
        label: "predict",
        rows: json!("rows"),
        positions: "gen_pos",
        slots: "gen_slots",
        kv_len: "step_kv",
        causal: false,
        step: true,
    };
    tower.forward(&mut g, &step_pass);
    g.gemm_rows("predict.head", (buf("x0"), buf_at("h", h * 2), buf(&head)), np.clone(), (PATCH_DIM, h));
    bias_act(&mut g, "predict.head.bias", ("x0", 0), &head_b, (patches.clone(), PATCH_DIM), 0);
    let predict = g.take();

    // The sampler's pieces.
    let quads = json!({"ceil_div": [n.clone(), 4]});
    let gaussian = |g: &mut Gen, label: &str| {
        g.launch(
            label,
            "hidream_gaussian",
            [json!({"ceil_div": [quads.clone(), THREADS]}), json!(1), json!(1)],
            THREADS,
            vec![arg("out buffer<f32>", "noise"), ("i32", json!({"expr": n.clone()})), arg("in buffer<u32>", "draw")],
        );
    };
    let flow = |g: &mut Gen, label: &str, x0: &str, noise: &str| {
        g.launch(
            label,
            "hidream_flow_step",
            [json!({"ceil_div": [n.clone(), THREADS]}), json!(1), json!(1)],
            THREADS,
            vec![
                io("z"),
                inb(x0),
                inf(noise),
                arg("in buffer<u64>", "sums"),
                ("i32", json!({"expr": n.clone()})),
                inf("step"),
            ],
        );
    };
    let zero = |g: &mut Gen, label: &str| {
        g.launch(label, "hidream_zero_sums", [json!(1), json!(1), json!(1)], 2, vec![arg("out buffer<u64>", "sums")]);
    };
    let moments = |g: &mut Gen, label: &str, noise: &str| {
        zero(g, &format!("{label}.zero"));
        g.launch(
            &format!("{label}.moments"),
            "hidream_moments",
            [json!(256), json!(1), json!(1)],
            THREADS,
            vec![inf(noise), ("i32", json!({"expr": n.clone()})), arg("inout buffer<u64>", "sums")],
        );
    };
    gaussian(&mut g, "start.noise");
    zero(&mut g, "start.zero");
    flow(&mut g, "start.flow", "x0", "noise");
    let start = g.take();
    gaussian(&mut g, "advance.noise");
    moments(&mut g, "advance", "noise");
    flow(&mut g, "advance.flow", "x0", "noise");
    let advance = g.take();
    moments(&mut g, "advance_with", "noise_in");
    flow(&mut g, "advance_with.flow", "x0", "noise_in");
    let advance_with = g.take();
    let groups = json!({"mul": ["patches", PATCH_DIM / 8]});
    g.launch(
        "set_z",
        "hidream_copy",
        [json!({"ceil_div": [groups.clone(), THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![inb("z_in"), outb("z"), ("i32", json!({"expr": groups}))],
    );
    let set_z = g.take();
    g.launch(
        "rgb",
        "hidream_rgb",
        [json!({"ceil_div": [n.clone(), THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![inb("z"), arg("out buffer<u8>", "rgb"), ini("width"), ("i32", json!({"expr": n.clone()}))],
    );
    let rgb = g.take();

    let Limits { max_text, max_patches } = limits;
    let max_rows = max_text.max(max_patches + 1);
    let max_kv = max_text + max_patches;
    let min0 = json!({"min": 0});
    g.input("ids", json!(["text"]), json!({"index_into": embed}));
    g.input("text_pos", json!(["text", 3]), min0.clone());
    g.input("text_slots", json!(["text"]), json!({"min": 0, "max": max_kv - 1}));
    g.input("gen_pos", json!(["rows", 3]), min0.clone());
    g.input("gen_slots", json!(["rows"]), json!({"min": 0, "max": max_kv - 1}));
    g.input("prefill_kv", json!([1]), json!({"min": 1, "max": max_text}));
    g.input("step_kv", json!([1]), json!({"min": 1, "max": max_kv}));
    g.input("width", json!([1]), json!({"min": crate::config::PATCH}));
    g.buffer("freq", "bf16", json!([1, TIMESTEP_FREQUENCIES]), "input");
    g.buffer("step", "f32", json!([3]), "input");
    g.buffer("draw", "u32", json!([3]), "input");
    g.buffer("z_in", "bf16", json!(["patches", PATCH_DIM]), "input");
    g.buffer("noise_in", "f32", json!(["patches", PATCH_DIM]), "input");
    for i in 0..cfg.num_hidden_layers {
        for kv in ["k", "v"] {
            g.buffer(&format!("{kv}{i}"), "bf16", json!([max_kv, hk * d]), "carry");
        }
    }
    g.buffer("z", "bf16", json!([max_patches, PATCH_DIM]), "carry");
    g.buffer("x0", "bf16", json!([max_patches, PATCH_DIM]), "carry");
    g.buffer("rgb", "u8", json!(["patches", PATCH_DIM]), "output");
    for (name, width) in [
        ("h", h),
        ("res", h),
        ("qkv", cfg.qkv_width()),
        ("attn", cfg.num_attention_heads * d),
        ("gate_up", 2 * inter),
        ("act", inter),
    ] {
        g.buffer(name, "bf16", json!([max_rows, width]), "workspace");
    }
    g.buffer("bottleneck", "bf16", json!([max_patches, bottleneck]), "workspace");
    g.buffer("t_hidden", "bf16", json!([1, h]), "workspace");
    g.buffer("noise", "f32", json!([max_patches, PATCH_DIM]), "workspace");
    g.buffer("sums", "u64", json!([2]), "workspace");

    let mut programs = serde_json::Map::new();
    programs.insert("prefill".into(), json!({"calls": prefill}));
    programs.insert("start".into(), json!({"calls": start}));
    let one = |rows: &str| json!({"groups": 1, "rows": rows});
    programs.insert("predict".into(), json!({"batch": one("rows"), "graph": true, "calls": predict}));
    programs.insert("advance".into(), json!({"batch": one("patches"), "graph": true, "calls": advance}));
    programs.insert("advance_with".into(), json!({"calls": advance_with}));
    programs.insert("set_z".into(), json!({"calls": set_z}));
    programs.insert("rgb".into(), json!({"calls": rgb}));
    for shape in step_shapes(cfg) {
        g.gemm_op(&name(shape), gemms.launches(shape));
    }
    g.finish(&mut programs);
    let (buffers, ops, tensors) = g.into_parts();
    let mut vars =
        json!({"text": {"max": max_text}, "patches": {"max": max_patches}, "rows": {"max": max_patches + 1}});
    vars.as_object_mut().expect("an object").extend(gemms.vars());
    let manifest = json!({
        "schema_version": 5,
        "model": "hidream-o1",
        "vars": vars,
        "states": {},
        "buffers": buffers,
        "modules": {"hidream": {"source": format!("hidream-{}.cubin", &sha[..12]), "sha256": sha}},
        "ops": ops,
        "programs": programs,
    });
    Ok((manifest, tensors))
}

/// `y = x · wᵀ`; a step GEMM runs on its shape's op, whose algorithm [`Gemms`] sets.
fn gemm(g: &mut Gen, label: &str, operands: (Value, Value, Value), rows: Value, shape: (usize, usize), step: bool) {
    if step {
        g.gemm_rows_on(&name(shape), label, operands, rows, shape);
    } else {
        g.gemm_rows(label, operands, rows, shape);
    }
}

/// A row count as an `i32` argument.
fn rows_arg(rows: &Value) -> (&'static str, Value) {
    ("i32", count(rows))
}

/// An argument of `ty`, e.g. `("in buffer<bf16>", "h")`.
fn arg(ty: &'static str, name: &str) -> (&'static str, Value) {
    (ty, buf(name))
}

fn arg_at(ty: &'static str, name: &str, offset: usize) -> (&'static str, Value) {
    (ty, buf_at(name, offset))
}
