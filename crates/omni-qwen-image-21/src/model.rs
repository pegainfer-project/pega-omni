//! Qwen Image 2.1's DiT as one kern manifest: 32 single-stream blocks with
//! shared modulation, block-causal prefill / full-attention decode, and a
//! deterministic flow-matching Euler sampler.
//!
//! Programs over two vars: `text` (prompt tokens) and `img` (target patches).
//! - `prefill` (eager): txt_in → mod(t=0) → 32 blocks causal → KV caches.
//! - `decode` (graph): img_in → mod(t=σ) → 32 blocks full → norm_out → pred.
//! - `start` (eager): Philox noise → z.
//! - `step` (graph): Euler z += (σ_next−σ)·pred.
//! - `set_z` (eager): host latent for golden replay.

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

use crate::config::Config;
use crate::config::Transformer;
use crate::gemm::Gemms;
use crate::gemm::name;
use crate::gemm::step_shapes;
use crate::prompt;
use crate::weights::Checkpoint;

const QWEN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen.cubin"));
const H: usize = 4096;
const D: usize = 128;
const HQ: usize = 32;
const HK: usize = 32;
const INTER: usize = 12288;
const LAYERS: usize = 32;
const EPS: f32 = 1e-6;
const THREADS: u32 = 256;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_text: usize,
    pub max_patches: usize,
}

pub struct Model {
    rt: Runtime,
    limits: Limits,
    grid: (usize, usize),
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
        let cfg = Config::load(dir)?;
        let sha = hex(&sha2::Sha256::digest(QWEN));
        let (manifest, tensors) = generate(dir, &cfg.transformer, &sha, limits, gemms)?;
        let verified = kern_manifest::verify(kern_manifest::Manifest::from_json(&manifest.to_string())?)
            .map_err(|e| anyhow::anyhow!("manifest: {e}"))?;
        let kernels = kernels_dir(&[("qwen", &sha, QWEN)])?;
        let mut rt =
            Runtime::load(&verified, Some(&kernels), device, Some(Capacity { tokens: Some(1), seqs: 1 }), None)?;
        rt.load_weights(&tensors)?;
        if std::env::var("OMNI_EAGER").is_ok() {
            rt.set_eager(true);
        }
        let picks = gemms.vars().into_keys().map(|v| (v, 1)).collect();
        Ok(Self { rt, limits, grid: (0, 0), picks })
    }

    fn img(&self) -> usize {
        self.grid.0 * self.grid.1
    }

    pub fn prefill(&mut self, embeds: &[u8], text_len: usize, grid: (usize, usize)) -> Result<()> {
        let patches = grid.0 * grid.1;
        ensure!(
            text_len >= 1 && text_len <= self.limits.max_text,
            "{text_len} text tokens exceeds {}",
            self.limits.max_text
        );
        ensure!(patches >= 1 && patches <= self.limits.max_patches, "{grid:?} exceeds {}", self.limits.max_patches);
        let v = vars(&[("text", text_len)]);
        let positions: Vec<i32> = (0..text_len).flat_map(prompt::text_position).collect();
        let slots: Vec<i32> = (0..text_len as i32).collect();
        let rt = &mut self.rt;
        rt.write_input_at("txt_embeds", embeds, &v)?;
        rt.write_input_at("text_pos", &ints(&positions), &v)?;
        rt.write_input_at("text_slots", &ints(&slots), &v)?;
        rt.write_input("prefill_kv", &ints(&[text_len as i32]))?;
        rt.issue("prefill", &v)?;

        let v = vars(&[("img", patches)]);
        let positions: Vec<i32> = prompt::image_positions(text_len, grid.0, grid.1).into_iter().flatten().collect();
        let slots: Vec<i32> = (text_len..text_len + patches).map(|s| s as i32).collect();
        rt.write_input_at("img_pos", &ints(&positions), &v)?;
        rt.write_input_at("img_slots", &ints(&slots), &v)?;
        rt.write_input("decode_kv", &ints(&[(text_len + patches) as i32]))?;
        self.grid = grid;
        Ok(())
    }

    pub fn decode(&mut self, sigma: f32) -> Result<()> {
        let freq: Vec<u8> =
            prompt::timestep_sinusoid(sigma).iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        self.rt.write_input("freq", &freq)?;
        let mut v = vars(&[("img", self.img())]);
        v.extend(self.picks.clone());
        self.rt.issue("decode", &v)?;
        if std::env::var("OMNI_DEBUG_PRED").is_ok() {
            self.rt.synchronize()?;
            let pred = self.rt.read_buffer_prefix("pred", self.img() * 64 * 2)?;
            let dir =
                std::env::var("OMNI_DEBUG_DIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
            std::fs::write(format!("{dir}/debug_pred_{sigma:.4}.bin"), &pred).ok();
        }
        Ok(())
    }

    pub fn start(&mut self, seed: u64) -> Result<()> {
        let draw: Vec<u8> = [seed as u32, (seed >> 32) as u32, 0u32].iter().flat_map(|x| x.to_le_bytes()).collect();
        self.rt.write_input("draw", &draw)?;
        self.rt.issue("start", &vars(&[("img", self.img())]))?;
        Ok(())
    }

    pub fn step(&mut self, sigma: f32, sigma_next: f32) -> Result<()> {
        self.rt.write_input("sigmas", &floats(&[sigma, sigma_next]))?;
        self.rt.issue("step", &vars(&[("img", self.img())]))?;
        Ok(())
    }

    pub fn set_z(&mut self, z: &[u8]) -> Result<()> {
        let v = vars(&[("img", self.img())]);
        self.rt.write_input_at("z_in", z, &v)?;
        self.rt.issue("set_z", &v)?;
        Ok(())
    }

    pub fn pred(&self) -> Result<Vec<u8>> {
        Ok(self.rt.read_buffer_prefix("pred", self.img() * 64 * 2)?)
    }

    pub fn latent(&self) -> Result<Vec<u8>> {
        Ok(self.rt.read_buffer_prefix("z", self.img() * 64 * 2)?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.rt.synchronize()?)
    }
}

fn generate(dir: &Path, cfg: &Transformer, sha: &str, limits: Limits, gemms: &Gemms) -> Result<(Value, HostTensors)> {
    let ck = Checkpoint::open(&dir.join("transformer"))?;
    let mut g = Gen::default();
    let one =
        |g: &mut Gen, n: &str, t: &str, s: &[usize]| -> Result<String> { Ok(g.weight(n, s, &ck.expect(t, s)?.data)) };

    // Global weights
    let img_in = one(&mut g, "img_in", "img_in.weight", &[H, 64])?;
    let proj_out = one(&mut g, "proj_out", "proj_out.weight", &[64, H])?;
    let mod_w = one(&mut g, "mod_w", "modulation.1.weight", &[4 * H, H])?;
    let nout_w = one(&mut g, "nout_w", "norm_out.linear.weight", &[H, H])?;
    let t_fc1 = one(&mut g, "t_fc1", "time_text_embed.timestep_embedder.linear_1.weight", &[H, 256])?;
    let t_fc2 = one(&mut g, "t_fc2", "time_text_embed.timestep_embedder.linear_2.weight", &[H, H])?;
    let txt_norm = one(&mut g, "txt_norm", "txt_in.text_norm.weight", &[H])?;
    let txt_in_w = one(&mut g, "txt_in_w", "txt_in.in_layer.weight", &[H, H])?;
    let txt_out_w = one(&mut g, "txt_out_w", "txt_in.out_layer.weight", &[H, H])?;

    // Per-layer weights
    struct Layer {
        qkv: String,
        qn: String,
        kn: String,
        o: String,
        gate_up: String,
        down: String,
    }
    let layers: Vec<Layer> = (0..LAYERS)
        .map(|i| {
            let p = format!("transformer_blocks.{i}");
            let w = |n: &str, s: &[usize]| ck.expect(&format!("{p}.{n}"), s);
            let n = |s: &str| format!("l{i}.{s}");
            Ok(Layer {
                qkv: g.weight(
                    &n("qkv"),
                    &[3 * H, H],
                    &concat_rows(&[
                        w("attn.to_q.weight", &[H, H])?,
                        w("attn.to_k.weight", &[H, H])?,
                        w("attn.to_v.weight", &[H, H])?,
                    ]),
                ),
                qn: g.weight(&n("qn"), &[D], &w("attn.norm_q.weight", &[D])?.data),
                kn: g.weight(&n("kn"), &[D], &w("attn.norm_k.weight", &[D])?.data),
                o: g.weight(&n("o"), &[H, H], &w("attn.to_out.0.weight", &[H, H])?.data),
                gate_up: g.weight(
                    &n("gate_up"),
                    &[2 * INTER, H],
                    &concat_rows(&[
                        w("img_mlp.gate_layer.weight", &[INTER, H])?,
                        w("img_mlp.proj.weight", &[INTER, H])?,
                    ]),
                ),
                down: g.weight(&n("down"), &[H, INTER], &w("img_mlp.out.weight", &[H, INTER])?.data),
            })
        })
        .collect::<Result<_>>()?;

    // RoPE tables (host f32)
    let (rope_cos, rope_sin) = rope_tables();
    let cos_buf = g.weight_f32("rope_cos", rope_cos);
    let sin_buf = g.weight_f32("rope_sin", rope_sin);
    // t=0 sinusoid constant: cos(0)=1, sin(0)=0
    let sin0: Vec<f32> = [vec![1.0f32; 128], vec![0.0f32; 128]].concat();
    let sin0_buf = g.weight("sinusoid0", &[1, 256], &sin0);

    let max_rows = limits.max_text.max(limits.max_patches);
    let per_row = |rows: &Value| [rows.clone(), json!(1), json!(1)];
    let norm_block = (H / 8) as u32;

    // --- Block emission helper ---
    let block = |g: &mut Gen,
                 label: &str,
                 i: usize,
                 rows: &Value,
                 pos: &str,
                 slots: &str,
                 kv_len: &str,
                 causal: bool,
                 step: bool| {
        let l = &layers[i];
        let at = |s: &str| format!("{label}.l{i}.{s}");
        let n = count(rows);
        // ln_mod with scale1
        g.launch(
            &at("ln1"),
            "qwen_image_21_ln_mod",
            per_row(rows),
            norm_block,
            vec![inb("h"), ("in buffer<bf16>", buf_at("params", 0)), outb("normed"), i32a(H), f32a(EPS)],
        );
        gemm(g, &at("qkv"), (buf("qkv"), buf("normed"), buf(&l.qkv)), n.clone(), (3 * H, H), step);
        g.launch(
            &at("rope"),
            "qwen_image_21_qk_rope",
            [rows.clone(), json!(12), json!(1)],
            THREADS,
            vec![
                io("qkv"),
                i32a(3 * H),
                inb(&l.qn),
                inb(&l.kn),
                ini(pos),
                ini(slots),
                io(&format!("k{i}")),
                io(&format!("v{i}")),
                ("in buffer<f32>", buf(&cos_buf)),
                ("in buffer<f32>", buf(&sin_buf)),
                i32a(HQ),
                i32a(HK),
                f32a(EPS),
            ],
        );
        let blocks = json!({"ceil_div": [rows, 128]});
        g.launch_shared(
            &at("attn"),
            "qwen_image_21_attend",
            [blocks, json!(HK), json!(1)],
            THREADS,
            4 * 64 * 128 * 2,
            vec![
                inb("qkv"),
                inb(&format!("k{i}")),
                inb(&format!("v{i}")),
                outb("attn"),
                ("i32", count(rows)),
                ini(kv_len),
                i32a(3 * H),
                i32a(causal as i32),
                f32a((D as f32).powf(-0.5) * std::f32::consts::LOG2_E),
            ],
        );
        gemm(g, &at("o"), (buf("normed"), buf("attn"), buf(&l.o)), n.clone(), (H, H), step);
        // res_gate with gate1
        let total = json!({"mul": [rows, H / 8]});
        g.launch(
            &at("res1"),
            "qwen_image_21_res_gate",
            [json!({"ceil_div": [total.clone(), THREADS]}), json!(1), json!(1)],
            THREADS,
            vec![
                inb("h"),
                inb("normed"),
                ("in buffer<bf16>", buf_at("params", H * 2)),
                outb("h"),
                i32a(H),
                ("i32", json!({"expr": total.clone()})),
            ],
        );
        // ln_mod with scale2
        g.launch(
            &at("ln2"),
            "qwen_image_21_ln_mod",
            per_row(rows),
            norm_block,
            vec![inb("h"), ("in buffer<bf16>", buf_at("params", 2 * H * 2)), outb("normed"), i32a(H), f32a(EPS)],
        );
        gemm(g, &at("gate_up"), (buf("gate_up"), buf("normed"), buf(&l.gate_up)), n.clone(), (2 * INTER, H), step);
        let groups = json!({"mul": [rows, INTER / 8]});
        g.launch(
            &at("silu_mul"),
            "qwen_image_21_silu_mul",
            [json!({"ceil_div": [groups.clone(), THREADS]}), json!(1), json!(1)],
            THREADS,
            vec![inb("gate_up"), outb("act"), i32a(INTER), ("i32", json!({"expr": groups}))],
        );
        gemm(g, &at("down"), (buf("normed"), buf("act"), buf(&l.down)), n.clone(), (H, INTER), step);
        // res_gate with gate2
        g.launch(
            &at("res2"),
            "qwen_image_21_res_gate",
            [json!({"ceil_div": [total.clone(), THREADS]}), json!(1), json!(1)],
            THREADS,
            vec![
                inb("h"),
                inb("normed"),
                ("in buffer<bf16>", buf_at("params", 3 * H * 2)),
                outb("h"),
                i32a(H),
                ("i32", json!({"expr": total})),
            ],
        );
    };

    // --- Modulation emission ---
    let modulation = |g: &mut Gen, label: &str, freq_src: &str, step: bool| {
        let at = |s: &str| format!("{label}.{s}");
        gemm(g, &at("t_fc1"), (buf("t_emb"), buf(freq_src), buf(&t_fc1)), json!({"i32": 1}), (H, 256), false);
        g.launch(
            &at("silu1"),
            "qwen_image_21_silu",
            [json!(2), json!(1), json!(1)],
            THREADS,
            vec![inb("t_emb"), outb("silu_temb"), ("i32", json!({"expr": json!(H / 8)}))],
        );
        gemm(g, &at("t_fc2"), (buf("temb"), buf("silu_temb"), buf(&t_fc2)), json!({"i32": 1}), (H, H), false);
        g.launch(
            &at("silu2"),
            "qwen_image_21_silu",
            [json!(2), json!(1), json!(1)],
            THREADS,
            vec![inb("temb"), outb("silu_temb"), ("i32", json!({"expr": json!(H / 8)}))],
        );
        gemm(g, &at("mod"), (buf("mod_raw"), buf("silu_temb"), buf(&mod_w)), json!({"i32": 1}), (4 * H, H), false);
        g.launch(
            &at("mod_prep"),
            "qwen_image_21_mod_prep",
            [json!(2), json!(1), json!(1)],
            THREADS,
            vec![inb("mod_raw"), outb("params"), i32a(H)],
        );
        if step {
            gemm(g, &at("nout"), (buf("nout_raw"), buf("silu_temb"), buf(&nout_w)), json!({"i32": 1}), (H, H), false);
            g.launch(
                &at("one_plus"),
                "qwen_image_21_one_plus",
                [json!(2), json!(1), json!(1)],
                THREADS,
                vec![inb("nout_raw"), outb("s_out"), ("i32", json!({"expr": json!(H / 8)}))],
            );
        }
    };

    // === PREFILL ===
    let text = json!("text");
    g.launch(
        "prefill.txt_norm",
        "qwen_image_21_zero_rms_norm",
        per_row(&text),
        norm_block,
        vec![inb("txt_embeds"), inb(&txt_norm), outb("txt_normed"), i32a(H), f32a(EPS)],
    );
    gemm(&mut g, "prefill.txt_in", (buf("txt_hidden"), buf("txt_normed"), buf(&txt_in_w)), count(&text), (H, H), false);
    let txt_total = json!({"mul": ["text", H / 8]});
    g.launch(
        "prefill.gelu",
        "qwen_image_21_gelu_tanh",
        [json!({"ceil_div": [txt_total.clone(), THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![io("txt_hidden"), ("i32", json!({"expr": txt_total}))],
    );
    gemm(&mut g, "prefill.txt_out", (buf("h"), buf("txt_hidden"), buf(&txt_out_w)), count(&text), (H, H), false);
    modulation(&mut g, "prefill", &sin0_buf, false);
    for i in 0..LAYERS {
        block(&mut g, "prefill", i, &text, "text_pos", "text_slots", "prefill_kv", true, false);
    }
    let prefill = g.take();

    // === DECODE ===
    let img = json!("img");
    gemm(&mut g, "decode.img_in", (buf("h"), buf("z"), buf(&img_in)), count(&img), (H, 64), false);
    modulation(&mut g, "decode", "freq", true);
    for i in 0..LAYERS {
        block(&mut g, "decode", i, &img, "img_pos", "img_slots", "decode_kv", false, true);
    }
    g.launch(
        "decode.norm_out",
        "qwen_image_21_ln_mod",
        per_row(&img),
        norm_block,
        vec![inb("h"), inb("s_out"), outb("normed"), i32a(H), f32a(EPS)],
    );
    gemm(&mut g, "decode.proj_out", (buf("pred"), buf("normed"), buf(&proj_out)), count(&img), (64, H), false);
    let decode = g.take();

    // === START ===
    let n = json!({"mul": ["img", 64]});
    let quads = json!({"ceil_div": [n.clone(), 4]});
    g.launch(
        "start.init_z",
        "qwen_image_21_init_z",
        [json!({"ceil_div": [quads, THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![outb("z"), ("i32", json!({"expr": n.clone()})), ("in buffer<u32>", buf("draw"))],
    );
    let start = g.take();

    // === STEP ===
    let groups8 = json!({"mul": ["img", 8]}); // img*64/8
    g.launch(
        "step.euler",
        "qwen_image_21_euler",
        [json!({"ceil_div": [groups8.clone(), THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![io("z"), inb("pred"), ("i32", json!({"expr": n})), inf("sigmas")],
    );
    let step = g.take();

    // === SET_Z ===
    g.launch(
        "set_z.copy",
        "qwen_image_21_copy",
        [json!({"ceil_div": [groups8, THREADS]}), json!(1), json!(1)],
        THREADS,
        vec![inb("z_in"), outb("z"), ("i32", json!({"expr": json!({"mul": ["img", 8]})}))],
    );
    let set_z = g.take();

    // === BUFFERS ===
    let Limits { max_text, max_patches } = limits;
    let max_kv = max_text + max_patches;
    let min0 = json!({"min": 0});
    g.buffer("txt_embeds", "bf16", json!(["text", H]), "input");
    g.input("text_pos", json!(["text", 3]), min0.clone());
    g.input("text_slots", json!(["text"]), json!({"min": 0, "max": max_kv - 1}));
    g.input("img_pos", json!(["img", 3]), min0.clone());
    g.input("img_slots", json!(["img"]), json!({"min": 0, "max": max_kv - 1}));
    g.input("prefill_kv", json!([1]), json!({"min": 1, "max": max_text}));
    g.input("decode_kv", json!([1]), json!({"min": 1, "max": max_kv}));
    g.buffer("freq", "bf16", json!([1, 256]), "input");
    g.buffer("draw", "u32", json!([3]), "input");
    g.buffer("sigmas", "f32", json!([2]), "input");
    g.buffer("z_in", "bf16", json!(["img", 64]), "input");
    for i in 0..LAYERS {
        g.buffer(&format!("k{i}"), "bf16", json!([max_kv, H]), "carry");
        g.buffer(&format!("v{i}"), "bf16", json!([max_kv, H]), "carry");
    }
    g.buffer("z", "bf16", json!([max_patches, 64]), "carry");
    g.buffer("pred", "bf16", json!([max_patches, 64]), "carry");
    for (bn, w) in [("h", H), ("normed", H), ("attn", H), ("qkv", 3 * H), ("gate_up", 2 * INTER), ("act", INTER)] {
        g.buffer(bn, "bf16", json!([max_rows, w]), "workspace");
    }
    g.buffer("txt_normed", "bf16", json!([max_text, H]), "workspace");
    g.buffer("txt_hidden", "bf16", json!([max_text, H]), "workspace");
    for bn in ["t_emb", "temb", "silu_temb", "nout_raw", "s_out"] {
        g.buffer(bn, "bf16", json!([1, H]), "workspace");
    }
    g.buffer("mod_raw", "bf16", json!([1, 4 * H]), "workspace");
    g.buffer("params", "bf16", json!([4, H]), "workspace");

    // === PROGRAMS ===
    let mut programs = serde_json::Map::new();
    let batch = |rows: &str| json!({"groups": 1, "rows": rows});
    programs.insert("prefill".into(), json!({"calls": prefill}));
    programs.insert("decode".into(), json!({"batch": batch("img"), "graph": true, "calls": decode}));
    programs.insert("start".into(), json!({"calls": start}));
    programs.insert("step".into(), json!({"batch": batch("img"), "graph": true, "calls": step}));
    programs.insert("set_z".into(), json!({"calls": set_z}));
    for shape in step_shapes(cfg) {
        g.gemm_op(&name(shape), gemms.launches(shape));
    }
    g.finish(&mut programs);
    let (buffers, ops, tensors) = g.into_parts();
    let mut vars = json!({"text": {"max": max_text}, "img": {"max": max_patches}});
    vars.as_object_mut().expect("object").extend(gemms.vars());
    let manifest = json!({
        "schema_version": 5, "model": "qwen-image-21",
        "vars": vars, "states": {}, "buffers": buffers,
        "modules": {"qwen": {"source": format!("qwen-{}.cubin", &sha[..12]), "sha256": sha}},
        "ops": ops, "programs": programs,
    });
    Ok((manifest, tensors))
}

fn gemm(g: &mut Gen, label: &str, operands: (Value, Value, Value), rows: Value, shape: (usize, usize), step: bool) {
    if step {
        g.gemm_rows_on(&name(shape), label, operands, rows, shape);
    } else {
        g.gemm_rows(label, operands, rows, shape);
    }
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let axes = [16usize, 56, 56];
    let table_len = 9216;
    let mut cos = vec![0.0f32; table_len * 64];
    let mut sin = vec![0.0f32; table_len * 64];
    let mut offset = 0;
    for &dim in &axes {
        for i in 0..dim / 2 {
            let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / dim as f32);
            for p in 0..table_len {
                let pos = if p < 8192 { p as f32 } else { p as f32 - 9216.0 };
                let angle = pos * freq;
                let idx = p * 64 + offset + i;
                cos[idx] = angle.cos();
                sin[idx] = angle.sin();
            }
        }
        offset += dim / 2;
    }
    (cos, sin)
}
