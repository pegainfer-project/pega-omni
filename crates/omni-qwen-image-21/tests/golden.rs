//! The engine against the official implementation's recorded run
//! (`tools/qwen_image_21/golden.py`): the RoPE positions of the joint sequence
//! must match exactly, the teacher-forced velocity prediction at each recorded
//! step must agree with the reference's, and the finished picture, sampled on
//! the reference's starting latent, must land above [`PSNR_DB`].
//!
//! The sampler is deterministic, so replaying `z_0` reproduces the reference's
//! picture without needing its RNG: the engine's own Philox draw is a different
//! noise field, and a picture is only comparable on the same one.
//!
//! Needs a GPU, `OMNI_QWEN_IMAGE_21_MODEL` (the pipeline directory),
//! `OMNI_QWEN_IMAGE_21_GOLDEN` (the golden file) and `OMNI_QWEN_IMAGE_21_PYTHON`
//! (the interpreter the VAE sidecar runs on); skipped when any is unset.

use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use half::f16;
use memmap2::Mmap;
use omni_engine::image::Size;
use omni_qwen_image_21::config::ROPE_TABLE;
use omni_qwen_image_21::gemm::Gemms;
use omni_qwen_image_21::model::Limits;
use omni_qwen_image_21::model::Model;
use omni_qwen_image_21::prompt;
use omni_qwen_image_21::sidecar::Sidecar;
use safetensors::Dtype;
use safetensors::SafeTensors;

/// The picture PSNR the engine must reach against the reference's own bf16 run.
/// Eight Euler steps compound bf16 rounding, but a wrong kernel falls far below:
/// three quarters of the value cache left unwritten lands at 10 dB.
const PSNR_DB: f64 = 30.0;
/// Cosine similarity the teacher-forced prediction must reach. The reference
/// tiles attention differently (one call over the joint sequence, against our
/// prefill plus decode), so bf16 rounding keeps this just below 1.
const PRED_COS: f64 = 0.99;

fn paths() -> Option<(PathBuf, PathBuf, String)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_QWEN_IMAGE_21_MODEL")
        .zip(get("OMNI_QWEN_IMAGE_21_GOLDEN"))
        .zip(std::env::var("OMNI_QWEN_IMAGE_21_PYTHON").ok());
    if found.is_none() {
        eprintln!("skipped: set OMNI_QWEN_IMAGE_21_MODEL, OMNI_QWEN_IMAGE_21_GOLDEN and OMNI_QWEN_IMAGE_21_PYTHON");
    }
    found.map(|((m, g), p)| (m, g, p))
}

struct Golden<'a>(SafeTensors<'a>);

impl Golden<'_> {
    fn raw(&self, name: &str) -> (Dtype, Vec<usize>, &[u8]) {
        let t = self.0.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        (t.dtype(), t.shape().to_vec(), t.data())
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        match self.raw(name) {
            (Dtype::F32, _, b) => b.as_chunks().0.iter().map(|&x| f32::from_le_bytes(x)).collect(),
            (Dtype::F16, _, b) => b.as_chunks().0.iter().map(|&x| f16::from_le_bytes(x).to_f32()).collect(),
            (Dtype::BF16, _, b) => b.as_chunks().0.iter().map(|&x| bf16::from_le_bytes(x).to_f32()).collect(),
            (d, ..) => panic!("{name}: unexpected {d:?}"),
        }
    }

    fn i32(&self, name: &str) -> (Vec<usize>, Vec<i32>) {
        let (d, shape, b) = self.raw(name);
        assert_eq!(d, Dtype::I32, "{name}");
        (shape, b.as_chunks().0.iter().map(|&x| i32::from_le_bytes(x)).collect())
    }
}

/// Cosine similarity, and the largest and mean absolute difference.
fn compare(ours: &[f32], reference: &[f32]) -> (f64, f32, f64) {
    assert_eq!(ours.len(), reference.len());
    let (mut dot, mut a2, mut b2, mut max, mut sum) = (0f64, 0f64, 0f64, 0f32, 0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        dot += f64::from(a) * f64::from(b);
        a2 += f64::from(a) * f64::from(a);
        b2 += f64::from(b) * f64::from(b);
        max = max.max((a - b).abs());
        sum += f64::from((a - b).abs());
    }
    (dot / (a2.sqrt() * b2.sqrt()), max, sum / ours.len() as f64)
}

fn psnr(ours: &[u8], reference: &[u8]) -> f64 {
    assert_eq!(ours.len(), reference.len());
    let mse = ours.iter().zip(reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>()
        / ours.len() as f64;
    10.0 * (255.0f64.powi(2) / mse).log10()
}

fn std(v: &[f32]) -> f64 {
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
    (v.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

#[test]
fn engine_matches_the_official_run() {
    let Some((model_dir, golden_path, python)) = paths() else { return };
    let file = std::fs::File::open(&golden_path).unwrap();
    let map = unsafe { Mmap::map(&file).unwrap() };
    let (_, meta) = SafeTensors::read_metadata(&map).unwrap();
    let meta = meta.metadata().clone().unwrap();
    let golden = Golden(SafeTensors::deserialize(&map).unwrap());
    let size: Size = meta["size"].parse().unwrap();
    let grid = prompt::grid(size);
    let patches = grid.0 * grid.1;
    let text_len: usize = meta["text_len"].parse().unwrap();
    let sigmas: Vec<f32> = meta["sigmas"].split(',').map(|s| s.parse().unwrap()).collect();
    let steps: Vec<usize> = meta["steps"].split(',').map(|s| s.parse().unwrap()).collect();

    // The joint sequence's positions: text advances all three axes, the picture
    // freezes the frame axis and centres the spatial ones on zero. The golden
    // records them signed; the engine reads a negative position from the tail
    // of the rotary table.
    let (shape, want_pos) = golden.i32("rope_pos");
    let total = text_len + patches;
    assert_eq!(shape, [3, total]);
    let ours: Vec<[i32; 3]> =
        (0..text_len).map(prompt::text_position).chain(prompt::image_positions(text_len, grid.0, grid.1)).collect();
    for axis in 0..3 {
        let want = &want_pos[axis * total..(axis + 1) * total];
        for (n, (got, &w)) in ours.iter().map(|p| p[axis]).zip(want).enumerate() {
            let row = if w < 0 { w + ROPE_TABLE as i32 } else { w };
            assert_eq!(got, row, "RoPE axis {axis} token {n}");
        }
    }

    let t = Instant::now();
    let limits = Limits { max_text: text_len, max_patches: patches };
    let mut model = Model::load(0, &model_dir, limits, &Gemms::Heuristic).unwrap();
    eprintln!("loaded in {:.1?}", t.elapsed());
    let (dtype, shape, embeds) = golden.raw("encoder_hidden");
    assert_eq!((dtype, shape.as_slice()), (Dtype::BF16, &[text_len, 4096][..]));
    model.prefill(embeds, text_len, grid).unwrap();

    // Teacher-forced: each recorded step reads the reference's own latent.
    for &k in &steps {
        model.set_z(golden.raw(&format!("z_{k}")).2).unwrap();
        let t = Instant::now();
        model.decode(sigmas[k]).unwrap();
        let ours = omni_kern::bf16s(&model.pred().unwrap());
        let elapsed = t.elapsed();
        let reference = golden.f32(&format!("pred_{k}"));
        let (cos, max, mean) = compare(&ours, &reference);
        let sigma = sigmas[k];
        eprintln!(
            "step {k} (sigma={sigma:.4}): pred cosine {cos:.6}, std {:.4} against {:.4}, max |diff| {max:.4}, \
             mean |diff| {mean:.5}, {elapsed:.1?}",
            std(&ours),
            std(&reference),
        );
        assert!(cos > PRED_COS, "step {k}: pred cosine {cos} against the reference");
    }

    // The whole schedule on the reference's starting latent.
    let t = Instant::now();
    model.set_z(golden.raw("z_0").2).unwrap();
    for k in 0..sigmas.len() - 1 {
        model.decode(sigmas[k]).unwrap();
        model.step(sigmas[k], sigmas[k + 1]).unwrap();
    }
    model.synchronize().unwrap();
    let latent = model.latent().unwrap();
    let (cos, max, mean) = compare(&omni_kern::bf16s(&latent), &golden.f32("z_final"));
    eprintln!(
        "replayed {} steps in {:.1?}: z_final cosine {cos:.6}, max |diff| {max:.4}, mean |diff| {mean:.5}",
        sigmas.len() - 1,
        t.elapsed()
    );
    assert!(cos > PRED_COS, "z_final cosine {cos} against the reference");

    // The picture, through the same VAE the reference decodes with.
    let socket = std::env::temp_dir().join(format!("qwen21-golden-{}.sock", std::process::id()));
    let mut sidecar = Sidecar::spawn(&python, &model_dir, 0, &socket).unwrap();
    let image = sidecar.decode(size.height, size.width, &latent).unwrap();
    let reference = golden.raw("image").2;
    let db = psnr(&image, reference);
    eprintln!("picture PSNR to the reference {db:.2} dB");
    assert!(db > PSNR_DB, "picture PSNR {db} dB, the gate is {PSNR_DB} dB");
    assert_eq!(image.len(), size.height as usize * size.width as usize * 3);
}
