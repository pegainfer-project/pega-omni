//! The engine against the official implementation's recorded run
//! (`tools/hidream_o1/golden.py`): prompt tokens and M-RoPE positions must
//! match exactly; the teacher-forced x0 prediction at each recorded step must
//! be as close to the float32 model as the reference's own bf16 run is, and the
//! finished picture, sampled on the reference's noise, must land within the
//! spread of equally accurate implementations ([`MARGIN_DB`]).
//!
//! Needs a GPU, `OMNI_HIDREAM_O1_MODEL` (checkpoint directory) and
//! `OMNI_HIDREAM_O1_GOLDEN` (the golden file); skipped when either is unset.

use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use half::f16;
use memmap2::Mmap;
use omni_engine::image::Size;
use omni_hidream_o1::config::PATCH_DIM;
use omni_hidream_o1::gemm::Gemms;
use omni_hidream_o1::gemm::Pins;
use omni_hidream_o1::model::Limits;
use omni_hidream_o1::model::Model;
use omni_hidream_o1::prompt;
use omni_hidream_o1::prompt::Tokenizer;
use omni_hidream_o1::sampler;
use omni_hidream_o1::sampler::Noise;
use omni_hidream_o1::sampler::TIMESTEPS;
use safetensors::Dtype;
use safetensors::SafeTensors;

/// How far below the reference's PSNR to float32 the engine's picture may
/// land. The 28 steps compound rounding: equally accurate numerics span
/// several dB (the reference's own eager attention lands 1 dB below its sdpa
/// run, and the two agree with each other only to 26.4 dB), while a sampler
/// bug falls far below (an unclipped draw 16.7 dB, sigma off by a step
/// 11.6 dB). The teacher-forced steps above are where accuracy is held.
const MARGIN_DB: f64 = 3.0;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_HIDREAM_O1_MODEL").zip(get("OMNI_HIDREAM_O1_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_HIDREAM_O1_MODEL and OMNI_HIDREAM_O1_GOLDEN");
    }
    found
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

    fn bf16(&self, name: &str) -> Vec<bf16> {
        let (d, _, b) = self.raw(name);
        assert_eq!(d, Dtype::BF16, "{name}");
        b.as_chunks().0.iter().map(|&x| bf16::from_le_bytes(x)).collect()
    }

    fn i64(&self, name: &str) -> (Vec<usize>, Vec<i64>) {
        let (d, shape, b) = self.raw(name);
        assert_eq!(d, Dtype::I64, "{name}");
        (shape, b.as_chunks().0.iter().map(|&x| i64::from_le_bytes(x)).collect())
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

#[test]
fn engine_matches_the_official_run() {
    let Some((model_dir, golden_path)) = paths() else { return };
    let file = std::fs::File::open(&golden_path).unwrap();
    let map = unsafe { Mmap::map(&file).unwrap() };
    let (_, meta) = SafeTensors::read_metadata(&map).unwrap();
    let meta = meta.metadata().clone().unwrap();
    let golden = Golden(SafeTensors::deserialize(&map).unwrap());
    let size: Size = meta["size"].parse().unwrap();
    let grid = prompt::grid(size);
    let patches = grid.0 * grid.1;

    let ids = Tokenizer::load(&model_dir).unwrap().encode(&meta["prompt"]).unwrap();
    let (_, want_ids) = golden.i64("input_ids");
    assert_eq!(ids.iter().map(|&i| i64::from(i)).collect::<Vec<_>>(), want_ids, "prompt tokens");

    let (shape, want_pos) = golden.i64("position_ids");
    let total = ids.len() + patches;
    assert_eq!(shape, [3, total]);
    let ours: Vec<[i32; 3]> =
        (0..ids.len()).map(prompt::text_position).chain(prompt::patch_positions(grid.0, grid.1)).collect();
    for axis in 0..3 {
        let got: Vec<i64> = ours.iter().map(|p| i64::from(p[axis])).collect();
        assert_eq!(got, want_pos[axis * total..(axis + 1) * total], "M-RoPE axis {axis}");
    }

    let t = Instant::now();
    // The algorithms a server would pin with `--gemm-algos`, when given.
    let gemms = match std::env::var_os("OMNI_HIDREAM_O1_GEMM_ALGOS") {
        Some(path) => Pins::load(path.as_ref(), 0).unwrap(),
        None => Gemms::Heuristic,
    };
    let limits = Limits { max_text: ids.len(), max_patches: patches };
    let mut model = Model::load(0, &model_dir, limits, &gemms).unwrap();
    eprintln!("loaded in {:.1?}", t.elapsed());
    model.prefill(&ids, grid).unwrap();

    for k in meta["steps"].split(',').map(|s| s.parse::<usize>().unwrap()) {
        model.set_z(&golden.bf16(&format!("z_{k}"))).unwrap();
        let t = Instant::now();
        model.predict(f32::from(TIMESTEPS[k])).unwrap();
        let ours: Vec<f32> = model.x0().unwrap().iter().map(|x| x.to_f32()).collect();
        let elapsed = t.elapsed();
        let truth = golden.f32(&format!("x0_fp32_{k}"));
        let (reference, ..) = compare(&golden.f32(&format!("x0_{k}")), &truth);
        let (cos, max, mean) = compare(&ours, &truth);
        eprintln!(
            "step {k:2} (t={}): x0 cosine to float32 {cos:.6} (reference bf16 {reference:.6}), max |diff| {max:.4}, \
             mean |diff| {mean:.5}, {elapsed:.1?}",
            TIMESTEPS[k]
        );
        assert!(cos > reference - 0.002, "step {k}: x0 cosine {cos} against the reference's {reference}");
    }

    // The reference's draws; the last step (sigma 0) reads none.
    let draw = |k: u32| {
        let name = format!("noise_{k}");
        if golden.0.tensor(&name).is_ok() { golden.f32(&name) } else { vec![0.0; patches * PATCH_DIM] }
    };
    let t = Instant::now();
    assert!(sampler::sample(&mut model, Noise::Given(&draw), &|| true).unwrap());
    let ours = model.rgb().unwrap();
    let (truth, reference) = (golden.raw("image_fp32").2, golden.raw("image").2);
    let (db, floor) = (psnr(&ours, truth), psnr(reference, truth));
    eprintln!(
        "replayed 28 steps in {:.1?}: picture PSNR to float32 {db:.2} dB (reference bf16 {floor:.2} dB), \
         to the reference {:.2} dB",
        t.elapsed(),
        psnr(&ours, reference)
    );
    assert!(db > floor - MARGIN_DB, "picture PSNR {db} dB against the reference's {floor} dB");
}
