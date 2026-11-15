//! The engine against the Hugging Face run recorded by `tools/higgs_audio/golden.py`.
//!
//! Needs `OMNI_HIGGS_AUDIO_MODEL` (the checkpoint directory) and `OMNI_HIGGS_AUDIO_GOLDEN` (the
//! golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_higgs_audio::config::CODEBOOKS;
use omni_higgs_audio::config::VOCAB;
use omni_higgs_audio::delay::BOC;
use omni_higgs_audio::delay::Delay;
use omni_higgs_audio::delay::Row;
use omni_higgs_audio::delay::undelay;
use omni_higgs_audio::model::Draw;
use omni_higgs_audio::model::Limits;
use omni_higgs_audio::model::Model;
use omni_higgs_audio::model::Sampling;
use safetensors::SafeTensors;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_HIGGS_AUDIO_MODEL").zip(get("OMNI_HIGGS_AUDIO_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_HIGGS_AUDIO_MODEL and OMNI_HIGGS_AUDIO_GOLDEN");
    }
    found
}

struct Golden(Vec<u8>);

impl Golden {
    fn open(path: &std::path::Path) -> Self {
        Self(std::fs::read(path).unwrap())
    }

    fn i32(&self, name: &str) -> Vec<i32> {
        let t = SafeTensors::deserialize(&self.0).unwrap();
        t.tensor(name).unwrap().data().as_chunks().0.iter().map(|&b| i32::from_le_bytes(b)).collect()
    }

    fn rows(&self, name: &str) -> Vec<Row> {
        self.i32(name).as_chunks::<CODEBOOKS>().0.to_vec()
    }

    fn text(&self) -> String {
        let (_, meta) = SafeTensors::read_metadata(&self.0).unwrap();
        meta.metadata().clone().unwrap()["text"].clone()
    }
}

#[test]
fn the_ramp_and_the_frames_are_the_references() {
    let Some((_, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    let (codes, mut delay) = (golden.rows("codes"), Delay::default());
    for row in &codes {
        assert!(row[delay.forced_from()..].iter().all(|&c| c == BOC), "{row:?} after {delay:?}");
        delay.advance(row[0]);
    }
    assert_eq!(undelay(&codes), golden.rows("frames"));
}

impl Golden {
    fn f32(&self, name: &str) -> Vec<f32> {
        let t = SafeTensors::deserialize(&self.0).unwrap();
        t.tensor(name).unwrap().data().as_chunks().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
    }
}

/// Distance between two logit rows after removing each one's mean, against the reference's spread.
fn relative_error(got: &[f32], want: &[f32]) -> f64 {
    let centered = |x: &[f32]| {
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        x.iter().map(|&v| v as f64 - mean).collect::<Vec<_>>()
    };
    let (g, w) = (centered(got), centered(want));
    let diff: f64 = g.iter().zip(&w).map(|(a, b)| (a - b) * (a - b)).sum();
    let scale: f64 = w.iter().map(|b| b * b).sum();
    (diff / scale).sqrt()
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().fold(0, |best, (i, &v)| if v > x[best] { i } else { best })
}

/// Signal to error power of `got` against `want`, in dB.
fn snr(got: &[f32], want: &[f32]) -> f64 {
    let signal: f64 = want.iter().map(|&x| x as f64 * x as f64).sum();
    let noise: f64 = got.iter().zip(want).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}

/// Steps the third stream joins after: it is prefilled next to two running rows.
const LATE: usize = 7;

/// Logits and argmax agreement of every stream's every step.
#[derive(Default)]
struct Agreement {
    worst: f64,
    flips: usize,
    draws: usize,
}

impl Agreement {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        self.worst = self.worst.max(relative_error(got, want));
        self.flips += (0..CODEBOOKS)
            .filter(|&b| argmax(&got[b * VOCAB..(b + 1) * VOCAB]) != argmax(&want[b * VOCAB..(b + 1) * VOCAB]))
            .count();
        self.draws += CODEBOOKS;
    }
}

#[test]
fn the_model_follows_the_reference() {
    let Some((model, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    let (ids, codes, want) = (golden.i32("prompt_ids"), golden.rows("codes"), golden.f32("logits"));
    let limits =
        Limits { max_batch: 3, max_tokens: 512, kv_gib: 2.0, sampling: Sampling { temperature: 0.0, top_k: 0 } };
    let mut m = Model::load(0, &model, limits).unwrap();
    assert_eq!(m.tokenizer.prompt(&golden.text()), ids, "the prompt");
    let mut seqs: Vec<_> = (0..3).map(|_| m.open(ids.len(), codes.len() + 1).unwrap().unwrap()).collect();
    let join = [0, 0, LATE];
    let force = |s: usize| Draw { uniforms: [0.5; CODEBOOKS], force: Some(codes[s]) };
    let book = CODEBOOKS * VOCAB;
    let mut agreement = Agreement::default();
    for t in 0..codes.len() + LATE {
        let fresh: Vec<usize> = (0..3).filter(|&i| join[i] == t).collect();
        let old: Vec<usize> = (0..3).filter(|&i| join[i] < t && t - join[i] < codes.len()).collect();
        let mut check = |m: &Model, which: &[usize], rows: &[[i32; CODEBOOKS]]| {
            for (j, &i) in which.iter().enumerate() {
                let s = t - join[i];
                assert_eq!(rows[j], codes[s], "stream {i} step {s} did not take the forced row");
                agreement.add(&m.logits(j).unwrap(), &want[s * book..(s + 1) * book]);
            }
        };
        if !fresh.is_empty() {
            let mut rows: Vec<_> = seqs
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| fresh.contains(i))
                .map(|(_, seq)| (seq, &ids[..], force(0)))
                .collect();
            let out = m.start(&mut rows).unwrap();
            check(&m, &fresh, &out.rows);
        }
        if !old.is_empty() {
            let mut rows: Vec<_> = seqs
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| old.contains(i))
                .map(|(i, seq)| (seq, force(t - join[i])))
                .collect();
            let out = m.step(&mut rows).unwrap();
            check(&m, &old, &out.rows);
        }
    }
    let Agreement { worst, flips, draws } = agreement;
    eprintln!("logits: worst relative error {worst:.4}, {flips} argmax flips in {draws} draws");
    // The same model on CUDA through torch, against its CPU run, measures 0.0075 and 7.6% flips.
    assert!(worst < 0.015, "worst relative error {worst}");
    assert!((flips as f64) < 0.15 * draws as f64, "{flips} argmax flips in {draws} draws");

    for (codes, wav) in [("codec_codes", "codec_wav"), ("frames", "frames_wav")] {
        let (got, want) = (m.decode(&golden.rows(codes)).unwrap(), golden.f32(wav));
        assert_eq!(got.len(), want.len(), "{wav}");
        eprintln!("{wav}: {:.1} dB against the reference's", snr(&got, &want));
        assert!(snr(&got, &want) > 30.0, "{wav}");
    }
}
