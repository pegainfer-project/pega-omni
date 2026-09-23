//! The model against the official implementation's recorded run
//! (`tools/qwen3_tts/golden.py`): tokenizer ids, prompt embeddings,
//! teacher-forced talker and code-predictor logits, and the decoded waveform,
//! all through the serving path (`prefill`, `first`, `decode`) with the
//! recorded frames forced in place of the draws.
//!
//! Needs a GPU, `OMNI_QWEN3_TTS_MODEL` (checkpoint directory) and
//! `OMNI_QWEN3_TTS_GOLDEN` (the golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_qwen3_tts::model::Draw;
use omni_qwen3_tts::model::Limits;
use omni_qwen3_tts::model::Model;
use omni_qwen3_tts::model::Seq;
use omni_qwen3_tts::prompt;
use omni_qwen3_tts::talker::GROUPS;
use omni_qwen3_tts::weights::File;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_QWEN3_TTS_MODEL").zip(get("OMNI_QWEN3_TTS_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_QWEN3_TTS_MODEL and OMNI_QWEN3_TTS_GOLDEN");
    }
    found
}

struct Golden {
    file: File,
    text: String,
    speaker: String,
    language: String,
}

impl Golden {
    fn open(path: &std::path::Path) -> Golden {
        let bytes = std::fs::read(path).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let m = meta.metadata().clone().unwrap();
        Golden {
            file: File::open(path).unwrap(),
            text: m["text"].clone(),
            speaker: m["speaker"].clone(),
            language: m["language"].clone(),
        }
    }

    fn f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let t = self.file.host(name).unwrap();
        (t.shape, t.data)
    }

    fn i32(&self, name: &str) -> Vec<i32> {
        self.file.ints(name).unwrap()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// Worst per-row cosine and the fraction of rows whose argmax agrees.
fn agreement(ours: &[f32], theirs: &[f32], width: usize) -> (f64, f64) {
    let pairs: Vec<_> = ours.chunks(width).zip(theirs.chunks(width)).collect();
    let worst = pairs.iter().map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
    let agree = pairs.iter().filter(|(a, b)| argmax(a) == argmax(b)).count() as f64 / pairs.len() as f64;
    (worst, agree)
}

/// Three streams of the same prompt and frames: `a` and `b` in lockstep
/// (their audio must be identical), `c` starting `LAG` steps later, so it
/// is prefilled next to running rows and shares calls of other sizes.
const LAG: usize = 7;

#[test]
fn matches_the_official_run() {
    let Some((dir, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    let limits = Limits { max_batch: 3, max_tokens: 1024, kv_tokens: 4096 };
    let mut model = Model::load(0, &dir, limits).unwrap();
    let config = model.config.clone();

    let text_ids = model.tokenizer.assistant(&golden.text);
    assert_eq!(text_ids, golden.i32("text_ids"), "tokenizer ids");
    let voice = prompt::voice(&config.model, &golden.speaker, &golden.language).unwrap();
    let prompt = prompt::assemble(&config.model, voice, &text_ids, None);
    let (embed_shape, embeds) = golden.f32("prefill_embeds");
    assert_eq!(prompt.len(), embed_shape[0], "prompt length");

    let frames: Vec<[i32; GROUPS]> = golden.i32("codes").chunks(GROUPS).map(|c| c.try_into().unwrap()).collect();
    let n = frames.len();
    let (_, talker_logits) = golden.f32("talker_logits");
    let (_, predictor_logits) = golden.f32("predictor_logits");
    let (h, vocab, p_vocab) =
        (config.model.talker_config.stack.hidden_size, config.model.talker_config.stack.vocab_size, 2048);
    let spf = config.samples_per_frame;

    let mut seqs: Vec<Seq> = (0..3).map(|_| model.open(prompt.len(), n + 1).unwrap().unwrap()).collect();
    let starts = [0, 0, LAG];
    let draw = |t: usize| Draw { uniforms: [0.5; GROUPS], force: frames.get(t).copied() };
    let mut ours_talker = Vec::new();
    let mut ours_predictor = Vec::new();
    let mut wav = vec![Vec::new(); 3];
    for step in 0..=n + LAG {
        let at: Vec<Option<usize>> = starts.iter().map(|&s| step.checked_sub(s).filter(|&t| t <= n)).collect();
        for fresh in [true, false] {
            let who: Vec<usize> = (0..3).filter(|&k| at[k].is_some_and(|t| (t == 0) == fresh)).collect();
            if who.is_empty() {
                continue;
            }
            let picked = seqs.iter_mut().enumerate().filter(|(k, _)| who.contains(k)).map(|(_, s)| s);
            let out = if fresh {
                model.start(&mut picked.map(|s| (s, &prompt, draw(0))).collect::<Vec<_>>()).unwrap()
            } else {
                let rows = picked.zip(&who).map(|(s, &k)| (s, draw(at[k].unwrap())));
                model.step(&mut rows.collect::<Vec<_>>()).unwrap()
            };
            for (i, &k) in who.iter().enumerate() {
                match at[k].unwrap() {
                    t if t < n => {
                        assert_eq!(out.codes[i], frames[t], "stream {k} frame {t} is not the forced one");
                        wav[k].extend_from_slice(&out.wav[i * spf..(i + 1) * spf]);
                    }
                    _ => assert!(model.is_end(&out.codes[i]), "the recorded run ends after {n} frames"),
                }
            }
            if who[0] == 0 {
                let t = at[0].unwrap();
                if t == 0 {
                    let (worst, _) = agreement(&model.embeds(prompt.len()).unwrap(), &embeds, h);
                    eprintln!("prompt embeddings: worst cosine {worst:.6}");
                    assert!(worst > 0.999, "prompt embeddings diverge: worst cosine {worst}");
                }
                ours_talker.extend(model.talker_logits(0).unwrap());
                if t < n {
                    ours_predictor.extend(model.predictor_logits(0).unwrap());
                }
            }
        }
    }

    let (worst, agree) = agreement(&ours_talker, &talker_logits, vocab);
    eprintln!("talker logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.999 && agree > 0.95, "talker logits diverge");
    let (worst, agree) = agreement(&ours_predictor, &predictor_logits, p_vocab);
    eprintln!("predictor logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.998 && agree > 0.9, "predictor logits diverge");

    let (_, reference) = golden.f32("wav");
    assert_eq!(wav[0], wav[1], "two streams of the same frames in one batch decode differently");
    for (name, audio) in [("streamed waveform", &wav[0]), ("lagging stream", &wav[2])] {
        let snr = snr(audio, &reference);
        eprintln!("{name}: snr {snr:.2} dB");
        assert!(snr > 30.0, "{name} snr {snr:.2} dB");
    }
}

fn snr(ours: &[f32], theirs: &[f32]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    let noise: f64 = ours.iter().zip(theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    let signal: f64 = theirs.iter().map(|&x| (x as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}
