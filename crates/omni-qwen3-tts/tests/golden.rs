//! The model against the official implementation's recorded run
//! (`tools/qwen3_tts/golden.py`): tokenizer ids, prompt embeddings,
//! teacher-forced talker and code-predictor logits, and the decoded waveform,
//! all through the serving path (`prefill`, `first`, `decode`) with the
//! recorded frames forced in place of the draws. The same test replays one
//! stream, then three streams sequentially, so single-row algorithms and mixed
//! batch sizes are covered without parallel GPU contexts.
//!
//! Needs a GPU, `OMNI_QWEN3_TTS_MODEL` (checkpoint directory) and
//! `OMNI_QWEN3_TTS_GOLDEN` (the golden file); skipped when either is unset.

use std::path::Path;
use std::path::PathBuf;

use omni_qwen3_tts::config::ModelType;
use omni_qwen3_tts::model::Draw;
use omni_qwen3_tts::model::GROUPS;
use omni_qwen3_tts::model::Limits;
use omni_qwen3_tts::model::Model;
use omni_qwen3_tts::model::Seq;
use omni_qwen3_tts::prompt;
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
    instructions: Option<String>,
    model_type: ModelType,
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
            instructions: m.get("instructions").cloned(),
            model_type: m
                .get("tts_model_type")
                .map(|kind| serde_json::from_value(serde_json::Value::String(kind.clone())).unwrap())
                .unwrap_or(ModelType::CustomVoice),
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
    for streams in [1, 3] {
        compare(&dir, &golden, streams);
    }
}

fn compare(dir: &Path, golden: &Path, streams: usize) {
    eprintln!("streams={streams}: replay official golden");
    let golden = Golden::open(golden);
    let limits = Limits { max_batch: streams, max_tokens: 1024, kv_gib: 0.5 };
    let gemm_algos =
        std::env::var_os("OMNI_QWEN3_TTS_GEMM_ALGOS").filter(|p| !p.is_empty()).map(std::path::PathBuf::from);
    let mut model = Model::load(0, dir, limits, gemm_algos.as_deref()).unwrap();
    let config = model.config.clone();
    assert_eq!(config.model.tts_model_type, golden.model_type, "golden checkpoint family");

    let text_ids = model.tokenizer.assistant(&golden.text);
    assert_eq!(text_ids, golden.i32("text_ids"), "tokenizer ids");
    let voice = prompt::voice(&config.model, &golden.speaker, &golden.language).unwrap();
    let instruct = golden.instructions.as_deref().filter(|s| !s.is_empty()).map(|s| model.tokenizer.instruct(s));
    if golden.instructions.is_some() {
        assert_eq!(instruct.as_deref().unwrap_or_default(), golden.i32("instruct_ids"), "instruction tokenizer ids");
    }
    let prompt = prompt::assemble(&config.model, voice, &text_ids, instruct.as_deref());
    let (embed_shape, embeds) = golden.f32("prefill_embeds");
    assert_eq!(prompt.len(), embed_shape[0], "prompt length");

    let frames: Vec<[i32; GROUPS]> = golden.i32("codes").chunks(GROUPS).map(|c| c.try_into().unwrap()).collect();
    let n = frames.len();
    let (_, talker_logits) = golden.f32("talker_logits");
    let (_, predictor_logits) = golden.f32("predictor_logits");
    let t = &config.model.talker_config;
    let (h, vocab, p_vocab) = (t.stack.hidden_size, t.stack.vocab_size, t.code_predictor_config.vocab_size);
    let spf = config.samples_per_frame;
    assert_eq!(talker_logits.len(), (n + 1) * vocab, "reference must include the terminal EOS-step logits");
    assert_eq!(predictor_logits.len(), n * (GROUPS - 1) * p_vocab, "reference predictor logits per frame");

    let mut seqs: Vec<Seq> = (0..streams).map(|_| model.open(prompt.len(), n + 1).unwrap().unwrap()).collect();
    let starts = if streams == 1 { vec![0] } else { vec![0, 0, LAG] };
    let lag = if streams == 1 { 0 } else { LAG };
    // The reference sampled EOS with its own RNG. Force that final decision too;
    // its raw talker logits are still compared below, independently of sampling.
    let mut end = [0; GROUPS];
    end[0] = config.model.talker_config.codec_eos_token_id;
    let draw = |t: usize| Draw { uniforms: [0.5; GROUPS], force: Some(frames.get(t).copied().unwrap_or(end)) };
    let mut ours_talker = Vec::new();
    let mut ours_predictor = Vec::new();
    let mut wav = vec![Vec::new(); streams];
    for step in 0..=n + lag {
        let at: Vec<Option<usize>> = starts.iter().map(|&s| step.checked_sub(s).filter(|&t| t <= n)).collect();
        for fresh in [true, false] {
            let who: Vec<usize> = (0..streams).filter(|&k| at[k].is_some_and(|t| (t == 0) == fresh)).collect();
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
                    eprintln!("streams={streams}: prompt embeddings: worst cosine {worst:.6}");
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
    eprintln!("streams={streams}: talker logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.999 && agree > 0.95, "talker logits diverge");
    let (worst, agree) = agreement(&ours_predictor, &predictor_logits, p_vocab);
    eprintln!("streams={streams}: predictor logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.998 && agree > 0.9, "predictor logits diverge");

    let (_, reference) = golden.f32("wav");
    if streams == 3 {
        assert_eq!(wav[0], wav[1], "two streams of the same frames in one batch decode differently");
    }
    for (k, audio) in wav.iter().enumerate() {
        let snr = snr(audio, &reference);
        eprintln!("streams={streams} stream={k}: snr {snr:.2} dB");
        assert!(snr > 30.0, "streams={streams} stream={k} snr {snr:.2} dB");
    }
}

fn snr(ours: &[f32], theirs: &[f32]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    let noise: f64 = ours.iter().zip(theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    let signal: f64 = theirs.iter().map(|&x| (x as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}
