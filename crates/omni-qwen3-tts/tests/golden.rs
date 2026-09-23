//! The model against the official implementation's recorded run
//! (`tools/qwen3_tts/golden.py`): tokenizer ids, prompt embeddings,
//! teacher-forced talker and code-predictor logits, and the decoded waveform.
//!
//! Needs a GPU, `OMNI_QWEN3_TTS_MODEL` (checkpoint directory) and
//! `OMNI_QWEN3_TTS_GOLDEN` (the golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_cuda::Gpu;
use omni_qwen3_tts::codec::Codec;
use omni_qwen3_tts::config::Config;
use omni_qwen3_tts::engine::chunk_due;
use omni_qwen3_tts::prompt;
use omni_qwen3_tts::prompt::Tokenizer;
use omni_qwen3_tts::talker::Frame;
use omni_qwen3_tts::talker::GROUPS;
use omni_qwen3_tts::talker::Input;
use omni_qwen3_tts::talker::Limits;
use omni_qwen3_tts::talker::Probe;
use omni_qwen3_tts::talker::Row;
use omni_qwen3_tts::talker::Sampling;
use omni_qwen3_tts::talker::Talker;
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

#[test]
fn matches_the_official_run() {
    let Some((model, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    let config = Config::load(&model).unwrap();
    let tokenizer = Tokenizer::load(&model).unwrap();

    let text_ids = tokenizer.assistant(&golden.text);
    assert_eq!(text_ids, golden.i32("text_ids"), "tokenizer ids");

    let voice = prompt::voice(&config.model, &golden.speaker, &golden.language).unwrap();
    let prompt = prompt::assemble(&config.model, voice, &text_ids, None);
    let (embed_shape, embeds) = golden.f32("prefill_embeds");
    assert_eq!(prompt.len(), embed_shape[0], "prompt length");

    let gpu = Gpu::new(0).unwrap();
    gpu.bind().unwrap();
    let limits = Limits { max_batch: 1, max_tokens: 256, pages: 64, page_size: 16 };
    let mut talker =
        Talker::load(&gpu, &File::open(&model.join("model.safetensors")).unwrap(), &config.model, limits).unwrap();
    let codes = golden.i32("codes");
    let frames: Vec<[i32; GROUPS]> = codes.chunks(GROUPS).map(|c| c.try_into().unwrap()).collect();
    let (_, talker_logits) = golden.f32("talker_logits");
    let (_, predictor_logits) = golden.f32("predictor_logits");
    let (vocab, p_vocab) = (config.model.talker_config.stack.vocab_size, 2048);

    let g = &config.generation;
    let sampling = Sampling {
        temperature: g.temperature,
        top_k: g.top_k,
        repetition_penalty: g.repetition_penalty,
        sub_temperature: g.subtalker_temperature,
        sub_top_k: g.subtalker_top_k,
    };
    let pages: Vec<i32> = (0..64).collect();
    let seen = vec![0u32; talker.seen_words()];
    let mut ours_talker = Vec::new();
    let mut ours_predictor = Vec::new();
    let mut cached = 0;
    for t in 0..=frames.len() {
        let input = match t {
            0 => Input::Prompt(&prompt),
            _ => Input::Frame(&frames[t - 1]),
        };
        let row = Row { pages: &pages, cached, input, sampling, seen: &seen, generated: t, uniforms: [0.5; GROUPS] };
        let mut probe = Probe { force: frames.get(t).map(|f| vec![*f]).unwrap_or_default(), ..Probe::default() };
        let out = talker.step(&gpu, &[row], Some(&mut probe)).unwrap();
        if t == 0 {
            let (worst, _) = agreement(&probe.inputs, &embeds, config.model.talker_config.stack.hidden_size);
            eprintln!("prompt embeddings: worst cosine {worst:.6}");
            assert!(worst > 0.999, "prompt embeddings diverge: worst cosine {worst}");
        }
        cached += prompt.len() * (t == 0) as usize + (t > 0) as usize;
        ours_talker.extend(probe.talker_logits);
        if t < frames.len() {
            ours_predictor.extend(probe.predictor_logits.into_iter().flatten());
        } else {
            assert_eq!(out, vec![Frame::End], "the recorded run ends after {} frames", frames.len());
        }
    }

    let (worst, agree) = agreement(&ours_talker, &talker_logits, vocab);
    eprintln!("talker logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.999 && agree > 0.95, "talker logits diverge");
    let (worst, agree) = agreement(&ours_predictor, &predictor_logits, p_vocab);
    eprintln!("predictor logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.998 && agree > 0.9, "predictor logits diverge");

    let (_, wav) = golden.f32("wav");
    let file = File::open(&model.join("speech_tokenizer/model.safetensors")).unwrap();
    let mut codec = Codec::load(&gpu, &file, &config.codec, config.samples_per_frame, frames.len()).unwrap();
    let whole = codec.decode(&gpu, &codes).unwrap();
    let snr = snr(&whole, &wav);
    eprintln!("waveform: snr {snr:.2} dB");
    assert!(snr > 30.0, "waveform snr {snr:.2} dB");

    let (spf, ctx) = (config.samples_per_frame, config.codec.sliding_window);
    let mut streamed = Vec::new();
    let mut emitted = 0;
    while let Some(n) = chunk_due(frames.len(), emitted, true, 2, 8) {
        let start = emitted.saturating_sub(ctx);
        let wav = codec.decode(&gpu, &codes[start * GROUPS..(emitted + n) * GROUPS]).unwrap();
        streamed.extend_from_slice(&wav[(emitted - start) * spf..]);
        emitted += n;
    }
    let chunked = self::snr(&streamed, &wav);
    eprintln!("streamed waveform: snr {chunked:.2} dB");
    assert!(chunked > 30.0, "streamed waveform snr {chunked:.2} dB");
}

fn snr(ours: &[f32], theirs: &[f32]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    let noise: f64 = ours.iter().zip(theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    let signal: f64 = theirs.iter().map(|&x| (x as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}
