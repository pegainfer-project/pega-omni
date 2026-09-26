//! The host side of a session against the reference's recorded run
//! (`tools/personaplex/golden.py`): tokenizer ids, the prompt's input rows,
//! and the per-step token bookkeeping of the whole conversation.
//!
//! Needs `OMNI_PERSONAPLEX_MODEL` (checkpoint directory) and
//! `OMNI_PERSONAPLEX_GOLDEN` (the golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_kern::weights::File;
use omni_personaplex::config::CODEBOOKS;
use omni_personaplex::config::CONTEXT;
use omni_personaplex::config::DRAWN;
use omni_personaplex::config::STREAMS;
use omni_personaplex::model::max_instructions_chars;
use omni_personaplex::prompt;
use omni_personaplex::tokenizer::Detok;
use omni_personaplex::tokenizer::Tokenizer;
use omni_personaplex::voice::Voice;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_PERSONAPLEX_MODEL").zip(get("OMNI_PERSONAPLEX_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_PERSONAPLEX_MODEL and OMNI_PERSONAPLEX_GOLDEN");
    }
    found
}

fn metadata(path: &std::path::Path) -> std::collections::BTreeMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    meta.metadata().clone().unwrap().into_iter().collect()
}

fn rows<const N: usize>(file: &File, name: &str) -> Vec<[i32; N]> {
    file.ints(name).unwrap().as_chunks::<N>().0.to_vec()
}

#[test]
fn tokenizer_matches_sentencepiece() {
    let Some((model, _)) = paths() else { return };
    let tok = Tokenizer::load(&model.join("tokenizer_spm_32k_3.model")).unwrap();
    let cases: [(&str, &[u32]); 6] = [
        ("Hello, world!", &[11725, 261, 671, 430]),
        ("  two  spaces and 1234 digits", &[260, 260, 368, 260, 3697, 267, 260, 265, 270, 278, 281, 11535]),
        (
            "naïve café — 東京 🙂",
            &[572, 16650, 518, 29301, 729, 260, 234, 161, 181, 232, 190, 176, 260, 244, 163, 157, 134],
        ),
        ("<system> You are Tom. <system>", &[607, 4831, 578, 493, 298, 2608, 263, 607, 4831, 578]),
        ("end.\nnew line", &[606, 263, 14, 2706, 619]),
        ("don't let it sit, it's done", &[551, 286, 303, 1131, 296, 6485, 261, 296, 286, 266, 954]),
    ];
    for (text, ids) in cases {
        assert_eq!(tok.encode(text).unwrap(), ids, "{text:?}");
        let mut d = Detok::default();
        let back: String = ids.iter().filter_map(|&i| d.push(&tok, i)).collect();
        assert_eq!(back, text.trim_start(), "{text:?} round trip");
    }
}

#[test]
fn prompt_and_steps_match_the_reference() {
    let Some((model, golden)) = paths() else { return };
    let meta = metadata(&golden);
    let file = File::open(&golden).unwrap();
    let tok = Tokenizer::load(&model.join("tokenizer_spm_32k_3.model")).unwrap();
    let ids = tok.encode(&prompt::wrap(&meta["prompt"])).unwrap();
    let want: Vec<u32> = file.ints("prompt_ids").unwrap().iter().map(|&i| i as u32).collect();
    assert_eq!(ids, want);

    let voice = Voice::load(&model.join(format!("voices/{}.pt", meta["voice"]))).unwrap();
    assert_eq!(voice.steps().to_string(), meta["voice_steps"]);
    let p = prompt::build(&voice, &ids).unwrap();
    assert_eq!(p.rows, rows::<STREAMS>(&file, "prompt_inputs"));

    let inputs = rows::<STREAMS>(&file, "inputs");
    let drawn = rows::<DRAWN>(&file, "sampled");
    let caller = rows::<CODEBOOKS>(&file, "caller_codes");
    let agent = rows::<CODEBOOKS>(&file, "agent_codes");
    assert_eq!(p.first, inputs[0]);
    assert!(p.first_force.iter().zip(&drawn[0]).all(|(&f, &d)| f < 0 || f == d), "{:?}", p.first_force);
    let (mut row, mut pending) = (p.first, p.pending);
    for j in 0..inputs.len() {
        assert_eq!(row, inputs[j], "step {j} input");
        let s = prompt::advance(&row, &pending, &drawn[j], &caller[j]);
        assert_eq!(s.frame, agent[j], "step {j} agent frame");
        (row, pending) = (s.row, s.pending);
    }
}

#[test]
fn the_longest_allowed_instructions_fit_the_context_with_any_voice() {
    let Some((model, _)) = paths() else { return };
    let tok = Tokenizer::load(&model.join("tokenizer_spm_32k_3.model")).unwrap();
    let voices: Vec<Voice> = std::fs::read_dir(model.join("voices"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "pt"))
        .map(|p| Voice::load(&p).unwrap())
        .collect();
    let longest = voices.iter().max_by_key(|v| v.steps()).unwrap();
    let max = max_instructions_chars(CONTEXT - 1, longest.steps());
    eprintln!("{} voices, the longest {} steps: {max} characters", voices.len(), longest.steps());
    for c in ["🙂", "東", "a", " ", "\u{0}"] {
        let ids = tok.encode(&prompt::wrap(&c.repeat(max))).unwrap();
        let p = prompt::build(longest, &ids).unwrap();
        assert!(p.len() < CONTEXT, "{max} x {c:?}: {} rows", p.len());
    }
}
