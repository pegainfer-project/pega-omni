//! The model against the reference's recorded run
//! (`tools/personaplex/golden.py`), through the serving path (`prefill`,
//! then one `tick` per frame) with the recorded draws and caller codes
//! forced in place of sampling and the encoder: Helium's input rows, text
//! and depformer logits, the emitted agent frames, the encoder's latent and
//! codes of the caller's voiced frames, and the decoded agent audio.
//!
//! Three sessions share every call: `a` and `b` in lockstep (their outputs
//! must be identical), `c` starting `LAG` ticks later, so it is prefilled
//! between ticks of the others and runs in calls of other sizes.
//!
//! Needs a GPU, `OMNI_PERSONAPLEX_MODEL` (checkpoint directory) and
//! `OMNI_PERSONAPLEX_GOLDEN` (the golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_kern::weights::File;
use omni_personaplex::config::CARD;
use omni_personaplex::config::CODEBOOKS;
use omni_personaplex::config::DRAWN;
use omni_personaplex::config::FRAME;
use omni_personaplex::config::MIMI_DIM;
use omni_personaplex::config::STREAMS;
use omni_personaplex::config::Sampling;
use omni_personaplex::config::TEXT_VOCAB;
use omni_personaplex::model::Limits;
use omni_personaplex::model::Model;
use omni_personaplex::model::Step;

const LAG: usize = 5;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_PERSONAPLEX_MODEL").zip(get("OMNI_PERSONAPLEX_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_PERSONAPLEX_MODEL and OMNI_PERSONAPLEX_GOLDEN");
    }
    found
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// Signal to noise of `ours` against `theirs`, in dB.
fn snr(ours: &[f32], theirs: &[f32]) -> f64 {
    let signal: f64 = theirs.iter().map(|&x| (x as f64).powi(2)).sum();
    let noise: f64 = ours.iter().zip(theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}

/// Frames quieter than this (about -60 dBFS) encode to latents dominated by
/// rounding: bf16 moves them by a cosine of up to 0.1, so they are not compared.
fn voiced(pcm: &[f32]) -> bool {
    (pcm.iter().map(|&x| x * x).sum::<f32>() / pcm.len() as f32).sqrt() > 1e-3
}

#[derive(Default)]
struct Tally {
    worst_text: f64,
    worst_audio: f64,
    text_agree: usize,
    audio_agree: usize,
    caller_agree: [usize; CODEBOOKS],
    voiced: usize,
    voiced_agree: usize,
    worst_voiced_latent: f64,
    steps: usize,
    pcm: Vec<f32>,
}

#[test]
fn teacher_forced_run_matches_the_reference() {
    let Some((dir, golden)) = paths() else { return };
    let meta = {
        let bytes = std::fs::read(&golden).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        meta.metadata().clone().unwrap().into_iter().collect::<std::collections::BTreeMap<_, _>>()
    };
    let file = File::open(&golden).unwrap();
    let rows = |name: &str| file.ints(name).unwrap();
    let floats = |name: &str| file.host(name).unwrap().data;
    let (inputs, drawn, caller, agent) = (rows("inputs"), rows("sampled"), rows("caller_codes"), rows("agent_codes"));
    let (text_logits, audio_logits) = (floats("text_logits"), floats("audio_logits"));
    let (caller_pcm, agent_pcm, latent) = (floats("caller_pcm"), floats("agent_pcm"), floats("caller_latent"));
    let steps = inputs.len() / STREAMS;

    let mut model = Model::load(0, &dir, Limits { max_sessions: 4, max_prefill: 512 }, Sampling::default()).unwrap();
    let prefix = model.prompt(&meta["voice"], &meta["prompt"]).unwrap();
    let mut sessions: Vec<_> = (0..3).map(|_| model.open().unwrap()).collect();
    let mut tallies: Vec<Tally> = (0..3)
        .map(|_| Tally { worst_text: 1.0, worst_audio: 1.0, worst_voiced_latent: 1.0, ..Tally::default() })
        .collect();
    {
        let (ab, _) = sessions.split_at_mut(2);
        let [a, b] = ab else { unreachable!() };
        model.start(&mut [(a, &prefix), (b, &prefix)]).unwrap();
    }
    let step_of = |j: usize| Step {
        pcm: &caller_pcm[j * FRAME..(j + 1) * FRAME],
        force: drawn[j * DRAWN..(j + 1) * DRAWN].try_into().unwrap(),
        force_caller: caller[j * CODEBOOKS..(j + 1) * CODEBOOKS].try_into().unwrap(),
        seed: j as u64,
    };
    for tick in 0..steps + LAG {
        if tick == LAG {
            model.start(&mut [(&mut sessions[2], &prefix)]).unwrap();
        }
        let live: Vec<(usize, usize)> = (0..3)
            .filter_map(|s| {
                let j = if s == 2 { tick.checked_sub(LAG)? } else { tick };
                (j < steps).then_some((s, j))
            })
            .collect();
        let mut rows_in: Vec<_> = {
            let mut refs: Vec<Option<&mut _>> = sessions.iter_mut().map(Some).collect();
            live.iter().map(|&(s, j)| (refs[s].take().unwrap(), step_of(j))).collect()
        };
        let out = model.tick(&mut rows_in).unwrap();
        for (n, &(s, j)) in live.iter().enumerate() {
            let t = &mut tallies[s];
            assert_eq!(
                model.input_row(n).unwrap().as_slice(),
                &inputs[j * STREAMS..(j + 1) * STREAMS],
                "session {s} step {j} input row"
            );
            assert_eq!(
                &out.emitted[n][1..],
                &agent[j * CODEBOOKS..(j + 1) * CODEBOOKS],
                "session {s} step {j} agent frame"
            );
            let tl = model.text_logits(n).unwrap();
            let want = &text_logits[j * TEXT_VOCAB..(j + 1) * TEXT_VOCAB];
            t.worst_text = t.worst_text.min(cosine(&tl, want));
            t.text_agree += (argmax(&tl) == argmax(want)) as usize;
            let al = model.audio_logits(n).unwrap();
            for k in 0..CODEBOOKS {
                let (ours, theirs) =
                    (&al[k * CARD..(k + 1) * CARD], &audio_logits[(j * CODEBOOKS + k) * CARD..][..CARD]);
                t.worst_audio = t.worst_audio.min(cosine(ours, theirs));
                t.audio_agree += (argmax(ours) == argmax(theirs)) as usize;
            }
            let codes = model.caller_codes(n).unwrap();
            if voiced(&caller_pcm[j * FRAME..(j + 1) * FRAME]) {
                let ours = model.caller_latent(n).unwrap();
                t.worst_voiced_latent =
                    t.worst_voiced_latent.min(cosine(&ours, &latent[j * MIMI_DIM..(j + 1) * MIMI_DIM]));
                t.voiced += 1;
                t.voiced_agree += (codes[0] == caller[j * CODEBOOKS]) as usize;
            }
            for k in 0..CODEBOOKS {
                t.caller_agree[k] += (codes[k] == caller[j * CODEBOOKS + k]) as usize;
            }
            t.pcm.extend(&out.pcm[n * FRAME..(n + 1) * FRAME]);
            t.steps += 1;
        }
    }
    for (s, t) in tallies.iter().enumerate() {
        let n = t.steps as f64;
        eprintln!(
            "session {s}: text cos ≥ {:.5}, argmax {:.3}; audio cos ≥ {:.5}, argmax {:.3}; voiced caller latent cos ≥ {:.5}, semantic {:.3} of {}; all caller codes {:?}; pcm SNR {:.1} dB",
            t.worst_text,
            t.text_agree as f64 / n,
            t.worst_audio,
            t.audio_agree as f64 / (n * CODEBOOKS as f64),
            t.worst_voiced_latent,
            t.voiced_agree as f64 / t.voiced as f64,
            t.voiced,
            t.caller_agree.map(|a| (a as f64 / n * 100.0).round() / 100.0),
            snr(&t.pcm, &agent_pcm),
        );
    }
    assert_eq!(tallies[0].pcm, tallies[1].pcm, "lockstep sessions diverged");
    let batching = snr(&tallies[2].pcm, &tallies[0].pcm);
    assert!(batching > 40.0, "a session in other batches decodes {batching:.1} dB from its lockstep twin");
    for t in &tallies {
        assert_eq!(t.steps, steps);
        assert!(t.worst_text > 0.995, "text logits cosine {}", t.worst_text);
        assert!(t.worst_audio > 0.995, "audio logits cosine {}", t.worst_audio);
        assert!(t.text_agree as f64 / steps as f64 > 0.97);
        assert!(t.audio_agree as f64 / (steps * CODEBOOKS) as f64 > 0.97);
        assert!(t.worst_voiced_latent > 0.98, "voiced caller latent cosine {}", t.worst_voiced_latent);
        assert!(
            t.voiced_agree as f64 / t.voiced as f64 > 0.9,
            "voiced semantic caller codes {}/{}",
            t.voiced_agree,
            t.voiced
        );
        assert!(snr(&t.pcm, &agent_pcm) > 25.0, "decoded audio SNR {:.1} dB", snr(&t.pcm, &agent_pcm));
    }
}
