//! The engine against the Hugging Face run recorded by `tools/higgs_audio/golden.py`.
//!
//! Needs `OMNI_HIGGS_AUDIO_MODEL` (the checkpoint directory) and `OMNI_HIGGS_AUDIO_GOLDEN` (the
//! golden file); skipped when either is unset.

use std::path::PathBuf;

use omni_higgs_audio::config::CODEBOOKS;
use omni_higgs_audio::config::Config;
use omni_higgs_audio::delay::BOC;
use omni_higgs_audio::delay::Delay;
use omni_higgs_audio::delay::Row;
use omni_higgs_audio::delay::undelay;
use omni_higgs_audio::prompt::Tokenizer;
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
fn the_config_is_the_one_the_engine_runs() {
    let Some((model, _)) = paths() else { return };
    let config = Config::load(&model).unwrap();
    assert_eq!(config.stack.num_hidden_layers, 36);
}

#[test]
fn the_prompt_is_the_references() {
    let Some((model, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    assert_eq!(Tokenizer::load(&model).unwrap().prompt(&golden.text()), golden.i32("prompt_ids"));
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
