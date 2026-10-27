use std::collections::BTreeMap;

use omni_engine::Draft;
use omni_engine::EngineInfo;

fn info(designed: bool) -> EngineInfo {
    EngineInfo {
        model: "speech".into(),
        sample_rate: 24_000,
        voices: if designed { Default::default() } else { ["ryan".into()].into() },
        instructions_required: designed,
        extra: BTreeMap::new(),
        speeds: 1.0..=1.0,
        max_input_chars: 4096,
    }
}

#[test]
fn instruction_driven_speech_needs_a_non_blank_description() {
    for instructions in [None, Some(""), Some(" \t\n\u{3000}")] {
        let draft = Draft {
            input: "Hello.".into(),
            voice: "alloy".into(),
            instructions: instructions.map(str::to_owned),
            ..Default::default()
        };
        assert_eq!(info(true).check(draft).unwrap_err().param, "instructions");
    }
}

#[test]
fn instruction_driven_speech_ignores_the_sdk_voice_name() {
    for voice in ["alloy", "ryan", "", "not-a-preset"] {
        let draft = Draft {
            input: "Hello.".into(),
            voice: voice.into(),
            instructions: Some(" A low, slow voice. ".into()),
            ..Default::default()
        };
        let speech = info(true).check(draft).unwrap();
        assert_eq!(speech.instructions.as_deref(), Some(" A low, slow voice. "));
    }
}

#[test]
fn preset_voices_still_require_a_known_speaker_and_allow_no_instruction() {
    let draft = Draft { input: "Hello.".into(), voice: "ryan".into(), ..Default::default() };
    assert!(info(false).check(draft.clone()).is_ok());
    let unknown = Draft { voice: "alloy".into(), ..draft };
    assert_eq!(info(false).check(unknown).unwrap_err().param, "voice");
}
