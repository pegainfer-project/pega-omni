use omni_qwen3_tts::config::Model;
use omni_qwen3_tts::config::ModelType;
use omni_qwen3_tts::prompt;
use omni_qwen3_tts::prompt::NO_CODEC;
use serde_json::json;

fn model(kind: &str) -> Model {
    // Reduced checkpoint config: the official 1.7B shapes and special ids.
    let stack = json!({
        "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6, "vocab_size": 3072
    });
    let mut talker = stack;
    talker["code_predictor_config"] = json!({
        "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 5,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6, "vocab_size": 2048
    });
    for (key, value) in [
        ("num_code_groups", 16),
        ("codec_bos_id", 2149),
        ("codec_eos_token_id", 2150),
        ("codec_pad_id", 2148),
        ("codec_think_id", 2154),
        ("codec_nothink_id", 2155),
        ("codec_think_bos_id", 2156),
        ("codec_think_eos_id", 2157),
    ] {
        talker[key] = json!(value);
    }
    talker["codec_language_id"] = json!({"english": 2050, "chinese": 2055, "sichuan_dialect": 2062});
    let custom = kind == "custom_voice";
    talker["spk_id"] = if custom { json!({"ryan": 3061, "eric": 2875}) } else { json!({}) };
    talker["spk_is_dialect"] = if custom { json!({"ryan": false, "eric": "sichuan_dialect"}) } else { json!({}) };
    serde_json::from_value(json!({
        "tts_model_type": kind, "tts_bos_token_id": 151672, "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671, "talker_config": talker
    }))
    .unwrap()
}

// Role prefix (3 ids), two content tokens, and the template suffix (5 ids).
const TEXT: &[i32] = &[151644, 77091, 198, 100, 101, 151645, 198, 151644, 77091, 198];
const INSTRUCT: &[i32] = &[151644, 872, 198, 200, 151645, 198];

#[test]
fn voice_design_omits_the_speaker_position_and_prepends_instructions() {
    let model = model("voice_design");
    assert_eq!(model.tts_model_type, ModelType::VoiceDesign);
    assert!(model.talker_config.spk_id.is_empty());
    for language in ["auto", "english"] {
        let voice = prompt::voice(&model, "alloy", language).unwrap();
        assert_eq!(voice.speaker, None);
        let p = prompt::assemble(&model, voice, TEXT, Some(INSTRUCT));
        let mut text = INSTRUCT.to_vec();
        text.extend_from_slice(&TEXT[..3]);
        text.extend(if language == "auto" { vec![151671; 3] } else { vec![151671; 4] });
        text.extend([151672, 100, 101, 151673, 151671]);
        assert_eq!(p.text, text);
        let mut codec = vec![NO_CODEC; INSTRUCT.len() + 3];
        codec.extend(if language == "auto" { vec![2155, 2156, 2157] } else { vec![2154, 2156, 2050, 2157] });
        codec.extend([2148, 2148, 2148, 2148, 2149]);
        assert_eq!(p.codec, codec);
        assert_eq!(p.text.len(), p.codec.len());
    }
}

#[test]
fn custom_voice_keeps_its_speaker_position_and_dialect_rules() {
    let model = model("custom_voice");
    let voice = prompt::voice(&model, "ryan", "english").unwrap();
    assert_eq!(voice.speaker, Some(3061));
    let p = prompt::assemble(&model, voice, TEXT, None);
    assert_eq!(p.text, [151644, 77091, 198, 151671, 151671, 151671, 151671, 151671, 151672, 100, 101, 151673, 151671]);
    assert_eq!(p.codec, [NO_CODEC, NO_CODEC, NO_CODEC, 2154, 2156, 2050, 2157, 3061, 2148, 2148, 2148, 2148, 2149]);
    for language in ["auto", "chinese"] {
        assert_eq!(prompt::voice(&model, "eric", language).unwrap().language, Some(2062));
    }
    assert_eq!(prompt::voice(&model, "eric", "english").unwrap().language, Some(2050));
    assert!(prompt::voice(&model, "not-a-speaker", "auto").is_err());
}

#[test]
fn voice_design_ignores_voice_names_but_checks_languages() {
    let model = model("voice_design");
    let voice = prompt::voice(&model, "alloy", "auto").unwrap();
    for speaker in ["", "ryan", "not-a-speaker"] {
        assert_eq!(prompt::voice(&model, speaker, "auto").unwrap(), voice);
    }
    assert!(prompt::voice(&model, "alloy", "not-a-language").is_err());
}
