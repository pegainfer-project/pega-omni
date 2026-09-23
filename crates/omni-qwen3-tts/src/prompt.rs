//! The talker's prompt: text tokens and the dual-track sequence built from them.
//!
//! Every prompt position carries a text-track token, projected through the
//! talker's text MLP, and optionally a codec-track token added on top. The
//! layout is the official non-streaming one (the whole text is known up front):
//!
//! ```text
//! [instruct…]  <|im_start|>assistant\n  pad…pad bos   text… eos   pad
//!  -            -                        think… spk    pad… pad    bos
//! ```
//!
//! After the prompt, every decode step's text track is `<tts_pad>`.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde_json::Value;
use serde_json::json;

use crate::config::Model;

/// No codec-track token at this position.
pub const NO_CODEC: i32 = -1;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prompt {
    pub text: Vec<i32>,
    pub codec: Vec<i32>,
}

impl Prompt {
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn push(&mut self, text: i32, codec: i32) {
        self.text.push(text);
        self.codec.push(codec);
    }
}

/// A speaker's codec token and the language tag it speaks under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Voice {
    pub speaker: i32,
    pub language: Option<i32>,
}

/// Resolves a speaker and a language name (`auto` lets the model infer it). A
/// dialect speaker asked for Chinese or `auto` speaks its dialect.
pub fn voice(model: &Model, speaker: &str, language: &str) -> Result<Voice, String> {
    let t = &model.talker_config;
    let speaker_id = *t.spk_id.get(speaker).ok_or_else(|| format!("unknown speaker `{speaker}`"))?;
    let dialect = t.spk_is_dialect.get(speaker).and_then(Value::as_str);
    let language = match (language, dialect) {
        ("auto" | "chinese", Some(d)) => Some(d),
        ("auto", None) => None,
        (l, _) => Some(l),
    };
    let language = match language {
        None => None,
        Some(l) => Some(*t.codec_language_id.get(l).ok_or_else(|| format!("unknown language `{l}`"))?),
    };
    Ok(Voice { speaker: speaker_id, language })
}

/// Lays out the prompt for `text` (the template-tokenized text, see
/// [`Tokenizer::assistant`]) and an optional instruction (see
/// [`Tokenizer::instruct`]).
pub fn assemble(model: &Model, voice: Voice, text: &[i32], instruct: Option<&[i32]>) -> Prompt {
    let t = &model.talker_config;
    let (bos, eos, pad) = (model.tts_bos_token_id, model.tts_eos_token_id, model.tts_pad_token_id);
    let mut codec: Vec<i32> = match voice.language {
        None => vec![t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id],
        Some(l) => vec![t.codec_think_id, t.codec_think_bos_id, l, t.codec_think_eos_id],
    };
    codec.extend([voice.speaker, t.codec_pad_id, t.codec_bos_id]);

    let mut p = Prompt::default();
    instruct.unwrap_or_default().iter().for_each(|&id| p.push(id, NO_CODEC));
    text[..3].iter().for_each(|&id| p.push(id, NO_CODEC));
    for (i, &c) in codec[..codec.len() - 1].iter().enumerate() {
        p.push(if i + 2 < codec.len() { pad } else { bos }, c);
    }
    text[3..text.len() - 5].iter().for_each(|&id| p.push(id, t.codec_pad_id));
    p.push(eos, t.codec_pad_id);
    p.push(pad, t.codec_bos_id);
    p
}

/// The Qwen2 byte-level BPE the checkpoint ships as `vocab.json` + `merges.txt`.
pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// Builds the tokenizer the way `transformers` converts `Qwen2Tokenizer`:
    /// NFC, the Qwen2 split regex, byte-level BPE, and the special tokens of
    /// `tokenizer_config.json` at their declared ids.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).with_context(|| format!("reading {name}"));
        let vocab: Value = serde_json::from_str(&read("vocab.json")?)?;
        let merges: Vec<String> =
            read("merges.txt")?.lines().filter(|l| !l.starts_with("#version")).map(str::to_owned).collect();
        let config: Value = serde_json::from_str(&read("tokenizer_config.json")?)?;
        let added: Vec<Value> = config["added_tokens_decoder"]
            .as_object()
            .context("tokenizer_config.json has no added_tokens_decoder")?
            .iter()
            .map(|(id, t)| {
                json!({ "id": id.parse::<u64>().unwrap_or(u64::MAX), "content": t["content"], "single_word": false,
                        "lstrip": false, "rstrip": false, "normalized": false, "special": t["special"] })
            })
            .collect();
        let byte_level =
            json!({ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false });
        let spec = json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": { "type": "NFC" },
            "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
                { "type": "Split", "behavior": "Isolated", "invert": false, "pattern": { "Regex":
                  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" } },
                byte_level,
            ] },
            "post_processor": null,
            "decoder": byte_level,
            "model": { "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                       "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                       "vocab": vocab, "merges": merges },
        });
        let inner: tokenizers::Tokenizer = spec.to_string().parse().map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        for t in config["added_tokens_decoder"].as_object().into_iter().flatten() {
            let content = t.1["content"].as_str().unwrap_or_default();
            if inner.token_to_id(content).map(|i| i.to_string()).as_deref() != Some(t.0.as_str()) {
                bail!("special token {content} did not land at id {}", t.0);
            }
        }
        Ok(Self(inner))
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        let encoding = self.0.encode(text, false).expect("byte-level BPE encodes any string");
        encoding.get_ids().iter().map(|&i| i as i32).collect()
    }

    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    pub fn assistant(&self, text: &str) -> Vec<i32> {
        self.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"))
    }

    /// `<|im_start|>user\n{instruct}<|im_end|>\n`.
    pub fn instruct(&self, instruct: &str) -> Vec<i32> {
        self.encode(&format!("<|im_start|>user\n{instruct}<|im_end|>\n"))
    }
}
