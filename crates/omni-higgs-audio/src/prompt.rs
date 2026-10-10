//! The zero-shot prompt: `<|tts|> <|text|> {text} <|audio|>`.
//!
//! `<|tts|>` selects speech generation over recognition; leaving it out gives fluent speech
//! that is not the text.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    tts: i32,
    text: i32,
    audio: i32,
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let id = |t: &str| inner.token_to_id(t).map(|i| i as i32).with_context(|| format!("tokenizer has no {t}"));
        Ok(Self { tts: id("<|tts|>")?, text: id("<|text|>")?, audio: id("<|audio|>")?, inner })
    }

    pub fn prompt(&self, text: &str) -> Vec<i32> {
        let encoding = self.inner.encode(text, false).expect("byte-level BPE encodes any string");
        let mut ids = vec![self.tts, self.text];
        ids.extend(encoding.get_ids().iter().map(|&i| i as i32));
        ids.push(self.audio);
        ids
    }
}
