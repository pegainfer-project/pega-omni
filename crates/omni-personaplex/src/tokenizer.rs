//! The text tokenizer: the checkpoint's SentencePiece model
//! (`tokenizer_spm_32k_3.model`: unigram, byte fallback, a dummy `▁` prefix,
//! whitespace and digit splitting, identity normalization) as a
//! `tokenizers` unigram with the matching normalizer and pre-tokenizers.
//!
//! The `.model` file is a `ModelProto` protobuf; only its pieces (text, score,
//! type) are read, by a reader small enough not to need a protobuf crate.
//!
//! [`Detok`] turns the model's text stream back into text as it arrives,
//! holding back byte-fallback pieces until they complete a UTF-8 character
//! and dropping the stream's leading whitespace (the `▁` every first word
//! carries).

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use tokenizers::DecoderWrapper;
use tokenizers::NormalizerWrapper;
use tokenizers::PostProcessorWrapper;
use tokenizers::PreTokenizerWrapper;
use tokenizers::TokenizerImpl;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::Prepend;
use tokenizers::normalizers::Replace;
use tokenizers::normalizers::Sequence;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::pre_tokenizers::metaspace::Metaspace;
use tokenizers::pre_tokenizers::metaspace::PrependScheme;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;

const SPACE: char = '▁';

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    Byte(u8),
}

/// A SentencePiece `ModelProto`'s pieces.
fn pieces(bytes: &[u8]) -> Result<Vec<(String, f32, Kind)>> {
    fields(bytes)?
        .into_iter()
        .filter(|(field, _)| *field == 1)
        .map(|(_, value)| {
            let Value::Bytes(piece) = value else { bail!("a piece that is not a message") };
            let mut text = String::new();
            let (mut score, mut kind) = (0.0, 1);
            for (field, value) in fields(piece)? {
                match (field, value) {
                    (1, Value::Bytes(b)) => {
                        text = String::from_utf8(b.to_vec()).context("a piece that is not UTF-8")?
                    }
                    (2, Value::Fixed32(v)) => score = f32::from_bits(v),
                    (3, Value::Varint(v)) => kind = v,
                    _ => {}
                }
            }
            let kind = match kind {
                1 | 4 => Kind::Normal,
                2 => Kind::Unknown,
                3 | 5 => Kind::Control,
                6 => Kind::Byte(
                    text.strip_prefix("<0x")
                        .and_then(|h| h.strip_suffix('>'))
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                        .with_context(|| format!("byte piece `{text}`"))?,
                ),
                other => bail!("piece `{text}` has unknown type {other}"),
            };
            Ok((text, score, kind))
        })
        .collect()
}

enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

/// The top-level fields of a protobuf message, in order.
fn fields(mut b: &[u8]) -> Result<Vec<(u64, Value<'_>)>> {
    fn varint(b: &mut &[u8]) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = b.split_first().context("truncated varint")?;
            *b = rest;
            v |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Ok(v);
            }
        }
        bail!("varint longer than 64 bits")
    }
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
        ensure!(b.len() >= n, "truncated field");
        let (head, rest) = b.split_at(n);
        *b = rest;
        Ok(head)
    }
    let mut out = Vec::new();
    while !b.is_empty() {
        let key = varint(&mut b)?;
        let value = match key & 7 {
            0 => Value::Varint(varint(&mut b)?),
            1 => {
                take(&mut b, 8)?;
                Value::Fixed64
            }
            2 => {
                let n = varint(&mut b)? as usize;
                Value::Bytes(take(&mut b, n)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(take(&mut b, 4)?.try_into()?)),
            other => bail!("unsupported protobuf wire type {other}"),
        };
        out.push((key >> 3, value));
    }
    Ok(out)
}

type Spm = TokenizerImpl<Unigram, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;

pub struct Tokenizer {
    spm: Spm,
    kinds: Vec<Kind>,
    texts: Vec<String>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    /// A tokenizer from a SentencePiece `ModelProto`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let pieces = pieces(bytes)?;
        let unk = pieces.iter().position(|p| p.2 == Kind::Unknown).context("no unknown piece")?;
        let vocab = pieces.iter().map(|(t, s, _)| (t.clone(), *s as f64)).collect();
        let model = Unigram::from(vocab, Some(unk), true).map_err(|e| anyhow::anyhow!("unigram: {e}"))?;
        let mut spm = Spm::new(model);
        let _ = spm.with_normalizer(Some(NormalizerWrapper::Sequence(Sequence::new(vec![
            Prepend::new(SPACE.into()).into(),
            Replace::new(" ", SPACE.to_string()).map_err(|e| anyhow::anyhow!("{e}"))?.into(),
        ]))));
        spm.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(PreSequence::new(vec![
            Metaspace::new(SPACE, PrependScheme::Never, true).into(),
            Digits::new(true).into(),
        ]))));
        Ok(Self { kinds: pieces.iter().map(|p| p.2).collect(), texts: pieces.into_iter().map(|p| p.0).collect(), spm })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.spm.encode(text, false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec())
    }
}

/// A session's text stream turned back into text, token by token.
#[derive(Default)]
pub struct Detok {
    bytes: Vec<u8>,
    spoke: bool,
}

impl Detok {
    /// The text `id` completes, if any, never empty; control and unknown
    /// pieces (the model's padding among them) produce nothing.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> Option<String> {
        match tok.kinds.get(id as usize)? {
            Kind::Normal => self.bytes.extend(tok.texts[id as usize].replace(SPACE, " ").bytes()),
            Kind::Byte(b) => self.bytes.push(*b),
            Kind::Control | Kind::Unknown => return None,
        }
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(e) => e.valid_up_to() + e.error_len().unwrap_or(0),
        };
        let done: Vec<u8> = self.bytes.drain(..valid).collect();
        let text = String::from_utf8_lossy(&done);
        let text = if self.spoke { &text } else { text.trim_start() };
        self.spoke |= !text.is_empty();
        Some(text.to_string()).filter(|s| !s.is_empty())
    }
}
