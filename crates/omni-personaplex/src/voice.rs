//! Voices: the reference's precomputed voice prompts (`voices/<name>.pt`).
//!
//! A voice file is what the reference saves after running a speaker's audio
//! through the model: the LM input embedding of every voice-prompt step and
//! the token ring after the last one. It is a `torch.save` zip with two
//! stored tensors, `<name>/data/0` (bf16 `[steps, 1, 1, dim]`) and
//! `<name>/data/1` (i64 `[1, streams, ring]`). Their sizes identify them, so
//! the pickle that describes them is never read.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use crate::config::DIM;
use crate::config::RING;
use crate::config::STREAMS;

pub struct Voice {
    /// `[steps, DIM]` bf16, little-endian.
    pub(crate) embeddings: Vec<u8>,
    /// The token ring after the voice prompt, `[STREAMS][RING]`.
    pub(crate) ring: [[i32; RING]; STREAMS],
}

impl Voice {
    pub fn steps(&self) -> usize {
        self.embeddings.len() / (2 * DIM)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    /// A voice from the bytes of its `.pt` file.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let entries = stored(bytes).context("reading a zip")?;
        let find = |suffix: &str| {
            entries
                .iter()
                .find(|(name, _)| name.ends_with(suffix))
                .map(|e| e.1)
                .with_context(|| format!("no `{suffix}`"))
        };
        let embeddings = find("/data/0")?;
        let ring = find("/data/1")?;
        ensure!(
            !embeddings.is_empty() && embeddings.len().is_multiple_of(2 * DIM),
            "embeddings of {} bytes are not bf16 rows of {DIM}",
            embeddings.len()
        );
        ensure!(ring.len() == 8 * STREAMS * RING, "a token ring of {} bytes", ring.len());
        let ids: Vec<i32> = ring.as_chunks::<8>().0.iter().map(|&b| i64::from_le_bytes(b) as i32).collect();
        Ok(Self {
            embeddings: embeddings.to_vec(),
            ring: std::array::from_fn(|k| std::array::from_fn(|t| ids[k * RING + t])),
        })
    }
}

/// The stored (uncompressed) entries of a zip archive, by name, from its central directory.
fn stored(b: &[u8]) -> Result<Vec<(String, &[u8])>> {
    let u16_at = |at: usize| -> Result<usize> {
        Ok(u16::from_le_bytes(b.get(at..at + 2).context("truncated")?.try_into()?) as usize)
    };
    let u32_at = |at: usize| -> Result<usize> {
        Ok(u32::from_le_bytes(b.get(at..at + 4).context("truncated")?.try_into()?) as usize)
    };
    let end = (0..b.len().saturating_sub(21))
        .rev()
        .find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .context("no end of central directory")?;
    let (count, mut at) = (u16_at(end + 10)?, u32_at(end + 16)?);
    (0..count)
        .map(|_| {
            ensure!(u32_at(at)? == 0x0201_4b50, "bad central directory entry");
            let (method, size) = (u16_at(at + 10)?, u32_at(at + 20)?);
            let (name_len, extra, comment) = (u16_at(at + 28)?, u16_at(at + 30)?, u16_at(at + 32)?);
            let local = u32_at(at + 42)?;
            let name = String::from_utf8_lossy(b.get(at + 46..at + 46 + name_len).context("truncated")?).into_owned();
            at += 46 + name_len + extra + comment;
            if method != 0 {
                bail!("`{name}` is compressed");
            }
            ensure!(u32_at(local)? == 0x0403_4b50, "bad local header of `{name}`");
            let data = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
            Ok((name, b.get(data..data + size).context("truncated entry")?))
        })
        .collect()
}
