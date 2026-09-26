//! The serving loop: one picture at a time, first come first served.
//!
//! A 2048 x 2048 picture is a 4097-row forward per step, which fills the GPU
//! by itself, so requests are not batched. A request's `n` pictures run back
//! to back with seeds `seed`, `seed + 1`, ...; the same seed gives the same
//! picture. A client that goes away stops its picture at the next step.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::image::Event;
use omni_engine::image::Handle;
use omni_engine::image::ImageInfo;
use omni_engine::image::Inbox;
use omni_engine::image::Rgb;
use omni_engine::image::Size;

use crate::gemm::Gemms;
use crate::model::Limits;
use crate::model::Model;
use crate::prompt;
use crate::prompt::Tokenizer;
use crate::sampler;

/// The default of the longest prompt served, in characters.
pub const MAX_PROMPT_CHARS: usize = 2000;

/// What the runtime is sized for: text tokens of the worst case of
/// byte-level BPE (four tokens a character) over `max_prompt_chars` plus the
/// template, and the patches of the largest size.
pub fn limits(max_prompt_chars: usize) -> Limits {
    let max_patches = prompt::sizes().map(|s| prompt::grid(s).0 * prompt::grid(s).1).max().unwrap_or(0);
    Limits { max_text: 4 * max_prompt_chars + 32, max_patches }
}

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
}

impl Engine {
    /// One picture of `prompt` from `seed`, or `None` when `keep_going` said stop.
    pub fn picture(
        &mut self,
        prompt: &str,
        size: Size,
        seed: u64,
        keep_going: &dyn Fn() -> bool,
    ) -> Result<Option<Rgb>> {
        let ids = self.tokenizer.encode(prompt)?;
        self.model.prefill(&ids, prompt::grid(size))?;
        if !sampler::sample(&mut self.model, sampler::Noise::Seed(seed), keep_going)? {
            return Ok(None);
        }
        Ok(Some(Rgb { size, pixels: Bytes::from(self.model.rgb()?) }))
    }
}

fn info(name: &str, max_n: u32, max_prompt_chars: usize) -> ImageInfo {
    ImageInfo {
        model: name.into(),
        sizes: prompt::sizes().collect(),
        default_size: Size { width: 2048, height: 2048 },
        max_n,
        max_prompt_chars,
        extra: BTreeMap::from([("seed".to_string(), Extra::Integer(0..=i64::MAX))]),
    }
}

/// Loads the checkpoint at `dir` onto `device` on the engine's own thread and
/// serves from there until every [`Handle`] is dropped.
pub fn start(
    device: usize,
    dir: PathBuf,
    name: String,
    (max_n, max_prompt_chars): (u32, usize),
    queue: usize,
    gemms: Gemms,
) -> Result<(Handle, JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-hidream-o1".into()).spawn(move || {
        let loaded = (|| {
            let model = Model::load(device, &dir, limits(max_prompt_chars), &gemms)
                .with_context(|| format!("load {}", dir.display()))?;
            let tokenizer = Tokenizer::load(&dir)?;
            let (handle, inbox) = omni_engine::image::channel(info(&name, max_n, max_prompt_chars), queue);
            anyhow::Ok((handle, inbox, Engine { model, tokenizer }))
        })();
        match loaded {
            Ok((handle, inbox, engine)) => {
                let _ = tx.send(Ok(handle));
                run(&inbox, engine);
            }
            Err(e) => {
                let _ = tx.send(Err(e));
            }
        }
    })?;
    let handle = rx.recv().context("the engine thread died while loading")??;
    Ok((handle, thread))
}

fn run(inbox: &Inbox, mut engine: Engine) {
    while let Ok(sub) = inbox.rx.recv() {
        inbox.load.running.store(1, Ordering::Relaxed);
        let g = sub.generation;
        let seed = g.extra.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(rand_seed);
        let keep_going = || !sub.sink.is_closed();
        // `Err(None)`: the client went away; `Err(Some(e))`: the picture failed.
        let outcome = (0..u64::from(g.n)).try_for_each(|i| {
            match engine.picture(&g.prompt, g.size, seed.wrapping_add(i), &keep_going) {
                Ok(Some(rgb)) => sub.sink.send(Event::Image(rgb)).map_err(|_| None),
                Ok(None) => Err(None),
                Err(e) => Err(Some(e)),
            }
        });
        if let Err(Some(e)) = &outcome {
            tracing::error!(id = g.id, "generation failed: {e:#}");
        }
        let finish = if outcome.is_ok() { Finish::Complete } else { Finish::Aborted };
        let _ = sub.sink.send(Event::Done(finish));
        inbox.load.running.store(0, Ordering::Relaxed);
    }
}

/// A seed for a request that did not name one.
fn rand_seed() -> u64 {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    (nanos as u64) & (i64::MAX as u64)
}
