//! The serving loop: one picture at a time, first come first served.
//!
//! A 1024×1024 picture is 4096 target tokens through 32 DiT blocks per step,
//! which fills the GPU. The sidecar (Qwen3-VL + VAE) shares the same device;
//! execution is sequential: encode → DiT → decode.

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

use crate::config::Config;
use crate::gemm::Gemms;
use crate::model::Limits;
use crate::model::Model;
use crate::prompt;
use crate::sampler;
use crate::sidecar::Sidecar;

pub const MAX_PROMPT_CHARS: usize = 2000;

pub fn limits(max_prompt_chars: usize) -> Limits {
    let max_patches = prompt::sizes()
        .map(|s| {
            let g = prompt::grid(s);
            g.0 * g.1
        })
        .max()
        .unwrap_or(0);
    Limits { max_text: 4 * max_prompt_chars + 32, max_patches }
}

pub struct Engine {
    model: Model,
    sidecar: Sidecar,
    sigmas: Vec<f32>,
}

impl Engine {
    pub fn picture(
        &mut self,
        prompt: &str,
        size: Size,
        seed: u64,
        keep_going: &dyn Fn() -> bool,
    ) -> Result<Option<Rgb>> {
        let t0 = std::time::Instant::now();
        let (text_len, embeds) = self.sidecar.encode(prompt)?;
        let t_enc = t0.elapsed();
        let t1 = std::time::Instant::now();
        self.model.prefill(&embeds, text_len, prompt::grid(size))?;
        if !sampler::sample(&mut self.model, &self.sigmas, sampler::Noise::Seed(seed), keep_going)? {
            return Ok(None);
        }
        self.model.synchronize()?;
        let t_dit = t1.elapsed();
        let t2 = std::time::Instant::now();
        let z = self.model.latent()?;
        let rgb = self.sidecar.decode(size.height, size.width, &z)?;
        let t_vae = t2.elapsed();
        if std::env::var("OMNI_TIMING").is_ok() {
            eprintln!("TIMING encode {:?} dit {:?} vae {:?}", t_enc, t_dit, t_vae);
        }
        Ok(Some(Rgb { size, pixels: Bytes::from(rgb) }))
    }
}

fn info(name: &str, max_n: u32, max_prompt_chars: usize) -> ImageInfo {
    ImageInfo {
        model: name.into(),
        sizes: prompt::sizes().collect(),
        default_size: Size { width: 1024, height: 1024 },
        max_n,
        max_prompt_chars,
        extra: BTreeMap::from([("seed".to_string(), Extra::Integer(0..=i64::MAX))]),
    }
}

pub fn start(
    device: usize,
    dir: PathBuf,
    name: String,
    (max_n, max_prompt_chars): (u32, usize),
    queue: usize,
    gemms: Gemms,
    python: String,
) -> Result<(Handle, JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-qwen-image-21".into()).spawn(move || {
        let loaded = (|| {
            let cfg = Config::load(&dir)?;
            let sigmas = cfg.sigmas();
            let model = Model::load(device, &dir, limits(max_prompt_chars), &gemms)
                .with_context(|| format!("load DiT {}", dir.display()))?;
            let socket = std::env::temp_dir().join(format!("qwen21-sidecar-{device}.sock"));
            let sidecar = Sidecar::spawn(&python, &dir, device, &socket).context("spawn sidecar")?;
            let (handle, inbox) = omni_engine::image::channel(info(&name, max_n, max_prompt_chars), queue);
            anyhow::Ok((handle, inbox, Engine { model, sidecar, sigmas }))
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

fn rand_seed() -> u64 {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    (nanos as u64) & (i64::MAX as u64)
}
