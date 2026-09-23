//! A CPU-only image engine with the timing shape of a diffusion model.
//!
//! One request at a time, first come first served: a request costs `steps`
//! denoising steps per picture, each `step_cost` long, and its pictures leave
//! together once the last one is done, the way a GPU engine that runs one
//! sequence per forward behaves. A picture is a gradient keyed by the request's
//! `seed` and its index, so a client can tell pictures apart and check that the
//! same seed gives the same bytes.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Duration;

use bytes::Bytes;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::image::Event;
use omni_engine::image::Generation;
use omni_engine::image::ImageInfo;
use omni_engine::image::Inbox;
use omni_engine::image::Rgb;
use omni_engine::image::Size;

/// The image sim's timing and output shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageProfile {
    pub sizes: BTreeSet<Size>,
    pub default_size: Size,
    pub max_n: u32,
    pub steps: u32,
    pub step_cost: Duration,
}

impl Default for ImageProfile {
    fn default() -> Self {
        let square = Size { width: 64, height: 64 };
        Self {
            sizes: [square, Size { width: 96, height: 64 }].into(),
            default_size: square,
            max_n: 4,
            steps: 4,
            step_cost: Duration::ZERO,
        }
    }
}

impl ImageProfile {
    pub fn info(&self, model: &str) -> ImageInfo {
        ImageInfo {
            model: model.into(),
            sizes: self.sizes.clone(),
            default_size: self.default_size,
            max_n: self.max_n,
            max_prompt_chars: 4096,
            extra: BTreeMap::from([("seed".into(), Extra::Integer(0..=i64::from(u32::MAX)))]),
        }
    }
}

/// The picture a request with `seed` gets at position `index`.
pub fn picture(size: Size, seed: u64, index: u32) -> Rgb {
    let key = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(u64::from(index));
    let [r0, g0, b0, ..] = key.to_le_bytes();
    let (w, h) = (size.width as usize, size.height as usize);
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.extend([r0.wrapping_add((x * 255 / w) as u8), g0.wrapping_add((y * 255 / h) as u8), b0]);
        }
    }
    Rgb { size, pixels: Bytes::from(pixels) }
}

pub fn spawn(inbox: Inbox, profile: ImageProfile) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("omni-sim-image".into())
        .spawn(move || run(&inbox, &profile))
        .expect("spawn omni-sim-image")
}

fn run(inbox: &Inbox, profile: &ImageProfile) {
    let load = &inbox.load;
    let mut queue = VecDeque::new();
    loop {
        if queue.is_empty() {
            match inbox.rx.recv() {
                Ok(sub) => queue.push_back(sub),
                Err(_) => return,
            }
        }
        queue.extend(inbox.rx.try_iter());
        let Some(sub) = queue.pop_front() else { continue };
        load.waiting.store(queue.len(), Ordering::Relaxed);
        load.running.store(1, Ordering::Relaxed);
        let Generation { size, n, extra, .. } = sub.generation;
        std::thread::sleep(profile.step_cost * profile.steps * n);
        let seed = extra.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
        let delivered = (0..n).all(|i| sub.sink.send(Event::Image(picture(size, seed, i))).is_ok());
        if delivered {
            let _ = sub.sink.send(Event::Done(Finish::Complete));
        }
        load.running.store(0, Ordering::Relaxed);
    }
}
