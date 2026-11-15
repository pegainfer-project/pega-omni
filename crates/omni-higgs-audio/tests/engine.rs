use omni_higgs_audio::config::SAMPLES_PER_FRAME;
use omni_higgs_audio::engine::HOLDBACK;
use omni_higgs_audio::engine::MAX_FRAMES;
use omni_higgs_audio::engine::OVERLAP;
use omni_higgs_audio::engine::Window;
use omni_higgs_audio::engine::window;
use proptest::prelude::*;

/// A chunk of a request: the frames that existed when it was due, the frames already sent, and its window.
struct Chunk {
    generated: usize,
    emitted: usize,
    window: Window,
}

impl Chunk {
    fn frames(&self) -> usize {
        (self.window.keep.end - self.window.keep.start) / SAMPLES_PER_FRAME
    }
}

/// What a request emits as its frames arrive one per step, then as it finishes.
fn chunks(total: usize, first: usize, steady: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut emitted = 0;
    for generated in 0..=total {
        while let Some(window) = window(generated, emitted, generated == total, first, steady) {
            let chunk = Chunk { generated, emitted, window };
            emitted += chunk.frames();
            out.push(chunk);
        }
    }
    out
}

proptest! {
    #[test]
    fn every_frame_is_emitted_once_in_order(total in 0usize..400, first in 1usize..40, steady in 1usize..60) {
        let mut next = 0;
        for chunk in chunks(total, first, steady) {
            prop_assert_eq!(chunk.emitted, next);
            prop_assert!(chunk.frames() > 0);
            next += chunk.frames();
        }
        prop_assert_eq!(next, total);
    }

    #[test]
    fn a_chunk_is_bounded_and_only_the_last_is_short(total in 1usize..400, first in 1usize..40, steady in 1usize..60) {
        let c = chunks(total, first, steady);
        let target = |i: usize| if i == 0 { first } else { steady };
        prop_assert!(c.iter().enumerate().all(|(i, chunk)| chunk.frames() <= target(i)));
        prop_assert!(c[..c.len() - 1].iter().enumerate().all(|(i, chunk)| chunk.frames() == target(i)));
    }

    #[test]
    fn a_window_holds_its_chunk_between_the_codecs_limits(total in 1usize..400, first in 1usize..40, steady in 1usize..60) {
        for c in chunks(total, first, steady) {
            let (w, end) = (&c.window, c.emitted + c.frames());
            prop_assert!(w.frames.start <= c.emitted && end <= w.frames.end && w.frames.end <= c.generated);
            prop_assert!(w.frames.len() <= MAX_FRAMES);
            prop_assert!(w.frames.start >= c.emitted.saturating_sub(OVERLAP));
            prop_assert!(w.frames.end - end <= HOLDBACK);
            prop_assert_eq!(w.keep.start, (c.emitted - w.frames.start) * SAMPLES_PER_FRAME);
        }
    }

    #[test]
    fn a_running_request_holds_back_the_frames_that_lack_context(total in 1usize..400, first in 1usize..40, steady in 1usize..60) {
        for c in chunks(total, first, steady).iter().filter(|c| c.generated < total) {
            prop_assert!(c.emitted + c.frames() + HOLDBACK <= c.generated);
        }
    }
}
