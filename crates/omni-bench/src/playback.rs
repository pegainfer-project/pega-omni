//! A real-time player fed by arriving audio, to measure what a listener hears.
//!
//! Playback starts when the first audio arrives and advances with the wall
//! clock. When a new piece arrives after the player has run out of buffered
//! audio, the gap between running dry and that arrival is a stall; playback
//! resumes with the new piece. Only audio arrival times matter, so the model is
//! exact for any chunking.

use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Playback {
    first: Option<Instant>,
    /// Wall time at which the buffered audio runs out.
    dry_at: Option<Instant>,
    audio: f64,
    stall: Duration,
}

impl Playback {
    /// Records `seconds` of audio arriving at `at`; arrivals must be in time order.
    pub fn arrive(&mut self, at: Instant, seconds: f64) {
        self.first.get_or_insert(at);
        let start = match self.dry_at {
            Some(dry) if at > dry => {
                self.stall += at - dry;
                at
            }
            Some(dry) => dry,
            None => at,
        };
        self.dry_at = Some(start + Duration::from_secs_f64(seconds));
        self.audio += seconds;
    }

    pub fn first(&self) -> Option<Instant> {
        self.first
    }

    pub fn audio_seconds(&self) -> f64 {
        self.audio
    }

    pub fn stall(&self) -> Duration {
        self.stall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stalls_only_when_the_buffer_runs_dry() {
        let t0 = Instant::now();
        let ms = |n| t0 + Duration::from_millis(n);
        let mut p = Playback::default();
        p.arrive(ms(0), 0.080);
        p.arrive(ms(50), 0.320);
        p.arrive(ms(300), 0.320);
        assert_eq!((p.stall(), p.audio_seconds()), (Duration::ZERO, 0.72));
        p.arrive(ms(1000), 0.080);
        assert_eq!(p.stall(), Duration::from_millis(1000 - 720));
    }
}
