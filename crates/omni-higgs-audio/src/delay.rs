//! The delay pattern: codebook `c` of a frame is drawn `c` steps after codebook 0, so a step
//! draws one code per codebook but every frame is spread over eight steps.
//!
//! [`Delay`] is the sampler's state machine at temperature-independent level: which codebooks
//! of the next step are not drawn (they read [`BOC`]), and when generation ends. The first
//! `CODEBOOKS` steps are the ramp; after it, codebook 0 drawing [`EOC`] starts a countdown of
//! `CODEBOOKS - 2` more steps. [`undelay`] puts the drawn rows back into frames.

use crate::config::CODEBOOKS;

/// Codec begin and end, the last two codes of every codebook.
pub const BOC: i32 = 1024;
pub const EOC: i32 = 1025;

/// One row of the delayed codes: a code per codebook.
pub type Row = [i32; CODEBOOKS];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Delay {
    steps: usize,
    countdown: Option<usize>,
    done: bool,
}

impl Delay {
    /// The first codebook the next step does not draw: it and every later one read [`BOC`].
    pub fn forced_from(&self) -> usize {
        (self.steps + 1).min(CODEBOOKS)
    }

    /// Takes the next step's codebook 0 draw; returns whether generation has ended. The step
    /// that ends it still yields its row.
    pub fn advance(&mut self, first: i32) -> bool {
        if self.done {
            return true;
        }
        if self.steps < CODEBOOKS {
            self.steps += 1;
        } else if let Some(left) = self.countdown {
            self.countdown = Some(left - 1);
            self.done = left <= 1;
        } else if first == EOC {
            self.countdown = Some(CODEBOOKS - 2);
        }
        self.done
    }
}

/// The frames inside delayed rows: frame `t` takes codebook `c` from row `t + c`. A run shorter
/// than a frame has none.
pub fn undelay(rows: &[Row]) -> Vec<Row> {
    let frames = rows.len().saturating_sub(CODEBOOKS - 1);
    (0..frames).map(|t| std::array::from_fn(|c| rows[t + c][c])).collect()
}
