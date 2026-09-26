//! A session's prompt: what Helium reads before the conversation, and the
//! token state the conversation starts from.
//!
//! The reference feeds the model through a ring of token slots (`LMGen`'s
//! cache and its `provided` flags): each step writes the tokens it is given
//! at `offset + delay`, reads its input from `offset - 1`, and draws whatever
//! at `offset` nobody provided. The prompt is the voice prompt's embeddings,
//! 0.5 s of silence, the role prompt one text token per step, and 0.5 s of
//! silence; every token of it is provided, so it is a pure function of the
//! voice and the text, and [`build`] computes it by running that ring on the
//! host. What the ring holds at the end is where the conversation starts.
//!
//! During the conversation the device keeps the same state in closed form
//! (`kernels/lm.cu`): the next input row and the caller's pending acoustic
//! codes.

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use crate::config::CARD;
use crate::config::CODEBOOKS;
use crate::config::DELAYS;
use crate::config::DRAWN;
use crate::config::RING;
use crate::config::SILENCE;
use crate::config::SILENCE_STEPS;
use crate::config::SINE;
use crate::config::STREAMS;
use crate::config::TEXT_PAD;
use crate::config::TEXT_VOCAB;
use crate::voice::Voice;

/// The reference's first input: text and audio "initial" tokens.
const INITIAL: [i32; STREAMS] = {
    let mut row = [CARD as i32; STREAMS];
    row[0] = TEXT_VOCAB as i32;
    row
};
/// A ring slot nobody provided, which the model would draw: never valid in a prompt.
const UNPROVIDED: i32 = -2;

/// The `<system>`-wrapped role prompt, as the reference's server builds it.
pub fn wrap(instructions: &str) -> String {
    let t = instructions.trim();
    if t.starts_with("<system>") && t.ends_with("<system>") { t.to_string() } else { format!("<system> {t} <system>") }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    /// Rows `0..voice_steps` are the voice's embeddings.
    pub voice_steps: usize,
    /// The token rows after them.
    pub rows: Vec<[i32; STREAMS]>,
    /// The first conversation step's input.
    pub first: [i32; STREAMS],
    /// The caller's acoustic codes the first step passes on (they arrive a step late).
    pub pending: [i32; CODEBOOKS - 1],
    /// What the first step takes instead of drawing, text then the agent's
    /// codebooks; `-1` where it draws.
    pub first_force: [i32; DRAWN],
}

impl Prompt {
    /// Helium positions the prompt fills; the conversation's step `j` is at `len() + j`.
    pub fn len(&self) -> usize {
        self.voice_steps + self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct Ring {
    cache: [[i32; RING]; STREAMS],
    provided: [[bool; RING]; STREAMS],
    offset: usize,
}

impl Ring {
    /// One `LMGen.step` with `given` tokens: the input row it reads, or none
    /// on the very first call, which only primes the ring.
    fn step(&mut self, given: &[i32; STREAMS]) -> Option<[i32; STREAMS]> {
        for k in 0..STREAMS {
            let at = (self.offset + DELAYS[k]) % RING;
            self.cache[k][at] = given[k];
            self.provided[k][at] = true;
            if self.offset <= DELAYS[k] {
                self.cache[k][self.offset % RING] = INITIAL[k];
                self.provided[k][self.offset % RING] = true;
            }
        }
        if self.offset == 0 {
            (0..STREAMS).for_each(|k| self.cache[k][0] = INITIAL[k]);
            self.offset = 1;
            return None;
        }
        let (input, target) = ((self.offset - 1) % RING, self.offset % RING);
        let row = std::array::from_fn(|k| self.cache[k][input]);
        for k in 0..STREAMS {
            self.provided[k][input] = false;
            if !self.provided[k][target] {
                self.cache[k][target] = UNPROVIDED;
            }
        }
        self.offset += 1;
        Some(row)
    }
}

fn given(text: i32, agent: [i32; CODEBOOKS], caller: [i32; CODEBOOKS]) -> [i32; STREAMS] {
    std::array::from_fn(|k| match k {
        0 => text,
        k if k <= CODEBOOKS => agent[k - 1],
        k => caller[k - 1 - CODEBOOKS],
    })
}

/// The prompt of `voice` with the role prompt `text` (token ids).
pub fn build(voice: &Voice, text: &[u32]) -> Result<Prompt> {
    let mut ring = Ring { cache: [[0; RING]; STREAMS], provided: [[false; RING]; STREAMS], offset: 0 };
    let dummy = given(TEXT_PAD, [CARD as i32; CODEBOOKS], [CARD as i32; CODEBOOKS]);
    for _ in 0..voice.steps() {
        if ring.step(&dummy).is_none() {
            ring.step(&dummy);
        }
    }
    ring.cache = voice.ring;
    let texts = std::iter::repeat_n(TEXT_PAD, SILENCE_STEPS)
        .chain(text.iter().map(|&t| t as i32))
        .chain(std::iter::repeat_n(TEXT_PAD, SILENCE_STEPS));
    let rows: Vec<[i32; STREAMS]> =
        texts.map(|t| ring.step(&given(t, SILENCE, SINE)).expect("the ring is primed")).collect();
    ensure!(rows.iter().flatten().all(|&t| t != UNPROVIDED), "the prompt reads a token nobody provided");
    let (last, next) = ((ring.offset - 1) % RING, ring.offset % RING);
    let first: [i32; STREAMS] = std::array::from_fn(|k| ring.cache[k][last]);
    let caller = |k: usize| (ring.provided[k][next], ring.cache[k][next]);
    if caller(1 + CODEBOOKS).0 {
        bail!("the prompt provides the caller's first semantic code");
    }
    let pending = std::array::from_fn(|i| caller(2 + CODEBOOKS + i));
    ensure!(pending.iter().all(|p| p.0), "the prompt leaves the caller's acoustic codes undrawn");
    Ok(Prompt {
        voice_steps: voice.steps(),
        rows,
        first,
        pending: pending.map(|p| p.1),
        first_force: std::array::from_fn(|k| if ring.provided[k][next] { ring.cache[k][next] } else { -1 }),
    })
}

/// A conversation step's token bookkeeping, the host spec of `lm_advance`: step `j`
/// reads `row`, draws `drawn` (text, then the agent's codebooks), and hears
/// the caller's frame `j` as `caller`. It emits the agent frame whose
/// semantic code was drawn a step earlier (the acoustic codes lag by one),
/// and the next step reads the new row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Advance {
    pub row: [i32; STREAMS],
    pub pending: [i32; CODEBOOKS - 1],
    /// The text token spoken with `frame`, and the agent's frame.
    pub text: i32,
    pub frame: [i32; CODEBOOKS],
}

pub fn advance(
    row: &[i32; STREAMS],
    pending: &[i32; CODEBOOKS - 1],
    drawn: &[i32; DRAWN],
    caller: &[i32; CODEBOOKS],
) -> Advance {
    Advance {
        row: std::array::from_fn(|k| match k {
            k if k <= CODEBOOKS => drawn[k],
            k if k == 1 + CODEBOOKS => caller[0],
            k => pending[k - 2 - CODEBOOKS],
        }),
        pending: std::array::from_fn(|i| caller[i + 1]),
        text: row[0],
        frame: std::array::from_fn(|i| if i == 0 { row[1] } else { drawn[1 + i] }),
    }
}
