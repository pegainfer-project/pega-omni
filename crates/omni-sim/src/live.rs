//! A CPU-only live engine with the clock and the session shape of a
//! full-duplex model.
//!
//! [`Echo`] is the decision core, one per session: the agent's frame is the
//! caller's own, [`LiveProfile::echo_frames`] late, and every
//! [`LiveProfile::word_frames`] frames in which the caller was heard it says a
//! word of [`WORDS`], so a person on the demo page can hear and see the loop
//! work end to end. [`LiveProfile::cost`] is what a tick costs.
//!
//! [`spawn_live`] is the shell: [`drive`] on a thread of its own, sleeping out
//! each tick's cost.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::thread::JoinHandle;
use std::time::Duration;

use omni_engine::live::Jitter;
use omni_engine::live::Line;
use omni_engine::live::LiveInbox;
use omni_engine::live::LiveInfo;
use omni_engine::live::LiveSubmission;
use omni_engine::live::Ticker;
use omni_engine::live::drive;
use omni_engine::live::s16le;

use crate::VOICES;

pub const WORDS: [&str; 6] = ["I", "hear", "you", "loud", "and", "clear."];

#[derive(Clone, Debug, PartialEq)]
pub struct LiveProfile {
    pub sample_rate: u32,
    pub frame_samples: u32,
    pub max_sessions: usize,
    pub max_frames: u64,
    /// Frames of caller audio the jitter buffer holds before dropping the oldest.
    pub jitter_frames: usize,
    /// Frames of caller audio buffered before playout starts.
    pub prebuffer_frames: usize,
    /// How late the echo is.
    pub echo_frames: usize,
    /// Frames per spoken word while the caller is heard.
    pub word_frames: u64,
    pub tick_base: Duration,
    pub tick_per_session: Duration,
}

impl Default for LiveProfile {
    /// PersonaPlex's shape: 24 kHz, 80 ms frames, four-minute sessions.
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            frame_samples: 1920,
            max_sessions: 64,
            max_frames: 3000,
            jitter_frames: 4,
            prebuffer_frames: 2,
            echo_frames: 6,
            word_frames: 6,
            tick_base: Duration::ZERO,
            tick_per_session: Duration::ZERO,
        }
    }
}

impl LiveProfile {
    pub fn check(self) -> Result<Self, String> {
        if self.sample_rate == 0 || self.frame_samples == 0 {
            return Err("sample_rate and frame_samples must be positive".into());
        }
        if self.max_sessions == 0 || self.max_frames == 0 || self.jitter_frames == 0 || self.word_frames == 0 {
            return Err("max_sessions, max_frames, jitter_frames and word_frames must be positive".into());
        }
        if !(1..=self.jitter_frames).contains(&self.prebuffer_frames) {
            return Err("prebuffer_frames must be between 1 and jitter_frames".into());
        }
        Ok(self)
    }

    /// Whole frames in `length`.
    pub fn frames_in(&self, length: Duration) -> u64 {
        omni_engine::live::frames_in(length, self.sample_rate, self.frame_samples as usize)
    }

    pub fn info(&self, model: &str) -> LiveInfo {
        LiveInfo {
            model: model.into(),
            sample_rate: self.sample_rate,
            frame_samples: self.frame_samples,
            voices: VOICES.iter().map(|v| v.to_string()).collect(),
            default_voice: VOICES[0].into(),
            default_instructions: "You echo the caller.".into(),
            max_instructions_chars: 4096,
            max_sessions: self.max_sessions,
            max_frames: self.max_frames,
        }
    }

    /// What a tick of `sessions` sessions costs.
    pub fn cost(&self, sessions: usize) -> Duration {
        self.tick_base + self.tick_per_session * sessions as u32
    }
}

/// One session's agent.
#[derive(Clone, Debug)]
pub struct Echo {
    /// The caller's last `echo_frames` frames, oldest first.
    delay: VecDeque<Vec<f32>>,
    word_frames: u64,
    heard: u64,
    words: usize,
}

impl Echo {
    pub fn new(profile: &LiveProfile) -> Self {
        let silence = vec![0.0; profile.frame_samples as usize];
        let delay = std::iter::repeat_n(silence, profile.echo_frames).collect();
        Self { delay, word_frames: profile.word_frames, heard: 0, words: 0 }
    }

    /// The agent's frame answering the caller's frame `heard`, and the text it speaks with it.
    pub fn step(&mut self, heard: Vec<f32>) -> (Vec<f32>, Option<String>) {
        let loud = heard.iter().map(|x| x * x).sum::<f32>() / heard.len() as f32 > 1e-4;
        self.delay.push_back(heard);
        let out = self.delay.pop_front().expect("the frame just queued");
        self.heard += loud as u64;
        let word = (loud && self.heard.is_multiple_of(self.word_frames)).then(|| {
            let w = WORDS[self.words % WORDS.len()];
            self.words += 1;
            if self.words == 1 { w.to_string() } else { format!(" {w}") }
        });
        (out, word)
    }
}

struct Call {
    line: Line,
    echo: Echo,
}

struct Engine {
    profile: LiveProfile,
    calls: BTreeMap<u64, Call>,
}

impl Ticker for Engine {
    type Error = Infallible;

    fn sessions(&self) -> usize {
        self.calls.len()
    }

    fn take(&mut self, s: LiveSubmission) -> Result<(), Infallible> {
        let p = &self.profile;
        let jitter = Jitter::new(p.frame_samples as usize, p.prebuffer_frames, p.jitter_frames);
        let id = s.id;
        if let Some(line) = Line::start(s, jitter, p.max_frames) {
            self.calls.insert(id, Call { line, echo: Echo::new(p) });
        }
        Ok(())
    }

    fn tick(&mut self, skipped: u64) -> Result<(), Infallible> {
        let ended: Vec<_> = self.calls.iter_mut().filter_map(|(&id, c)| Some((id, c.line.listen()?))).collect();
        ended.into_iter().for_each(|(id, reason)| self.calls.remove(&id).expect("a call").line.close(reason));
        std::thread::sleep(self.profile.cost(self.calls.len()));
        let gone: Vec<u64> = self
            .calls
            .iter_mut()
            .filter_map(|(&id, c)| {
                let (out, word) = c.echo.step(c.line.hear(skipped));
                (!c.line.say(s16le(&out), word)).then_some(id)
            })
            .collect();
        gone.iter().for_each(|id| drop(self.calls.remove(id)));
        Ok(())
    }
}

/// Runs the live sim on its own thread until every handle is dropped and the last session ends.
pub fn spawn_live(inbox: LiveInbox, profile: LiveProfile) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("omni-sim-live".into())
        .spawn(move || {
            let Ok(()) = drive(&inbox, &mut Engine { profile, calls: BTreeMap::new() });
        })
        .expect("spawn")
}
