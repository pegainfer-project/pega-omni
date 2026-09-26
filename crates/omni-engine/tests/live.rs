use std::collections::BTreeSet;
use std::time::Duration;

use bytes::Bytes;
use omni_engine::live::Clock;
use omni_engine::live::CloseReason;
use omni_engine::live::Closed;
use omni_engine::live::Due;
use omni_engine::live::Jitter;
use omni_engine::live::Line;
use omni_engine::live::LiveInfo;
use omni_engine::live::LiveSubmission;
use omni_engine::live::Output;
use omni_engine::live::SessionDraft;
use omni_engine::live::due;
use omni_engine::live::s16le;
use proptest::prelude::*;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;

fn pcm(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn info() -> LiveInfo {
    LiveInfo {
        model: "live".into(),
        sample_rate: 24_000,
        frame_samples: 1920,
        voices: BTreeSet::from(["a".to_string(), "b".to_string()]),
        default_voice: "a".into(),
        default_instructions: "be kind".into(),
        max_instructions_chars: 8,
        max_sessions: 4,
        max_frames: 100,
    }
}

proptest! {
    #[test]
    fn jitter_conserves_samples_bounds_latency_and_counts_each_dry_spell_once(
        chunks in prop::collection::vec(prop::collection::vec(any::<i16>(), 0..700), 0..40),
        ticks in prop::collection::vec((0usize..3, 0u64..3), 0..40),
        frame in 1usize..400,
        (prebuffer, max_frames) in (1usize..5).prop_flat_map(|m| (1..=m, Just(m))),
    ) {
        let mut j = Jitter::new(frame, prebuffer, max_frames);
        let (mut pushed, mut taken, mut underruns) = (0u64, 0u64, 0u64);
        let mut playing = false;
        for (i, c) in chunks.iter().enumerate() {
            j.push(&pcm(c));
            pushed += c.len() as u64;
            prop_assert!(j.len() <= frame * max_frames);
            let (pops, skip) = ticks.get(i).copied().unwrap_or((1, 0));
            let before = j.len();
            j.skip(skip);
            prop_assert_eq!(before - j.len(), before.min(skip as usize * frame));
            taken += (before - j.len()) as u64;
            for _ in 0..pops {
                let before = j.len();
                prop_assert_eq!(j.pop().len(), frame);
                let plays = (playing || before >= frame * prebuffer) && before >= frame;
                prop_assert_eq!(before - j.len(), if plays { frame } else { 0 });
                underruns += (playing && !plays) as u64;
                playing = plays;
                taken += (before - j.len()) as u64;
            }
            prop_assert_eq!(pushed, taken + j.dropped() + j.len() as u64);
            prop_assert_eq!(j.underruns(), underruns);
        }
    }

    #[test]
    fn s16le_inverts_playout(samples in prop::collection::vec(any::<i16>(), 1..500)) {
        let mut j = Jitter::new(samples.len(), 1, 1);
        j.push(&pcm(&samples));
        prop_assert_eq!(s16le(&j.pop()).to_vec(), pcm(&samples));
    }

    #[test]
    fn jitter_is_indifferent_to_where_bytes_split(samples in prop::collection::vec(any::<i16>(), 0..3000), cuts in prop::collection::vec(0usize..6000, 0..20)) {
        let bytes = pcm(&samples);
        let mut whole = Jitter::new(160, 1, 64);
        whole.push(&bytes);
        let mut cuts: Vec<usize> = cuts.into_iter().map(|c| c.min(bytes.len())).collect();
        cuts.sort();
        let mut split = Jitter::new(160, 1, 64);
        let mut at = 0;
        for c in cuts.into_iter().chain([bytes.len()]) {
            split.push(&bytes[at..c]);
            at = c;
        }
        let drain = |j: &mut Jitter| (0..samples.len() / 160 + 1).map(|_| j.pop()).collect::<Vec<_>>();
        prop_assert_eq!(drain(&mut whole), drain(&mut split));
    }

    #[test]
    fn a_clock_that_resumes_runs_at_once_and_then_on_time(
        idle_us in 0u64..1_000_000_000,
        period_us in 1u64..200_000,
        lag_us in prop::collection::vec(0u64..400_000, 1..20),
    ) {
        let period = Duration::from_micros(period_us);
        let mut clock = Clock::new(period);
        clock.resume(Duration::from_micros(idle_us));
        let at = Duration::from_micros(idle_us);
        let Due::Run { tick, skipped: 0 } = clock.poll(at) else { panic!("a resumed clock waits") };
        let mut ran = BTreeSet::from([tick]);
        let mut at = at;
        for lag in lag_us {
            at += Duration::from_micros(lag);
            if let Due::Run { tick, skipped } = clock.poll(at) {
                prop_assert!(ran.insert(tick), "tick {} ran twice", tick);
                prop_assert!(period * tick as u32 <= at);
                prop_assert!(skipped == 0 || !ran.contains(&(tick - 1)));
            }
        }
    }

    #[test]
    fn the_clock_never_drifts_or_bursts(next in 0u64..10_000, elapsed_us in 0u64..1_000_000_000, period_us in 1u64..200_000) {
        let (elapsed, period) = (Duration::from_micros(elapsed_us), Duration::from_micros(period_us));
        match due(next, elapsed, period) {
            Due::Wait(d) => {
                prop_assert!(!d.is_zero());
                prop_assert_eq!(elapsed + d, period * next as u32);
            }
            Due::Run { tick, skipped } => {
                prop_assert_eq!(tick, next + skipped);
                prop_assert!(period * tick as u32 <= elapsed && elapsed < period * (tick as u32 + 1));
            }
        }
    }
}

#[test]
fn an_underrun_is_silence_and_keeps_the_partial_frame() {
    let mut j = Jitter::new(4, 1, 2);
    j.push(&pcm(&[0, 0, 0, 0, 16384, 16384, 16384]));
    assert_eq!(j.pop(), vec![0.0; 4]);
    assert_eq!((j.pop(), j.underruns(), j.len()), (vec![0.0; 4], 1, 3));
    j.push(&pcm(&[-32768]));
    assert_eq!((j.pop(), j.underruns()), (vec![0.5, 0.5, 0.5, -1.0], 1));
}

#[test]
fn playout_waits_for_the_prebuffer_and_refills_it_after_running_dry() {
    let mut j = Jitter::new(2, 2, 4);
    j.push(&pcm(&[1, 1]));
    assert_eq!((j.pop(), j.len(), j.underruns()), (vec![0.0; 2], 2, 0));
    j.push(&pcm(&[2, 2]));
    assert_eq!((j.pop(), j.len()), (vec![1.0 / 32768.0; 2], 2));
    assert_eq!((j.pop(), j.len()), (vec![2.0 / 32768.0; 2], 0));
    j.push(&pcm(&[3, 3]));
    assert_eq!((j.pop(), j.underruns()), (vec![3.0 / 32768.0; 2], 0));
    assert_eq!((j.pop(), j.underruns()), (vec![0.0; 2], 1));
    j.push(&pcm(&[4, 4]));
    assert_eq!((j.pop(), j.len(), j.underruns()), (vec![0.0; 2], 2, 1));
}

#[test]
fn late_audio_drops_the_oldest() {
    let mut j = Jitter::new(2, 1, 2);
    j.push(&pcm(&[1, 2, 3, 4, 5, 6]));
    assert_eq!((j.dropped(), j.pop()), (2, vec![3.0 / 32768.0, 4.0 / 32768.0]));
}

#[test]
fn sessions_default_and_name_what_is_wrong() {
    let info = info();
    let s = info.check(SessionDraft::default()).unwrap();
    assert_eq!((s.voice.as_str(), s.instructions.as_str()), ("a", "be kind"));
    let bad_voice = info.check(SessionDraft { voice: Some("c".into()), instructions: None }).unwrap_err();
    let long = info.check(SessionDraft { voice: None, instructions: Some("é".repeat(9)) }).unwrap_err();
    assert_eq!((bad_voice.param, long.param), ("audio.output.voice", "instructions"));
    assert!(info.check(SessionDraft { voice: Some("b".into()), instructions: Some("é".repeat(8)) }).is_ok());
    assert_eq!((info.frame_ms(0), info.frame_ms(1), info.frame_ms(25)), (0, 80, 2000));
}

fn submission(info: &LiveInfo) -> (LiveSubmission, UnboundedSender<Bytes>, UnboundedReceiver<Output>) {
    let (audio, input) = unbounded_channel();
    let (sink, output) = unbounded_channel();
    let session = info.check(SessionDraft::default()).unwrap();
    (LiveSubmission { id: 1, session, input, sink }, audio, output)
}

fn drain(out: &mut UnboundedReceiver<Output>) -> Vec<Output> {
    std::iter::from_fn(|| out.try_recv().ok()).collect()
}

#[test]
fn a_line_numbers_its_frames_skips_overrun_audio_and_expires() {
    let (s, audio, mut out) = submission(&info());
    let mut line = Line::start(s, Jitter::new(2, 1, 8), 3).unwrap();
    assert!(matches!(drain(&mut out)[..], [Output::Started]));
    audio.send(Bytes::from(pcm(&[1, 1, 2, 2, 3, 3]))).unwrap();
    assert_eq!(line.listen(), None);
    assert_eq!(line.hear(1), vec![2.0 / 32768.0; 2]);
    assert!(line.say(Bytes::from_static(&[0; 4]), Some(String::new())));
    assert_eq!(line.hear(0), vec![3.0 / 32768.0; 2]);
    assert!(line.say(Bytes::from_static(&[0; 4]), Some("hi".into())));
    assert_eq!(line.hear(0), vec![0.0; 2]);
    assert!(line.say(Bytes::from_static(&[0; 4]), None));
    let frames: Vec<(u64, bool)> = drain(&mut out)
        .into_iter()
        .map(|o| match o {
            Output::Audio { frame, .. } => (frame, false),
            Output::Text { frame, delta } if delta == "hi" => (frame, true),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(frames, [(0, false), (1, false), (1, true), (2, false)]);
    assert_eq!(line.listen(), Some(CloseReason::Expired));
    line.close(CloseReason::Expired);
    let closed = Closed { reason: CloseReason::Expired, frames: 3, underruns: 1, dropped: 0 };
    assert!(matches!(drain(&mut out)[..], [Output::Closed(c)] if c == closed));
}

#[test]
fn a_line_hears_a_hangup_and_a_refusal_never_starts() {
    let (s, audio, mut out) = submission(&info());
    let mut line = Line::start(s, Jitter::new(2, 1, 8), 100).unwrap();
    drop(audio);
    assert_eq!(line.listen(), Some(CloseReason::Hangup));
    let (s, _audio, out2) = submission(&info());
    drop(out2);
    assert!(Line::start(s, Jitter::new(2, 1, 8), 100).is_none());
    let (s, _audio, mut busy) = submission(&info());
    s.refuse(CloseReason::Busy);
    assert!(matches!(drain(&mut busy)[..], [Output::Closed(c)] if c == Closed::refused(CloseReason::Busy)));
    drop(line);
    assert!(matches!(drain(&mut out)[..], [Output::Started]));
}
