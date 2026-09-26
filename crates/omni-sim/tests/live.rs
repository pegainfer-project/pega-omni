use std::time::Duration;

use bytes::Bytes;
use omni_engine::live::CloseReason;
use omni_engine::live::Output;
use omni_engine::live::SessionDraft;
use omni_sim::live::Echo;
use omni_sim::live::LiveProfile;
use omni_sim::live::WORDS;
use proptest::prelude::*;

proptest! {
    #[test]
    fn the_echo_is_the_caller_late_and_says_a_word_per_word_frames_heard(
        loud in prop::collection::vec(any::<bool>(), 1..200),
        echo_frames in 0usize..8,
        word_frames in 1u64..8,
    ) {
        let profile = LiveProfile { frame_samples: 4, echo_frames, word_frames, ..LiveProfile::default() };
        let mut echo = Echo::new(&profile);
        let frame = |k: usize| if loud[k] { vec![0.1 + k as f32 / 1000.0; 4] } else { vec![0.0; 4] };
        let mut said = String::new();
        for k in 0..loud.len() {
            let (out, word) = echo.step(frame(k));
            prop_assert_eq!(out, if k >= echo_frames { frame(k - echo_frames) } else { vec![0.0; 4] });
            let heard = loud[..=k].iter().filter(|&&l| l).count() as u64;
            prop_assert_eq!(word.is_some(), loud[k] && heard.is_multiple_of(word_frames));
            said.extend(word);
        }
        let words = loud.iter().filter(|&&l| l).count() / word_frames as usize;
        let want: Vec<&str> = WORDS.iter().copied().cycle().take(words).collect();
        prop_assert_eq!(said, want.join(" "));
    }
}

#[test]
fn tick_cost_and_session_length_follow_the_profile() {
    let profile = LiveProfile {
        tick_base: Duration::from_micros(500),
        tick_per_session: Duration::from_micros(100),
        ..LiveProfile::default()
    };
    assert_eq!(profile.cost(3), Duration::from_micros(800));
    assert_eq!(profile.frames_in(Duration::from_secs(240)), 3000);
    assert_eq!(profile.frames_in(Duration::from_millis(500)), 6);
}

#[test]
fn a_session_starts_echoes_refuses_the_next_and_acknowledges_a_hangup() {
    let profile = LiveProfile { echo_frames: 0, max_sessions: 1, ..LiveProfile::default() };
    let info = profile.info("sim");
    let (handle, inbox) = omni_engine::live::live_channel(info.clone(), 4);
    let engine = omni_sim::live::spawn_live(inbox, profile);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let session = info.check(SessionDraft::default()).unwrap();
        let omni_engine::live::Opened { audio, output: mut out, .. } = handle.start(session.clone()).unwrap();
        assert!(matches!(out.recv().await, Some(Output::Started)));
        let mut busy = handle.start(session).unwrap().output;
        assert!(matches!(busy.recv().await, Some(Output::Closed(c)) if c.reason == CloseReason::Busy));

        let tone: Bytes = (0..1920 * 4).flat_map(|i| (((i % 48) as i16 - 24) * 1000).to_le_bytes()).collect();
        audio.send(tone).unwrap();
        let mut frames = Vec::new();
        let mut loud = false;
        while !loud {
            if let Output::Audio { frame, pcm } = out.recv().await.unwrap() {
                frames.push(frame);
                loud = pcm.iter().any(|&b| b != 0);
            }
        }
        assert_eq!(frames, (0..frames.len() as u64).collect::<Vec<_>>());

        drop(audio);
        let closed = loop {
            if let Output::Closed(c) = out.recv().await.unwrap() {
                break c;
            }
        };
        assert_eq!(closed.reason, CloseReason::Hangup);
        assert!(closed.frames >= frames.len() as u64, "{closed:?}");
    });
    drop(handle);
    engine.join().unwrap();
}
