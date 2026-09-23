use std::collections::BTreeMap;
use std::time::Duration;

use omni_engine::Draft;
use omni_engine::Event;
use omni_engine::Finish;
use omni_engine::Speech;
use omni_sim::Emission;
use omni_sim::Profile;
use omni_sim::Sim;
use proptest::prelude::*;
use serde_json::json;

fn speech(id: u64, frames: u32) -> Speech {
    let info = Profile::default().info("sim");
    let extra = BTreeMap::from([("frames".to_string(), json!(frames))]);
    let mut s = info.check(Draft { input: "x".into(), voice: "alloy".into(), extra, ..Draft::default() }).unwrap();
    s.id = id;
    s
}

/// The reference: request `i` gets a `first` chunk, then `steady` chunks, then a short tail and one `Done`.
fn expected_chunks(planned: u32, first: u32, steady: u32) -> Vec<u32> {
    let head = planned.min(first);
    let rest = planned - head;
    let mut chunks = vec![head];
    chunks.extend(std::iter::repeat_n(steady, (rest / steady) as usize));
    if !rest.is_multiple_of(steady) {
        chunks.push(rest % steady);
    }
    chunks
}

proptest! {
    #[test]
    fn every_request_gets_its_frames_in_the_planned_chunks(
        plans in prop::collection::vec(1u32..60, 1..40),
        first in 1u32..6,
        steady in 1u32..9,
        max_batch in 1usize..12,
    ) {
        let profile = Profile { first_chunk_frames: first, chunk_frames: steady, max_batch, ..Profile::default() };
        let mut sim = Sim::new(profile.check().unwrap());
        for (i, &p) in plans.iter().enumerate() {
            sim.admit(&speech(i as u64, p));
        }
        let mut chunks: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        let mut done: BTreeMap<u64, u32> = BTreeMap::new();
        let mut steps = 0;
        while !sim.is_idle() {
            let step = sim.step();
            prop_assert!(step.rows <= max_batch);
            for e in step.emissions {
                match e {
                    Emission::Chunk { id, frames } => {
                        prop_assert!(!done.contains_key(&id), "chunk after done");
                        chunks.entry(id).or_default().push(frames);
                    }
                    Emission::Done { id, frames, .. } => {
                        prop_assert!(done.insert(id, frames).is_none(), "two dones");
                    }
                }
            }
            steps += 1;
            prop_assert!(steps <= 60 * plans.len() + 1, "no progress");
        }
        for (i, &p) in plans.iter().enumerate() {
            let id = i as u64;
            prop_assert_eq!((chunks.get(&id).cloned(), done.get(&id).copied()), (Some(expected_chunks(p, first, steady)), Some(p)));
        }
    }

    #[test]
    fn admission_is_first_come_first_served(plans in prop::collection::vec(1u32..20, 2..30), max_batch in 1usize..6) {
        let mut sim = Sim::new(Profile { max_batch, ..Profile::default() });
        for (i, &p) in plans.iter().enumerate() {
            sim.admit(&speech(i as u64, p));
        }
        let mut first_seen = Vec::new();
        while !sim.is_idle() {
            for e in sim.step().emissions {
                if let Emission::Chunk { id, .. } = e && !first_seen.contains(&id) {
                    first_seen.push(id);
                }
            }
        }
        let mut sorted = first_seen.clone();
        sorted.sort();
        prop_assert_eq!(first_seen, sorted);
    }
}

#[test]
fn step_cost_follows_the_profile() {
    let profile = Profile {
        step_base: Duration::from_micros(100),
        step_per_row: Duration::from_micros(10),
        prefill_per_char: Duration::from_micros(1),
        ..Profile::default()
    };
    let mut sim = Sim::new(profile);
    sim.admit(&speech(1, 3));
    sim.admit(&speech(2, 3));
    let admitted = sim.step();
    let steady = sim.step();
    assert_eq!((sim.cost(&admitted), sim.cost(&steady)), (Duration::from_micros(122), Duration::from_micros(120)));
}

#[test]
fn a_dropped_receiver_retires_the_request_and_the_engine_moves_on() {
    let profile = Profile { max_batch: 1, ..Profile::default() };
    let (handle, inbox) = omni_engine::channel(profile.info("sim"), 16);
    let engine = omni_sim::spawn(inbox, profile);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let abandoned = handle.submit(speech(0, 4000)).unwrap();
        drop(abandoned);
        let mut events = handle.submit(speech(0, 9)).unwrap();
        let mut audio = 0;
        let done = loop {
            match events.recv().await.expect("engine alive") {
                Event::Audio(pcm) => audio += pcm.len(),
                Event::Done(d) => break d,
            }
        };
        assert_eq!((done.finish, done.frames, audio), (Finish::Complete, 9, 9 * 1920 * 2));
    });
    drop(handle);
    engine.join().unwrap();
}
