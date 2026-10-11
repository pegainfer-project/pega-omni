use omni_higgs_audio::config::CODEBOOKS;
use omni_higgs_audio::delay::BOC;
use omni_higgs_audio::delay::Delay;
use omni_higgs_audio::delay::EOC;
use omni_higgs_audio::delay::Row;
use omni_higgs_audio::delay::undelay;
use proptest::prelude::*;

/// Frames spread over delayed rows: codebook `c` starts `c` rows late and ends the run in [`EOC`].
fn delayed(frames: &[Row]) -> Vec<Row> {
    (0..frames.len() + CODEBOOKS - 1)
        .map(|r| {
            std::array::from_fn(|c| match r.checked_sub(c) {
                None => BOC,
                Some(t) => frames.get(t).map_or(EOC, |f| f[c]),
            })
        })
        .collect()
}

/// The step that ends a run whose codebook 0 draws `firsts`, if any.
fn ends_at(firsts: &[i32]) -> Option<usize> {
    let mut delay = Delay::default();
    firsts.iter().position(|&f| delay.advance(f))
}

proptest! {
    #[test]
    fn undelay_inverts_the_pattern(frames in prop::collection::vec(prop::array::uniform8(0..BOC), 0..40)) {
        prop_assert_eq!(undelay(&delayed(&frames)), frames);
    }

    #[test]
    fn a_run_shorter_than_a_frame_has_none(len in 0usize..CODEBOOKS) {
        prop_assert!(undelay(&vec![[0; CODEBOOKS]; len]).is_empty());
    }

    #[test]
    fn the_run_ends_a_countdown_after_the_first_end_past_the_ramp(
        firsts in prop::collection::vec(0..=EOC, 0..80),
    ) {
        let first_end = firsts.iter().skip(CODEBOOKS).position(|&f| f == EOC).map(|i| i + CODEBOOKS);
        let expected = first_end.map(|e| e + CODEBOOKS - 2).filter(|&e| e < firsts.len());
        prop_assert_eq!(ends_at(&firsts), expected);
    }

    #[test]
    fn an_ended_run_stays_ended(firsts in prop::collection::vec(0..=EOC, 40..80), later in 0..=EOC) {
        let mut delay = Delay::default();
        let ended = firsts.iter().any(|&f| delay.advance(f));
        prop_assert!(!ended || delay.advance(later));
    }
}

#[test]
fn the_ramp_unlocks_one_codebook_a_step() {
    let mut delay = Delay::default();
    let forced: Vec<usize> = (0..CODEBOOKS + 2)
        .map(|_| {
            let from = delay.forced_from();
            delay.advance(EOC);
            from
        })
        .collect();
    assert_eq!(forced, [1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);
}

#[test]
fn an_end_inside_the_ramp_is_not_one() {
    let mut delay = Delay::default();
    assert!((0..CODEBOOKS).all(|_| !delay.advance(EOC)));
}
