use omni_qwen3_tts::engine::chunk_due;
use omni_qwen3_tts::engine::frame_cap;
use omni_qwen3_tts::engine::pages_for;
use proptest::prelude::*;

/// Chunks a request emits as its frames arrive one per step, then as it finishes.
fn chunks(total: usize, first: usize, steady: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut emitted = 0;
    for generated in 0..=total {
        while let Some(n) = chunk_due(generated, emitted, generated == total, first, steady) {
            out.push(n);
            emitted += n;
        }
    }
    out
}

proptest! {
    #[test]
    fn every_frame_is_emitted_once_in_bounded_chunks(total in 0usize..200, first in 1usize..16, steady in 1usize..32) {
        let c = chunks(total, first, steady);
        prop_assert_eq!(c.iter().sum::<usize>(), total);
        prop_assert!(c.first().is_none_or(|&n| n <= first));
        prop_assert!(c.iter().skip(1).all(|&n| n <= steady));
        prop_assert!(c.iter().all(|&n| n > 0));
    }

    #[test]
    fn only_the_last_chunk_is_short(total in 1usize..200, first in 1usize..16, steady in 1usize..32) {
        let c = chunks(total, first, steady);
        let full = |i: usize, n: usize| n == if i == 0 { first } else { steady };
        prop_assert!(c[..c.len() - 1].iter().enumerate().all(|(i, &n)| full(i, n)));
    }

    #[test]
    fn frame_cap_respects_the_model(chars in 0usize..10_000, max in 1usize..10_000) {
        prop_assert!(frame_cap(chars, max) <= max);
    }

    #[test]
    fn pages_cover_the_tokens(tokens in 0usize..100_000, page in 1usize..64) {
        let p = pages_for(tokens, page);
        prop_assert!(p * page >= tokens && p * page < tokens + page);
    }
}
