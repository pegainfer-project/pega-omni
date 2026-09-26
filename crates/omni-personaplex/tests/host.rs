//! The host side without a checkpoint: prefill batching, the instruction
//! bound, the tokenizer and [`Detok`] on a hand-built SentencePiece model, and
//! a voice file and its prompt from a hand-built zip.

use omni_personaplex::config::DIM;
use omni_personaplex::config::RING;
use omni_personaplex::config::SILENCE_STEPS;
use omni_personaplex::config::STREAMS;
use omni_personaplex::engine::batches;
use omni_personaplex::model::max_instructions_chars;
use omni_personaplex::prompt;
use omni_personaplex::tokenizer::Detok;
use omni_personaplex::tokenizer::Tokenizer;
use omni_personaplex::voice::Voice;
use proptest::prelude::*;

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn bytes_field(field: u64, body: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(body.len() as u64, out);
    out.extend(body);
}

/// A SentencePiece `ModelProto` with `pieces` (text, type); scores favour longer pieces.
fn model_proto(pieces: &[(String, u64)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (text, kind) in pieces {
        let mut piece = Vec::new();
        bytes_field(1, text.as_bytes(), &mut piece);
        varint(2 << 3 | 5, &mut piece);
        piece.extend((text.chars().count() as f32 - 10.0).to_le_bytes());
        varint(3 << 3, &mut piece);
        varint(*kind, &mut piece);
        bytes_field(1, &piece, &mut out);
    }
    out
}

const WORDS: [&str; 4] = ["▁", "▁hello", "▁world", "é"];

fn tokenizer() -> Tokenizer {
    let control = ["<unk>", "<s>", "</s>"].iter().zip([2, 3, 3]).map(|(t, k)| (t.to_string(), k));
    let bytes = (0..=255u8).map(|b| (format!("<0x{b:02X}>"), 6));
    let words = WORDS.iter().map(|w| (w.to_string(), 1));
    Tokenizer::parse(&model_proto(&control.chain(bytes).chain(words).collect::<Vec<_>>())).unwrap()
}

fn word(w: &str) -> u32 {
    3 + 256 + WORDS.iter().position(|x| *x == w).unwrap() as u32
}

/// A zip of `entries`, stored, the way `torch.save` writes one.
fn zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    let u16s = |v: &[u16], o: &mut Vec<u8>| v.iter().for_each(|x| o.extend(x.to_le_bytes()));
    let u32s = |v: &[u32], o: &mut Vec<u8>| v.iter().for_each(|x| o.extend(x.to_le_bytes()));
    for (name, data) in entries {
        let (at, size, len) = (out.len() as u32, data.len() as u32, name.len() as u16);
        u32s(&[0x0403_4b50], &mut out);
        u16s(&[20, 0, 0, 0, 0], &mut out);
        u32s(&[0, size, size], &mut out);
        u16s(&[len, 0], &mut out);
        out.extend(name.as_bytes());
        out.extend(data);
        u32s(&[0x0201_4b50], &mut central);
        u16s(&[20, 20, 0, 0, 0, 0], &mut central);
        u32s(&[0, size, size], &mut central);
        u16s(&[len, 0, 0, 0, 0], &mut central);
        u32s(&[0, at], &mut central);
        central.extend(name.as_bytes());
    }
    let (at, size, n) = (out.len() as u32, central.len() as u32, entries.len() as u16);
    out.extend(central);
    u32s(&[0x0605_4b50], &mut out);
    u16s(&[0, 0, n, n], &mut out);
    u32s(&[size, at], &mut out);
    u16s(&[0], &mut out);
    out
}

fn voice(steps: usize) -> Voice {
    let embeddings = vec![0u8; steps * DIM * 2];
    let ring: Vec<u8> = (0..STREAMS * RING).flat_map(|i| (i as i64 + 1).to_le_bytes()).collect();
    Voice::parse(&zip(&[("v/data.pkl", vec![0x80]), ("v/data/0", embeddings), ("v/data/1", ring)])).unwrap()
}

proptest! {
    #[test]
    fn prefill_batches_cover_every_prefix_in_order_and_fill_each_call(
        lens in prop::collection::vec(1usize..300, 0..40),
        max in 1usize..1000,
    ) {
        let b = batches(&lens, max);
        prop_assert_eq!(b.iter().sum::<usize>(), lens.len());
        let mut at = 0;
        for &n in &b {
            let rows: usize = lens[at..at + n].iter().sum();
            prop_assert!(n >= 1 && (rows <= max || n == 1), "{:?} over {}", &lens[at..at + n], max);
            at += n;
            prop_assert!(at == lens.len() || rows + lens[at] > max, "a call that had room");
        }
    }

    #[test]
    fn instructions_within_the_bound_always_fit(
        text in "\\PC{0,400}",
        rows in 100usize..3000,
        voice_steps in 1usize..60,
    ) {
        let max = max_instructions_chars(rows, voice_steps);
        let text: String = text.chars().take(max).collect();
        let ids = tokenizer().encode(&prompt::wrap(&text)).unwrap();
        prop_assert!(voice_steps + 2 * SILENCE_STEPS + ids.len() <= rows, "{} ids for {:?}", ids.len(), text);
    }

    #[test]
    fn wrapping_is_idempotent(text in "\\PC{0,40}") {
        let once = prompt::wrap(&text);
        prop_assert_eq!(prompt::wrap(&once), once);
    }

    #[test]
    fn a_prompt_is_the_voice_then_the_text_between_silences(
        text in prop::collection::vec(0u32..32_000, 0..60),
        steps in 1usize..8,
    ) {
        let p = prompt::build(&voice(steps), &text).unwrap();
        prop_assert_eq!(p.len(), steps + 2 * SILENCE_STEPS + text.len());
        let said: Vec<u32> = p.rows[1 + SILENCE_STEPS..][..text.len()].iter().map(|r| r[0] as u32).collect();
        prop_assert_eq!(said, text);
    }
}

#[test]
fn detok_drops_the_leading_space_and_holds_split_characters() {
    let tok = tokenizer();
    let mut d = Detok::default();
    let smile = "🙂".bytes().map(|b| 3 + b as u32);
    let ids = [1, word("▁"), word("▁hello"), 2, word("▁world")].into_iter().chain(smile).chain([word("é")]);
    let out: Vec<Option<String>> = ids.map(|i| d.push(&tok, i)).collect();
    let want = [None, None, Some("hello"), None, Some(" world"), None, None, None, Some("🙂"), Some("é")];
    assert_eq!(out, want.map(|w| w.map(String::from)));
    assert_eq!(tok.encode("hello world").unwrap(), [word("▁hello"), word("▁world")]);
}

#[test]
fn a_voice_file_needs_both_tensors_whole() {
    assert_eq!(voice(3).steps(), 3);
    let short = zip(&[("v/data/0", vec![0; DIM * 2]), ("v/data/1", vec![0; 8])]);
    assert!(Voice::parse(&short).err().unwrap().to_string().contains("token ring"));
    assert!(Voice::parse(&zip(&[("v/data/1", vec![0; 8])])).is_err());
    assert!(Voice::parse(b"not a zip").is_err());
}
