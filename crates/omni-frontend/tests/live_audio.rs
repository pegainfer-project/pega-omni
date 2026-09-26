use bytes::Bytes;
use omni_frontend::live_audio::Format;
use omni_frontend::live_audio::Resampler;
use omni_frontend::live_audio::Transcoder;
use proptest::prelude::*;

fn le(samples: &[i16]) -> Bytes {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn samples(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<2>().0.iter().map(|&p| i16::from_le_bytes(p)).collect()
}

fn sine(hz: f64, rate: u32, n: usize, delay: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 - delay as f64) * hz / rate as f64).map(|t| (t * std::f64::consts::TAU).sin()).collect()
}

/// The largest error of `got` (full scale 1.0) against a `hz` sine at `rate`
/// of amplitude `amp`, delayed by `delay` samples, past the first `skip`.
fn worst(got: &[f32], hz: f64, rate: u32, amp: f64, delay: usize, skip: usize) -> f64 {
    let want = sine(hz, rate, got.len(), delay);
    got.iter().zip(want).skip(skip).map(|(&g, w)| (g as f64 - amp * w).abs()).fold(0.0, f64::max)
}

#[test]
fn a_tone_keeps_its_pitch_and_level_across_every_rate_pair() {
    for (from, to) in [(16_000, 24_000), (24_000, 16_000), (8_000, 24_000), (24_000, 8_000)] {
        let mut r = Resampler::new(from, to);
        let input: Vec<f32> = sine(440.0, from, from as usize, 0).iter().map(|&x| (0.5 * x) as f32).collect();
        let out = r.push(&input);
        assert_eq!(out.len(), to as usize, "{from} -> {to}");
        let err = worst(&out, 440.0, to, 0.5, r.delay(), 2 * r.delay());
        assert!(err < 2e-3, "{from} -> {to}: off by {err}");
    }
}

#[test]
fn what_the_wire_rate_cannot_carry_is_filtered_not_folded() {
    let mut r = Resampler::new(24_000, 16_000);
    let input: Vec<f32> = sine(10_000.0, 24_000, 24_000, 0).iter().map(|&x| (0.5 * x) as f32).collect();
    let out = r.push(&input);
    let peak = out.iter().skip(1000).fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak < 0.01, "a 10 kHz tone leaks {peak} into 16 kHz");
}

#[test]
fn pcm_at_the_engine_rate_passes_through_untouched() {
    let mut t = Transcoder::new(Format::Pcm(24_000), 24_000);
    let pcm = le(&[1, -2, 300, i16::MIN, i16::MAX]);
    assert_eq!(t.decode(pcm.clone()).unwrap(), pcm);
    assert_eq!(t.encode(pcm.clone()), pcm);
    assert_eq!(t.decode(Bytes::from_static(&[1, 2, 3])).unwrap_err().code, "invalid_audio");
}

#[test]
fn g711_round_trips_every_code_it_decodes() {
    for format in [Format::Pcmu, Format::Pcma] {
        let mut t = Transcoder::new(format, 8_000);
        let codes: Bytes = (0..=255u8).collect();
        let linear = t.decode(codes).unwrap();
        let coded = t.encode(linear.clone());
        let again = t.decode(coded).unwrap();
        assert_eq!(samples(&again), samples(&linear), "{format:?}");
    }
    let mut ulaw = Transcoder::new(Format::Pcmu, 8_000);
    let mut alaw = Transcoder::new(Format::Pcma, 8_000);
    assert_eq!(
        (ulaw.encode(le(&[0])), alaw.encode(le(&[0]))),
        (Bytes::from_static(&[0xff]), Bytes::from_static(&[0xd5]))
    );
    assert_eq!(samples(&ulaw.decode(Bytes::from_static(&[0x00])).unwrap()), [-32124]);
}

#[test]
fn g711_meets_the_engine_rate() {
    let mut t = Transcoder::new(Format::Pcmu, 24_000);
    assert_eq!(t.decode(Bytes::from(vec![0xff; 800])).unwrap().len(), 2400 * 2);
    assert_eq!(t.encode(le(&[0; 1920])).len(), 640);
}

fn splits(len: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(0..=len, 0..8).prop_map(|mut cuts| {
        cuts.sort_unstable();
        cuts
    })
}

proptest! {
    /// However the network splits the caller's audio, the engine hears the same samples.
    #[test]
    fn decoding_does_not_depend_on_how_the_audio_was_split(
        audio in prop::collection::vec(any::<i16>(), 0..2000),
        cuts in splits(2000),
        format in prop::sample::select(vec![Format::Pcm(16_000), Format::Pcm(24_000), Format::Pcmu, Format::Pcma]),
    ) {
        let wire: Vec<u8> = match format {
            Format::Pcm(_) => le(&audio).to_vec(),
            _ => audio.iter().map(|&s| s as u8).collect(),
        };
        let width = format.bytes_per_sample();
        let bounds: Vec<usize> = std::iter::once(0)
            .chain(cuts.iter().map(|&c| (c * width).min(wire.len())))
            .chain(std::iter::once(wire.len()))
            .collect();
        let mut whole = Transcoder::new(format, 24_000);
        let mut split = Transcoder::new(format, 24_000);
        let at_once = if wire.is_empty() { Bytes::new() } else { whole.decode(Bytes::from(wire.clone())).unwrap() };
        let pieces: Vec<u8> = bounds
            .windows(2)
            .filter(|w| w[1] > w[0])
            .flat_map(|w| split.decode(Bytes::copy_from_slice(&wire[w[0]..w[1]])).unwrap().to_vec())
            .collect();
        prop_assert_eq!(pieces, at_once.to_vec());
    }

    /// Every whole 10 ms of input yields exactly its duration at the other rate.
    #[test]
    fn resampling_conserves_duration(sizes in prop::collection::vec(0..700usize, 1..12)) {
        let mut r = Resampler::new(16_000, 24_000);
        let (mut fed, mut got) = (0, 0);
        for n in sizes {
            fed += n;
            got += r.push(&vec![0.1; n]).len();
            prop_assert_eq!(got, fed / 160 * 240);
        }
    }
}
