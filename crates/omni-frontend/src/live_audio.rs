//! The live socket's audio formats, between the wire and the engine.
//!
//! A GPT-Live session picks its wire format once, in `session.audio.format`:
//! PCM16 at 16 or 24 kHz, or G.711 (μ-law, A-law) at 8 kHz. The engine hears
//! and speaks s16le at [`LiveInfo::sample_rate`](omni_engine::live::LiveInfo),
//! whatever the model is, so a [`Transcoder`] converts each direction: the
//! caller's audio to the engine's rate, the agent's back to the wire's.
//!
//! Rates meet through [`Resampler`], rubato's FFT resampler with both sides
//! fixed at 10 ms of input: output depends only on the samples, never on how
//! the network split them, and each direction adds a constant 5 ms of delay
//! (half an FFT block). A wire format at the engine's rate passes through
//! untouched.

use audio_codec_algorithms::decode_alaw;
use audio_codec_algorithms::decode_ulaw;
use audio_codec_algorithms::encode_alaw;
use audio_codec_algorithms::encode_ulaw;
use bytes::Bytes;
use rubato::FixedSync;
use rubato::Resampler as _;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use serde_json::Value;
use serde_json::json;

use crate::live_protocol::LiveError;

/// A wire format a session may pick; only the protocol parser builds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// PCM16 little-endian at 16000 or 24000 Hz.
    Pcm(u32),
    /// G.711 μ-law at 8000 Hz.
    Pcmu,
    /// G.711 A-law at 8000 Hz.
    Pcma,
}

pub const G711_RATE: u32 = 8000;

impl Format {
    /// What a session that names no format gets.
    pub const DEFAULT: Self = Self::Pcm(24_000);

    pub fn rate(self) -> u32 {
        match self {
            Self::Pcm(rate) => rate,
            Self::Pcmu | Self::Pcma => G711_RATE,
        }
    }

    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Pcm(_) => 2,
            Self::Pcmu | Self::Pcma => 1,
        }
    }

    pub fn to_json(self) -> Value {
        let kind = match self {
            Self::Pcm(_) => "audio/pcm",
            Self::Pcmu => "audio/pcmu",
            Self::Pcma => "audio/pcma",
        };
        json!({"type": kind, "rate": self.rate()})
    }

    fn samples(self, wire: &[u8]) -> Vec<i16> {
        match self {
            Self::Pcm(_) => le_samples(wire),
            Self::Pcmu => wire.iter().map(|&b| decode_ulaw(b)).collect(),
            Self::Pcma => wire.iter().map(|&b| decode_alaw(b)).collect(),
        }
    }

    fn wire(self, samples: &[i16]) -> Bytes {
        match self {
            Self::Pcm(_) => le_bytes(samples),
            Self::Pcmu => samples.iter().map(|&s| encode_ulaw(s)).collect(),
            Self::Pcma => samples.iter().map(|&s| encode_alaw(s)).collect(),
        }
    }
}

fn le_samples(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<2>().0.iter().map(|&p| i16::from_le_bytes(p)).collect()
}

fn le_bytes(samples: &[i16]) -> Bytes {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// A mono stream from one rate to another, fed in chunks of any size.
pub struct Resampler {
    fft: rubato::Fft<f32>,
    pending: Vec<f32>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        let chunk = (from as usize / 100).max(1);
        let fft = rubato::Fft::new(from as usize, to as usize, chunk, 1, FixedSync::Both)
            .expect("positive rates and chunk make a resampler");
        Self { fft, pending: Vec::new() }
    }

    /// Output samples of the delay every stream starts with.
    pub fn delay(&self) -> usize {
        self.fft.output_delay()
    }

    /// The output of every whole chunk `samples` completes; the rest waits for the next push.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(samples);
        let (take, give) = (self.fft.input_frames_next(), self.fft.output_frames_next());
        let chunks = self.pending.len() / take;
        let mut out = vec![0.0; chunks * give];
        for (input, output) in self.pending.chunks_exact(take).zip(out.chunks_exact_mut(give)) {
            let input = InterleavedSlice::new(input, 1, take).expect("a mono chunk");
            let mut output = InterleavedSlice::new_mut(output, 1, give).expect("a mono chunk");
            self.fft.process_into_buffer(&input, &mut output, None).expect("buffers of the resampler's sizes");
        }
        self.pending.drain(..chunks * take);
        out
    }
}

/// `samples` through `r`, if the rates differ.
fn convert(r: &mut Option<Resampler>, samples: Vec<i16>) -> Vec<i16> {
    match r {
        None => samples,
        Some(r) => r
            .push(&samples.iter().map(|&s| s as f32 / 32768.0).collect::<Vec<_>>())
            .iter()
            .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect(),
    }
}

/// One session's audio both ways: wire bytes to engine s16le and back.
pub struct Transcoder {
    format: Format,
    inbound: Option<Resampler>,
    outbound: Option<Resampler>,
}

impl Transcoder {
    pub fn new(format: Format, engine_rate: u32) -> Self {
        let resampler = |from, to| (from != to).then(|| Resampler::new(from, to));
        Self { format, inbound: resampler(format.rate(), engine_rate), outbound: resampler(engine_rate, format.rate()) }
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// The caller's audio as engine s16le, or why `wire` is not audio of this format.
    pub fn decode(&mut self, wire: Bytes) -> Result<Bytes, LiveError> {
        if !wire.len().is_multiple_of(self.format.bytes_per_sample()) {
            let message = format!("PCM16 audio must contain an even number of bytes, got {}", wire.len());
            return Err(LiveError::invalid("invalid_audio", Some("audio"), message));
        }
        Ok(match (&self.inbound, self.format) {
            (None, Format::Pcm(_)) => wire,
            _ => le_bytes(&convert(&mut self.inbound, self.format.samples(&wire))),
        })
    }

    /// The agent's engine s16le in the wire format.
    pub fn encode(&mut self, pcm: Bytes) -> Bytes {
        match (&self.outbound, self.format) {
            (None, Format::Pcm(_)) => pcm,
            _ => self.format.wire(&convert(&mut self.outbound, le_samples(&pcm))),
        }
    }
}
