//! Decode an inline PCM16 WAV and deliver mono PCM at the engine's reference rate.
//! Check duration at the source rate, then compensate for resampler delay so the
//! normalized clip spans the same interval as the input.

use std::fmt::Display;
use std::io::Read;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use hound::SampleFormat;
use hound::WavReader;
use hound::WavSpec;
use omni_engine::ReferenceAudioInfo;
use omni_engine::ReferencePcm;
use rubato::Fft;
use rubato::FixedSync;
use rubato::Resampler as _;
use rubato::audioadapter_buffers::direct::InterleavedSlice;

use crate::protocol::ApiError;

const WAV_PREFIX: &str = "data:audio/wav;base64,";

pub(crate) fn decode(uri: &str, info: &ReferenceAudioInfo) -> Result<ReferencePcm, ApiError> {
    let encoded = uri.strip_prefix(WAV_PREFIX).ok_or_else(|| invalid(format!("expected the `{WAV_PREFIX}` prefix")))?;
    let wav = STANDARD.decode(encoded).map_err(|err| invalid(format!("invalid base64 ({err})")))?;
    let (source, spec) = wav_pcm(&wav)?;
    let channels = usize::from(spec.channels);
    let frames = source.len() / (channels * 2);
    let sample_rate = info.sample_rate;
    info.check_duration(frames, spec.sample_rate)?;

    let pcm = if channels == 1 && spec.sample_rate == sample_rate {
        Bytes::copy_from_slice(source)
    } else {
        let samples = source.as_chunks::<2>().0.chunks_exact(channels).map(|frame| {
            let sum: f32 = frame.iter().map(|&sample| f32::from(i16::from_le_bytes(sample))).sum();
            sum / channels as f32
        });
        if spec.sample_rate == sample_rate {
            samples.flat_map(quantize).collect()
        } else {
            resample(samples.collect(), spec.sample_rate, sample_rate)?
        }
    };
    Ok(ReferencePcm { pcm, sample_rate })
}

fn invalid(message: impl Display) -> ApiError {
    ApiError::invalid(Some("ref_audio"), format!("invalid `ref_audio` ({message})"))
}

fn wav_pcm(wav: &[u8]) -> Result<(&[u8], WavSpec), ApiError> {
    let mut chunks = wav;
    let declared_len = hound::read_wave_header(&mut chunks).map_err(invalid)?;
    if declared_len != wav.len() as u64 {
        return Err(invalid(format!("RIFF declares {declared_len} bytes, got {}", wav.len())));
    }
    // Bound every chunk against the actual RIFF buffer, including pads and chunks after data.
    // Hound handles the format header. Passing just fmt and the data header avoids its
    // treatment of odd-sized metadata chunks and keeps the original PCM as a borrowed slice.
    let mut format = None;
    let mut data = None;
    while !chunks.is_empty() {
        let (header, rest) = chunks.split_first_chunk::<8>().ok_or_else(|| invalid("truncated WAV chunk header"))?;
        let size = u32::from_le_bytes(*header.last_chunk::<4>().expect("chunk length field")) as usize;
        let (payload, tail) = rest.split_at_checked(size).ok_or_else(|| invalid("truncated WAV chunk payload"))?;
        match &header[..4] {
            b"fmt " if format.is_none() => format = Some(&chunks[..8 + size]),
            b"data" if format.is_some() && data.is_none() => data = Some((&header[..], payload)),
            b"fmt " | b"data" => return Err(invalid("expected one fmt chunk followed by one data chunk")),
            _ => {}
        }
        chunks = tail.get(size % 2..).ok_or_else(|| invalid("missing WAV chunk pad byte"))?;
    }

    let format = format.ok_or_else(|| invalid("missing WAV fmt chunk"))?;
    let (data_header, pcm) = data.ok_or_else(|| invalid("missing WAV data chunk"))?;
    let fmt = format[8..].first_chunk::<16>().ok_or_else(|| invalid("truncated WAV fmt chunk"))?;
    let reader = WavReader::new(wav[..12].chain(format).chain(data_header)).map_err(invalid)?;
    let spec = reader.spec();
    if reader.into_inner().read(&mut [0]).map_err(invalid)? != 0 {
        return Err(invalid("malformed WAV fmt chunk"));
    }
    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 || !matches!(spec.channels, 1 | 2) {
        return Err(invalid(format!(
            "expected mono or stereo PCM16 WAV, got {:?} {}-bit WAV with {} channel(s)",
            spec.sample_format, spec.bits_per_sample, spec.channels
        )));
    }
    let &[.., align_lo, align_hi, bits_lo, bits_hi] = fmt;
    let align = u16::from_le_bytes([align_lo, align_hi]);
    let bits = u16::from_le_bytes([bits_lo, bits_hi]);
    let frame_bytes = spec.channels * 2;
    if align != frame_bytes || bits != 16 {
        return Err(invalid(format!(
            "expected {frame_bytes} bytes per frame with 16-bit samples, got {align} bytes and {bits} bits"
        )));
    }
    if !matches!(spec.sample_rate, 16_000 | 24_000 | 44_100 | 48_000) {
        return Err(invalid(format!("expected WAV at 16000, 24000, 44100 or 48000 Hz, got {} Hz", spec.sample_rate)));
    }
    Ok((pcm, spec))
}

fn quantize(sample: f32) -> [u8; 2] {
    (sample.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16).to_le_bytes()
}

fn resample(mut samples: Vec<f32>, from: u32, to: u32) -> Result<Bytes, ApiError> {
    let mut fft = Fft::new(from as usize, to as usize, from as usize / 100, 1, FixedSync::Both).map_err(invalid)?;
    let take = fft.input_frames_next();
    let give = fft.output_frames_next();
    let delay = fft.output_delay();
    let frames = (samples.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let end = delay + frames;
    // Run whole blocks, padding with enough zeros to flush the filter's tail past `end`.
    let blocks = end.div_ceil(give);
    samples.resize(blocks * take, 0.0);
    let mut output = vec![0.0; blocks * give];
    for (input, output) in samples.chunks_exact(take).zip(output.chunks_exact_mut(give)) {
        let input = InterleavedSlice::new(input, 1, take).expect("a mono chunk");
        let mut output = InterleavedSlice::new_mut(output, 1, give).expect("a mono chunk");
        fft.process_into_buffer(&input, &mut output, None).map_err(invalid)?;
    }
    Ok(output[delay..end].iter().copied().flat_map(quantize).collect())
}
