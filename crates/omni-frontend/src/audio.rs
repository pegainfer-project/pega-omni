//! Turning an engine's event stream into a response body.
//!
//! Every response is streamed: audio leaves as soon as the engine emits it,
//! whether the client reads it incrementally or waits for the end. A wav
//! stream's length is unknown when its header is written, so the header carries
//! the streaming sentinel (`0xFFFFFFFF`) in both size fields, as other
//! streaming TTS servers do; readers take the data to the end of the body.
//! The header is sent together with the first audio, so the first byte a client
//! sees is also its first packet of sound.
//!
//! An engine abort, or an engine that disappears before `Done`, ends the body
//! with an error so the client sees a truncated transfer instead of a short but
//! well-formed file.

use std::convert::Infallible;
use std::time::Instant;

use axum::body::Body;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::response::sse;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use bytes::BytesMut;
use futures_util::Stream;
use futures_util::StreamExt;
use futures_util::stream;
use omni_engine::Done;
use omni_engine::Event;
use omni_engine::Finish;
use serde_json::json;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::protocol::Delivery;
use crate::protocol::Format;
use crate::protocol::Framing;

pub const WAV_HEADER_LEN: usize = 44;

/// A mono s16le wav header with streaming sizes.
pub fn wav_header(sample_rate: u32) -> [u8; WAV_HEADER_LEN] {
    let mut h = [0u8; WAV_HEADER_LEN];
    let unknown = u32::MAX.to_le_bytes();
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&unknown);
    h[8..16].copy_from_slice(b"WAVEfmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&1u16.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&unknown);
    h
}

#[derive(Debug, thiserror::Error)]
pub enum Broken {
    #[error("the engine aborted the request")]
    Aborted,
    #[error("the engine stopped before finishing the request")]
    Vanished,
}

/// What a finished request reports to the metrics.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub first_packet: Option<Instant>,
    /// PCM samples delivered, headers excluded.
    pub samples: u64,
    pub done: Option<Done>,
}

/// The encoded audio pieces of one request, then its `Done`.
enum Piece {
    Audio(Bytes),
    Done(Done),
}

struct Cursor {
    events: UnboundedReceiver<Event>,
    header: Option<[u8; WAV_HEADER_LEN]>,
    outcome: Outcome,
    report: Option<Box<dyn FnOnce(Outcome) + Send>>,
}

impl Drop for Cursor {
    fn drop(&mut self) {
        if let Some(report) = self.report.take() {
            report(self.outcome);
        }
    }
}

/// Encoded pieces in order; a wav header rides on the first audio piece.
fn pieces(
    events: UnboundedReceiver<Event>,
    delivery: Delivery,
    sample_rate: u32,
    report: Box<dyn FnOnce(Outcome) + Send>,
) -> impl Stream<Item = Result<Piece, Broken>> + Send {
    let header = (delivery.format == Format::Wav).then(|| wav_header(sample_rate));
    let cursor = Cursor {
        events,
        header,
        outcome: Outcome { first_packet: None, samples: 0, done: None },
        report: Some(report),
    };
    stream::unfold(Some(cursor), |cursor| async move {
        let mut c = cursor?;
        match c.events.recv().await {
            Some(Event::Audio(pcm)) => {
                c.outcome.first_packet.get_or_insert_with(Instant::now);
                c.outcome.samples += pcm.len() as u64 / 2;
                let piece = match c.header.take() {
                    Some(h) => {
                        let mut b = BytesMut::with_capacity(WAV_HEADER_LEN + pcm.len());
                        b.extend_from_slice(&h);
                        b.extend_from_slice(&pcm);
                        b.freeze()
                    }
                    None => pcm,
                };
                Some((Ok(Piece::Audio(piece)), Some(c)))
            }
            Some(Event::Done(d)) => {
                c.outcome.done = Some(d);
                match d.finish {
                    Finish::Complete => Some((Ok(Piece::Done(d)), None)),
                    Finish::Aborted => Some((Err(Broken::Aborted), None)),
                }
            }
            None => Some((Err(Broken::Vanished), None)),
        }
    })
}

/// The response for an accepted request; `report` runs once when the stream ends or the client leaves.
pub fn respond(
    events: UnboundedReceiver<Event>,
    delivery: Delivery,
    sample_rate: u32,
    report: Box<dyn FnOnce(Outcome) + Send>,
) -> Response {
    let pieces = pieces(events, delivery, sample_rate, report);
    match delivery.framing {
        Framing::Audio => {
            let bytes = pieces.filter_map(|p| async move {
                match p {
                    Ok(Piece::Audio(b)) => Some(Ok(b)),
                    Ok(Piece::Done(_)) => None,
                    Err(e) => Some(Err(e)),
                }
            });
            let mime = match delivery.format {
                Format::Wav => "audio/wav",
                Format::Pcm => "audio/pcm",
            };
            ([(header::CONTENT_TYPE, mime)], Body::from_stream(bytes)).into_response()
        }
        Framing::Sse => {
            let events = pieces.map(|p| p.map(sse_event)).map(|r| r.or_else(|e| Ok::<_, Infallible>(sse_error(&e))));
            sse::Sse::new(events).into_response()
        }
    }
}

fn sse_event(p: Piece) -> sse::Event {
    let data = match p {
        Piece::Audio(b) => json!({ "type": "speech.audio.delta", "audio": STANDARD.encode(&b) }),
        Piece::Done(d) => json!({
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": d.input_units,
                "output_tokens": d.frames,
                "total_tokens": d.input_units + d.frames,
            },
        }),
    };
    sse::Event::default().data(data.to_string())
}

fn sse_error(e: &Broken) -> sse::Event {
    let data = json!({ "type": "error", "error": { "type": "server_error", "message": e.to_string() } });
    sse::Event::default().data(data.to_string())
}
