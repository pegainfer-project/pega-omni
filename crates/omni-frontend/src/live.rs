//! The live socket: one GPT-Live session between a WebSocket and a
//! [`LiveHandle`].
//!
//! A socket holds at most one session, from `session.start` until someone
//! ends it: the client (`session.close`, answered by `session.closed` once the
//! engine has acknowledged the hangup with its final counts, or after
//! [`CLOSE_GRACE`]), the engine (expiry, abort), or the network (the socket
//! drops; nobody is left to tell). A session the server has no room for is an
//! error and leaves the socket open for another `session.start`, whether the
//! queue or the engine refused it. Muting forwards silence in place of the
//! caller's audio, so the engine's timeline keeps running without underruns.
//!
//! Protocol decisions are [`crate::live_protocol`]'s; this is the shell.

use std::time::Duration;
use std::time::Instant;

use axum::extract::ws::Message;
use axum::extract::ws::WebSocket;
use bytes::Bytes;
use omni_engine::Rejected;
use omni_engine::live::CloseReason;
use omni_engine::live::LiveHandle;
use omni_engine::live::Output;
use omni_engine::live::Session;
use omni_engine::live::SessionDraft;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;

use crate::live_protocol::ClientEvent;
use crate::live_protocol::ClosedReason;
use crate::live_protocol::LiveError;
use crate::live_protocol::audio_event;
use crate::live_protocol::closed_event;
use crate::live_protocol::error_event;
use crate::live_protocol::mute_event;
use crate::live_protocol::parse;
use crate::live_protocol::stamp;
use crate::live_protocol::started_event;
use crate::live_protocol::transcript_event;
use crate::live_protocol::usage_event;

const USAGE_EVERY: Duration = Duration::from_secs(5);
/// How long `session.close` waits for the engine's final counts.
const CLOSE_GRACE: Duration = Duration::from_secs(2);

struct Line {
    id: u64,
    session: Session,
    /// `None` once the client asked to close.
    audio: Option<UnboundedSender<Bytes>>,
    output: UnboundedReceiver<Output>,
    started: Option<Instant>,
    muted: bool,
    closing: Option<Instant>,
}

impl Line {
    fn seconds(&self) -> f64 {
        self.started.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }
}

struct Socket {
    ws: WebSocket,
    sent: u64,
}

impl Socket {
    async fn send(&mut self, event: Value) -> bool {
        self.sent += 1;
        self.ws.send(Message::Text(stamp(event, self.sent).to_string().into())).await.is_ok()
    }

    async fn error(&mut self, e: &LiveError) -> bool {
        self.send(error_event(e)).await
    }
}

/// Counts a session in `omni_live_sessions` while it lives.
struct Gauge;

impl Gauge {
    fn new() -> Self {
        metrics::gauge!("omni_live_sessions").increment(1.0);
        Self
    }
}

impl Drop for Gauge {
    fn drop(&mut self) {
        metrics::gauge!("omni_live_sessions").decrement(1.0);
    }
}

fn record(outcome: &'static str, seconds: f64) {
    metrics::counter!("omni_live_sessions_total", "outcome" => outcome).increment(1);
    metrics::counter!("omni_live_session_seconds_total").increment(seconds.round() as u64);
}

fn busy(live: &LiveHandle) -> LiveError {
    let n = live.info.max_sessions;
    LiveError::server("server_busy", format!("all {n} live sessions are in use; retry later"))
}

/// Serves one socket to the end.
pub async fn serve(ws: WebSocket, live: LiveHandle) {
    let mut socket = Socket { ws, sent: 0 };
    let mut line: Option<Line> = None;
    let mut usage = tokio::time::interval(USAGE_EVERY);
    usage.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let _gauge = Gauge::new();
    loop {
        let grace = line.as_ref().and_then(|l| l.closing).map(|t| t + CLOSE_GRACE);
        tokio::select! {
            frame = socket.ws.recv() => {
                let text = match frame {
                    Some(Ok(Message::Text(t))) => t,
                    Some(Ok(Message::Binary(_))) => {
                        let e = LiveError::invalid("invalid_event", None, "events are JSON text frames".into());
                        if !socket.error(&e).await { break; }
                        continue;
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                    Some(Ok(Message::Close(_)) | Err(_)) | None => {
                        if let Some(l) = &line { record("remote_hangup", l.seconds()); }
                        return;
                    }
                };
                let received = match parse(&text, live.info.sample_rate) {
                    Ok(r) => r,
                    Err(e) => {
                        if !socket.error(&e).await { break; }
                        continue;
                    }
                };
                let id = received.client_event_id;
                let reply = match (received.event, line.as_mut()) {
                    (ClientEvent::Start { model, draft }, None) => match open(&live, model, draft) {
                        Ok(l) => {
                            line = Some(l);
                            None
                        }
                        Err(e) => Some(error_event(&e.answering(id))),
                    },
                    (ClientEvent::Start { .. }, Some(_)) => Some(error_event(
                        &LiveError::invalid("session_already_started", Some("type"), "this socket already has a session".into())
                            .answering(id),
                    )),
                    (_, None) => Some(error_event(
                        &LiveError::invalid("session_not_started", Some("type"), "send `session.start` first".into())
                            .answering(id),
                    )),
                    (ClientEvent::Append(pcm), Some(l)) => {
                        if let Some(a) = &l.audio {
                            let _ = a.send(if l.muted { Bytes::from(vec![0; pcm.len()]) } else { pcm });
                        }
                        None
                    }
                    (ClientEvent::Mute, Some(l)) => {
                        l.muted = true;
                        Some(mute_event(true, id))
                    }
                    (ClientEvent::Unmute, Some(l)) => {
                        l.muted = false;
                        Some(mute_event(false, id))
                    }
                    (ClientEvent::Close, Some(l)) => {
                        l.audio = None;
                        l.closing.get_or_insert_with(Instant::now);
                        None
                    }
                };
                if let Some(event) = reply && !socket.send(event).await {
                    break;
                }
            }
            out = next(&mut line) => {
                let l = line.as_mut().expect("a session produced output");
                let (event, end) = match out {
                    Some(Output::Started) => {
                        l.started = Some(Instant::now());
                        (started_event(&live.info, l.id, &l.session), None)
                    }
                    Some(Output::Audio { frame, pcm }) => {
                        (audio_event(&pcm, live.info.frame_ms(frame), live.info.frame_ms(frame + 1)), None)
                    }
                    Some(Output::Text { frame, delta }) => {
                        (transcript_event(&delta, live.info.frame_ms(frame), live.info.frame_ms(frame + 1)), None)
                    }
                    Some(Output::Closed(c)) => {
                        metrics::counter!("omni_live_underruns_total").increment(c.underruns);
                        metrics::counter!("omni_live_dropped_samples_total").increment(c.dropped);
                        match c.reason {
                            CloseReason::Busy => {
                                record("busy", 0.0);
                                line = None;
                                if !socket.error(&busy(&live)).await { break; }
                                continue;
                            }
                            CloseReason::Expired => (closed_event(ClosedReason::Expired, l.seconds()), Some("expired")),
                            CloseReason::Hangup => {
                                (closed_event(ClosedReason::CloseRequested, l.seconds()), Some("close_requested"))
                            }
                            CloseReason::Aborted => {
                                let e = LiveError::server("server_error", "the engine ended the session".into());
                                let _ = socket.error(&e).await;
                                (closed_event(ClosedReason::ConnectionLost, l.seconds()), Some("aborted"))
                            }
                        }
                    }
                    None => {
                        let e = LiveError::server("server_error", "the engine stopped".into());
                        let _ = socket.error(&e).await;
                        (closed_event(ClosedReason::ConnectionLost, l.seconds()), Some("aborted"))
                    }
                };
                let delivered = socket.send(event).await;
                if let Some(outcome) = end {
                    record(outcome, l.seconds());
                    break;
                }
                if !delivered {
                    record("remote_hangup", l.seconds());
                    return;
                }
            }
            _ = usage.tick(), if line.as_ref().is_some_and(|l| l.started.is_some() && l.closing.is_none()) => {
                let seconds = line.as_ref().map_or(0.0, Line::seconds);
                if !socket.send(usage_event(seconds)).await { break; }
            }
            _ = sleep_until(grace), if grace.is_some() => {
                let seconds = line.as_ref().map_or(0.0, Line::seconds);
                let _ = socket.send(closed_event(ClosedReason::CloseRequested, seconds)).await;
                record("close_requested", seconds);
                break;
            }
        }
    }
    let _ = socket.ws.send(Message::Close(None)).await;
}

async fn next(line: &mut Option<Line>) -> Option<Output> {
    match line {
        Some(l) => l.output.recv().await,
        None => std::future::pending().await,
    }
}

async fn sleep_until(at: Option<Instant>) {
    match at {
        Some(t) => tokio::time::sleep_until(t.into()).await,
        None => std::future::pending().await,
    }
}

fn open(live: &LiveHandle, model: Option<String>, draft: SessionDraft) -> Result<Line, LiveError> {
    if let Some(m) = model.filter(|m| *m != live.info.model) {
        let message = format!("model `{m}` is not served here; this server serves `{}`", live.info.model);
        return Err(LiveError::invalid("model_not_found", Some("session.model"), message));
    }
    let session = live.info.check(draft).map_err(LiveError::session)?;
    let opened = live.start(session).map_err(|e| match e {
        Rejected::Full => busy(live),
        Rejected::Stopped => LiveError::server("server_error", e.to_string()),
    })?;
    Ok(Line {
        id: opened.id,
        session: opened.session,
        audio: Some(opened.audio),
        output: opened.output,
        started: None,
        muted: false,
        closing: None,
    })
}
