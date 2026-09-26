//! `omni-bench duplex`: concurrent live sessions over the GPT-Live WebSocket.
//!
//! Each session opens `/v1/live/sessions`, waits for `session.started`, then
//! streams the caller's audio at real time in `--chunk-ms` chunks (a 24 kHz
//! mono wav, looped, or a synthetic speech-like signal: voiced syllables
//! between pauses, different per session) for `--seconds`, and closes.
//! Meanwhile it records every output frame's arrival:
//!
//! - lateness: how far behind the session timeline a frame arrived, i.e.
//!   arrival − first arrival − (`start_ms` − first `start_ms`), which a
//!   real-time server keeps near zero however long the session runs;
//! - underrun: a player that starts `--playout-ms` after the first frame and
//!   plays the agent's stream in real time stalls whenever the next frame is
//!   late ([`Playback`]).
//!
//! `session.start` declares the input and output format at 24 kHz, so a
//! server at another rate refuses the session instead of mishearing it.
//!
//! A level is clean when every session ran and none stalled. `--ramp 1,2,4,…`
//! runs levels in order and stops at the first unclean one; the last clean
//! level is the capacity it reports.

use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::bail;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use futures_util::SinkExt;
use futures_util::StreamExt;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use tokio_tungstenite::tungstenite::Message;

use crate::Percentiles;
use crate::percentiles;
use crate::playback::Playback;

const RATE: u32 = 24_000;

#[derive(clap::Args, Clone)]
pub struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    base_url: String,
    /// Sent as `session.model` when given.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    voice: Option<String>,
    #[arg(long)]
    instructions: Option<String>,
    /// Concurrent sessions (ignored with `--ramp`).
    #[arg(long, default_value_t = 8)]
    sessions: usize,
    /// Comma-separated session counts to run in order, stopping at the first unclean level.
    #[arg(long)]
    ramp: Option<String>,
    /// Caller audio per session, in seconds.
    #[arg(long, default_value_t = 20.0)]
    seconds: f64,
    /// 24 kHz mono s16 wav to stream (looped); without it, a synthetic speech-like signal.
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    chunk_ms: u64,
    /// Delay between session starts, so admission is not one burst.
    #[arg(long, default_value_t = 20)]
    stagger_ms: u64,
    /// The player's startup buffer.
    #[arg(long, default_value_t = 120)]
    playout_ms: u64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "")]
    label: String,
}

#[derive(Debug, Serialize)]
struct Record {
    ok: bool,
    error: Option<String>,
    started_ms: Option<f64>,
    frames: usize,
    audio_s: f64,
    underrun_ms: f64,
    late_p99_ms: f64,
    late_max_ms: f64,
    close_reason: Option<String>,
    transcript: String,
    #[serde(skip)]
    lateness_ms: Vec<f64>,
}

#[derive(Serialize)]
struct Level {
    label: String,
    sessions: usize,
    ok: usize,
    failed: usize,
    errors: std::collections::BTreeMap<String, usize>,
    clean: bool,
    started_ms: Percentiles,
    lateness_ms: Percentiles,
    underrun_ms: Percentiles,
    underrun_sessions: usize,
    frames: usize,
    audio_s: f64,
}

/// A 24 kHz mono s16 wav's samples.
fn read_wav(path: &PathBuf) -> anyhow::Result<Vec<i16>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("{} is not a wav file", path.display());
    }
    let (mut at, mut format, mut data) = (12, None, None);
    while at + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into()?) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match &bytes[at..at + 4] {
            b"fmt " if body.len() >= 16 => {
                format = Some((
                    u16::from_le_bytes(body[0..2].try_into()?),
                    u16::from_le_bytes(body[2..4].try_into()?),
                    u32::from_le_bytes(body[4..8].try_into()?),
                    u16::from_le_bytes(body[14..16].try_into()?),
                ))
            }
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + len % 2;
    }
    match (format, data) {
        (Some((1, 1, RATE, 16)), Some(d)) => Ok(d.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect()),
        (Some(f), Some(_)) => {
            bail!("{}: need PCM16 mono at {RATE} Hz, got (format, channels, rate, bits) {f:?}", path.display())
        }
        _ => bail!("{}: no fmt or data chunk", path.display()),
    }
}

/// Speech-shaped test audio: syllables (a harmonic buzz at a drifting pitch
/// under a smooth envelope) in phrases separated by pauses, seeded.
fn synthetic(seconds: f64, seed: u64) -> Vec<i16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let total = (seconds * RATE as f64) as usize;
    let mut out = Vec::with_capacity(total);
    while out.len() < total {
        for _ in 0..rng.random_range(3..12) {
            let len = (rng.random_range(0.12..0.28) * RATE as f64) as usize;
            let (f0, drift) = (rng.random_range(95.0..230.0), rng.random_range(-0.4..0.4));
            let formant: f64 = rng.random_range(2.0..6.0);
            let mut phase = 0.0f64;
            for i in 0..len {
                let t = i as f64 / len as f64;
                phase += std::f64::consts::TAU * f0 * (1.0 + drift * t) / RATE as f64;
                let buzz: f64 = (1..8).map(|h| (phase * h as f64).sin() / (1.0 + (h as f64 - formant).abs())).sum();
                out.push((buzz * (std::f64::consts::PI * t).sin().powi(2) * 5000.0) as i16);
            }
            out.extend(std::iter::repeat_n(0, (rng.random_range(0.02..0.08) * RATE as f64) as usize));
        }
        out.extend(std::iter::repeat_n(0, (rng.random_range(0.3..1.2) * RATE as f64) as usize));
    }
    out.truncate(total);
    out
}

fn ws_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    let base = base.strip_prefix("http").map_or(base.to_string(), |rest| format!("ws{rest}"));
    format!("{base}/v1/live/sessions")
}

async fn session(args: &Args, audio: Vec<i16>) -> Record {
    let mut record = Record {
        ok: false,
        error: None,
        started_ms: None,
        frames: 0,
        audio_s: 0.0,
        underrun_ms: 0.0,
        late_p99_ms: 0.0,
        late_max_ms: 0.0,
        close_reason: None,
        transcript: String::new(),
        lateness_ms: Vec::new(),
    };
    let budget = Duration::from_secs_f64(args.seconds + 60.0);
    let result = tokio::time::timeout(budget, drive(args, audio, &mut record)).await;
    record.error = match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("{e:#}")),
        Err(_) => Some("timeout".into()),
    };
    record.ok = record.error.is_none();
    let p = percentiles(record.lateness_ms.iter().copied());
    (record.late_p99_ms, record.late_max_ms) = (p.p99, p.max);
    record
}

async fn drive(args: &Args, audio: Vec<i16>, record: &mut Record) -> anyhow::Result<()> {
    let connected = Instant::now();
    let (ws, _) = tokio_tungstenite::connect_async(ws_url(&args.base_url)).await.context("connect")?;
    let (mut tx, mut rx) = ws.split();
    let mut session = json!({});
    if let Some(m) = &args.model {
        session["model"] = json!(m);
    }
    if let Some(i) = &args.instructions {
        session["instructions"] = json!(i);
    }
    let format = json!({"type": "audio/pcm", "rate": RATE});
    session["audio"] = json!({"input": {"format": format}, "output": {"format": format}});
    if let Some(v) = &args.voice {
        session["audio"]["output"]["voice"] = json!(v);
    }
    tx.send(Message::Text(json!({"type": "session.start", "session": session}).to_string().into())).await?;

    let chunk = (RATE as u64 * args.chunk_ms / 1000) as usize;
    let chunks = (args.seconds * 1000.0 / args.chunk_ms as f64).ceil() as usize;
    let pace = Duration::from_millis(args.chunk_ms);
    let mut writer: Option<tokio::task::JoinHandle<anyhow::Result<()>>> = None;
    let mut sink = Some(tx);
    let mut playback = Playback::default();
    let playout = Duration::from_millis(args.playout_ms);
    let mut first: Option<(Instant, u64)> = None;
    let mut played_to = 0u64;

    while let Some(msg) = rx.next().await {
        let text = match msg? {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };
        let at = Instant::now();
        let v: Value = serde_json::from_str(&text).context("event is not JSON")?;
        match v["type"].as_str() {
            Some("session.started") => {
                record.started_ms = Some(connected.elapsed().as_secs_f64() * 1e3);
                let mut tx = sink.take().context("two session.started")?;
                let audio = audio.clone();
                writer = Some(tokio::spawn(async move {
                    let start = tokio::time::Instant::now();
                    for k in 0..chunks {
                        tokio::time::sleep_until(start + pace * k as u32).await;
                        let pcm: Vec<u8> = (0..chunk)
                            .map(|i| audio[(k * chunk + i) % audio.len()])
                            .flat_map(i16::to_le_bytes)
                            .collect();
                        let e = json!({"type": "session.input_audio.append", "audio": STANDARD.encode(&pcm)});
                        tx.send(Message::Text(e.to_string().into())).await?;
                    }
                    tx.send(Message::Text(json!({"type": "session.close"}).to_string().into())).await?;
                    Ok(())
                }));
            }
            Some("session.output_audio.delta") => {
                let (start, end) =
                    (v["start_ms"].as_u64().context("start_ms")?, v["end_ms"].as_u64().context("end_ms")?);
                let (t0, s0) = *first.get_or_insert((at, start));
                if start < played_to {
                    bail!("start_ms {start} goes back before {played_to}");
                }
                let expected = t0 + Duration::from_millis(start - s0);
                record.lateness_ms.push(at.saturating_duration_since(expected).as_secs_f64() * 1e3);
                let seconds = (end - played_to.max(s0)) as f64 / 1e3;
                playback.arrive(at.max(t0 + playout), seconds);
                played_to = end;
                record.frames += 1;
            }
            Some("session.output_transcript.delta") => record.transcript.push_str(v["delta"].as_str().unwrap_or("")),
            Some("session.closed") => {
                record.close_reason = v["reason"].as_str().map(String::from);
                break;
            }
            Some("error") => bail!("server error: {}", v["error"]),
            _ => {}
        }
    }
    if let Some(w) = writer {
        w.await.context("the writer panicked")?.context("send")?;
    }
    record.audio_s = playback.audio_seconds();
    record.underrun_ms = playback.stall().as_secs_f64() * 1e3;
    match record.close_reason.as_deref() {
        Some("close_requested") => Ok(()),
        Some(r) => bail!("session closed: {r}"),
        None => bail!("socket ended without session.closed"),
    }
}

fn level(label: &str, records: &[Record]) -> Level {
    let ok: Vec<&Record> = records.iter().filter(|r| r.ok).collect();
    let mut errors = std::collections::BTreeMap::new();
    for e in records.iter().filter_map(|r| r.error.as_ref()) {
        *errors.entry(e.chars().take(80).collect::<String>()).or_insert(0) += 1;
    }
    let underrun_sessions = ok.iter().filter(|r| r.underrun_ms > 1.0).count();
    Level {
        label: label.into(),
        sessions: records.len(),
        ok: ok.len(),
        failed: records.len() - ok.len(),
        errors,
        clean: ok.len() == records.len() && underrun_sessions == 0,
        started_ms: percentiles(ok.iter().filter_map(|r| r.started_ms)),
        lateness_ms: percentiles(ok.iter().flat_map(|r| r.lateness_ms.iter().copied())),
        underrun_ms: percentiles(ok.iter().map(|r| r.underrun_ms)),
        underrun_sessions,
        frames: ok.iter().map(|r| r.frames).sum(),
        audio_s: ok.iter().map(|r| r.audio_s).sum(),
    }
}

fn print(l: &Level) {
    let row = |name: &str, p: &Percentiles| {
        println!("  {name:<12} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}", p.mean, p.p50, p.p90, p.p99, p.max)
    };
    let verdict = if l.clean { "clean" } else { "NOT clean" };
    println!("{} sessions: ok {} failed {} · {verdict}", l.sessions, l.ok, l.failed);
    for (e, n) in &l.errors {
        println!("  {n} x {e}");
    }
    println!("  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "mean", "p50", "p90", "p99", "max");
    row("started ms", &l.started_ms);
    row("late ms", &l.lateness_ms);
    row("underrun ms", &l.underrun_ms);
    println!(
        "  frames {} ({:.1} audio-s), sessions with underrun > 1 ms: {}",
        l.frames, l.audio_s, l.underrun_sessions
    );
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let levels: Vec<usize> = match &args.ramp {
        Some(r) => r
            .split(',')
            .map(|n| n.trim().parse().context("--ramp is comma-separated counts"))
            .collect::<Result<_, _>>()?,
        None => vec![args.sessions],
    };
    let base = match &args.input {
        Some(p) => Some(read_wav(p)?),
        None => None,
    };
    let mut report = Vec::new();
    let mut records_out = Vec::new();
    let mut capacity = None;
    for (li, &n) in levels.iter().enumerate() {
        let tasks: Vec<_> = (0..n)
            .map(|i| {
                let args = args.clone();
                let audio = base.clone().unwrap_or_else(|| synthetic(args.seconds.min(60.0), args.seed + i as u64));
                let delay = Duration::from_millis(args.stagger_ms * i as u64);
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    session(&args, audio).await
                })
            })
            .collect();
        let mut records = Vec::with_capacity(n);
        for t in tasks {
            records.push(t.await?);
        }
        let l = level(&args.label, &records);
        print(&l);
        let clean = l.clean;
        report.push(l);
        records_out.push(records);
        if !clean {
            break;
        }
        capacity = Some(n);
        if li + 1 < levels.len() {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    if args.ramp.is_some() {
        println!("largest clean level: {}", capacity.map_or("none".into(), |n| n.to_string()));
    }
    if let Some(path) = &args.out {
        let out = json!({
            "config": {
                "base_url": args.base_url, "model": args.model, "voice": args.voice, "levels": levels,
                "seconds": args.seconds, "input": args.input, "chunk_ms": args.chunk_ms,
                "stagger_ms": args.stagger_ms, "playout_ms": args.playout_ms, "seed": args.seed,
            },
            "levels": report,
            "largest_clean": capacity,
            "records": records_out,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&out)?).with_context(|| format!("write {}", path.display()))?;
    }
    if report.iter().any(|l| l.failed > 0) {
        std::process::exit(2);
    }
    Ok(())
}
