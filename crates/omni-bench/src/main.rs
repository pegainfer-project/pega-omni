//! `omni-bench`: load generator for `POST /v1/audio/speech`.
//!
//! Requests arrive open-loop (Poisson at `--request-rate`, or all at once for
//! `inf`) under a `--max-concurrency` cap. For every request it records time to
//! first audio packet (TTFP), end-to-end latency, audio seconds received, the
//! real-time factor, and playback underrun: a player that starts at the first
//! packet and plays in real time, stalling whenever the next audio has not
//! arrived. Stall time is what a listener hears as a gap.
//!
//! `omni-bench duplex` loads a live endpoint instead ([`duplex`]).

mod duplex;
mod playback;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use hdrhistogram::Histogram;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::Semaphore;

use crate::playback::Playback;

#[derive(Parser)]
#[command(
    name = "omni-bench",
    version,
    about = "Load generator for OpenAI-compatible speech endpoints",
    args_conflicts_with_subcommands = true
)]
struct Top {
    #[command(subcommand)]
    mode: Option<Mode>,
    #[command(flatten)]
    speech: Cli,
}

#[derive(clap::Subcommand)]
enum Mode {
    /// Concurrent GPT-Live sessions over `/v1/live/sessions`, paced at real time.
    Duplex(duplex::Args),
}

#[derive(clap::Args, Clone)]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    base_url: String,
    #[arg(long, default_value = "pega-omni-sim")]
    model: String,
    #[arg(long, default_value_t = 1000)]
    num_requests: usize,
    /// Requests per second, or `inf` to release every request at once.
    #[arg(long, default_value = "inf")]
    request_rate: String,
    #[arg(long, default_value_t = 256)]
    max_concurrency: usize,
    /// One prompt per line; without it, synthetic prompts of `--input-chars` characters.
    #[arg(long)]
    prompts: Option<PathBuf>,
    #[arg(long, default_value_t = 120)]
    input_chars: usize,
    #[arg(long, default_value = "alloy")]
    voice: String,
    #[arg(long, value_enum, default_value_t = Format::Pcm)]
    response_format: Format,
    #[arg(long, value_enum, default_value_t = Framing::Audio)]
    stream_format: Framing,
    /// JSON object sent as the request's `extra` (engine-specific options).
    #[arg(long)]
    extra: Option<String>,
    /// PCM sample rate of the server's audio, used for audio durations.
    #[arg(long, default_value_t = 24_000)]
    sample_rate: u32,
    /// Requests sent and discarded before measuring.
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    timeout: u64,
    /// Write the summary and per-request records as JSON here.
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "")]
    label: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Format {
    Wav,
    Pcm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Framing {
    Audio,
    Sse,
}

#[derive(Clone, Debug, Serialize)]
struct Record {
    ok: bool,
    error: Option<String>,
    ttfp_ms: Option<f64>,
    e2e_ms: f64,
    audio_s: f64,
    underrun_ms: f64,
}

fn prompts(cli: &Cli) -> anyhow::Result<Vec<String>> {
    if let Some(p) = &cli.prompts {
        let text = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let lines: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
        if lines.is_empty() {
            bail!("{} has no prompts", p.display());
        }
        return Ok(lines);
    }
    const WORDS: [&str; 16] = [
        "harbor", "lantern", "quietly", "river", "morning", "copper", "gathered", "window", "orchard", "distant",
        "velvet", "signal", "winter", "measured", "garden", "thunder",
    ];
    Ok((0..cli.num_requests.max(1))
        .map(|i| {
            let mut s = String::with_capacity(cli.input_chars + 16);
            let mut k = i;
            while s.len() < cli.input_chars {
                s.push_str(WORDS[k % WORDS.len()]);
                s.push(' ');
                k = k.wrapping_mul(31).wrapping_add(7);
            }
            s.truncate(cli.input_chars);
            s
        })
        .collect())
}

struct Shot {
    client: reqwest::Client,
    url: String,
    body_template: Value,
    framing: Framing,
    header_len: usize,
    bytes_per_second: f64,
    timeout: Duration,
}

impl Shot {
    async fn fire(&self, input: &str) -> Record {
        let started = Instant::now();
        let mut body = self.body_template.clone();
        body["input"] = Value::String(input.to_string());
        let mut playback = Playback::default();
        let result = tokio::time::timeout(self.timeout, self.stream(&body, &mut playback)).await;
        let e2e = started.elapsed();
        let error = match result {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("timeout".into()),
        };
        Record {
            ok: error.is_none(),
            error,
            ttfp_ms: playback.first().map(|t| t.duration_since(started).as_secs_f64() * 1e3),
            e2e_ms: e2e.as_secs_f64() * 1e3,
            audio_s: playback.audio_seconds(),
            underrun_ms: playback.stall().as_secs_f64() * 1e3,
        }
    }

    async fn stream(&self, body: &Value, playback: &mut Playback) -> anyhow::Result<()> {
        let resp = self.client.post(&self.url).json(body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("HTTP {status}: {}", text.chars().take(200).collect::<String>());
        }
        match self.framing {
            Framing::Audio => {
                let mut skip = self.header_len;
                let mut chunks = resp.bytes_stream();
                while let Some(c) = chunks.next().await {
                    let c = c?;
                    let n = c.len().saturating_sub(skip);
                    skip = skip.saturating_sub(c.len());
                    if n > 0 {
                        playback.arrive(Instant::now(), n as f64 / self.bytes_per_second);
                    }
                }
                Ok(())
            }
            Framing::Sse => {
                let mut events = resp.bytes_stream().eventsource();
                let mut skip = self.header_len;
                while let Some(e) = events.next().await {
                    let e = e?;
                    let v: Value = serde_json::from_str(&e.data).context("event is not JSON")?;
                    match v["type"].as_str() {
                        Some("speech.audio.delta") => {
                            use base64::Engine as _;
                            let b64 = v["audio"].as_str().context("delta without audio")?;
                            let pcm = base64::engine::general_purpose::STANDARD.decode(b64)?;
                            let n = pcm.len().saturating_sub(skip);
                            skip = skip.saturating_sub(pcm.len());
                            if n > 0 {
                                playback.arrive(Instant::now(), n as f64 / self.bytes_per_second);
                            }
                        }
                        Some("speech.audio.done") => return Ok(()),
                        Some("error") => bail!("server error event: {}", v["error"]),
                        t => bail!("unexpected event type {t:?}"),
                    }
                }
                bail!("stream ended without speech.audio.done")
            }
        }
    }
}

#[derive(Serialize)]
struct Summary {
    label: String,
    requests: usize,
    ok: usize,
    failed: usize,
    errors: std::collections::BTreeMap<String, usize>,
    duration_s: f64,
    request_throughput: f64,
    audio_throughput: f64,
    ttfp_ms: Percentiles,
    e2e_ms: Percentiles,
    rtf: Percentiles,
    underrun_ms: Percentiles,
    underrun_requests: usize,
}

#[derive(Serialize, Default)]
struct Percentiles {
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

/// Percentiles of `values`, kept at three significant digits in micro-units.
fn percentiles(values: impl Iterator<Item = f64>) -> Percentiles {
    let mut h = Histogram::<u64>::new(3).expect("3 significant digits");
    for v in values {
        h.record((v * 1e3).round().max(0.0) as u64).expect("auto-resizing histogram");
    }
    if h.is_empty() {
        return Percentiles::default();
    }
    let at = |q: f64| h.value_at_quantile(q) as f64 / 1e3;
    Percentiles { mean: h.mean() / 1e3, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: h.max() as f64 / 1e3 }
}

fn summarize(label: &str, records: &[Record], duration: Duration) -> Summary {
    let ok: Vec<&Record> = records.iter().filter(|r| r.ok).collect();
    let mut errors = std::collections::BTreeMap::new();
    for r in records.iter().filter_map(|r| r.error.as_ref()) {
        *errors.entry(r.chars().take(80).collect::<String>()).or_insert(0) += 1;
    }
    let secs = duration.as_secs_f64();
    Summary {
        label: label.into(),
        requests: records.len(),
        ok: ok.len(),
        failed: records.len() - ok.len(),
        errors,
        duration_s: secs,
        request_throughput: ok.len() as f64 / secs,
        audio_throughput: ok.iter().map(|r| r.audio_s).sum::<f64>() / secs,
        ttfp_ms: percentiles(ok.iter().filter_map(|r| r.ttfp_ms)),
        e2e_ms: percentiles(ok.iter().map(|r| r.e2e_ms)),
        rtf: percentiles(ok.iter().filter(|r| r.audio_s > 0.0).map(|r| r.e2e_ms / 1e3 / r.audio_s)),
        underrun_ms: percentiles(ok.iter().map(|r| r.underrun_ms)),
        underrun_requests: ok.iter().filter(|r| r.underrun_ms > 1.0).count(),
    }
}

fn print(s: &Summary) {
    let row = |name: &str, p: &Percentiles| {
        println!("{name:<14} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}", p.mean, p.p50, p.p90, p.p99, p.max)
    };
    println!("requests {} ok {} failed {} in {:.3} s", s.requests, s.ok, s.failed, s.duration_s);
    for (e, n) in &s.errors {
        println!("  {n} x {e}");
    }
    println!(
        "request throughput {:.1} req/s, audio throughput {:.1} audio-s/s",
        s.request_throughput, s.audio_throughput
    );
    println!("{:<14} {:>10} {:>10} {:>10} {:>10} {:>10}", "", "mean", "p50", "p90", "p99", "max");
    row("ttfp ms", &s.ttfp_ms);
    row("e2e ms", &s.e2e_ms);
    row("rtf", &s.rtf);
    row("underrun ms", &s.underrun_ms);
    println!("requests with underrun > 1 ms: {}", s.underrun_requests);
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let rate: Option<f64> = match cli.request_rate.as_str() {
        "inf" => None,
        r => Some(r.parse().context("--request-rate is a number or `inf`")?),
    };
    let extra: Option<Value> = cli.extra.as_deref().map(serde_json::from_str).transpose().context("--extra is JSON")?;
    let prompts = Arc::new(prompts(&cli)?);
    let format = match cli.response_format {
        Format::Wav => "wav",
        Format::Pcm => "pcm",
    };
    let stream_format = match cli.stream_format {
        Framing::Audio => "audio",
        Framing::Sse => "sse",
    };
    let mut template = json!({
        "model": cli.model,
        "voice": cli.voice,
        "response_format": format,
        "stream_format": stream_format,
    });
    if let Some(e) = extra {
        template["extra"] = e;
    }
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(cli.max_concurrency)
        .tcp_nodelay(true)
        .build()
        .context("build the HTTP client")?;
    let shot = Arc::new(Shot {
        client,
        url: format!("{}/v1/audio/speech", cli.base_url.trim_end_matches('/')),
        body_template: template,
        framing: cli.stream_format,
        header_len: if cli.response_format == Format::Wav { 44 } else { 0 },
        bytes_per_second: cli.sample_rate as f64 * 2.0,
        timeout: Duration::from_secs(cli.timeout),
    });

    if cli.warmup > 0 {
        let warm = futures_util::future::join_all((0..cli.warmup).map(|i| {
            let (shot, prompts) = (shot.clone(), prompts.clone());
            async move { shot.fire(&prompts[i % prompts.len()]).await }
        }))
        .await;
        let failed = warm.iter().filter(|r| !r.ok).count();
        if failed > 0 {
            bail!("{failed}/{} warmup requests failed: {:?}", cli.warmup, warm.iter().find_map(|r| r.error.clone()));
        }
    }

    let gate = Arc::new(Semaphore::new(cli.max_concurrency));
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let gaps = rate.map(|r| Exp::new(r).context("--request-rate must be positive")).transpose()?;
    let started = Instant::now();
    let mut next = started;
    let mut tasks = Vec::with_capacity(cli.num_requests);
    for i in 0..cli.num_requests {
        if let Some(g) = &gaps {
            next += Duration::from_secs_f64(g.sample(&mut rng));
            tokio::time::sleep_until(next.into()).await;
        }
        let (shot, prompts, gate) = (shot.clone(), prompts.clone(), gate.clone());
        tasks.push(tokio::spawn(async move {
            let _permit = gate.acquire_owned().await.expect("semaphore never closes");
            shot.fire(&prompts[i % prompts.len()]).await
        }));
    }
    let mut records = Vec::with_capacity(tasks.len());
    for t in tasks {
        records.push(t.await?);
    }
    let duration = started.elapsed();

    let summary = summarize(&cli.label, &records, duration);
    print(&summary);
    if let Some(path) = &cli.out {
        let report = json!({
            "config": {
                "base_url": cli.base_url, "model": cli.model, "num_requests": cli.num_requests,
                "request_rate": cli.request_rate, "max_concurrency": cli.max_concurrency,
                "input_chars": cli.input_chars, "response_format": cli.response_format,
                "stream_format": cli.stream_format, "extra": cli.extra, "warmup": cli.warmup, "seed": cli.seed,
            },
            "summary": summary,
            "records": records,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    if summary.failed > 0 {
        std::process::exit(2);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let top = Top::parse();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    match top.mode {
        Some(Mode::Duplex(args)) => rt.block_on(duplex::run(args)),
        None => rt.block_on(run(top.speech)),
    }
}
