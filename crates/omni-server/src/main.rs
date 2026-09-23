//! `pega-omni`: the server binary. Each subcommand wires one engine behind the front end.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use metrics_exporter_prometheus::PrometheusBuilder;
use omni_sim::Profile;

#[derive(Parser)]
#[command(name = "pega-omni", version, about = "OpenAI-compatible speech serving")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the CPU-only simulated engine.
    Sim(SimArgs),
    /// Serve Qwen3-TTS (12Hz CustomVoice) on one GPU.
    #[cfg(feature = "qwen3-tts")]
    Qwen3Tts(Qwen3TtsArgs),
}

#[derive(Args)]
struct Serve {
    #[arg(long, default_value = "127.0.0.1:8000")]
    listen: SocketAddr,
    /// Tokio worker threads; defaults to one per core.
    #[arg(long)]
    workers: Option<usize>,
    /// Submissions the engine may hold before new requests get 429.
    #[arg(long, default_value_t = 65_536)]
    queue: usize,
    /// Leave `/metrics` off and record nothing.
    #[arg(long)]
    no_metrics: bool,
    /// Sample this process's CPU for the whole run and write a flamegraph SVG here on exit.
    #[cfg(feature = "cpu-profile")]
    #[arg(long)]
    cpu_profile: Option<std::path::PathBuf>,
}

#[derive(Args)]
struct SimArgs {
    #[command(flatten)]
    serve: Serve,
    #[arg(long, default_value = "pega-omni-sim")]
    model: String,
    #[arg(long, default_value_t = 24_000)]
    sample_rate: u32,
    #[arg(long, default_value_t = 12.5)]
    frame_rate: f64,
    #[arg(long, default_value_t = 0.8)]
    frames_per_char: f64,
    #[arg(long, default_value_t = 4096)]
    max_frames: u32,
    #[arg(long, default_value_t = 1)]
    first_chunk_frames: u32,
    #[arg(long, default_value_t = 4)]
    chunk_frames: u32,
    #[arg(long, default_value_t = 256)]
    max_batch: usize,
    /// Fixed cost of one step, in microseconds.
    #[arg(long, default_value_t = 0)]
    step_base_us: u64,
    /// Added step cost per running request, in microseconds.
    #[arg(long, default_value_t = 0)]
    step_per_row_us: u64,
    /// Added step cost per admitted input character, in microseconds.
    #[arg(long, default_value_t = 0)]
    prefill_per_char_us: u64,
}

#[cfg(feature = "qwen3-tts")]
#[derive(Args)]
struct Qwen3TtsArgs {
    #[command(flatten)]
    serve: Serve,
    /// Checkpoint directory (config.json, model.safetensors, speech_tokenizer/).
    #[arg(long)]
    model_path: std::path::PathBuf,
    /// The model name clients send; defaults to the checkpoint directory's name.
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 64)]
    max_batch: usize,
    /// Talker tokens per step, prompts included.
    #[arg(long, default_value_t = 8192)]
    max_step_tokens: usize,
    /// Talker KV cache size.
    #[arg(long, default_value_t = 16.0)]
    kv_gib: f64,
    #[arg(long, default_value_t = 2)]
    first_chunk_frames: usize,
    #[arg(long, default_value_t = 8)]
    chunk_frames: usize,
    #[arg(long, default_value_t = 4096)]
    max_input_chars: usize,
}

#[cfg(feature = "qwen3-tts")]
impl Qwen3TtsArgs {
    fn start(&self) -> anyhow::Result<(omni_engine::Handle, std::thread::JoinHandle<()>, String)> {
        use omni_qwen3_tts::engine;
        let opts = engine::Options {
            max_batch: self.max_batch,
            max_step_tokens: self.max_step_tokens,
            kv_gib: self.kv_gib,
            first_chunk_frames: self.first_chunk_frames,
            chunk_frames: self.chunk_frames,
            ..engine::Options::default()
        };
        let name = self.model.clone().unwrap_or_else(|| {
            self.model_path.file_name().map_or("qwen3-tts".into(), |n| n.to_string_lossy().into_owned())
        });
        let (handle, thread) = engine::start(
            self.device,
            self.model_path.clone(),
            opts,
            name.clone(),
            self.max_input_chars,
            self.serve.queue,
        )?;
        Ok((handle, thread, name))
    }
}

impl SimArgs {
    fn profile(&self) -> anyhow::Result<Profile> {
        Profile {
            sample_rate: self.sample_rate,
            frame_rate: self.frame_rate,
            frames_per_char: self.frames_per_char,
            max_frames: self.max_frames,
            first_chunk_frames: self.first_chunk_frames,
            chunk_frames: self.chunk_frames,
            max_batch: self.max_batch,
            step_base: Duration::from_micros(self.step_base_us),
            step_per_row: Duration::from_micros(self.step_per_row_us),
            prefill_per_char: Duration::from_micros(self.prefill_per_char_us),
        }
        .check()
        .map_err(anyhow::Error::msg)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let (serve, (handle, engine, model)) = match Cli::parse().command {
        Command::Sim(args) => {
            let profile = args.profile()?;
            let (handle, inbox) = omni_engine::channel(profile.info(&args.model), args.serve.queue);
            (args.serve, (handle, omni_sim::spawn(inbox, profile), args.model))
        }
        #[cfg(feature = "qwen3-tts")]
        Command::Qwen3Tts(args) => {
            let started = args.start()?;
            (args.serve, started)
        }
    };
    let serve = &serve;
    #[cfg(feature = "cpu-profile")]
    let sampler = serve
        .cpu_profile
        .as_ref()
        .map(|_| pprof::ProfilerGuardBuilder::default().frequency(999).blocklist(&["libc", "libgcc", "vdso"]).build())
        .transpose()
        .context("start the CPU profiler")?;

    let mut rt = tokio::runtime::Builder::new_multi_thread();
    if let Some(n) = serve.workers {
        rt.worker_threads(n);
    }
    rt.enable_all().build()?.block_on(async {
        let prometheus = (!serve.no_metrics)
            .then(|| PrometheusBuilder::new().install_recorder())
            .transpose()
            .context("install the metrics recorder")?;
        let app = omni_frontend::router(handle, prometheus);
        let listener =
            tokio::net::TcpListener::bind(serve.listen).await.with_context(|| format!("bind {}", serve.listen))?;
        tracing::info!("serving `{model}` on http://{}", serve.listen);
        omni_frontend::serve(listener, app, shutdown()).await?;
        anyhow::Ok(())
    })?;

    #[cfg(feature = "cpu-profile")]
    if let (Some(guard), Some(path)) = (sampler, &serve.cpu_profile) {
        let report = guard.report().build().context("build the CPU profile")?;
        let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        report.flamegraph(file).context("write the flamegraph")?;
        tracing::info!("CPU profile written to {}", path.display());
    }
    drop(engine);
    Ok(())
}

async fn shutdown() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
    tokio::select! {
        _ = ctrl_c => {}
        _ = term.recv() => {}
    }
}
