//! The Qwen3-VL text encoder and the VAE decoder, in a process of their own.
//!
//! The tokenizer and the text tower stay out of this crate: the sidecar owns
//! them and hands over the prompt's embeddings ([`TEXT_DIM`] channels, the
//! checkpoint's `context_in_dim`) and their token count, which is all the
//! transformer's text projection reads. It decodes the finished latents too.
//! Both run on the same GPU as the transformer, one after the other, so the
//! sidecar serves one request at a time and the engine calls it from one
//! thread: two callers would interleave their frames on the one socket.
//!
//! [`Sidecar`] owns the child: it materializes `python/sidecar.py` from the
//! binary, waits for its `READY` line, and speaks the little-endian frames the
//! script documents. A failed request is an `Err` for the engine to abort that
//! generation with; a broken pipe is the same, and the next request finds out.
//! Dropping it asks the script to exit and then makes sure it did.

use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::ChildStderr;
use std::process::ChildStdout;
use std::process::Command;
use std::process::Stdio;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use sha2::Digest;

use crate::config::LATENT_DIM;

/// The script the child runs, carried in the binary so a deployment is one file.
const SCRIPT: &str = include_str!("../python/sidecar.py");

/// Channels of the text tower's embeddings, checked against the sidecar at the
/// handshake: the transformer's text projection is built for this width.
pub const TEXT_DIM: usize = 4096;
/// The longest text prefix the transformer is sized for, in tokens.
pub const MAX_TEXT: usize = 2048;

const PING: u32 = 1;
const ENCODE: u32 = 2;
const DECODE: u32 = 3;
const SHUTDOWN: u32 = 4;

/// Loading two checkpoints is minutes on a cold page cache.
const LOAD_TIMEOUT: Duration = Duration::from_secs(600);
/// One request, so a wedged sidecar aborts the generation instead of the engine.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// The largest response there is: a token count and [`MAX_TEXT`] embeddings of
/// [`TEXT_DIM`] bf16. A picture is a few MiB.
const MAX_RESPONSE: usize = 4 + MAX_TEXT * TEXT_DIM * 2;
/// How long a child gets to exit on its own before it is killed.
const EXIT_GRACE: Duration = Duration::from_secs(5);

/// Tells concurrent materializations in one process apart.
static SCRATCH: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct Sidecar {
    child: Child,
    stream: UnixStream,
    text_dim: usize,
}

impl Sidecar {
    /// Runs `python` on the embedded script against the pipeline directory at
    /// `model_dir`, on GPU `device` — the one the transformer runs on — and
    /// returns once the models are loaded and the socket answers.
    pub fn spawn(python: &str, model_dir: &Path, device: usize, socket_path: &Path) -> Result<Self> {
        let script = materialize()?;
        if let Some(parent) = socket_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut child = Command::new(python)
            .arg(&script)
            .arg("--model-dir")
            .arg(model_dir)
            .arg("--device")
            .arg(device.to_string())
            .arg("--socket")
            .arg(socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning {python} {}", script.display()))?;
        match handshake(&mut child, socket_path) {
            Ok((stream, text_dim)) => Ok(Self { child, stream, text_dim }),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                // A killed child cannot run its own cleanup.
                let _ = std::fs::remove_file(socket_path);
                Err(e)
            }
        }
    }

    /// The prompt's embeddings: `(text_len, text_len * text_dim bf16 bytes)`,
    /// the text tower's last hidden state before its final norm.
    pub fn encode(&mut self, prompt: &str) -> Result<(usize, Vec<u8>)> {
        let mut payload = Vec::with_capacity(4 + prompt.len());
        payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
        payload.extend_from_slice(prompt.as_bytes());
        let body = request(&mut self.stream, ENCODE, &payload)?;
        let tokens = word(&body, 0)? as usize;
        let embeddings = body.get(4..).context("the sidecar's encode response has no token count")?;
        ensure!(tokens <= MAX_TEXT, "the sidecar returned {tokens} text tokens, the engine takes at most {MAX_TEXT}");
        ensure!(
            embeddings.len() == tokens * self.text_dim * 2,
            "{tokens} text tokens of {} channels is {} bytes, the sidecar sent {}",
            self.text_dim,
            tokens * self.text_dim * 2,
            embeddings.len()
        );
        Ok((tokens, embeddings.to_vec()))
    }

    /// The picture's RGB bytes, row-major and unpadded: `height * width * 3`.
    /// `latent` is `rows * 64` bf16, one row per 16 x 16 cell, raster order.
    pub fn decode(&mut self, height: u32, width: u32, latent: &[u8]) -> Result<Vec<u8>> {
        let row = LATENT_DIM * 2;
        ensure!(latent.len().is_multiple_of(row), "{} latent bytes is not a multiple of {row}", latent.len());
        let rows = (latent.len() / row) as u32;
        let mut payload = Vec::with_capacity(12 + latent.len());
        payload.extend_from_slice(&height.to_le_bytes());
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&rows.to_le_bytes());
        payload.extend_from_slice(latent);
        let body = request(&mut self.stream, DECODE, &payload)?;
        let (got_h, got_w) = (word(&body, 0)?, word(&body, 4)?);
        let rgb = body.get(8..).context("the sidecar's decode response has no dimensions")?;
        ensure!(
            (got_h, got_w) == (height, width),
            "the sidecar decoded {got_h}x{got_w} for a requested {height}x{width}: both sides must be multiples of 32"
        );
        let want = got_h as usize * got_w as usize * 3;
        ensure!(rgb.len() == want, "a {got_h}x{got_w} picture is {want} bytes, the sidecar sent {}", rgb.len());
        Ok(rgb.to_vec())
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // Ask for a clean exit, then make sure of it: a sidecar that outlives
        // the engine keeps the GPU memory it loaded.
        let _ = request(&mut self.stream, SHUTDOWN, &[]);
        let deadline = Instant::now() + EXIT_GRACE;
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(20)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The embedded script under its hash, in the per-user cache the cubins use, so
/// a rebuild replaces it and two engines share one copy.
fn materialize() -> Result<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let dir = var("XDG_CACHE_HOME")
        .or_else(|| var("HOME").map(|h| h.join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("pega-omni/sidecar");
    let sha = omni_kern::hex(&sha2::Sha256::digest(SCRIPT)[..6]);
    let path = dir.join(format!("qwen-image-21-{sha}.py"));
    if !path.exists() {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        // Placed by rename so a concurrent spawn never reads a half-written
        // script; the scratch name is unique to this call, not just this process.
        let scratch = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".qwen-image-21-{sha}.{}.{scratch}", std::process::id()));
        std::fs::write(&tmp, SCRIPT).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("placing {}", path.display()))?;
    }
    Ok(path)
}

/// The child's stderr, line by line, into this process's log.
fn pump(stderr: ChildStderr) -> Result<()> {
    thread::Builder::new()
        .name("qwen21-sidecar-stderr".into())
        .spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if !line.trim().is_empty() {
                    tracing::warn!(target: "omni_qwen_image_21::sidecar", "{line}");
                }
            }
        })
        .context("spawning the sidecar's stderr reader")?;
    Ok(())
}

/// Waits for `READY`, which the script prints once both models are on the GPU
/// and the socket listens. stdout is read on its own thread so the wait can
/// time out; that thread drains the pipe for the rest of the child's life.
fn ready(child: &mut Child, stdout: ChildStdout) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    thread::Builder::new()
        .name("qwen21-sidecar-stdout".into())
        .spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(Some(line)).is_err() {
                    break;
                }
            }
            let _ = tx.send(None);
        })
        .context("spawning the sidecar's stdout reader")?;
    let deadline = Instant::now() + LOAD_TIMEOUT;
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Some(line)) if line.trim() == "READY" => return Ok(()),
            Ok(Some(_)) => {}
            Ok(None) => bail!("the sidecar exited while loading: {:?}", child.try_wait()?),
            Err(_) => bail!("the sidecar did not load within {}s", LOAD_TIMEOUT.as_secs()),
        }
    }
}

/// Connects and checks the text tower's width before the first generation.
fn handshake(child: &mut Child, socket_path: &Path) -> Result<(UnixStream, usize)> {
    let stderr = child.stderr.take().context("the sidecar's stderr is not piped")?;
    let stdout = child.stdout.take().context("the sidecar's stdout is not piped")?;
    pump(stderr)?;
    ready(child, stdout)?;
    let mut stream =
        UnixStream::connect(socket_path).with_context(|| format!("connecting to {}", socket_path.display()))?;
    stream.set_read_timeout(Some(REQUEST_TIMEOUT)).context("setting the sidecar's read timeout")?;
    stream.set_write_timeout(Some(REQUEST_TIMEOUT)).context("setting the sidecar's write timeout")?;
    let body = request(&mut stream, PING, &[])?;
    let dim = word(&body, 0)? as usize;
    ensure!(body.len() == 4, "the sidecar's ping response is {} bytes, expected 4", body.len());
    ensure!(dim == TEXT_DIM, "the sidecar's text tower is {dim} channels, the transformer reads {TEXT_DIM}");
    Ok((stream, dim))
}

/// One frame out and one back. A non-zero status is the sidecar's message, and
/// an I/O error leaves the stream unusable: the engine aborts the generation.
fn request(stream: &mut UnixStream, kind: u32, payload: &[u8]) -> Result<Vec<u8>> {
    let mut frame = Vec::with_capacity(8 + payload.len());
    frame.extend_from_slice(&kind.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(payload);
    stream.write_all(&frame).context("writing to the sidecar")?;
    let mut head = [0u8; 12];
    stream.read_exact(&mut head).context("reading the sidecar's response")?;
    let (answered, status) = (word(&head, 0)?, word(&head, 4)?);
    ensure!(answered == kind, "the sidecar answered kind {answered} to kind {kind}");
    let len = word(&head, 8)? as usize;
    ensure!(len <= MAX_RESPONSE, "the sidecar's {len} byte response is over the {MAX_RESPONSE} byte limit");
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).context("reading the sidecar's response body")?;
    if status != 0 {
        bail!("the sidecar refused kind {kind}: {}", String::from_utf8_lossy(&body));
    }
    Ok(body)
}

/// The little-endian word at `at`, or an error naming the short body.
fn word(body: &[u8], at: usize) -> Result<u32> {
    let bytes = body.get(at..at + 4).with_context(|| format!("a {} byte body has no word at {at}", body.len()))?;
    Ok(u32::from_le_bytes(bytes.try_into()?))
}
