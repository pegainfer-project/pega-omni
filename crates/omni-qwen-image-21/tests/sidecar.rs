//! The bridge against the real script, without torch or a checkpoint.
//!
//! `Sidecar::spawn` runs whatever interpreter it is handed, so the stand-in
//! here loads `python/sidecar.py` itself, replaces its two model classes with
//! ones that answer in bytes that say where they came from, and stubs the
//! `torch` the script imports at module scope. Everything else is the shipped
//! code on both sides: the flags, `READY`, the socket, the frames, a refused
//! request, and the cleanup on shutdown.
//!
//! Needs `python3` on the `PATH`; skipped without it.

use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use omni_qwen_image_21::sidecar::MAX_TEXT;
use omni_qwen_image_21::sidecar::Sidecar;
use omni_qwen_image_21::sidecar::TEXT_DIM;

/// Token `i` is `dim` copies of `[i, 0x3f]`; pixel `(x, y)` channel `c` is
/// `x + y + c`. `FAKE_DIM` is the width the text tower reports, and 0 makes the
/// script exit before it loads.
const DRIVER: &str = r#"
import importlib.util, os, struct, sys, types

torch = types.ModuleType("torch")
torch.Tensor = type("Tensor", (), {})
torch.device = lambda spec: spec
torch.bfloat16 = "bfloat16"
sys.modules["torch"] = torch

dim = int(os.environ.get("FAKE_DIM", "4096"))
if dim == 0:
    sys.stderr.write("no checkpoint here\n")
    sys.exit(3)

# argv[1] is the script the bridge materialized; main() takes only the flags.
spec = importlib.util.spec_from_file_location("sidecar", sys.argv[1])
sidecar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecar)
sys.argv = sys.argv[:1] + sys.argv[2:]

class Encoder:
    def __init__(self, directory, device):
        self.dim = dim

    def encode(self, prompt):
        if prompt == "boom":
            raise ValueError("the encoder exploded")
        tokens = max(1, len(prompt.split()))
        body = b"".join(bytes([i % 256, 0x3F]) * self.dim for i in range(tokens))
        return struct.pack("<I", tokens) + body

class Decoder:
    def __init__(self, directory, device):
        pass

    def decode(self, height, width, payload):
        rows = len(payload) // (64 * 2)
        assert rows == (height // 16) * (width // 16), (rows, height, width)
        rgb = bytes((x + y + c) % 256 for y in range(height) for x in range(width) for c in range(3))
        return struct.pack("<II", height, width) + rgb

sidecar.Encoder = Encoder
sidecar.Decoder = Decoder
sidecar.main()
"#;

/// A temporary directory holding the stand-in interpreter and its socket.
struct Rig {
    dir: PathBuf,
    python: PathBuf,
    socket: PathBuf,
}

impl Rig {
    fn new(name: &str, dim: usize) -> Option<Self> {
        if !python3() {
            eprintln!("skipped: python3 is not on the PATH");
            return None;
        }
        let dir = std::env::temp_dir().join(format!("omni-qwen21-sidecar-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let driver = dir.join("driver.py");
        std::fs::write(&driver, DRIVER).ok()?;
        let python = dir.join("python");
        let body = format!("#!/bin/sh\nexec env FAKE_DIM={dim} python3 {} \"$@\"\n", driver.display());
        std::fs::write(&python, body).ok()?;
        std::fs::set_permissions(&python, Permissions::from_mode(0o755)).ok()?;
        let socket = dir.join("sidecar.sock");
        Some(Self { dir, python, socket })
    }

    fn spawn(&self) -> anyhow::Result<Sidecar> {
        let python = self.python.to_str().expect("the interpreter path is utf-8");
        Sidecar::spawn(python, Path::new("/nonexistent-checkpoint"), 0, &self.socket)
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn python3() -> bool {
    Command::new("python3")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[test]
fn the_bridge_speaks_the_scripts_protocol() {
    let Some(rig) = Rig::new("protocol", TEXT_DIM) else { return };
    let mut sidecar = rig.spawn().expect("the handshake");

    let (tokens, embeddings) = sidecar.encode("a cat on a mat").expect("encode");
    assert_eq!(tokens, 5);
    assert_eq!(embeddings.len(), tokens * TEXT_DIM * 2);
    for (i, token) in embeddings.chunks(TEXT_DIM * 2).enumerate() {
        assert_eq!(token[0], i as u8, "token {i}");
        assert_eq!(token[1], 0x3f, "token {i}");
    }

    // 48 x 32 pixels is a 3 x 2 grid of 16 x 16 cells, one 64-channel row each.
    // Not square, so a height and width swapped on the wire shows up.
    let latent = vec![0u8; 6 * 64 * 2];
    let rgb = sidecar.decode(48, 32, &latent).expect("decode");
    assert_eq!(rgb.len(), 48 * 32 * 3);
    assert_eq!(rgb[0], 0);
    assert_eq!(rgb[3], 1, "pixel (1, 0)");
    assert_eq!(rgb[32 * 3], 1, "pixel (0, 1)");

    // A refused request is that generation's error, not the connection's.
    let err = sidecar.encode("boom").expect_err("the script refuses this prompt");
    assert!(format!("{err:#}").contains("the encoder exploded"), "{err:#}");
    assert_eq!(sidecar.encode("still here").expect("encode after a refusal").0, 2);

    // A malformed latent never reaches the wire.
    assert!(sidecar.decode(48, 32, &[0u8; 3]).is_err());
    assert_eq!(sidecar.decode(48, 32, &latent).expect("decode after a refusal").len(), 48 * 32 * 3);

    // The longest text prefix the engine takes is also the largest response.
    let longest = "w ".repeat(MAX_TEXT);
    let (tokens, embeddings) = sidecar.encode(longest.trim()).expect("the longest prompt");
    assert_eq!(tokens, MAX_TEXT);
    assert_eq!(embeddings.len(), MAX_TEXT * TEXT_DIM * 2);

    drop(sidecar);
    assert!(!rig.socket.exists(), "the script left its socket behind");
}

#[test]
fn a_text_tower_of_the_wrong_width_is_refused() {
    let Some(rig) = Rig::new("width", TEXT_DIM / 2) else { return };
    let err = rig.spawn().expect_err("the handshake checks the width");
    assert!(format!("{err:#}").contains("channels"), "{err:#}");
    assert!(!rig.socket.exists(), "a refused sidecar left its socket behind");
}

#[test]
fn a_sidecar_that_dies_while_loading_is_an_error() {
    let Some(rig) = Rig::new("dies", 0) else { return };
    let err = rig.spawn().expect_err("the child exited before READY");
    assert!(format!("{err:#}").contains("exited while loading"), "{err:#}");
}
