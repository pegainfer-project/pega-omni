//! Compiles `csrc/*.cu` into one static library against the FlashInfer
//! submodule's headers and the CCCL it vendors (FlashInfer needs a newer CCCL
//! than the toolkit ships). A fresh clone gets exactly those two submodules
//! initialized on first build; FlashInfer's other third-party trees (CUTLASS,
//! spdlog, NIXL) are not needed and never fetched.
//!
//! `OMNI_CUDA_ARCH` picks the target (default `103a`, GB300).

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let flashinfer = root.join("third_party/flashinfer");
    let cccl = flashinfer.join("3rdparty/cccl");
    ensure_submodule(&root.join("../.."), "crates/omni-cuda/third_party/flashinfer", &flashinfer.join("include"));
    ensure_submodule(&flashinfer, "3rdparty/cccl", &cccl.join("libcudacxx/include"));

    let arch = std::env::var("OMNI_CUDA_ARCH").unwrap_or_else(|_| "103a".into());
    println!("cargo:rerun-if-env-changed=OMNI_CUDA_ARCH");
    println!("cargo:rerun-if-changed=csrc");

    let sources: Vec<PathBuf> = std::fs::read_dir(root.join("csrc"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();
    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .debug(false)
        .opt_level(3)
        .files(&sources)
        .include(cccl.join("libcudacxx/include"))
        .include(cccl.join("cub"))
        .include(cccl.join("thrust"))
        .include(flashinfer.join("include"))
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .flag("-diag-suppress=20012,177,550")
        .warnings(false)
        .compile("omni_cuda_kernels");
}

fn ensure_submodule(repo: &Path, path: &str, probe: &Path) {
    if probe.exists() {
        return;
    }
    println!("cargo:warning=initializing submodule {path}");
    let status = Command::new("git")
        .args(["submodule", "update", "--init", "--depth", "1", path])
        .current_dir(repo)
        .status()
        .unwrap_or_else(|e| panic!("git is needed to fetch {path}: {e}"));
    assert!(
        status.success() && probe.exists(),
        "could not initialize {path}; run `git submodule update --init {path}`"
    );
}
