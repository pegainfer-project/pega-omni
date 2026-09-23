//! Compiles `kernels/codec.cu` to the cubin the codec's kern manifest pins.
//! `OMNI_CUDA_ARCH` picks the target (default `103a`, GB300).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let arch = std::env::var("OMNI_CUDA_ARCH").unwrap_or_else(|_| "103a".into());
    println!("cargo:rerun-if-env-changed=OMNI_CUDA_ARCH");
    println!("cargo:rerun-if-changed=kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("codec.cubin");
    let nvcc = cc::Build::new().cuda(true).get_compiler().path().to_path_buf();
    let status = Command::new(&nvcc)
        .args(["-cubin", "-O3", "-std=c++17"])
        .arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .arg("-o")
        .arg(&out)
        .arg("kernels/codec.cu")
        .status()
        .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed on kernels/codec.cu");
}
