//! Compiles `kernels/*.cu` to the cubins the model's kern manifest pins.
//! `OMNI_CUDA_ARCH` picks the target (default `103a`, GB300).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let arch = std::env::var("OMNI_CUDA_ARCH").unwrap_or_else(|_| "103a".into());
    println!("cargo:rerun-if-env-changed=OMNI_CUDA_ARCH");
    println!("cargo:rerun-if-changed=kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let nvcc = cc::Build::new().cuda(true).get_compiler().path().to_path_buf();
    for name in ["codec", "talker"] {
        let status = Command::new(&nvcc)
            .args(["-cubin", "-O3", "-std=c++17"])
            .arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
            .arg("-o")
            .arg(out.join(format!("{name}.cubin")))
            .arg(format!("kernels/{name}.cu"))
            .status()
            .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed on kernels/{name}.cu");
    }
}
