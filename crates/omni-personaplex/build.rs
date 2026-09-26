//! Compiles `kernels/*.cu` to the cubins the model's kern manifest pins.
//! `OMNI_CUDA_ARCH` picks the target (default `103a`, GB300). nvcc is `$NVCC`,
//! else `$CUDA_PATH/bin/nvcc`, else `/usr/local/cuda/bin/nvcc`, else `nvcc` on
//! the `PATH`.

use std::path::PathBuf;
use std::process::Command;

fn nvcc() -> PathBuf {
    let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    var("NVCC")
        .or_else(|| var("CUDA_PATH").map(|p| p.join("bin/nvcc")))
        .or_else(|| Some(PathBuf::from("/usr/local/cuda/bin/nvcc")).filter(|p| p.exists()))
        .unwrap_or_else(|| "nvcc".into())
}

fn main() {
    let arch = std::env::var("OMNI_CUDA_ARCH").unwrap_or_else(|_| "103a".into());
    for var in ["OMNI_CUDA_ARCH", "NVCC", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=../omni-kern/kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let nvcc = nvcc();
    for name in ["lm", "mimi"] {
        let status = Command::new(&nvcc)
            .args(["-cubin", "-O3", "-std=c++17", "-I../omni-kern/kernels"])
            .arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
            .arg("-o")
            .arg(out.join(format!("{name}.cubin")))
            .arg(format!("kernels/{name}.cu"))
            .status()
            .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed on kernels/{name}.cu");
    }
}
