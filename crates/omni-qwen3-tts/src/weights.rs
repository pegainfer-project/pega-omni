//! Reading checkpoint tensors into device buffers.
//!
//! Tensors arrive as bf16 or f32 and all leave as bf16, except parameters that
//! a kernel reads in f32 (SnakeBeta's). Layout transforms (fusing projections,
//! permuting conv kernels, folding scales) happen on the host at load, so the
//! forward pass never pays for them.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use memmap2::Mmap;
use omni_cuda::Buf;
use omni_cuda::Gpu;
use omni_cuda::bf16;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

/// A host tensor: shape and values.
pub struct Host {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct File {
    map: Mmap,
    start: usize,
    meta: Metadata,
}

impl File {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let (header, meta) = SafeTensors::read_metadata(&map)?;
        Ok(Self { map, start: 8 + header, meta })
    }

    fn raw(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8])> {
        let info = self.meta.info(name).with_context(|| format!("tensor {name} missing"))?;
        let (a, b) = info.data_offsets;
        Ok((info.dtype, info.shape.clone(), &self.map[self.start + a..self.start + b]))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.meta.info(name).with_context(|| format!("tensor {name} missing"))?.shape.clone())
    }

    pub fn host(&self, name: &str) -> Result<Host> {
        let (dtype, shape, bytes) = self.raw(name)?;
        let data = match dtype {
            Dtype::F32 => bytes.as_chunks().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
            Dtype::BF16 => bytes.as_chunks().0.iter().map(|&b| bf16::from_le_bytes(b).to_f32()).collect(),
            other => bail!("tensor {name}: unsupported dtype {other:?}"),
        };
        Ok(Host { shape, data })
    }

    /// An integer tensor, narrowed to i32.
    pub fn ints(&self, name: &str) -> Result<Vec<i32>> {
        let (dtype, _, bytes) = self.raw(name)?;
        match dtype {
            Dtype::I64 => Ok(bytes.as_chunks().0.iter().map(|&b| i64::from_le_bytes(b) as i32).collect()),
            Dtype::I32 => Ok(bytes.as_chunks().0.iter().map(|&b| i32::from_le_bytes(b)).collect()),
            other => bail!("tensor {name}: expected integers, got {other:?}"),
        }
    }

    /// A tensor of the given shape.
    pub fn expect(&self, name: &str, shape: &[usize]) -> Result<Host> {
        let t = self.host(name)?;
        ensure!(t.shape == shape, "tensor {name}: expected shape {shape:?}, got {:?}", t.shape);
        Ok(t)
    }
}

pub fn upload(gpu: &Gpu, data: &[f32]) -> Result<Buf<bf16>> {
    gpu.upload(&data.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>())
}

/// Row-concatenation of `[out_i, in]` matrices sharing `in`.
pub fn concat_rows(parts: &[Host]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.data.iter().copied()).collect()
}

/// A conv kernel `[out, in, k]` as the `[out, k·in]` tap-major matrix `im2col` rows multiply.
pub fn conv_taps(w: &Host) -> Vec<f32> {
    let (o, i, k) = (w.shape[0], w.shape[1], w.shape[2]);
    (0..o)
        .flat_map(|r| (0..k).flat_map(move |j| (0..i).map(move |c| (r, j, c))))
        .map(|(r, j, c)| w.data[(r * i + c) * k + j])
        .collect()
}

/// A transposed-conv kernel `[in, out, k]` as the `[k·out, in]` matrix whose
/// product with `[l, in]` rows `col2im` overlap-adds.
pub fn transposed_taps(w: &Host) -> Vec<f32> {
    let (i, o, k) = (w.shape[0], w.shape[1], w.shape[2]);
    (0..k)
        .flat_map(|j| (0..o).flat_map(move |r| (0..i).map(move |c| (j, r, c))))
        .map(|(j, r, c)| w.data[(c * o + r) * k + j])
        .collect()
}

/// Scales row `r` of a `[rows, cols]` matrix (or vector, `cols = 1`) by `s[r]`.
pub fn scale_rows(data: &[f32], s: &[f32]) -> Vec<f32> {
    let cols = data.len() / s.len();
    data.iter().enumerate().map(|(i, &x)| x * s[i / cols]).collect()
}
