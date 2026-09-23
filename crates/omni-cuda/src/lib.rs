//! The GPU layer engine crates build on: one device, one stream, device
//! buffers addressed by raw pointer, cuBLAS GEMMs and the fused kernels in
//! `csrc/`.
//!
//! Everything runs in issue order on a single stream, so buffers carry no
//! per-access synchronization: a `Buf` is an allocation plus its device
//! address, and every op takes addresses ([`Ptr`]). Shapes are the caller's
//! contract, stated on each op; the kernels check only what would otherwise be
//! undefined behaviour.
//!
//! Layout conventions: activations are row-major `[rows, cols]` bf16; linear
//! weights keep the PyTorch `[out, in]` layout, so `y = x · wᵀ`.
#![allow(
    clippy::too_many_arguments,
    reason = "a kernel launch takes its shapes and pointers; a struct per call would only move the list"
)]

use std::ffi::CStr;
use std::sync::Arc;

use anyhow::Result;
use anyhow::bail;
use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys as blas;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
pub use half::bf16;

mod ffi;

/// A device address. Offsets are in bytes; use [`Buf::at`] for element offsets.
pub type Ptr = u64;

/// A device allocation that owns its memory and knows its address.
pub struct Buf<T> {
    slice: CudaSlice<T>,
    ptr: Ptr,
}

impl<T> Buf<T> {
    pub fn ptr(&self) -> Ptr {
        self.ptr
    }

    /// Address of element `i`.
    pub fn at(&self, i: usize) -> Ptr {
        self.ptr + (i * size_of::<T>()) as u64
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.len() == 0
    }
}

/// Post-bias activation of [`Gpu::bias_act`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None = 0,
    Silu = 1,
    Gelu = 2,
}

/// The draw of [`Gpu::sample`]: one setting for every row, per-row device
/// arrays for what differs between rows.
pub struct SampleArgs {
    /// Zero picks the argmax.
    pub temperature: f32,
    /// Zero keeps every token.
    pub top_k: i32,
    /// Repetition penalty on the tokens set in `seen`, a per-row bitmap of
    /// `ceil(vocab/32)` u32 (0: no penalty).
    pub penalty: f32,
    pub seen: Ptr,
    /// Token range masked out, except `exempt`; rows flagged in `block_exempt`
    /// (u8, may be 0) mask `exempt` too.
    pub suppress: (u32, u32),
    pub exempt: i32,
    pub block_exempt: Ptr,
    pub uniform: Ptr,
}

/// Paged-KV attention inputs of [`Gpu::paged_prefill`]; see [`PrefillPlan`].
pub struct PagedKv {
    pub k_pool: Ptr,
    pub v_pool: Ptr,
    pub page_size: u32,
    pub page_indices: Ptr,
    pub page_indptr: Ptr,
    pub last_page_len: Ptr,
}

/// FlashInfer's tile schedule for one ragged batch: which request and which
/// query tile each thread block takes. Built on the host from query lengths,
/// uploaded into [`PlanBufs`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub cta_tile_q: u32,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
}

impl PrefillPlan {
    pub fn new(qo_lens: &[u32], group: u32, head_dim: u32) -> Self {
        let rows: u64 = qo_lens.iter().map(|&l| l as u64).sum();
        let avg = (rows * group as u64 / qo_lens.len().max(1) as u64) as i64;
        let cta_tile_q = unsafe { ffi::omni_prefill_cta_tile_q(avg, head_dim) } as u32;
        let (request_indices, qo_tile_indices) = qo_lens
            .iter()
            .enumerate()
            .flat_map(|(r, &len)| (0..(len * group).div_ceil(cta_tile_q)).map(move |t| (r as i32, t as i32)))
            .unzip();
        Self { cta_tile_q, request_indices, qo_tile_indices }
    }

    pub fn tiles(&self) -> usize {
        self.request_indices.len()
    }
}

/// A [`PrefillPlan`] on the device, as FlashInfer reads it: `q_indptr`
/// ([batch + 1]), `request_indices`, `qo_tile_indices`, `kv_tile_indices`
/// (zeros), each sized for the largest plan, the two scalars it dereferences,
/// and the plan's shape.
pub struct PlanBufs {
    pub q_indptr: Buf<i32>,
    pub request_indices: Buf<i32>,
    pub qo_tile_indices: Buf<i32>,
    pub kv_tile_indices: Buf<i32>,
    pub kv_chunk_size: Buf<i32>,
    pub total_rows: Buf<u32>,
    pub cta_tile_q: u32,
    pub tiles: usize,
}

impl PlanBufs {
    pub fn new(gpu: &Gpu, max_batch: usize, max_tiles: usize) -> Result<Self> {
        Ok(Self {
            q_indptr: gpu.alloc(max_batch + 1)?,
            request_indices: gpu.alloc(max_tiles)?,
            qo_tile_indices: gpu.alloc(max_tiles)?,
            kv_tile_indices: gpu.alloc(max_tiles)?,
            kv_chunk_size: gpu.alloc(1)?,
            total_rows: gpu.alloc(1)?,
            cta_tile_q: 0,
            tiles: 0,
        })
    }

    /// Uploads `plan` for a batch whose query prefix sums are `q_indptr`.
    pub fn set(&mut self, gpu: &Gpu, plan: &PrefillPlan, q_indptr: &[i32]) -> Result<()> {
        if plan.tiles() > self.request_indices.len() || q_indptr.len() > self.q_indptr.len() {
            bail!("plan of {} tiles exceeds its buffers", plan.tiles());
        }
        gpu.write(&mut self.q_indptr, 0, q_indptr)?;
        gpu.write(&mut self.request_indices, 0, &plan.request_indices)?;
        gpu.write(&mut self.qo_tile_indices, 0, &plan.qo_tile_indices)?;
        gpu.write(&mut self.total_rows, 0, &[*q_indptr.last().unwrap_or(&0) as u32])?;
        (self.cta_tile_q, self.tiles) = (plan.cta_tile_q, plan.tiles());
        Ok(())
    }
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

fn check(code: i32, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(code)) };
    bail!("{what}: {} ({code})", msg.to_string_lossy())
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        // One stream: issue order is the only ordering buffers need.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Self { stream, blas })
    }

    /// Makes this device current on the calling thread; kernels launch through
    /// the runtime API, which reads the thread's current context.
    pub fn bind(&self) -> Result<()> {
        Ok(self.stream.context().bind_to_thread()?)
    }

    fn s(&self) -> ffi::Stream {
        self.stream.cu_stream() as ffi::Stream
    }

    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<Buf<T>> {
        self.own(self.stream.alloc_zeros(len.max(1))?)
    }

    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<Buf<T>> {
        if host.is_empty() {
            bail!("upload of an empty slice");
        }
        self.own(self.stream.clone_htod(host)?)
    }

    fn own<T>(&self, slice: CudaSlice<T>) -> Result<Buf<T>> {
        let ptr = slice.device_ptr(&self.stream).0;
        Ok(Buf { slice, ptr })
    }

    /// Copies `host` into `buf` starting at element `offset`.
    pub fn write<T: DeviceRepr>(&self, buf: &mut Buf<T>, offset: usize, host: &[T]) -> Result<()> {
        if host.is_empty() {
            return Ok(());
        }
        let mut view = buf.slice.slice_mut(offset..offset + host.len());
        self.stream.memcpy_htod(host, &mut view)?;
        Ok(())
    }

    /// Copies `len` elements from `ptr` to the host, waiting for the stream.
    pub fn read<T: DeviceRepr + Default + Clone>(&self, ptr: Ptr, len: usize) -> Result<Vec<T>> {
        let mut host = vec![T::default(); len];
        if len > 0 {
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    host.as_mut_ptr().cast(),
                    ptr,
                    len * size_of::<T>(),
                    self.stream.cu_stream(),
                )
            };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                bail!("device to host copy: {r:?}");
            }
        }
        self.sync()?;
        Ok(host)
    }

    pub fn copy(&self, dst: Ptr, src: Ptr, bytes: usize) -> Result<()> {
        let r = unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream.cu_stream()) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            bail!("device copy: {r:?}");
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    /// `y[m, n] = x[m, k] · w[n, k]ᵀ`, contiguous, bf16 in and out, f32 accumulate.
    pub fn linear(&self, y: Ptr, x: Ptr, w: Ptr, m: usize, n: usize, k: usize) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let (alpha, beta) = (1f32, 0f32);
        let r = unsafe {
            blas::cublasGemmEx(
                *self.blas.handle(),
                blas::cublasOperation_t::CUBLAS_OP_T,
                blas::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                w as *const _,
                blas::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                x as *const _,
                blas::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta as *const f32).cast(),
                y as *mut _,
                blas::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if r != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!("cublasGemmEx m={m} n={n} k={k}: {r:?}");
        }
        Ok(())
    }

    pub fn rms_norm(&self, x: Ptr, w: Ptr, out: Ptr, rows: usize, dim: usize, eps: f32) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        check(unsafe { ffi::omni_rms_norm(x, w, out, rows as u32, dim as u32, eps, self.s()) }, "rms_norm")
    }

    /// `residual += x; x = rms_norm(residual) · w`.
    pub fn add_rms_norm(&self, x: Ptr, residual: Ptr, w: Ptr, rows: usize, dim: usize, eps: f32) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        check(unsafe { ffi::omni_add_rms_norm(x, residual, w, rows as u32, dim as u32, eps, self.s()) }, "add_rms_norm")
    }

    /// Normalizes (when norm weights are non-zero) and rotates the Q and K heads
    /// of `qkv` rows (`[hq | hk | hk]` heads of `head_dim`, stride `ld`). With
    /// `slots` (i32 per row), K and V are written to the paged pools at that
    /// token slot; with `slots == 0`, K is rotated in place.
    pub fn qk_norm_rope(
        &self,
        qkv: Ptr,
        ld: usize,
        rows: usize,
        (hq, hk, head_dim): (usize, usize, usize),
        (q_norm, k_norm, eps): (Ptr, Ptr, f32),
        positions: Ptr,
        (slots, k_pool, v_pool): (Ptr, Ptr, Ptr),
        theta: f32,
    ) -> Result<()> {
        let code = unsafe {
            ffi::omni_qk_norm_rope(
                qkv,
                ld as u32,
                rows as u32,
                hq as u32,
                hk as u32,
                head_dim as u32,
                q_norm,
                k_norm,
                positions,
                slots,
                k_pool,
                v_pool,
                eps,
                theta,
                self.s(),
            )
        };
        check(code, "qk_norm_rope")
    }

    /// `out[r] = silu(gate_up[r, :inter]) · gate_up[r, inter:]`.
    pub fn silu_mul(&self, gate_up: Ptr, out: Ptr, rows: usize, inter: usize) -> Result<()> {
        check(unsafe { ffi::omni_silu_mul(gate_up, out, rows as u32, inter as u32, self.s()) }, "silu_mul")
    }

    /// `out[r] = (accumulate ? out[r] : 0) + bias + Σ_g tables[g][ids[r, g]]` over
    /// `dim` columns; `tables` is a device array of `groups` table addresses,
    /// ids below zero contribute nothing, `bias` may be 0.
    pub fn gather_sum(
        &self,
        (out, ld_out): (Ptr, usize),
        accumulate: bool,
        bias: Ptr,
        tables: Ptr,
        (ids, ld_ids, groups): (Ptr, usize, usize),
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let code = unsafe {
            ffi::omni_gather_sum(
                out,
                ld_out as u32,
                accumulate,
                bias,
                tables,
                ids,
                ld_ids as u32,
                groups as u32,
                rows as u32,
                dim as u32,
                self.s(),
            )
        };
        check(code, "gather_sum")
    }

    /// `out = act(x + bias) + residual`; `bias` and `residual` may be 0, `out` may be `x`.
    pub fn bias_act(
        &self,
        x: Ptr,
        bias: Ptr,
        residual: Ptr,
        out: Ptr,
        act: Act,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let code =
            unsafe { ffi::omni_bias_act(x, bias, residual, out, act as i32, rows as u32, cols as u32, self.s()) };
        check(code, "bias_act")
    }

    /// Causal attention over a ragged batch against paged K/V (head_dim 128).
    /// `q` rows have stride `q_stride`; `plan` describes the batch.
    pub fn paged_prefill(
        &self,
        (q, q_stride): (Ptr, usize),
        out: Ptr,
        kv: &PagedKv,
        plan: &PlanBufs,
        (rows, batch): (usize, usize),
        (hq, hk): (usize, usize),
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let code = unsafe {
            ffi::omni_paged_prefill_hd128(
                q,
                q_stride as u32,
                out,
                kv.k_pool,
                kv.v_pool,
                kv.page_indices,
                kv.page_indptr,
                kv.last_page_len,
                plan.q_indptr.ptr(),
                plan.request_indices.ptr(),
                plan.qo_tile_indices.ptr(),
                plan.kv_tile_indices.ptr(),
                plan.kv_chunk_size.ptr(),
                plan.total_rows.ptr(),
                hq as u32,
                hk as u32,
                kv.page_size,
                rows as u32,
                batch as u32,
                plan.tiles as u32,
                plan.cta_tile_q,
                (128f32).powf(-0.5),
                self.s(),
            )
        };
        check(code, "paged_prefill")
    }

    /// Causal self-attention of one sequence with a `window`-key sliding window
    /// (head_dim 64); `q`, `k`, `v` rows share stride `stride`.
    pub fn window_prefill(
        &self,
        (q, k, v): (Ptr, Ptr, Ptr),
        stride: usize,
        out: Ptr,
        len: usize,
        heads: usize,
        window: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let code = unsafe {
            ffi::omni_window_prefill_hd64(
                q,
                k,
                v,
                out,
                len as u32,
                heads as u32,
                stride as u32,
                window as i32 - 1,
                (64f32).powf(-0.5),
                self.s(),
            )
        };
        check(code, "window_prefill")
    }

    /// Picks one token per row from bf16 logits (`rows × vocab`, row stride `ld`).
    pub fn sample(
        &self,
        (logits, ld): (Ptr, usize),
        rows: usize,
        vocab: usize,
        args: &SampleArgs,
        out: Ptr,
    ) -> Result<()> {
        let code = unsafe {
            ffi::omni_sample(
                logits,
                ld as u32,
                rows as u32,
                vocab as u32,
                args.temperature,
                args.top_k,
                args.penalty,
                args.seen,
                args.suppress.0,
                args.suppress.1,
                args.exempt,
                args.block_exempt,
                args.uniform,
                out,
                self.s(),
            )
        };
        check(code, "sample")
    }

    /// `col[t, j·C + c] = snake(x[t − (K−1−j)·dilation, c] + bias[c])`, zero before
    /// the start; `bias` and the snake parameters (`a`, `inv_b`, f32) may be 0.
    pub fn im2col(
        &self,
        x: Ptr,
        bias: Ptr,
        (a, inv_b): (Ptr, Ptr),
        col: Ptr,
        (t, c): (usize, usize),
        k: usize,
        dilation: usize,
    ) -> Result<()> {
        let code = unsafe {
            ffi::omni_im2col(x, bias, a, inv_b, col, t as u32, c as u32, k as u32, dilation as u32, self.s())
        };
        check(code, "im2col")
    }

    /// Transposed-conv overlap-add: `z` is `[l, K·C]` tap-major, `out` is `[l·stride, C]`.
    pub fn col2im(&self, z: Ptr, bias: Ptr, out: Ptr, (l, c): (usize, usize), k: usize, stride: usize) -> Result<()> {
        let code = unsafe { ffi::omni_col2im(z, bias, out, l as u32, c as u32, k as u32, stride as u32, self.s()) };
        check(code, "col2im")
    }

    /// Depthwise causal conv (`w` `[C, K]`, bias `b`) then LayerNorm (`ln_w`, `ln_b`).
    pub fn dwconv_layernorm(
        &self,
        x: Ptr,
        (w, b): (Ptr, Ptr),
        (ln_w, ln_b): (Ptr, Ptr),
        out: Ptr,
        (t, c): (usize, usize),
        k: usize,
        eps: f32,
    ) -> Result<()> {
        let code = unsafe {
            ffi::omni_dwconv_layernorm(x, w, b, ln_w, ln_b, out, t as u32, c as u32, k as u32, eps, self.s())
        };
        check(code, "dwconv_layernorm")
    }

    /// SnakeBeta, the `C → 1` causal conv (`w` `[K, C]` tap-major) and a clamp to `[-1, 1]`, into f32 samples.
    pub fn conv_out(
        &self,
        x: Ptr,
        (a, inv_b): (Ptr, Ptr),
        (w, bias): (Ptr, Ptr),
        out: Ptr,
        (t, c): (usize, usize),
        k: usize,
    ) -> Result<()> {
        let code = unsafe { ffi::omni_conv_out(x, a, inv_b, w, bias, out, t as u32, c as u32, k as u32, self.s()) };
        check(code, "conv_out")
    }
}
