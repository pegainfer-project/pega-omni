//! Declarations of the `csrc/` entry points. Device pointers cross as `u64`
//! addresses; every function returns a `cudaError_t`.

use crate::Ptr;

pub type Stream = *mut std::ffi::c_void;

unsafe extern "C" {
    pub fn cudaGetErrorString(code: i32) -> *const std::ffi::c_char;

    pub fn omni_prefill_cta_tile_q(avg_packed_qo_len: i64, head_dim: u32) -> i32;

    pub fn omni_paged_prefill_hd128(
        q: Ptr,
        q_stride_n: u32,
        out: Ptr,
        k_pool: Ptr,
        v_pool: Ptr,
        page_indices: Ptr,
        page_indptr: Ptr,
        last_page_len: Ptr,
        q_indptr: Ptr,
        request_indices: Ptr,
        qo_tile_indices: Ptr,
        kv_tile_indices: Ptr,
        kv_chunk_size: Ptr,
        total_rows: Ptr,
        num_qo_heads: u32,
        num_kv_heads: u32,
        page_size: u32,
        rows: u32,
        batch: u32,
        tiles: u32,
        cta_tile_q: u32,
        sm_scale: f32,
        stream: Stream,
    ) -> i32;

    pub fn omni_rms_norm(x: Ptr, w: Ptr, out: Ptr, rows: u32, dim: u32, eps: f32, stream: Stream) -> i32;

    pub fn omni_add_rms_norm(x: Ptr, residual: Ptr, w: Ptr, rows: u32, dim: u32, eps: f32, stream: Stream) -> i32;

    pub fn omni_qk_norm_rope(
        qkv: Ptr,
        ld: u32,
        rows: u32,
        hq: u32,
        hk: u32,
        head_dim: u32,
        q_norm: Ptr,
        k_norm: Ptr,
        positions: Ptr,
        slots: Ptr,
        k_pool: Ptr,
        v_pool: Ptr,
        eps: f32,
        theta: f32,
        stream: Stream,
    ) -> i32;

    pub fn omni_silu_mul(gate_up: Ptr, out: Ptr, rows: u32, inter: u32, stream: Stream) -> i32;

    pub fn omni_gather_sum(
        out: Ptr,
        ld_out: u32,
        accumulate: bool,
        bias: Ptr,
        tables: Ptr,
        ids: Ptr,
        ld_ids: u32,
        groups: u32,
        rows: u32,
        dim: u32,
        stream: Stream,
    ) -> i32;

    pub fn omni_bias_act(
        x: Ptr,
        bias: Ptr,
        residual: Ptr,
        out: Ptr,
        act: i32,
        rows: u32,
        cols: u32,
        stream: Stream,
    ) -> i32;

    pub fn omni_sample(
        logits: Ptr,
        ld: u32,
        rows: u32,
        vocab: u32,
        temperature: f32,
        top_k: i32,
        penalty: f32,
        seen: Ptr,
        suppress_lo: u32,
        suppress_hi: u32,
        exempt: i32,
        block_exempt: Ptr,
        uniform: Ptr,
        out: Ptr,
        stream: Stream,
    ) -> i32;
}
