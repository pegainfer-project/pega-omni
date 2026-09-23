// The non-GEMM, non-attention pieces of a decoder layer, each fused as far as
// its producer and consumer allow:
//
// - `omni_rms_norm` / `omni_add_rms_norm`: FlashInfer's; the fused form folds
//   the residual add into the norm that follows it.
// - `omni_qk_norm_rope`: per-head RMSNorm of Q and K, rotary embedding, and the
//   K/V write into the paged cache, in one pass over the fused QKV rows.
// - `omni_silu_mul`: the gated MLP activation over fused gate|up rows.
// - `omni_gather_sum`: row gathers from one or more embedding tables, summed,
//   plus an optional bias and accumulator. Prompt assembly, frame embeddings
//   and row selection are all this one kernel.
// - `omni_bias_act`: `act(x + bias) + residual`, the epilogue every biased
//   linear or conv needs.
#include "common.cuh"

#include <flashinfer/norm.cuh>

extern "C" {

int omni_rms_norm(bf16* x, bf16* w, bf16* out, uint32_t rows, uint32_t dim, float eps, cudaStream_t stream) {
  OMNI_TRY({
    cudaError_t e = flashinfer::norm::RMSNorm(x, w, out, rows, dim, dim, dim, eps, false, stream);
    if (e != cudaSuccess) return (int)e;
  })
}

// residual += x; x = rms_norm(residual) * w
int omni_add_rms_norm(bf16* x, bf16* residual, bf16* w, uint32_t rows, uint32_t dim, float eps,
                      cudaStream_t stream) {
  OMNI_TRY({
    cudaError_t e = flashinfer::norm::FusedAddRMSNorm(x, residual, w, rows, dim, dim, dim, eps, false, stream);
    if (e != cudaSuccess) return (int)e;
  })
}

}  // extern "C"

// One warp per head. A lane owns D/64 elements from each half of the head so
// that every rotary pair (i, i + D/2) lives in one lane.
template <int D>
__global__ void qk_norm_rope_kernel(bf16* qkv, uint32_t ld, const bf16* q_norm, const bf16* k_norm,
                                    const int32_t* positions, const int32_t* slots, bf16* k_pool, bf16* v_pool,
                                    int hq, int hk, float eps, float theta) {
  constexpr int E = D / 64;
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= hq + 2 * hk) return;
  bf16* row = qkv + (size_t)n * ld + (size_t)head * D;
  const int32_t slot = slots ? slots[n] : -1;

  if (head >= hq + hk) {
    if (slot >= 0) {
      bf16* dst = v_pool + ((size_t)slot * hk + (head - hq - hk)) * D;
      for (int i = lane; i < D; i += 32) dst[i] = row[i];
    }
    return;
  }

  const bool is_q = head < hq;
  const bf16* w = is_q ? q_norm : k_norm;
  float lo[E], hi[E];
#pragma unroll
  for (int e = 0; e < E; ++e) {
    lo[e] = to_f32(row[lane * E + e]);
    hi[e] = to_f32(row[D / 2 + lane * E + e]);
  }
  if (w) {
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) ss += lo[e] * lo[e] + hi[e] * hi[e];
    const float r = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
    for (int e = 0; e < E; ++e) {
      lo[e] = to_f32(to_bf16(lo[e] * r)) * to_f32(w[lane * E + e]);
      hi[e] = to_f32(to_bf16(hi[e] * r)) * to_f32(w[D / 2 + lane * E + e]);
    }
  }
  const float pos = (float)positions[n];
  bf16* dst = row;
  if (!is_q && slot >= 0) dst = k_pool + ((size_t)slot * hk + (head - hq)) * D;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int i = lane * E + e;
    const float inv_freq = 1.f / powf(theta, (float)(2 * i) / D);
    float s, c;
    sincosf(pos * inv_freq, &s, &c);
    dst[i] = to_bf16(lo[e] * c - hi[e] * s);
    dst[D / 2 + i] = to_bf16(hi[e] * c + lo[e] * s);
  }
}

__global__ void silu_mul_kernel(const bf16* gate_up, bf16* out, uint32_t inter, size_t total) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t n = i / inter, c = i % inter;
    const float g = to_f32(gate_up[n * 2 * inter + c]);
    const float u = to_f32(gate_up[n * 2 * inter + inter + c]);
    out[i] = to_bf16(to_f32(to_bf16(g / (1.f + __expf(-g)))) * u);
  }
}

__global__ void gather_sum_kernel(bf16* out, uint32_t ld_out, bool accumulate, const bf16* bias,
                                  const bf16* const* tables, const int32_t* ids, uint32_t ld_ids, uint32_t groups,
                                  uint32_t dim) {
  const uint32_t n = blockIdx.x;
  bf16* dst = out + (size_t)n * ld_out;
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    float acc = accumulate ? to_f32(dst[c]) : 0.f;
    if (bias) acc += to_f32(bias[c]);
    for (uint32_t g = 0; g < groups; ++g) {
      const int32_t id = ids[(size_t)n * ld_ids + g];
      if (id >= 0) acc += to_f32(tables[g][(size_t)id * dim + c]);
    }
    dst[c] = to_bf16(acc);
  }
}

__device__ __forceinline__ float activate(float x, int act) {
  switch (act) {
    case 1: return x / (1.f + __expf(-x));
    case 2: return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));
    default: return x;
  }
}

__global__ void bias_act_kernel(const bf16* x, const bf16* bias, const bf16* residual, bf16* out, int act,
                                uint32_t cols, size_t total) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    float v = to_f32(x[i]);
    if (bias) v = to_f32(to_bf16(v + to_f32(bias[i % cols])));
    v = activate(v, act);
    if (residual) v = to_f32(to_bf16(v)) + to_f32(residual[i]);
    out[i] = to_bf16(v);
  }
}

static unsigned grid_for(size_t total) {
  const size_t blocks = (total + 255) / 256;
  return (unsigned)(blocks < 65535 * 8 ? blocks : 65535 * 8);
}

extern "C" {

// `qkv` rows hold [hq | hk | hk] heads of `head_dim`. Q is normed and rotated in
// place. With `slots`, K (normed, rotated) and V land in the paged pools at
// `slots[n]`; without, K is rotated in place and V is left alone. Null norm
// weights skip the norm.
int omni_qk_norm_rope(bf16* qkv, uint32_t ld, uint32_t rows, uint32_t hq, uint32_t hk, uint32_t head_dim,
                      const bf16* q_norm, const bf16* k_norm, const int32_t* positions, const int32_t* slots,
                      bf16* k_pool, bf16* v_pool, float eps, float theta, cudaStream_t stream) {
  if (rows == 0) return 0;
  constexpr int warps = 8;
  const dim3 grid(rows, (hq + 2 * hk + warps - 1) / warps);
  switch (head_dim) {
    case 64:
      qk_norm_rope_kernel<64><<<grid, warps * 32, 0, stream>>>(qkv, ld, q_norm, k_norm, positions, slots, k_pool,
                                                               v_pool, hq, hk, eps, theta);
      break;
    case 128:
      qk_norm_rope_kernel<128><<<grid, warps * 32, 0, stream>>>(qkv, ld, q_norm, k_norm, positions, slots, k_pool,
                                                                v_pool, hq, hk, eps, theta);
      break;
    default:
      return (int)cudaErrorInvalidValue;
  }
  return (int)cudaGetLastError();
}

int omni_silu_mul(const bf16* gate_up, bf16* out, uint32_t rows, uint32_t inter, cudaStream_t stream) {
  const size_t total = (size_t)rows * inter;
  if (total == 0) return 0;
  silu_mul_kernel<<<grid_for(total), 256, 0, stream>>>(gate_up, out, inter, total);
  return (int)cudaGetLastError();
}

int omni_gather_sum(bf16* out, uint32_t ld_out, bool accumulate, const bf16* bias, const bf16* const* tables,
                    const int32_t* ids, uint32_t ld_ids, uint32_t groups, uint32_t rows, uint32_t dim,
                    cudaStream_t stream) {
  if (rows == 0) return 0;
  gather_sum_kernel<<<rows, 256, 0, stream>>>(out, ld_out, accumulate, bias, tables, ids, ld_ids, groups, dim);
  return (int)cudaGetLastError();
}

// act: 0 identity, 1 SiLU, 2 exact GELU. `out` may alias `x`.
int omni_bias_act(const bf16* x, const bf16* bias, const bf16* residual, bf16* out, int act, uint32_t rows,
                  uint32_t cols, cudaStream_t stream) {
  const size_t total = (size_t)rows * cols;
  if (total == 0) return 0;
  bias_act_kernel<<<grid_for(total), 256, 0, stream>>>(x, bias, residual, out, act, cols, total);
  return (int)cudaGetLastError();
}

}  // extern "C"
