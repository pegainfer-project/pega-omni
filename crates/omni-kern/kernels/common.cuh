// Helpers every model's kernels share. Eight bf16 are 16 bytes, one
// vector load or store.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float round_bf16(float x) { return f32(to_bf16(x)); }

__device__ __forceinline__ void load8(const bf16* p, float* v) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = f32(e[k]);
}

__device__ __forceinline__ uint4 pack8(const float* v) {
  uint4 u;
  bf16* e = reinterpret_cast<bf16*>(&u);
#pragma unroll
  for (int k = 0; k < 8; ++k) e[k] = to_bf16(v[k]);
  return u;
}

__device__ __forceinline__ void store8(bf16* p, const float* v) { *reinterpret_cast<uint4*>(p) = pack8(v); }

__device__ __forceinline__ void copy8(bf16* dst, const bf16* src) {
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[32];
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  v = lane < (int)(blockDim.x >> 5) ? partial[lane] : 0.f;
  v = warp_sum(v);
  __syncthreads();
  return v;
}

// Sequence `s`'s slot of a per-sequence state: `state + lines[s] * stride`.
template <typename T>
__device__ __forceinline__ T* slot(void* state, const int32_t* lines, int s, int64_t stride) {
  return reinterpret_cast<T*>(static_cast<char*>(state) + (int64_t)lines[s] * stride);
}

// SiLU(gate) * up over fused [gate | up] rows, SiLU rounded to bf16; thread
// `i` of `total` takes eight channels.
__device__ __forceinline__ void silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t n = i / (inter / 8), c = i % (inter / 8) * 8;
  float g[8], u[8];
  load8(gate_up + n * 2 * inter + c, g);
  load8(gate_up + n * 2 * inter + inter + c, u);
#pragma unroll
  for (int k = 0; k < 8; ++k) g[k] = round_bf16(g[k] / (1.f + __expf(-g[k]))) * u[k];
  store8(out + (int64_t)i * 8, g);
}
