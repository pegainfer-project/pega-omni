// Shared by every kernel file: bf16 conversions and the FFI error convention.
//
// Every extern "C" entry point returns a cudaError_t as int (0 on success) and
// never throws: FlashInfer dispatchers can throw on an unsupported shape, which
// `OMNI_TRY` turns into cudaErrorInvalidValue.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float to_f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }

#define OMNI_TRY(...)                  \
  try {                                \
    __VA_ARGS__;                       \
  } catch (...) {                      \
    return (int)cudaErrorInvalidValue; \
  }                                    \
  return (int)cudaGetLastError();

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over a block of `blockDim.x` threads (a multiple of 32, at most 1024).
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[32];
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = blockDim.x >> 5;
  v = lane < warps ? partial[lane] : 0.f;
  v = warp_sum(v);
  __syncthreads();
  return v;
}
