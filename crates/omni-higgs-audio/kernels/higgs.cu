// The kernels around the Qwen3 body (which is `omni-qwen3`'s): the embedding gather, the fused
// eight-codebook embedding and the code draws, compiled to one cubin the kern runtime launches from
// the calls `model.rs` generates.
//
// Rows are tokens. Per-sequence state (the last frame's codes) is the sequence's slot of the
// per-sequence state, `state + lines[s] * stride`.
//
// Elementwise kernels take eight bf16 (16 bytes) per thread; every width here is a multiple of eight.
#include "common.cuh"

constexpr int kBooks = 8;

// out[r] = table[ids[r * ids_stride]], one block of dim / 8 threads per row.
extern "C" __global__ void higgs_gather(const int32_t* ids, int ids_stride, const bf16* table, bf16* out, int dim) {
  const int c = threadIdx.x * 8;
  const int64_t id = ids[(int64_t)blockIdx.x * ids_stride];
  *reinterpret_cast<uint4*>(out + (int64_t)blockIdx.x * dim + c) =
      *reinterpret_cast<const uint4*>(table + id * dim + c);
}

// A running sequence's next input: the sum of its last frame's eight codebook embeddings, codebook
// `b` at row `b * vocab + code` of the fused table, added in f32 and rounded once to bf16 like the
// reference's `sum`.
extern "C" __global__ void higgs_frame_embed(const void* state, const int32_t* lines, int64_t stride, const bf16* table,
                                             bf16* out, int dim, int vocab) {
  const int s = blockIdx.x, c = threadIdx.x * 8;
  const int32_t* codes = slot<int32_t>(const_cast<void*>(state), lines, s, stride);
  float acc[8], e[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
  for (int b = 0; b < kBooks; ++b) {
    load8(table + ((int64_t)b * vocab + codes[b]) * dim + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  store8(out + (int64_t)s * dim + c, acc);
}

// --- Code draws -----------------------------------------------------------------------------
//
// One block per (sequence, codebook) over the codebook's 1026 logits: temperature, top-k, softmax
// and a draw against the row's host-supplied uniform. A temperature of zero takes the argmax (lowest
// index on ties). Top-k keeps every value at or above the k-th largest (ties kept), found by a radix
// select over the float bits; the draw walks the kept probabilities in code order. Every codebook
// draws independently, as the reference does; the delay pattern reaches this kernel as `force`.
//
// The drawn (or forced, when `force` is non-negative) code goes to `codes[s][b]` and the first
// eight words of the sequence's slot.

constexpr int kSampleThreads = 1024;
constexpr int kPer = 4;

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float block_max(float v) {
  __shared__ float partial[32];
  v = warp_max(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  v = lane < (int)(blockDim.x >> 5) ? partial[lane] : -INFINITY;
  v = warp_max(v);
  __syncthreads();
  return v;
}

// Exclusive prefix of `v` over the block, and the total.
__device__ __forceinline__ float block_scan(float v, float* total) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) partial[warp] = x;
  __syncthreads();
  if (warp == 0) {
    float w = lane < (int)(blockDim.x >> 5) ? partial[lane] : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    partial[lane] = w;
  }
  __syncthreads();
  const float before = warp > 0 ? partial[warp - 1] : 0.f;
  *total = partial[(blockDim.x >> 5) - 1];
  __syncthreads();
  return before + x - v;
}

__device__ int draw(const float* v, int vocab, float temperature, int top_k, float u) {
  const int tid = threadIdx.x;
  __shared__ uint32_t hist[256];
  __shared__ uint32_t sel_prefix, sel_left;
  __shared__ int token, last;

  if (temperature <= 0.f) {
    __shared__ unsigned long long best;
    if (tid == 0) best = 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int i = tid * kPer + k;
      if (i < vocab) atomicMax(&best, (unsigned long long)order_key(v[k]) << 32 | (uint32_t)(4095 - i));
    }
    __syncthreads();
    return 4095 - (int)(best & 0xffffffffu);
  }

  float x[kPer];
  uint32_t key[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    x[k] = v[k] / temperature;
    key[k] = order_key(x[k]);
  }
  const int k_keep = top_k > 0 && top_k < vocab ? top_k : vocab;
  uint32_t prefix = 0, left = k_keep;
  if (k_keep < vocab) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
      __syncthreads();
      const uint32_t high = shift == 24 ? 0u : 0xffffffffu << (shift + 8);
#pragma unroll
      for (int k = 0; k < kPer; ++k)
        if (tid * kPer + k < vocab && ((key[k] ^ prefix) & high) == 0) atomicAdd(&hist[(key[k] >> shift) & 255], 1u);
      __syncthreads();
      if (tid < 32) {
        uint32_t c[8], sum = 0;
#pragma unroll
        for (int b = 0; b < 8; ++b) sum += c[b] = hist[tid * 8 + b];
        uint32_t upto = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_down_sync(0xffffffffu, upto, o);
          if (tid + o < 32) upto += y;
        }
        uint32_t above = upto - sum;
#pragma unroll
        for (int b = 7; b >= 0; --b) {
          if (above < left && above + c[b] >= left) {
            sel_prefix = prefix | (uint32_t)(tid * 8 + b) << shift;
            sel_left = left - above;
          }
          above += c[b];
        }
      }
      __syncthreads();
      prefix = sel_prefix;
      left = sel_left;
    }
  }
  const float m = block_max(fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3])));
  float p[kPer], own = 0.f;
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    p[k] = tid * kPer + k < vocab && key[k] >= prefix ? __expf(x[k] - m) : 0.f;
    own += p[k];
  }
  if (tid == 0) {
    token = -1;
    last = 0;
  }
  float total;
  float run = block_scan(own, &total);
  const float target = u * total;
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    if (p[k] > 0.f) {
      atomicMax(&last, tid * kPer + k);
      if (run <= target && target < run + p[k]) token = tid * kPer + k;
    }
    run += p[k];
  }
  __syncthreads();
  return token >= 0 ? token : last;
}

extern "C" __global__ void __launch_bounds__(kSampleThreads)
    higgs_sample(const bf16* logits, int vocab, float temperature, int top_k, const float* uniforms,
                 const int32_t* force, void* state, const int32_t* lines, int64_t stride, int32_t* codes) {
  const int s = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  int32_t* own = slot<int32_t>(state, lines, s, stride);
  const int forced = force[s * kBooks + b];
  int token = forced;
  if (forced < 0) {
    const bf16* l = logits + ((int64_t)s * kBooks + b) * vocab;
    float v[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int i = tid * kPer + k;
      v[k] = i < vocab ? f32(l[i]) : -INFINITY;
    }
    token = min(max(draw(v, vocab, temperature, top_k, uniforms[s * kBooks + b]), 0), vocab - 1);
  }
  __syncthreads();
  if (tid == 0) {
    codes[s * kBooks + b] = token;
    own[b] = token;
  }
}
