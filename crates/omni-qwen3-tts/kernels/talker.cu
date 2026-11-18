// The talker's and the code predictor's kernels besides the Qwen3 stack's (which are `omni-qwen3`'s):
// embedding gathers, the codec track, the frame embedding, the predictor's input and the sampler,
// compiled to one cubin the kern runtime launches from the calls `talker.rs` generates.
//
// Rows are tokens. Per-sequence state (the last frame's codes, the repetition bitmap) is the
// sequence's slot of the per-sequence state, `state + lines[s] * stride`.
//
// Elementwise kernels take eight bf16 (16 bytes) per thread; every width here
// is a multiple of eight.
#include "common.cuh"

constexpr int kGroups = 16;

// out[r] = table[ids[r * ids_stride]], one block of dim / 8 threads per row.
extern "C" __global__ void talker_gather(const int32_t* ids, int ids_stride, const bf16* table, bf16* out, int dim) {
  const int c = threadIdx.x * 8;
  const int64_t id = ids[(int64_t)blockIdx.x * ids_stride];
  *reinterpret_cast<uint4*>(out + (int64_t)blockIdx.x * dim + c) =
      *reinterpret_cast<const uint4*>(table + id * dim + c);
}

// out = table[id], one row.
extern "C" __global__ void talker_embed_id(const bf16* table, int id, bf16* out, int dim) {
  const int c = threadIdx.x * 8;
  *reinterpret_cast<uint4*>(out + c) = *reinterpret_cast<const uint4*>(table + (int64_t)id * dim + c);
}

// x = act(x + bias), rounded to bf16 in between; `total` counts 8-wide groups.
template <bool SILU>
__device__ void bias_act(bf16* x, const bf16* bias, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = i % (cols / 8) * 8;
  float v[8], b[8];
  load8(x + (int64_t)i * 8, v);
  load8(bias + c, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    v[k] = round_bf16(v[k] + b[k]);
    if (SILU) v[k] = v[k] / (1.f + __expf(-v[k]));
  }
  store8(x + (int64_t)i * 8, v);
}

extern "C" __global__ void talker_bias(bf16* x, const bf16* bias, int cols, int total) {
  bias_act<false>(x, bias, cols, total);
}

extern "C" __global__ void talker_bias_silu(bf16* x, const bf16* bias, int cols, int total) {
  bias_act<true>(x, bias, cols, total);
}

// The codec track of a prompt row on top of its projected text:
// x = x + bias + table[ids[r]] (no codec token when the id is negative).
extern "C" __global__ void talker_prompt_codec(bf16* x, const bf16* bias, const bf16* table, const int32_t* ids,
                                               int dim) {
  const int c = threadIdx.x * 8;
  bf16* row = x + (int64_t)blockIdx.x * dim + c;
  const int id = ids[blockIdx.x];
  float v[8], b[8], e[8];
  load8(row, v);
  load8(bias + c, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += b[k];
  if (id >= 0) {
    load8(table + (int64_t)id * dim + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] += e[k];
  }
  store8(row, v);
}

// A running sequence's next input: the text track's pad embedding plus the
// sum of its last frame's sixteen codebook embeddings, codebook 0 from the
// talker's table, 1..15 from the predictor's (`p_emb` [15 * p_vocab, dim]).
extern "C" __global__ void talker_frame_embed(const void* state, const int32_t* lines, int64_t stride,
                                              const bf16* pad, const bf16* codec_emb, const bf16* p_emb, bf16* out,
                                              int dim, int p_vocab) {
  const int s = blockIdx.x, c = threadIdx.x * 8;
  const int32_t* codes = slot<int32_t>(const_cast<void*>(state), lines, s, stride);
  float acc[8], e[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
  load8(pad + c, e);
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] += e[k];
  load8(codec_emb + (int64_t)codes[0] * dim + c, e);
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] += e[k];
  for (int g = 1; g < kGroups; ++g) {
    load8(p_emb + ((int64_t)(g - 1) * p_vocab + codes[g]) * dim + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  store8(out + (int64_t)s * dim + c, acc);
}

// Code-predictor input of codebook 1: per sequence, the talker's hidden state
// then the embedding of the codebook-0 code just drawn (rows 2s and 2s + 1).
extern "C" __global__ void talker_pred_input(const bf16* hidden, const bf16* codec_emb, const int32_t* codes,
                                             bf16* out, int dim) {
  const int s = blockIdx.x, c = threadIdx.x * 8;
  const bf16* src = blockIdx.y == 0 ? hidden + (int64_t)s * dim : codec_emb + (int64_t)codes[s * kGroups] * dim;
  *reinterpret_cast<uint4*>(out + ((int64_t)2 * s + blockIdx.y) * dim + c) =
      *reinterpret_cast<const uint4*>(src + c);
}

// --- Token selection ---------------------------------------------------------
//
// One block per row over a vocabulary of at most 4096, the logits processor
// chain of Hugging Face `generate` fused: repetition penalty and suppression
// on the raw logits, then temperature, top-k, softmax and a draw against the
// row's host-supplied uniform. A temperature of zero takes the argmax (lowest
// index on ties). Top-k keeps every value at or above the k-th largest (ties
// kept, like `TopKLogitsWarper`), found by a radix select over the float bits;
// the draw walks the kept probabilities in vocabulary order.
//
// The drawn (or forced, when `force` is non-negative) code goes to
// `codes[s][group]` and the sequence's slot, whose first sixteen words are its
// last frame and, with `penalize`, the next `ceil(vocab / 32)` the bitmap of
// codes under the repetition penalty.

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
    talker_sample(const bf16* logits, int vocab, float temperature, int top_k, float penalty, int suppress_lo,
                  int suppress_hi, int exempt, int min_frames, const int32_t* frames, const float* uniforms,
                  const int32_t* force, void* state, const int32_t* lines, int64_t stride, int penalize,
                  int32_t* codes, int group) {
  const int s = blockIdx.x, tid = threadIdx.x;
  int32_t* own = slot<int32_t>(state, lines, s, stride);
  uint32_t* seen = reinterpret_cast<uint32_t*>(own + kGroups);
  const int forced = force[s * kGroups + group];
  int token = forced;
  if (forced < 0) {
    const bf16* l = logits + (int64_t)s * vocab;
    const bool exempt_blocked = frames[s] < min_frames;
    float v[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int i = tid * kPer + k;
      float x = -INFINITY;
      if (i < vocab) {
        x = f32(l[i]);
        if (penalize && (seen[i / 32] >> (i % 32) & 1u)) x = x > 0.f ? x / penalty : x * penalty;
        const bool in_range = i >= suppress_lo && i < suppress_hi;
        if ((in_range && i != exempt) || (i == exempt && exempt_blocked)) x = -INFINITY;
      }
      v[k] = x;
    }
    token = min(max(draw(v, vocab, temperature, top_k, uniforms[s * kGroups + group]), 0), vocab - 1);
  }
  __syncthreads();
  if (tid == 0) {
    codes[s * kGroups + group] = token;
    own[group] = token;
    if (penalize) seen[token / 32] |= 1u << (token % 32);
  }
}
