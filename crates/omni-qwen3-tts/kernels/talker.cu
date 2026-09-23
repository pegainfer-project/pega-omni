// The talker's and the code predictor's kernels: every piece of a Qwen3 decoder
// step that is not a GEMM (those are cuBLASLt, named by the manifest), and the
// sampler, compiled to one cubin the kern runtime launches from the calls
// `talker.rs` generates.
//
// Rows are tokens. K and V of a row land at its token slot, laid out
// `[slot][K heads | V heads][128]`: a talker slot is a position in the
// sequence's pages of the layer's paged state, a code-predictor slot is
// `seq * span + pos` in a workspace, because the predictor sees each frame on
// its own. Per-sequence state (the last frame's codes, the repetition bitmap)
// is the sequence's slot of the per-sequence state, `state + lines[s] * stride`.
//
// Elementwise kernels take eight bf16 (16 bytes) per thread; every width here
// is a multiple of eight.
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

constexpr int kHead = 128;
constexpr int kGroups = 16;

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ void load8(const bf16* p, float* v) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = f32(e[k]);
}

__device__ __forceinline__ void store8(bf16* p, const float* v) {
  uint4 u;
  bf16* e = reinterpret_cast<bf16*>(&u);
#pragma unroll
  for (int k = 0; k < 8; ++k) e[k] = to_bf16(v[k]);
  *reinterpret_cast<uint4*>(p) = u;
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

__device__ __forceinline__ char* seq_slot(void* state, const int32_t* lines, int s, int64_t stride) {
  return static_cast<char*>(state) + (int64_t)lines[s] * stride;
}

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
    v[k] = f32(to_bf16(v[k] + b[k]));
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
  const int32_t* codes = reinterpret_cast<const int32_t*>(seq_slot(const_cast<void*>(state), lines, s, stride));
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

// FlashInfer's RMSNorm with the residual stream started: res = x,
// out = rms_norm(x) * w. One block of dim / 8 threads per row; `out` or `res`
// may alias `x`.
extern "C" __global__ void talker_norm_copy(const bf16* x, const bf16* w, bf16* out, bf16* res, int dim, float eps) {
  const int64_t at = (int64_t)blockIdx.x * dim + threadIdx.x * 8;
  float v[8], wv[8];
  load8(x + at, v);
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  store8(res + at, v);
  load8(w + threadIdx.x * 8, wv);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = v[k] * r * wv[k];
  store8(out + at, v);
}

// FlashInfer's FusedAddRMSNorm: res += x, then rms_norm(res) * w over the
// unrounded sum. The normed row `n` goes to `out[n / every]` when
// `n % every == which` (a strided selection of rows; `out` may be `x` with
// every = 1).
extern "C" __global__ void talker_add_norm(const bf16* x, bf16* res, const bf16* w, bf16* out, int dim, float eps,
                                           int every, int which) {
  const int n = blockIdx.x;
  const int64_t at = (int64_t)n * dim + threadIdx.x * 8;
  float v[8], rv[8], wv[8];
  load8(x + at, v);
  load8(res + at, rv);
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    v[k] += rv[k];
    ss += v[k] * v[k];
  }
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  store8(res + at, v);
  if (n % every != which) return;
  load8(w + threadIdx.x * 8, wv);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = v[k] * r * wv[k];
  store8(out + (int64_t)(n / every) * dim + threadIdx.x * 8, v);
}

// SiLU(gate) * up over fused [gate | up] rows.
extern "C" __global__ void talker_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t n = i / (inter / 8), c = i % (inter / 8) * 8;
  float g[8], u[8];
  load8(gate_up + n * 2 * inter + c, g);
  load8(gate_up + n * 2 * inter + inter + c, u);
#pragma unroll
  for (int k = 0; k < 8; ++k) g[k] = f32(to_bf16(g[k] / (1.f + __expf(-g[k])))) * u[k];
  store8(out + (int64_t)i * 8, g);
}

// Per-head RMSNorm of Q and K, rotary embedding at `pos`, and K and V to the
// row's slot of `kv`; Q stays in place. One warp per (row, head of q, k or v);
// a lane owns two elements of each half of a head, so a rotary pair shares it.
__device__ __forceinline__ void rope_kv(bf16* qkv, const bf16* q_norm, const bf16* k_norm, int n, int pos,
                                        int64_t slot, bf16* kv, int hq, int hk, float eps, float theta) {
  constexpr int E = kHead / 64;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= hq + 2 * hk) return;
  bf16* row = qkv + ((int64_t)n * (hq + 2 * hk) + head) * kHead;
  bf16* dst = kv + (slot * 2 * hk + head - hq) * kHead;
  if (head >= hq + hk) {
    for (int i = lane; i < kHead; i += 32) dst[i] = row[i];
    return;
  }
  const bf16* w = head < hq ? q_norm : k_norm;
  float lo[E], hi[E];
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    lo[e] = f32(row[lane * E + e]);
    hi[e] = f32(row[kHead / 2 + lane * E + e]);
    ss += lo[e] * lo[e] + hi[e] * hi[e];
  }
  const float r = rsqrtf(warp_sum(ss) / kHead + eps);
  if (head < hq) dst = row;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int i = lane * E + e;
    const float a = f32(to_bf16(lo[e] * r)) * f32(w[i]);
    const float b = f32(to_bf16(hi[e] * r)) * f32(w[kHead / 2 + i]);
    const float inv_freq = 1.f / powf(theta, (float)(2 * i) / kHead);
    float sn, cs;
    sincosf((float)pos * inv_freq, &sn, &cs);
    dst[i] = to_bf16(a * cs - b * sn);
    dst[kHead / 2 + i] = to_bf16(b * cs + a * sn);
  }
}

// Talker rows: position and token slot from the call's inputs.
extern "C" __global__ void talker_rope(bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos,
                                       const int32_t* slots, void* kv, int hq, int hk, float eps, float theta) {
  const int n = blockIdx.x;
  rope_kv(qkv, q_norm, k_norm, n, pos[n], slots[n], static_cast<bf16*>(kv), hq, hk, eps, theta);
}

// Code-predictor rows: `per` consecutive rows per sequence at positions
// `base`, `base + 1`, …, slot `seq * span + pos`.
extern "C" __global__ void talker_rope_dense(bf16* qkv, const bf16* q_norm, const bf16* k_norm, bf16* kv, int per,
                                             int base, int span, int hq, int hk, float eps, float theta) {
  const int n = blockIdx.x, pos = base + n % per;
  rope_kv(qkv, q_norm, k_norm, n, pos, (int64_t)(n / per) * span + pos, kv, hq, hk, eps, theta);
}

constexpr int kAttnWarps = 4;

// Row `n`'s G query heads sharing K/V head `blockIdx.y` against the `len`
// slots `slot_of(0..len)`, causal by construction. Each warp takes tiles of 32
// keys, a lane scoring one key, then the tile's V rows are accumulated with a
// lane owning four dims; the warps' online softmax states merge at the end.
template <int G, typename SlotOf>
__device__ __forceinline__ void attend(const bf16* qkv, const bf16* kv, bf16* out, int n, int len, int hq, int hk,
                                       float scale, SlotOf slot_of) {
  const int kh = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ float qs[G][kHead];
  __shared__ float ms[kAttnWarps][G], ls[kAttnWarps][G];
  __shared__ float os[kAttnWarps][G][kHead];
  const bf16* q = qkv + ((int64_t)n * (hq + 2 * hk) + kh * G) * kHead;
  for (int i = threadIdx.x; i < G * kHead; i += blockDim.x) qs[i / kHead][i % kHead] = f32(q[i]);
  __syncthreads();
  const int64_t per_slot = 2LL * hk * kHead;
  float m[G], l[G], acc[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.f;
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[g][d] = 0.f;
  }
  for (int base = warp * 32; base < len; base += kAttnWarps * 32) {
    const int t = base + lane;
    const bool valid = t < len;
    const int64_t slot = valid ? slot_of(t) : slot_of(base);
    const bf16* k = kv + slot * per_slot + kh * kHead;
    float sc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) sc[g] = 0.f;
#pragma unroll 4
    for (int c = 0; c < kHead; c += 8) {
      float kf[8];
      load8(k + c, kf);
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int e = 0; e < 8; ++e) sc[g] += qs[g][c + e] * kf[e];
    }
    float p[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float s = valid ? sc[g] * scale : -INFINITY;
      const float m2 = fmaxf(m[g], warp_max(s));
      const float corr = __expf(m[g] - m2);
      p[g] = __expf(s - m2);
      l[g] = l[g] * corr + warp_sum(p[g]);
#pragma unroll
      for (int d = 0; d < 4; ++d) acc[g][d] *= corr;
      m[g] = m2;
    }
    const int count = min(32, len - base);
    for (int j = 0; j < count; ++j) {
      const int64_t sj = __shfl_sync(0xffffffffu, slot, j);
      const uint2 u = *reinterpret_cast<const uint2*>(kv + sj * per_slot + (hk + kh) * kHead + lane * 4);
      const bf16* v = reinterpret_cast<const bf16*>(&u);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float pj = __shfl_sync(0xffffffffu, p[g], j);
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[g][d] += pj * f32(v[d]);
      }
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (lane == 0) {
      ms[warp][g] = m[g];
      ls[warp][g] = l[g];
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) os[warp][g][lane * 4 + d] = acc[g][d];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < G * kHead; i += blockDim.x) {
    const int g = i / kHead, d = i % kHead;
    float top = -INFINITY;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) top = fmaxf(top, ms[w][g]);
    float num = 0.f, den = 0.f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) {
      const float e = __expf(ms[w][g] - top);
      num += os[w][g][d] * e;
      den += ls[w][g] * e;
    }
    out[((int64_t)n * hq + kh * G + g) * kHead + d] = to_bf16(num / den);
  }
}

struct PagedSlots {
  const int32_t* pages;
  int page;
  __device__ int64_t operator()(int t) const { return (int64_t)pages[t / page] * page + t % page; }
};

struct DenseSlots {
  int64_t first;
  __device__ int64_t operator()(int t) const { return first + t; }
};

template <typename SlotOf>
__device__ __forceinline__ void attend_any(const bf16* qkv, const bf16* kv, bf16* out, int n, int len, int hq,
                                           int hk, float scale, SlotOf slot_of) {
  switch (hq / hk) {
    case 1: attend<1>(qkv, kv, out, n, len, hq, hk, scale, slot_of); break;
    case 2: attend<2>(qkv, kv, out, n, len, hq, hk, scale, slot_of); break;
    case 4: attend<4>(qkv, kv, out, n, len, hq, hk, scale, slot_of); break;
  }
}

// Talker rows over paged KV: row `n` belongs to sequence `ragged ? seq[n] : n`,
// sits at `pos[n]` and sees positions 0..pos[n], whose pages are
// `pages[indptr[s]..]`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    talker_attend(const bf16* qkv, const void* kv, const int32_t* pos, const int32_t* seq, int ragged,
                  const int32_t* indptr, const int32_t* pages, bf16* out, int hq, int hk, int page, float scale) {
  const int n = blockIdx.x;
  const int s = ragged ? seq[n] : n;
  attend_any(qkv, static_cast<const bf16*>(kv), out, n, pos[n] + 1, hq, hk, scale, PagedSlots{pages + indptr[s], page});
}

// Code-predictor rows over the dense workspace, laid out as `talker_rope_dense`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    talker_attend_dense(const bf16* qkv, const bf16* kv, bf16* out, int per, int base, int span, int hq, int hk,
                        float scale) {
  const int n = blockIdx.x;
  attend_any(qkv, kv, out, n, base + n % per + 1, hq, hk, scale, DenseSlots{(int64_t)(n / per) * span});
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
  int32_t* own = reinterpret_cast<int32_t*>(seq_slot(state, lines, s, stride));
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
