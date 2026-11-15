// The Qwen3 decoder stack's kernels: every piece of a layer that is not a GEMM (those are cuBLASLt,
// named by the manifest), compiled to one cubin the kern runtime launches from the calls
// `stack.rs` generates.
//
// Rows are tokens. K and V of a row land at its token slot, laid out
// `[slot][K heads | V heads][128]`: a paged slot is a position in the sequence's pages of the
// layer's paged state, a dense slot is `seq * span + pos` in a workspace.
//
// Elementwise kernels take eight bf16 (16 bytes) per thread; every width here is a multiple of
// eight.
#include "common.cuh"

constexpr int kHead = 128;

// RMSNorm that starts the residual stream: res = x, out = x *
// rsqrt(mean(x²) + eps) * w in f32. One block of dim / 8 threads per row;
// `out` or `res` may alias `x`.
extern "C" __global__ void qwen3_norm_copy(const bf16* x, const bf16* w, bf16* out, bf16* res, int dim, float eps) {
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

// Residual add then RMSNorm: res += x, stored as bf16; out = res *
// rsqrt(mean(res²) + eps) * w over the unrounded f32 sum. The normed row `n`
// goes to `out[n / every]` when `n % every == which` (a strided selection of
// rows; `out` may be `x` with every = 1).
extern "C" __global__ void qwen3_add_norm(const bf16* x, bf16* res, const bf16* w, bf16* out, int dim, float eps,
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

extern "C" __global__ void qwen3_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
  silu_mul(gate_up, out, inter, total);
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

// Paged rows: position and token slot from the call's inputs.
extern "C" __global__ void qwen3_rope(bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos,
                                       const int32_t* slots, void* kv, int hq, int hk, float eps, float theta) {
  const int n = blockIdx.x;
  rope_kv(qkv, q_norm, k_norm, n, pos[n], slots[n], static_cast<bf16*>(kv), hq, hk, eps, theta);
}

// Dense rows: `per` consecutive rows per sequence at positions
// `base`, `base + 1`, …, slot `seq * span + pos`.
extern "C" __global__ void qwen3_rope_dense(bf16* qkv, const bf16* q_norm, const bf16* k_norm, bf16* kv, int per,
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

// Paged rows: row `n` belongs to sequence `ragged ? seq[n] : n`,
// sits at `pos[n]` and sees positions 0..pos[n], whose pages are
// `pages[indptr[s]..]`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    qwen3_attend(const bf16* qkv, const void* kv, const int32_t* pos, const int32_t* seq, int ragged,
                  const int32_t* indptr, const int32_t* pages, int page, bf16* out, int hq, int hk, float scale) {
  const int n = blockIdx.x;
  const int s = ragged ? seq[n] : n;
  attend_any(qkv, static_cast<const bf16*>(kv), out, n, pos[n] + 1, hq, hk, scale, PagedSlots{pages + indptr[s], page});
}

// Dense rows over a workspace, laid out as `qwen3_rope_dense`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    qwen3_attend_dense(const bf16* qkv, const bf16* kv, int per, int base, int span, bf16* out, int hq, int hk,
                        float scale) {
  const int n = blockIdx.x;
  attend_any(qkv, kv, out, n, base + n % per + 1, hq, hk, scale, DenseSlots{(int64_t)(n / per) * span});
}
