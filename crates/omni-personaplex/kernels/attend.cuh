// Single-row attention over a set of key slots, shared by Helium (paged KV
// ring), the depformer (a dense workspace per frame) and Mimi (a ring in the
// session's state).
//
// A slot holds `[K heads | V heads][D]` for one position. Row `n`'s query
// head `h` (in `q`, `[rows][heads][D]` at `q_stride` per row) attends to
// `len` slots `slot_of(0..len)` with the head's K and V at `kv + slot *
// 2 * heads * D`. One block of four warps per (row, head): each warp takes
// tiles of 32 keys, a lane scoring one key, then accumulates the tile's V
// rows with a lane owning D / 32 dims; the warps' online-softmax states merge
// at the end. Order of slots does not matter (softmax is a sum).
#pragma once
#include "common.cuh"

constexpr int kAttnWarps = 4;

template <int D, typename SlotOf>
__device__ __forceinline__ void attend_row(const bf16* q, const bf16* kv, bf16* out, int heads, int len, float scale,
                                           SlotOf slot_of) {
  constexpr int E = D / 32;
  const int h = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ float qs[D];
  __shared__ float ms[kAttnWarps], ls[kAttnWarps];
  __shared__ float os[kAttnWarps][D];
  for (int i = threadIdx.x; i < D; i += blockDim.x) qs[i] = f32(q[h * D + i]);
  __syncthreads();
  const int64_t per_slot = 2LL * heads * D;
  float m = -INFINITY, l = 0.f, acc[E];
#pragma unroll
  for (int d = 0; d < E; ++d) acc[d] = 0.f;
  for (int base = warp * 32; base < len; base += kAttnWarps * 32) {
    const int t = base + lane;
    const bool valid = t < len;
    const int64_t slot = slot_of(valid ? t : base);
    const bf16* k = kv + slot * per_slot + h * D;
    float sc = 0.f;
#pragma unroll 4
    for (int c = 0; c < D; c += 8) {
      float kf[8];
      load8(k + c, kf);
#pragma unroll
      for (int e = 0; e < 8; ++e) sc += qs[c + e] * kf[e];
    }
    const float s = valid ? sc * scale : -INFINITY;
    const float m2 = fmaxf(m, warp_max(s));
    const float corr = __expf(m - m2);
    const float p = __expf(s - m2);
    l = l * corr + warp_sum(p);
#pragma unroll
    for (int d = 0; d < E; ++d) acc[d] *= corr;
    m = m2;
    const int count = min(32, len - base);
    for (int j = 0; j < count; ++j) {
      const int64_t sj = __shfl_sync(0xffffffffu, slot, j);
      const float pj = __shfl_sync(0xffffffffu, p, j);
      const bf16* v = kv + sj * per_slot + (heads + h) * D + lane * E;
#pragma unroll
      for (int d = 0; d < E; ++d) acc[d] += pj * f32(v[d]);
    }
  }
  if (lane == 0) {
    ms[warp] = m;
    ls[warp] = l;
  }
#pragma unroll
  for (int d = 0; d < E; ++d) os[warp][lane * E + d] = acc[d];
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float top = -INFINITY;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) top = fmaxf(top, ms[w]);
    float num = 0.f, den = 0.f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) {
      const float e = ms[w] == -INFINITY ? 0.f : __expf(ms[w] - top);
      num += os[w][d] * e;
      den += ls[w] * e;
    }
    out[h * D + d] = to_bf16(num / den);
  }
}

// Interleaved rotary embedding of one head at position `pos`: pair
// (x[2i], x[2i+1]) turns by pos * exp(i * coef), coef = -2 ln(period) / D
// (computed by the host in f64, as the reference does), in f32, rounded to
// bf16 into `dst`. A warp per head, lane `l` owning pairs l, l + 32, ...
template <int D>
__device__ __forceinline__ void rope_head(const bf16* src, bf16* dst, float pos, float coef) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = lane; i < D / 2; i += 32) {
    const float freq = expf((float)i * coef);
    float s, c;
    sincosf(freq * pos, &s, &c);
    const float r = f32(src[2 * i]), im = f32(src[2 * i + 1]);
    dst[2 * i] = to_bf16(r * c - im * s);
    dst[2 * i + 1] = to_bf16(r * s + im * c);
  }
}
