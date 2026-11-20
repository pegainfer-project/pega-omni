// Helium and the depformer: every piece of a step that is not a GEMM (those
// are cuBLASLt, named by the manifest), the samplers, and the token
// bookkeeping, compiled to one cubin the kern runtime launches from the calls
// `helium.rs` and `depformer.rs` generate.
//
// Rows are tokens: a prompt's positions in `prefill`, one per session in a
// tick. Helium's K and V of a row go to its slot of the layer's paged state
// (`[K heads | V heads][128]` per slot); a session's slots are a ring of
// CONTEXT positions, so a row at position p writes slot p % CONTEXT and reads
// the min(p + 1, CONTEXT) slots before it. The depformer's K and V live in a
// workspace, eight slots per session, rewritten every frame.
//
// A session's token state is the first 24 words of its slot of the `seq`
// state: the next step's input row (text, the agent's 8 codebooks, the
// caller's 8) and the caller's 7 acoustic codes that arrive a step late
// (`prompt::advance` is the same bookkeeping on the host).
#include <cuda_fp8.h>
#include "attend.cuh"

constexpr int kStreams = 17;
constexpr int kBooks = 8;
constexpr int kDrawn = 1 + kBooks;
constexpr int kCard = 2049;

// out = Σ audio_emb[k][ids[1 + k]] then + text_emb[ids[0]], each add rounded
// to bf16 (the reference's order); an id of -1 contributes nothing.
__device__ __forceinline__ void embed_ids(const int32_t* ids, const bf16* text_emb, const bf16* audio_emb, bf16* out,
                                          int dim) {
  const int c = threadIdx.x * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < kStreams; ++k) {
    const int stream = k < kStreams - 1 ? k + 1 : 0;
    const int id = ids[stream];
    if (id < 0) continue;
    const bf16* row = stream == 0 ? text_emb + (int64_t)id * dim : audio_emb + ((int64_t)k * kCard + id) * dim;
    float e[8];
    load8(row + c, e);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = round_bf16(acc[j] + e[j]);
  }
  store8(out + c, acc);
}

// Prompt rows: row `r` is voice embedding `voice[r]` when that is not
// negative, else the embedding of its 17 token ids. A block of dim / 8 per row.
extern "C" __global__ void lm_embed_prompt(const int32_t* ids, const int32_t* voice, const bf16* voices,
                                           const bf16* text_emb, const bf16* audio_emb, bf16* out, int dim) {
  pdl();
  const int64_t r = blockIdx.x;
  if (voice[r] >= 0) {
    const int c = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(out + r * dim + c) =
        *reinterpret_cast<const uint4*>(voices + (int64_t)voice[r] * dim + c);
    return;
  }
  embed_ids(ids + r * kStreams, text_emb, audio_emb, out + r * dim, dim);
}

// A tick's rows: each session's input row from its state, also copied to `rows`.
extern "C" __global__ void lm_embed_state(const void* state, const int32_t* lines, int64_t stride,
                                          const bf16* text_emb, const bf16* audio_emb, bf16* out, int32_t* rows,
                                          int dim) {
  pdl();
  const int s = blockIdx.x;
  const int32_t* row = slot<int32_t>(const_cast<void*>(state), lines, s, stride);
  if (threadIdx.x < kStreams) rows[s * kStreams + threadIdx.x] = row[threadIdx.x];
  embed_ids(row, text_emb, audio_emb, out + (int64_t)s * dim, dim);
}

// RMSNorm as the reference's `rms_norm_f32`: y = x * (alpha * rsqrt(eps +
// mean(x²))) in f32 over the bf16 row. One block of dim / 8 per row.
__device__ __forceinline__ void rms_norm_row(const float* v, const bf16* alpha, bf16* out, int dim, float eps) {
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  float a[8], y[8];
  load8(alpha + threadIdx.x * 8, a);
#pragma unroll
  for (int k = 0; k < 8; ++k) y[k] = v[k] * (a[k] * r);
  store8(out + threadIdx.x * 8, y);
}

extern "C" __global__ void lm_norm(const bf16* x, const bf16* alpha, bf16* out, int dim, float eps) {
  pdl();
  const int64_t at = (int64_t)blockIdx.x * dim;
  float v[8];
  load8(x + at + threadIdx.x * 8, v);
  rms_norm_row(v, alpha, out + at, dim, eps);
}

// x = bf16(x + y), then out = norm(x).
extern "C" __global__ void lm_add_norm(const bf16* y, bf16* x, const bf16* alpha, bf16* out, int dim, float eps) {
  pdl();
  const int64_t at = (int64_t)blockIdx.x * dim + threadIdx.x * 8;
  float v[8], u[8];
  load8(x + at, v);
  load8(y + at, u);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = round_bf16(v[k] + u[k]);
  store8(x + at, v);
  rms_norm_row(v, alpha, out + (int64_t)blockIdx.x * dim, dim, eps);
}

// x = bf16(x + y), eight values per thread (`total` counts the groups).
extern "C" __global__ void lm_add(const bf16* y, bf16* x, int total) {
  pdl();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8], u[8];
  load8(x + (int64_t)i * 8, v);
  load8(y + (int64_t)i * 8, u);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += u[k];
  store8(x + (int64_t)i * 8, v);
}

extern "C" __global__ void lm_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
  pdl();
  silu_mul(gate_up, out, inter, total);
}

constexpr int kHelium = 128;

// Helium rows [q | k | v] (heads of 128): rotary embedding of q in place,
// rotary + FP8 (E4M3) per-head quantization of k into the row's slot, and
// FP8 (E4M3) per-head quantization of v into the row's slot.
// Slot layout: [K heads (heads*128 B) | V heads (heads*128 B) | K scales (heads*4 B) | V scales (heads*4 B)].
extern "C" __global__ void lm_rope(bf16* qkv, const int32_t* pos, const int32_t* slots, void* kv, int heads,
                                   float coef) {
  pdl();
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (head >= 3 * heads) return;
  bf16* src = qkv + ((int64_t)n * 3 * heads + head) * kHelium;
  if (head < heads) {
    rope_head<kHelium>(src, src, (float)pos[n], coef);
    return;
  }
  const int lane = threadIdx.x & 31;
  const int64_t slot_bytes = 2LL * heads * kHelium + 2LL * heads * 4;
  uint8_t* slot_ptr = static_cast<uint8_t*>(kv) + (int64_t)slots[n] * slot_bytes;
  float* scales = reinterpret_cast<float*>(slot_ptr + 2LL * heads * kHelium);
  if (head < 2 * heads) {
    const int h = head - heads;
    const float p = (float)pos[n];
    float out_r[2], out_i[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int i = lane + k * 32;
      const float freq = expf((float)i * coef);
      float s, c;
      sincosf(freq * p, &s, &c);
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(src + 2 * i);
      const bf16* pair = reinterpret_cast<const bf16*>(&raw);
      const float r = f32(pair[0]), im = f32(pair[1]);
      out_r[k] = r * c - im * s;
      out_i[k] = r * s + im * c;
    }
    const float amax = warp_max(fmaxf(fmaxf(fabsf(out_r[0]), fabsf(out_i[0])),
                                      fmaxf(fabsf(out_r[1]), fabsf(out_i[1]))));
    const float sc = fmaxf(amax * (1.f / 448.f), 1e-12f);
    const float inv = 1.f / sc;
    uint8_t* dst = slot_ptr + (int64_t)h * kHelium;
    *reinterpret_cast<__nv_fp8x2_e4m3*>(dst + 2 * lane) = __nv_fp8x2_e4m3(make_float2(out_r[0] * inv, out_i[0] * inv));
    *reinterpret_cast<__nv_fp8x2_e4m3*>(dst + 2 * (lane + 32)) =
        __nv_fp8x2_e4m3(make_float2(out_r[1] * inv, out_i[1] * inv));
    if (lane == 0) scales[h] = sc;
  } else {
    const int h = head - 2 * heads;
    const uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
    const bf16* vp = reinterpret_cast<const bf16*>(&raw);
    const float v0 = f32(vp[0]), v1 = f32(vp[1]), v2 = f32(vp[2]), v3 = f32(vp[3]);
    const float amax = warp_max(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
    const float sc = fmaxf(amax * (1.f / 448.f), 1e-12f);
    const float inv = 1.f / sc;
    uint8_t* dst = slot_ptr + (int64_t)(heads + h) * kHelium;
    *reinterpret_cast<__nv_fp8x4_e4m3*>(dst + lane * 4) =
        __nv_fp8x4_e4m3(make_float4(v0 * inv, v1 * inv, v2 * inv, v3 * inv));
    if (lane == 0) scales[heads + h] = sc;
  }
}

struct PagedSlots16 {
  const int32_t* pages;
  __device__ __forceinline__ int64_t operator()(int t) const { return ((int64_t)pages[t >> 4] << 4) | (t & 15); }
};

struct PagedSlots {
  const int32_t* pages;
  int page;
  __device__ __forceinline__ int64_t operator()(int t) const { return (int64_t)pages[t / page] * page + t % page; }
};

template <typename SlotOf>
__device__ __forceinline__ void attend_row_fp8(const bf16* q, const uint8_t* kv, bf16* out, int heads, int len,
                                               float scale, SlotOf slot_of) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ float qs[kHelium];
  __shared__ float ms[kAttnWarps], ls[kAttnWarps];
  __shared__ float os[kAttnWarps][kHelium];
  qs[threadIdx.x] = f32(q[h * kHelium + threadIdx.x]);
  __syncthreads();
  const int64_t slot_bytes = 2LL * heads * kHelium + 2LL * heads * 4;
  const int64_t k_off = (int64_t)h * kHelium;
  const int64_t v_off = (int64_t)(heads + h) * kHelium + lane * 4;
  const int64_t ks_off = 2LL * heads * kHelium + (int64_t)h * 4;
  const int64_t vs_off = 2LL * heads * kHelium + (int64_t)(heads + h) * 4;
  float m = -INFINITY, l = 0.f, acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int base = warp * 32; base < len; base += kAttnWarps * 32) {
    const int t = base + lane;
    const bool valid = t < len;
    const int64_t slot = slot_of(valid ? t : base);
    const uint8_t* sp = kv + slot * slot_bytes;
    const uint8_t* k = sp + k_off;
    const float ks = *reinterpret_cast<const float*>(sp + ks_off);
    const float vs = *reinterpret_cast<const float*>(sp + vs_off);
    float sc = 0.f;
#pragma unroll 4
    for (int c = 0; c < kHelium; c += 8) {
      const uint2 raw = *reinterpret_cast<const uint2*>(k + c);
      const float4 k0 = static_cast<float4>(*reinterpret_cast<const __nv_fp8x4_e4m3*>(&raw.x));
      const float4 k1 = static_cast<float4>(*reinterpret_cast<const __nv_fp8x4_e4m3*>(&raw.y));
      sc += qs[c + 0] * k0.x + qs[c + 1] * k0.y + qs[c + 2] * k0.z + qs[c + 3] * k0.w +
            qs[c + 4] * k1.x + qs[c + 5] * k1.y + qs[c + 6] * k1.z + qs[c + 7] * k1.w;
    }
    const float s = valid ? sc * (ks * scale) : -INFINITY;
    const float m2 = fmaxf(m, warp_max(s));
    const float corr = __expf(m - m2);
    const float p = __expf(s - m2);
    l = l * corr + warp_sum(p);
    acc0 *= corr;
    acc1 *= corr;
    acc2 *= corr;
    acc3 *= corr;
    m = m2;
    const float pv = p * vs;
    const int count = min(32, len - base);
    for (int j = 0; j < count; ++j) {
      const int64_t sj = __shfl_sync(0xffffffffu, slot, j);
      const float pvj = __shfl_sync(0xffffffffu, pv, j);
      const uint32_t vraw = *reinterpret_cast<const uint32_t*>(kv + sj * slot_bytes + v_off);
      const float4 vf = static_cast<float4>(*reinterpret_cast<const __nv_fp8x4_e4m3*>(&vraw));
      acc0 += pvj * vf.x;
      acc1 += pvj * vf.y;
      acc2 += pvj * vf.z;
      acc3 += pvj * vf.w;
    }
  }
  if (lane == 0) {
    ms[warp] = m;
    ls[warp] = l;
  }
  os[warp][lane * 4 + 0] = acc0;
  os[warp][lane * 4 + 1] = acc1;
  os[warp][lane * 4 + 2] = acc2;
  os[warp][lane * 4 + 3] = acc3;
  __syncthreads();
  const int d = threadIdx.x;
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
  out[h * kHelium + d] = to_bf16(num / den);
}

// Row `n` (of session `ragged ? seq[n] : n`, at `pos[n]`) against its
// session's first min(pos + 1, context) ring slots, pages `pages[indptr[s]..]`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    lm_attend(const bf16* qkv, const void* kv, const int32_t* pos, const int32_t* seq, int ragged,
              const int32_t* indptr, const int32_t* pages, int page, bf16* out, int heads, float scale, int context) {
  pdl();
  const int n = blockIdx.x;
  const int s = ragged ? seq[n] : n;
  const int len = min(pos[n] + 1, context);
  const bf16* q = qkv + (int64_t)n * 3 * heads * kHelium;
  const uint8_t* kvb = static_cast<const uint8_t*>(kv);
  bf16* o = out + (int64_t)n * heads * kHelium;
  if (page == 16) {
    attend_row_fp8(q, kvb, o, heads, len, scale, PagedSlots16{pages + indptr[s]});
  } else {
    attend_row_fp8(q, kvb, o, heads, len, scale, PagedSlots{pages + indptr[s], page});
  }
}

// --- The depformer ------------------------------------------------------------

constexpr int kDep = 64;
constexpr int kDepSteps = 8;

// Step `k`'s input: its slice of the projected Helium output plus the
// embedding of the token the step before drew (the text token for step 0),
// rounded to bf16 into `x`, and immediately RMS-normed into `xn`.
// A block of dim / 8 per session.
extern "C" __global__ void lm_dep_embed_norm(const bf16* projected, int k, const bf16* table, const int32_t* drawn,
                                             const bf16* alpha, bf16* x, bf16* xn, int dim, float eps) {
  pdl();
  const int s = blockIdx.x, c = threadIdx.x * 8;
  const int64_t id = drawn[s * kDrawn + k];
  float v[8], e[8];
  load8(projected + ((int64_t)s * kDepSteps + k) * dim + c, v);
  load8(table + id * dim + c, e);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = round_bf16(v[j] + e[j]);
  store8(x + (int64_t)s * dim + c, v);
  rms_norm_row(v, alpha, xn + (int64_t)s * dim, dim, eps);
}

// Fused KV write + single-warp causal attention for depformer step `step` (0..7).
// One warp (32 lanes) per (session, head), each lane owning 2 dims of `kDep = 64`.
// Writes current step's K/V to `ws`, attends over steps `0..=step` in registers
// (current step's K/V reused from registers without re-reading `ws`), no shared mem.
extern "C" __global__ void __launch_bounds__(32)
    lm_dep_kv_attend(const bf16* qkv, bf16* ws, int step, bf16* out, int heads, float scale) {
  pdl();
  const int s = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
  const int width = heads * kDep;
  const bf16* row = qkv + (int64_t)s * 3 * width;
  const uint32_t q_raw = *reinterpret_cast<const uint32_t*>(row + h * kDep + lane * 2);
  const uint32_t k_raw = *reinterpret_cast<const uint32_t*>(row + width + h * kDep + lane * 2);
  const uint32_t v_raw = *reinterpret_cast<const uint32_t*>(row + 2 * width + h * kDep + lane * 2);

  const int64_t ws_base = (int64_t)s * kDepSteps * 2 * width;
  const int64_t cur_slot = ws_base + (int64_t)step * 2 * width;
  *reinterpret_cast<uint32_t*>(ws + cur_slot + h * kDep + lane * 2) = k_raw;
  *reinterpret_cast<uint32_t*>(ws + cur_slot + width + h * kDep + lane * 2) = v_raw;

  const bf16* qp = reinterpret_cast<const bf16*>(&q_raw);
  const float q0 = f32(qp[0]), q1 = f32(qp[1]);

  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
  for (int t = 0; t <= step; ++t) {
    uint32_t kr, vr;
    if (t < step) {
      const int64_t tslot = ws_base + (int64_t)t * 2 * width;
      kr = *reinterpret_cast<const uint32_t*>(ws + tslot + h * kDep + lane * 2);
      vr = *reinterpret_cast<const uint32_t*>(ws + tslot + width + h * kDep + lane * 2);
    } else {
      kr = k_raw;
      vr = v_raw;
    }
    const bf16* kp = reinterpret_cast<const bf16*>(&kr);
    const bf16* vp = reinterpret_cast<const bf16*>(&vr);
    const float sc = warp_sum(q0 * f32(kp[0]) + q1 * f32(kp[1])) * scale;
    const float m2 = fmaxf(m, sc);
    const float corr = __expf(m - m2);
    const float p = __expf(sc - m2);
    l = l * corr + p;
    a0 = a0 * corr + p * f32(vp[0]);
    a1 = a1 * corr + p * f32(vp[1]);
    m = m2;
  }

  uint32_t o_raw;
  bf16* op = reinterpret_cast<bf16*>(&o_raw);
  const float inv_l = 1.f / l;
  op[0] = to_bf16(a0 * inv_l);
  op[1] = to_bf16(a1 * inv_l);
  *reinterpret_cast<uint32_t*>(out + (int64_t)s * width + h * kDep + lane * 2) = o_raw;
}

// --- Sampling -------------------------------------------------------------------
//
// Gumbel-max over softmax(logits / T) restricted to top-k. Because `logits`
// are bf16 and T > 0 is monotone, top-k threshold selection runs directly on
// the 16-bit order-preserving bf16 key (2 radix passes instead of 4), with a
// warp-parallel 256-bin histogram scan. Vocabularies up to 2 * kSampleThreads
// (all audio codebooks, 2048) stay in registers for a single global read.

constexpr int kSampleThreads = 1024;

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ uint16_t bf16_order_key(bf16 v) {
  const uint16_t u = *reinterpret_cast<const uint16_t*>(&v);
  return (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
}

__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Warp-parallel descending scan of `hist[256]` to find the bucket containing `left`.
__device__ __forceinline__ void select_bucket(const uint32_t* hist, uint32_t prefix, uint32_t left, int shift,
                                              uint32_t* sel_prefix, uint32_t* sel_left) {
  const int tid = threadIdx.x;
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
        *sel_prefix = prefix | (uint32_t)(tid * 8 + b) << shift;
        *sel_left = left - above;
      }
      above += c[b];
    }
  }
}

extern "C" __global__ void __launch_bounds__(kSampleThreads)
    lm_sample(const bf16* logits, int vocab, int64_t ld, float temperature, int top_k, const int32_t* force,
              const int32_t* seeds, int col, int32_t* drawn) {
  pdl();
  const int s = blockIdx.x, tid = threadIdx.x;
  __shared__ uint32_t hist[256];
  __shared__ uint32_t sel_prefix, sel_left;
  __shared__ unsigned long long best;
  const int forced = force[s * kDrawn + col];
  if (forced >= 0) {
    if (tid == 0) drawn[s * kDrawn + col] = forced;
    return;
  }
  const bf16* l = logits + (int64_t)s * ld;
  const float inv_t = temperature > 0.f ? 1.f / temperature : 1.f;
  const int k_keep = top_k > 0 && top_k < vocab ? top_k : vocab;
  const uint32_t seed = mix((uint32_t)seeds[2 * s] ^ mix((uint32_t)seeds[2 * s + 1] + 0x9e3779b9u * (col + 1)));

  if (vocab <= 2 * kSampleThreads) {
    // Fast register-resident path for audio codebooks (vocab = 2048).
    bf16 rv[2];
    uint16_t rk[2];
    const int i0 = tid, i1 = tid + kSampleThreads;
    if (i0 < vocab) {
      rv[0] = l[i0];
      rk[0] = bf16_order_key(rv[0]);
    }
    if (i1 < vocab) {
      rv[1] = l[i1];
      rk[1] = bf16_order_key(rv[1]);
    }
    uint32_t prefix = 0;
    if (temperature > 0.f && k_keep < vocab) {
      uint32_t left = k_keep;
      for (int shift = 8; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const uint32_t high = shift == 8 ? 0u : 0xff00u;
        if (i0 < vocab && ((rk[0] ^ prefix) & high) == 0) atomicAdd(&hist[(rk[0] >> shift) & 255], 1u);
        if (i1 < vocab && ((rk[1] ^ prefix) & high) == 0) atomicAdd(&hist[(rk[1] >> shift) & 255], 1u);
        __syncthreads();
        select_bucket(hist, prefix, left, shift, &sel_prefix, &sel_left);
        __syncthreads();
        prefix = sel_prefix;
        left = sel_left;
      }
    }
    if (tid == 0) best = 0;
    __syncthreads();
    unsigned long long mine = 0;
    auto consider = [&](int idx, bf16 val, uint16_t key) {
      if (idx >= vocab || key < prefix) return;
      float score = f32(val) * inv_t;
      if (temperature > 0.f) {
        const float u = ((mix(seed ^ mix((uint32_t)idx)) >> 8) + 0.5f) * (1.f / 16777216.f);
        score -= __logf(-__logf(u));
      }
      const unsigned long long packed = (unsigned long long)order_key(score) << 32 | (uint32_t)(0xffffffffu - idx);
      mine = packed > mine ? packed : mine;
    };
    consider(i0, rv[0], rk[0]);
    consider(i1, rv[1], rk[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const unsigned long long other = __shfl_xor_sync(0xffffffffu, mine, o);
      mine = other > mine ? other : mine;
    }
    if ((tid & 31) == 0) atomicMax(&best, mine);
    __syncthreads();
    if (tid == 0) drawn[s * kDrawn + col] = (int32_t)(0xffffffffu - (uint32_t)(best & 0xffffffffu));
    return;
  }

  // General path for large vocabularies (text, vocab = 32000): 2-pass bf16 radix select.
  uint32_t prefix = 0;
  if (temperature > 0.f && k_keep < vocab) {
    uint32_t left = k_keep;
    for (int shift = 8; shift >= 0; shift -= 8) {
      if (tid < 256) hist[tid] = 0;
      __syncthreads();
      const uint32_t high = shift == 8 ? 0u : 0xff00u;
      for (int i = tid; i < vocab; i += blockDim.x) {
        const uint16_t key = bf16_order_key(l[i]);
        if (((key ^ prefix) & high) == 0) atomicAdd(&hist[(key >> shift) & 255], 1u);
      }
      __syncthreads();
      select_bucket(hist, prefix, left, shift, &sel_prefix, &sel_left);
      __syncthreads();
      prefix = sel_prefix;
      left = sel_left;
    }
  }
  if (tid == 0) best = 0;
  __syncthreads();
  unsigned long long mine = 0;
  for (int i = tid; i < vocab; i += blockDim.x) {
    const bf16 val = l[i];
    if (bf16_order_key(val) < prefix) continue;
    float score = f32(val) * inv_t;
    if (temperature > 0.f) {
      const float u = ((mix(seed ^ mix((uint32_t)i)) >> 8) + 0.5f) * (1.f / 16777216.f);
      score -= __logf(-__logf(u));
    }
    const unsigned long long packed = (unsigned long long)order_key(score) << 32 | (uint32_t)(0xffffffffu - i);
    mine = packed > mine ? packed : mine;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, mine, o);
    mine = other > mine ? other : mine;
  }
  if ((tid & 31) == 0) atomicMax(&best, mine);
  __syncthreads();
  if (tid == 0) drawn[s * kDrawn + col] = (int32_t)(0xffffffffu - (uint32_t)(best & 0xffffffffu));
}

// --- Token bookkeeping ------------------------------------------------------------

// After the draws: the frame the agent speaks now (the text token and
// semantic code drawn a step ago, the acoustic codes just drawn) into
// `emitted`, and the session's next input row and pending caller codes into
// its state. The caller's codes are the encoder's, or `force_caller` where
// that is not negative. One thread per session.
extern "C" __global__ void lm_advance(void* state, const int32_t* lines, int64_t stride, const int32_t* drawn,
                                      const int32_t* caller, const int32_t* force_caller, int32_t* emitted,
                                      int seqs) {
  pdl();
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= seqs) return;
  int32_t* row = slot<int32_t>(state, lines, s, stride);
  int32_t* pending = row + kStreams;
  const int32_t* d = drawn + s * kDrawn;
  int32_t heard[kBooks];
#pragma unroll
  for (int k = 0; k < kBooks; ++k) {
    const int f = force_caller[s * kBooks + k];
    heard[k] = f >= 0 ? f : caller[s * kBooks + k];
  }
  emitted[s * kDrawn] = row[0];
  emitted[s * kDrawn + 1] = row[1];
#pragma unroll
  for (int k = 1; k < kBooks; ++k) emitted[s * kDrawn + 1 + k] = d[1 + k];
#pragma unroll
  for (int k = 0; k < kDrawn; ++k) row[k] = d[k];
  row[kDrawn] = heard[0];
#pragma unroll
  for (int k = 0; k < kBooks - 1; ++k) {
    row[kDrawn + 1 + k] = pending[k];
    pending[k] = heard[1 + k];
  }
}

// A new session's token state: its first input row and pending caller codes.
extern "C" __global__ void lm_init(void* state, const int32_t* lines, int64_t stride, const int32_t* init) {
  pdl();
  const int s = blockIdx.x;
  int32_t* own = slot<int32_t>(state, lines, s, stride);
  if (threadIdx.x < kStreams + kBooks - 1) own[threadIdx.x] = init[s * (kStreams + kBooks - 1) + threadIdx.x];
}

// Copies `tokens` KV slots (`slot_bytes` each, 16-byte aligned) from `src_slot[r]`
// to `dst_slot[r]` within one layer's KV state, for prefix KV reuse.
extern "C" __global__ void lm_copy_kv(void* kv, const int32_t* src_slot, const int32_t* dst_slot, int slot_bytes) {
  pdl();
  const int r = blockIdx.x;
  const int64_t sb = slot_bytes;
  const uint4* src = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(kv) + (int64_t)src_slot[r] * sb);
  uint4* dst = reinterpret_cast<uint4*>(static_cast<uint8_t*>(kv) + (int64_t)dst_slot[r] * sb);
  const int n4 = slot_bytes / 16;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
}
