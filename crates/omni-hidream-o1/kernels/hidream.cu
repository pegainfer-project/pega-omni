// HiDream-O1's kernels: the Qwen3-VL text tower run as a pixel-space
// diffusion transformer, and the sampler around it. Every launch is a call in
// the model's kern manifest (`src/model.rs`); the GEMMs are cuBLASLt through
// kern's built-ins.
//
// What changes from one request or step to the next (key count, timestep,
// noise draw, picture width) is read from device buffers, so a captured step
// replays for every request of a grid size.
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

namespace {

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float round_bf16(float x) { return f32(to_bf16(x)); }

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

}  // namespace

// Row `ids[n]` of `table` into row `n` of `out`. One block of dim / 8 threads per row.
extern "C" __global__ void hidream_embed(const int32_t* ids, const bf16* table, bf16* out, int dim) {
  const int64_t c = threadIdx.x * 8;
  *reinterpret_cast<uint4*>(out + (int64_t)blockIdx.x * dim + c) =
      *reinterpret_cast<const uint4*>(table + (int64_t)ids[blockIdx.x] * dim + c);
}

// RMSNorm that starts the residual stream: res = x, out = x * rsqrt(mean(x²) +
// eps) * w in f32. One block of dim / 8 threads per row.
extern "C" __global__ void hidream_norm_copy(const bf16* x, const bf16* w, bf16* out, bf16* res, int dim, float eps) {
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

// res += x, stored as bf16; out = res * rsqrt(mean(res²) + eps) * w over the
// unrounded f32 sum. `out` may be `x`.
extern "C" __global__ void hidream_add_norm(const bf16* x, bf16* res, const bf16* w, bf16* out, int dim, float eps) {
  const int64_t at = (int64_t)blockIdx.x * dim + threadIdx.x * 8;
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
  load8(w + threadIdx.x * 8, wv);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = v[k] * r * wv[k];
  store8(out + at, v);
}

// Per-head RMSNorm of Q and K, then Qwen3-VL's interleaved M-RoPE; Q stays in
// place, K and V go to row `slots[n]` of `k_cache` / `v_cache` (`[slots, hk,
// 128]`). A row has three positions (t, h, w) at `positions[3n..]`: pair i
// rotates by the h position when i % 3 == 1, by the w position when i % 3 ==
// 2, while i < 3 * mrope_hw, and by t otherwise. One warp per (row, head); a
// lane owns two elements of each half of a head, so a rotary pair shares it.
extern "C" __global__ void hidream_qk_rope(bf16* qkv, int ld, const bf16* q_norm, const bf16* k_norm,
                                           const int32_t* positions, const int32_t* slots, bf16* k_cache,
                                           bf16* v_cache, int hq, int hk, int mrope_hw, float eps, float theta) {
  constexpr int D = 128, E = D / 64;
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= hq + 2 * hk) return;
  bf16* row = qkv + (int64_t)n * ld + (int64_t)head * D;
  const int64_t slot = slots[n];
  if (head >= hq + hk) {
    bf16* dst = v_cache + (slot * hk + (head - hq - hk)) * D;
    for (int i = lane; i < D; i += 32) dst[i] = row[i];
    return;
  }
  const bool is_q = head < hq;
  const bf16* w = is_q ? q_norm : k_norm;
  float lo[E], hi[E];
#pragma unroll
  for (int e = 0; e < E; ++e) {
    lo[e] = f32(row[lane * E + e]);
    hi[e] = f32(row[D / 2 + lane * E + e]);
  }
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) ss += lo[e] * lo[e] + hi[e] * hi[e];
  const float r = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
  for (int e = 0; e < E; ++e) {
    lo[e] = round_bf16(lo[e] * r) * f32(w[lane * E + e]);
    hi[e] = round_bf16(hi[e] * r) * f32(w[D / 2 + lane * E + e]);
  }
  const int32_t* pos = positions + (int64_t)n * 3;
  bf16* dst = is_q ? row : k_cache + (slot * hk + (head - hq)) * D;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int i = lane * E + e;
    const int axis = i < 3 * mrope_hw ? i % 3 : 0;
    const float inv_freq = 1.f / powf(theta, (float)(2 * i) / D);
    float s, c;
    sincosf((float)pos[axis] * inv_freq, &s, &c);
    dst[i] = to_bf16(lo[e] * c - hi[e] * s);
    dst[D / 2 + i] = to_bf16(hi[e] * c + lo[e] * s);
  }
}

// SiLU(gate) * up over fused [gate | up] rows, SiLU rounded to bf16; thread `i`
// of `total` takes eight channels.
extern "C" __global__ void hidream_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
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

// x = act(x + bias), the sum rounded to bf16 first; act 0 is identity, 1 SiLU.
// Thread `i` of `total` takes one value of a `cols`-wide row.
extern "C" __global__ void hidream_bias_act(bf16* x, const bf16* bias, int cols, int act, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = round_bf16(f32(x[i]) + f32(bias[i % cols]));
  if (act == 1) v = v / (1.f + __expf(-v));
  x[i] = to_bf16(v);
}

// ---------------------------------------------------------------------------
// Attention: FA2 on mma.sync at head_dim 128, four query heads per K/V head.
// A block owns 128 packed rows (32 query positions x the 4 query heads of one
// K/V head), eight warps of 16. Q goes into registers once; K and V stream
// through shared memory 64 keys at a time, double buffered with cp.async. S =
// Q K^T and O += P V are m16n8k16 bf16 MMAs with f32 accumulation, the softmax
// online in base 2 with P rounded to bf16 for the second MMA.

namespace attend {

constexpr int D = 128, G = 4, BM = 128, BN = 64, WARPS = BM / 16, THREADS = WARPS * 32, CHUNKS = D / 8;
// Q's staging tile shares the space of the two K and two V tiles.
constexpr int SMEM = 4 * BN * D * 2;

// Element offset of 16-byte chunk `c` of row `r`, XOR-swizzled so that eight
// rows' same chunk land in eight bank groups.
__device__ __forceinline__ int swz(int r, int c) { return r * D + ((c ^ (r & 7)) << 3); }

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes from global, zero-filled when `valid` is false.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
               "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void load_kv(bf16* ks, bf16* vs, const bf16* k, const bf16* v, int n0, int kv_len,
                                        int kv_stride) {
  for (int i = threadIdx.x; i < BN * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS, key = n0 + r;
    const bool ok = key < kv_len;
    const int64_t off = (int64_t)(ok ? key : 0) * kv_stride + c * 8;
    cp16(ks + swz(r, c), k + off, ok);
    cp16(vs + swz(r, c), v + off, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

}  // namespace attend

// Query rows `q + n * q_stride + h * 128` (heads 0..4 * hk) attend to the first
// `*kv_len` rows of `k` and `v` (`[slots, hk, 128]`); with `causal`, row n only
// to rows 0..=n. `out` is `[q_len, 4 * hk, 128]`. Grid `[ceil(q_len * 4 / 128),
// hk]`, block 256, `attend::SMEM` bytes of shared memory.
extern "C" __global__ void __launch_bounds__(attend::THREADS, 1)
    hidream_attend(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                   bf16* __restrict__ out, int q_len, const int32_t* kv_len_at, int q_stride, int causal,
                   float scale_log2) {
  using namespace attend;
  extern __shared__ __align__(128) bf16 smem[];
  bf16* qs = smem;
  bf16* ks[2] = {smem, smem + BN * D};
  bf16* vs[2] = {smem + 2 * BN * D, smem + 3 * BN * D};

  const int kv_len = *kv_len_at;
  const int kvh = blockIdx.y, hk = gridDim.y, kv_stride = hk * D;
  const int row0 = blockIdx.x * BM, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wrow = warp * 16;
  const bf16* kh = k + kvh * D;
  const bf16* vh = v + kvh * D;

  // Packed row r is position (row0 + r) / G, head kvh * G + (row0 + r) % G.
  for (int i = threadIdx.x; i < BM * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS, p = row0 + r, pos = p / G;
    const bool ok = pos < q_len;
    const int64_t off = (int64_t)(ok ? pos : 0) * q_stride + (kvh * G + p % G) * D + c * 8;
    cp16(qs + swz(r, c), q + off, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) ldsm_x4(qf[kk], qs + swz(wrow + (lane & 15), kk * 2 + (lane >> 4)));
  __syncthreads();

  // The last key any row of this block sees.
  const int last = causal ? min(kv_len, (min(row0 + BM, q_len * G) - 1) / G + 1) : kv_len;
  load_kv(ks[0], vs[0], kh, vh, 0, kv_len, kv_stride);

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  // This thread's rows, lane/4 (h = 0) and lane/4 + 8 (h = 1): their positions bound causal keys.
  const int pos0 = (row0 + wrow + (lane >> 2)) / G, pos1 = (row0 + wrow + (lane >> 2) + 8) / G;

  const int steps = (last + BN - 1) / BN;
  for (int s = 0; s < steps; ++s) {
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    if (s + 1 < steps) load_kv(ks[(s + 1) & 1], vs[(s + 1) & 1], kh, vh, (s + 1) * BN, kv_len, kv_stride);
    const bf16* kt = ks[s & 1];
    const bf16* vt = vs[s & 1];

    float sc[BN / 8][4];
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
      for (int j = 0; j < BN / 16; ++j) {
        uint32_t b[4];
        ldsm_x4(b, kt + swz(j * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
        mma(sc[2 * j], qf[kk], b[0], b[1]);
        mma(sc[2 * j + 1], qf[kk], b[2], b[3]);
      }
    }

    // Keys past kv_len, and with `causal` past a row's own position, do not count.
    const int n0 = s * BN;
    if (n0 + BN > kv_len || causal) {
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int key = n0 + j * 8 + (lane & 3) * 2;
        const int lim0 = causal ? min(kv_len, pos0 + 1) : kv_len, lim1 = causal ? min(kv_len, pos1 + 1) : kv_len;
        if (key >= lim0) sc[j][0] = -INFINITY;
        if (key + 1 >= lim0) sc[j][1] = -INFINITY;
        if (key >= lim1) sc[j][2] = -INFINITY;
        if (key + 1 >= lim1) sc[j][3] = -INFINITY;
      }
    }

    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) mx = fmaxf(mx, fmaxf(sc[j][2 * h], sc[j][2 * h + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      // A row with no key yet in this or an earlier step keeps m at -inf.
      const float m_new = fmaxf(m[h], mx * scale_log2);
      const float base = m_new == -INFINITY ? 0.f : m_new;
      alpha[h] = ex2(m[h] - base);
      m[h] = m_new;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        sc[j][2 * h] = ex2(sc[j][2 * h] * scale_log2 - base);
        sc[j][2 * h + 1] = ex2(sc[j][2 * h + 1] * scale_log2 - base);
        sum += sc[j][2 * h] + sc[j][2 * h + 1];
      }
      l[h] = l[h] * alpha[h] + sum;
    }
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }

    // P's accumulator layout is the A layout of the next MMA.
#pragma unroll
    for (int kc = 0; kc < BN / 16; ++kc) {
      const uint32_t a[4] = {pack(sc[2 * kc][0], sc[2 * kc][1]), pack(sc[2 * kc][2], sc[2 * kc][3]),
                             pack(sc[2 * kc + 1][0], sc[2 * kc + 1][1]), pack(sc[2 * kc + 1][2], sc[2 * kc + 1][3])};
      const int key = kc * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
      for (int dp = 0; dp < D / 16; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, vt + swz(key, dp * 2 + (lane >> 4)));
        mma(o[2 * dp], a, b[0], b[1]);
        mma(o[2 * dp + 1], a, b[2], b[3]);
      }
    }
  }

  const int hq = hk * G;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = l[h];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const float inv = 1.f / sum;
    const int p = row0 + wrow + (lane >> 2) + h * 8, pos = p / G;
    if (pos >= q_len) continue;
    bf16* dst = out + ((int64_t)pos * hq + kvh * G + p % G) * D + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < D / 8; ++i)
      *reinterpret_cast<uint32_t*>(dst + i * 8) = pack(o[i][2 * h] * inv, o[i][2 * h + 1] * inv);
  }
}

// ---------------------------------------------------------------------------
// The sampler. A step's scalars come from `draw` (`[key lo, key hi, stream]`)
// and `step` (`[sigma_next, noise scale, clip std]`).

namespace {

__device__ __forceinline__ uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t* hi) {
  *hi = __umulhi(a, b);
  return a * b;
}

// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3").
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, hi1;
    const uint32_t lo0 = mulhilo(0xD2511F53u, c.x, &hi0);
    const uint32_t lo1 = mulhilo(0xCD9E8D57u, c.z, &hi1);
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// (0, 1], so the logarithm below is finite.
__device__ __forceinline__ float unit(uint32_t x) { return ((float)(x >> 8) + 1.f) * (1.f / 16777216.f); }

}  // namespace

// `n` standard normals (Philox4x32-10 + Box-Muller), a pure function of (key,
// stream, index), so a seed reproduces its picture whatever the launch shape.
// Thread `i` writes normals 4i..4i+4.
extern "C" __global__ void hidream_gaussian(float* out, int n, const uint32_t* draw) {
  const int64_t quad = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (quad * 4 >= n) return;
  const uint4 r = philox(make_uint4((uint32_t)quad, (uint32_t)(quad >> 32), draw[2], 0u), make_uint2(draw[0], draw[1]));
  float z[4];
  const float two_pi = 6.28318530717958647692f;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const float u1 = unit(p ? r.z : r.x), u2 = unit(p ? r.w : r.y);
    const float radius = sqrtf(-2.f * logf(u1));
    float s, c;
    sincosf(two_pi * u2, &s, &c);
    z[2 * p] = radius * c;
    z[2 * p + 1] = radius * s;
  }
  for (int j = 0; j < 4 && quad * 4 + j < n; ++j) out[quad * 4 + j] = z[j];
}

// `sums` = 0, before `hidream_moments` adds to it.
extern "C" __global__ void hidream_zero_sums(unsigned long long* sums) { sums[threadIdx.x] = 0; }

// Sum and sum of squares of `x` added to `sums` (two f64), the sample standard
// deviation the sampler clips its noise to. Grid-stride.
extern "C" __global__ void hidream_moments(const float* x, int n, unsigned long long* sums) {
  float s = 0.f, ss = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    s += x[i];
    ss += x[i] * x[i];
  }
  s = block_sum(s);
  ss = block_sum(ss);
  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<double*>(sums), (double)s);
    atomicAdd(reinterpret_cast<double*>(sums) + 1, (double)ss);
  }
}

// One Euler step with fresh noise, `z = sigma_next * scale * clip(eps) + (1 -
// sigma_next) * x0`, rounded to bf16 the way the reference keeps its latent;
// a clip std of zero leaves the noise unclipped and `sums` unread, and at
// sigma_next = 1 `x0` is not read (the starting latent).
extern "C" __global__ void hidream_flow_step(bf16* z, const bf16* x0, const float* noise,
                                             const unsigned long long* sums, int n, const float* step) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float sigma_next = step[0], scale = step[1], clip_std = step[2];
  float eps = noise[i];
  if (clip_std > 0.f) {
    const double* sd = reinterpret_cast<const double*>(sums);
    const double mean = sd[0] / (double)n;
    const float std = (float)sqrt((sd[1] - mean * sd[0]) / (double)(n - 1));
    const float c = clip_std * std;
    eps = fminf(fmaxf(eps, -c), c);
  }
  const float keep = sigma_next == 1.f ? 0.f : (1.f - sigma_next) * f32(x0[i]);
  z[i] = to_bf16(sigma_next * eps * scale + keep);
}

// `total` groups of eight bf16 from `src` to `dst`, a group per thread.
extern "C" __global__ void hidream_copy(const bf16* src, bf16* dst, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
}

// Patch rows `[H/32 * W/32, 3 * 32 * 32]` in [-1, 1] (channel-major inside a
// patch) to 8-bit HWC RGB, `width` read from `size[0]`; thread `i` of `total`
// writes one byte.
extern "C" __global__ void hidream_rgb(const bf16* z, uint8_t* rgb, const int32_t* size, int total) {
  constexpr int P = 32;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int width = size[0];
  const int c = i % 3, x = (i / 3) % width, y = (int)(i / 3 / width);
  const int64_t patch = (int64_t)(y / P) * (width / P) + x / P;
  const float v = f32(z[patch * 3 * P * P + (int64_t)c * P * P + (y % P) * P + x % P]);
  const float t = round_bf16(v + 1.f) * 0.5f;
  rgb[i] = (uint8_t)fminf(fmaxf(rintf(t * 255.f), 0.f), 255.f);
}
