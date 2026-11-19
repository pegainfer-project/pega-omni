// Qwen Image 2.1 DiT kernels. Module name "qwen" (Gen::launch splits on '_').
// Per-step variables read from device buffers for CUDA graph replay safety.
#include "common.cuh"

// --- Elementwise kernels (uint4 vectorized, 8 bf16 per thread) ---

// LayerNorm (no affine) × (1+scale), the way the reference rounds it: the norm
// to bf16, then the product to bf16. fp32 stats. Grid [rows,1,1], block dim/8.
extern "C" __global__ void qwen_image_21_ln_mod(const bf16* x, const bf16* scale, bf16* out, int dim, float eps) {
  const int64_t at = (int64_t)blockIdx.x * dim + threadIdx.x * 8;
  float v[8];
  load8(x + at, v);
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) sum += v[k];
  const float mean = block_sum(sum) / dim;
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) { v[k] -= mean; ss += v[k] * v[k]; }
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  float sv[8];
  load8(scale + threadIdx.x * 8, sv);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = round_bf16(v[k] * r) * sv[k];
  store8(out + at, v);
}

// out = bf16(x + bf16(gate[c] * update)). Two bf16 rounds. out may alias x.
extern "C" __global__ void qwen_image_21_res_gate(const bf16* x, const bf16* update, const bf16* gate, bf16* out, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t at = (int64_t)i * 8;
  float xv[8], uv[8], gv[8];
  load8(x + at, xv);
  load8(update + at, uv);
  load8(gate + (at % cols), gv);
#pragma unroll
  for (int k = 0; k < 8; ++k) xv[k] += round_bf16(gv[k] * uv[k]);
  store8(out + at, xv);
}

// SwiGLU: out = bf16(bf16(silu(gate)) * up). gate_up is [rows, 2*inter].
extern "C" __global__ void qwen_image_21_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
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

// out = bf16(silu(x)).
extern "C" __global__ void qwen_image_21_silu(const bf16* x, bf16* out, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8];
  load8(x + (int64_t)i * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = v[k] / (1.f + __expf(-v[k]));
  store8(out + (int64_t)i * 8, v);
}

// mod [4*n] → params [4,n]: [1+s1, tanh(g1), 1+s2, tanh(g2)].
extern "C" __global__ void qwen_image_21_mod_prep(const bf16* mod, bf16* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n / 8) return;
  const int64_t c = (int64_t)i * 8;
  float s1[8], g1[8], s2[8], g2[8];
  load8(mod + c, s1);
  load8(mod + n + c, g1);
  load8(mod + 2 * n + c, s2);
  load8(mod + 3 * n + c, g2);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    s1[k] += 1.f; g1[k] = tanhf(g1[k]);
    s2[k] += 1.f; g2[k] = tanhf(g2[k]);
  }
  store8(out + c, s1);
  store8(out + n + c, g1);
  store8(out + 2 * n + c, s2);
  store8(out + 3 * n + c, g2);
}

// out = bf16(1 + x).
extern "C" __global__ void qwen_image_21_one_plus(const bf16* x, bf16* out, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8];
  load8(x + (int64_t)i * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += 1.f;
  store8(out + (int64_t)i * 8, v);
}

// Zero-centered RMSNorm: out = bf16(x * rsqrt(mean(x²)+eps) * (f32(w)+1)).
// The +1 is fp32. ONE bf16 round at end. Grid [rows,1,1], block dim/8.
extern "C" __global__ void qwen_image_21_zero_rms_norm(const bf16* x, const bf16* w, bf16* out, int dim, float eps) {
  const int64_t at = (int64_t)blockIdx.x * dim + threadIdx.x * 8;
  float v[8], wv[8];
  load8(x + at, v);
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  load8(w + threadIdx.x * 8, wv);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = v[k] * r * (wv[k] + 1.f);
  store8(out + at, v);
}

// In-place GELU-tanh. fp32 math.
extern "C" __global__ void qwen_image_21_gelu_tanh(bf16* x, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8];
  load8(x + (int64_t)i * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float x3 = v[k] * v[k] * v[k];
    v[k] = 0.5f * v[k] * (1.f + tanhf(0.7978845608f * (v[k] + 0.044715f * x3)));
  }
  store8(x + (int64_t)i * 8, v);
}

// Euler: z = bf16(f32(z) + (sigmas[1]-sigmas[0]) * f32(pred)). Device sigmas.
extern "C" __global__ void qwen_image_21_euler(bf16* z, const bf16* pred, int n, const float* sigmas) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n / 8) return;
  const int64_t at = (int64_t)i * 8;
  const float dt = sigmas[1] - sigmas[0];
  float zv[8], pv[8];
  load8(z + at, zv);
  load8(pred + at, pv);
#pragma unroll
  for (int k = 0; k < 8; ++k) zv[k] += dt * pv[k];
  store8(z + at, zv);
}

// Copy total/8 groups of 8 bf16.
extern "C" __global__ void qwen_image_21_copy(const bf16* src, bf16* dst, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
}

// --- Philox noise (bf16 output) ---

namespace {
__device__ __forceinline__ uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t* hi) {
  *hi = __umulhi(a, b); return a * b;
}
__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, hi1;
    const uint32_t lo0 = mulhilo(0xD2511F53u, c.x, &hi0);
    const uint32_t lo1 = mulhilo(0xCD9E8D57u, c.z, &hi1);
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u; k.y += 0xBB67AE85u;
  }
  return c;
}
__device__ __forceinline__ float unit(uint32_t x) { return ((float)(x >> 8) + 1.f) * (1.f / 16777216.f); }
}  // namespace

// n bf16 standard normals. Pure function of (key, stream, index).
extern "C" __global__ void qwen_image_21_init_z(bf16* out, int n, const uint32_t* draw) {
  const int64_t quad = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (quad * 4 >= n) return;
  const uint4 r = philox(make_uint4((uint32_t)quad, (uint32_t)(quad >> 32), draw[2], 0u), make_uint2(draw[0], draw[1]));
  const float two_pi = 6.28318530717958647692f;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const float u1 = unit(p ? r.z : r.x), u2 = unit(p ? r.w : r.y);
    const float radius = sqrtf(-2.f * logf(u1));
    float s, c;
    sincosf(two_pi * u2, &s, &c);
    const int64_t idx = quad * 4 + 2 * p;
    if (idx < n) out[idx] = to_bf16(radius * c);
    if (idx + 1 < n) out[idx + 1] = to_bf16(radius * s);
  }
}

// --- Fused QK-RMSNorm + RoPE + KV cache write ---
// Adjacent pairs (2i, 2i+1), axes concatenated [frame:0-7, height:8-35, width:36-63].
// cos/sin from precomputed [9216, 64] f32 tables. One warp per (row, head).
// Grid [rows, 12, 1], block 256. hq=32, hk=32, D=128.
extern "C" __global__ void qwen_image_21_qk_rope(bf16* qkv, int ld, const bf16* qn, const bf16* kn,
    const int32_t* pos, const int32_t* slots, bf16* kc, bf16* vc,
    const float* cos_table, const float* sin_table, int hq, int hk, float eps) {
  constexpr int D = 128;
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= hq + 2 * hk) return;
  bf16* row = qkv + (int64_t)n * ld + (int64_t)head * D;
  const int64_t slot = slots[n];

  // V heads: copy verbatim to cache. A lane owns four of the 128 channels.
  if (head >= hq + hk) {
    bf16* dst = vc + (slot * hk + (head - hq - hk)) * D;
    for (int i = lane; i < D; i += 32) dst[i] = row[i];
    return;
  }

  const bool is_q = head < hq;
  const bf16* w = is_q ? qn : kn;
  // Each lane owns 4 adjacent elements: lane*4 .. lane*4+3 = pairs 2*lane, 2*lane+1
  float v[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) v[e] = f32(row[lane * 4 + e]);

  // RMSNorm: fp32 sum, warp reduce, TWO bf16 rounds (the norm, then the weight)
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) ss += v[e] * v[e];
  const float r = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
  for (int e = 0; e < 4; ++e) v[e] = round_bf16(round_bf16(v[e] * r) * f32(w[lane * 4 + e]));

  // RoPE: adjacent pairs, fp32 complex multiply
  const int32_t* p = pos + (int64_t)n * 3;
  bf16* dst = is_q ? row : kc + (slot * hk + (head - hq)) * D;
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const int i = lane * 2 + pair;  // pair index 0..63
    const int axis = i < 8 ? 0 : (i < 36 ? 1 : 2);
    const int32_t idx = p[axis];
    const int64_t tidx = (int64_t)idx * 64 + i;
    const float c = cos_table[tidx], s = sin_table[tidx];
    const float a = v[pair * 2], b = v[pair * 2 + 1];
    dst[lane * 4 + pair * 2] = to_bf16(fmaf(a, c, -(b * s)));
    dst[lane * 4 + pair * 2 + 1] = to_bf16(fmaf(b, c, a * s));
  }
}

// ---------------------------------------------------------------------------
// Attention: FA2 on mma.sync at head_dim 128. Qwen Image 2.1 is MHA (32 query
// heads, 32 K/V heads), so a block owns 128 query positions of ONE head — the
// G=4 packing hidream uses degenerates to one row per position. Q goes into
// registers once; K and V stream through shared memory 64 keys at a time,
// double buffered with cp.async. S = Q K^T and O += P V are m16n8k16 bf16 MMAs
// with f32 accumulation, the softmax online in base 2 with P rounded to bf16
// for the second MMA.
//
// Query rows `q + n * q_stride + head * 128` attend to the first `*kv_len` rows
// of `k` and `v` (`[slots, 32, 128]`); with `causal`, row n only to rows 0..=n.
// `out` is `[q_len, 32, 128]`. Grid `[ceil(q_len / 128), 32]`, block 256,
// `attend::SMEM` bytes of shared memory.

namespace attend {
constexpr int D = 128, BM = 128, BN = 64, WARPS = BM / 16, THREADS = WARPS * 32, CHUNKS = D / 8;
// Q's staging tile shares the space of the two K and two V tiles.
constexpr int SMEM = 4 * BN * D * 2;

__device__ __forceinline__ int swz(int r, int c) { return r * D + ((c ^ (r & 7)) << 3); }
__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }

__device__ __forceinline__ void load_kv(bf16* ks, bf16* vs, const bf16* k, const bf16* v, int n0, int kv_len, int kv_stride) {
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

extern "C" __global__ void __launch_bounds__(attend::THREADS, 1)
qwen_image_21_attend(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ out, int q_len, const int32_t* kv_len_at, int q_stride, int causal, float scale_log2) {
  using namespace attend;
  extern __shared__ __align__(128) bf16 smem[];
  bf16* qs = smem;
  bf16* ks[2] = {smem, smem + BN * D};
  bf16* vs[2] = {smem + 2 * BN * D, smem + 3 * BN * D};

  const int kv_len = *kv_len_at;
  const int head = blockIdx.y, hk = gridDim.y, kv_stride = hk * D;
  const int row0 = blockIdx.x * BM, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wrow = warp * 16;
  const bf16* kh = k + head * D;
  const bf16* vh = v + head * D;

  // G=1: packed row r = query position row0+r, one head per block.
  for (int i = threadIdx.x; i < BM * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS, pos = row0 + r;
    const bool ok = pos < q_len;
    const int64_t off = (int64_t)(ok ? pos : 0) * q_stride + head * D + c * 8;
    cp16(qs + swz(r, c), q + off, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) ldsm_x4(qf[kk], qs + swz(wrow + (lane & 15), kk * 2 + (lane >> 4)));
  __syncthreads();

  const int last = causal ? min(kv_len, min(row0 + BM, q_len)) : kv_len;
  load_kv(ks[0], vs[0], kh, vh, 0, kv_len, kv_stride);

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  const int pos0 = row0 + wrow + (lane >> 2), pos1 = pos0 + 8;

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

    const int n0 = s * BN;
    if (n0 + BN > kv_len || causal) {
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int key = n0 + j * 8 + (lane & 3) * 2;
        const int lim0 = causal ? min(kv_len, pos0 + 1) : kv_len;
        const int lim1 = causal ? min(kv_len, pos1 + 1) : kv_len;
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
    for (int i = 0; i < D / 8; ++i) { o[i][0] *= alpha[0]; o[i][1] *= alpha[0]; o[i][2] *= alpha[1]; o[i][3] *= alpha[1]; }

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

  // Output: out[pos * hk * D + head * D + ...]
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = l[h];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const float inv = 1.f / sum;
    const int pos = row0 + wrow + (lane >> 2) + h * 8;
    if (pos >= q_len) continue;
    bf16* dst = out + ((int64_t)pos * hk + head) * D + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < D / 8; ++i)
      *reinterpret_cast<uint32_t*>(dst + i * 8) = pack(o[i][2 * h] * inv, o[i][2 * h + 1] * inv);
  }
}