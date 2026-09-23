// The streaming codec decoder's kernels, compiled to one cubin that the kern
// runtime launches from the manifest `codec::manifest` generates.
//
// A call decodes one frame for each of `seqs` sequences. At a stage running
// `T` rows per frame, global row `g` is row `t = g % T` of sequence `s = g / T`,
// whose absolute row index is `pos[s] * T + t`. Everything a causal layer needs
// from earlier frames lives in the sequence's slot of the per-sequence state
// (`state + lines[s] * stride`, plus the layer's offset the call adds):
//
// - the history of every causal conv: its transformed input's last `H` rows,
//   in two halves `[2][H][C]`; frame `p` reads rows `pT - H .. pT - 1` in order
//   from half `p & 1` and writes the next frame's into the other, so a launch
//   never overwrites what another of its threads still reads;
// - the second half of the last GEMM row of every transposed conv;
// - K and V of the last 72 frames of every attention layer, frame `p` at `p % 72`.
//
// A fresh slot is zeroed by its lease, so rows before the sequence start read
// as zero, the convs' causal padding.
//
// A bias is not always added where it arises: one that only feeds linear ops
// is folded through them at load, one that feeds a residual stream is carried
// to the stream's consumers, which add it on load (their `bias` params).
// Elementwise kernels move eight channels (16 bytes) per thread, so channel
// counts are multiples of 8.
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float round_bf16(float x) { return f32(to_bf16(x)); }

__device__ __forceinline__ void load8(const bf16* p, float* v) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = f32(e[k]);
}

__device__ __forceinline__ void loadf8(const float* p, float* v) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
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

// sin by the hardware approximation after a two-step reduction to [-π, π]
// (absolute error ~4e-7 there): the accurate sinf's instruction count, not
// memory, bounds every kernel that applies SnakeBeta.
__device__ __forceinline__ float fast_sin(float y) {
  const float k = rintf(y * 0.159154943091895336f);
  return __sinf(fmaf(-k, -1.74845553e-7f, fmaf(-k, 6.28318548202514648f, y)));
}

__device__ __forceinline__ float snake(float x, float a, float inv_b) {
  const float s = fast_sin(a * x);
  return x + inv_b * s * s;
}

// snake(v + bias) on the eight channels at `c`.
__device__ __forceinline__ void bias_snake8(float* v, const bf16* bias, const float* a, const float* inv_b, int c) {
  float b[8], av[8], ib[8];
  load8(bias + c, b);
  loadf8(a + c, av);
  loadf8(inv_b + c, ib);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = snake(v[k] + b[k], av[k], ib[k]);
}

__device__ __forceinline__ bf16* slot(void* state, const int32_t* lines, int s, int64_t stride) {
  return reinterpret_cast<bf16*>(static_cast<char*>(state) + (int64_t)lines[s] * stride);
}

// Row r: [codebook 0 | Σ codebooks 1..15], each `half` wide; `books` is
// [16 * rows_per_book, half]. Eight channels per thread, blockDim.x = half / 8.
extern "C" __global__ void codec_rvq(const int32_t* codes, const bf16* books, bf16* out, int half, int rows_per_book) {
  const int r = blockIdx.x, c = threadIdx.x * 8;
  const int32_t* code = codes + r * 16;
  copy8(out + (int64_t)r * 2 * half + c, books + (int64_t)code[0] * half + c);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int g = 1; g < 16; ++g) {
    float v[8];
    load8(books + ((int64_t)g * rows_per_book + code[g]) * half + c, v);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += v[k];
  }
  store8(out + (int64_t)r * 2 * half + half + c, acc);
}

// x = gelu(x + bias) in place, eight channels per thread (`total` counts the groups).
extern "C" __global__ void codec_bias_gelu(bf16* x, const bf16* bias, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8], b[8];
  load8(x + (int64_t)i * 8, v);
  load8(bias + i % (cols / 8) * 8, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float y = round_bf16(v[k] + b[k]);
    v[k] = 0.5f * y * (1.f + erff(y * 0.70710678118654752f));
  }
  store8(x + (int64_t)i * 8, v);
}

// out = snake(x + bias), eight channels per thread.
extern "C" __global__ void codec_bias_snake(const bf16* x, const bf16* bias, const float* a, const float* inv_b,
                                            bf16* out, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8];
  load8(x + (int64_t)i * 8, v);
  bias_snake8(v, bias, a, inv_b, i % (cols / 8) * 8);
  store8(out + (int64_t)i * 8, v);
}

// SiLU(gate) * up over fused [gate | up] rows, eight channels per thread.
extern "C" __global__ void codec_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
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

// FlashInfer's FusedAddRMSNorm, a warp per row: res += add; out =
// rms_norm(res) * w, the norm taken over the unrounded sum. `dim` a multiple
// of 256, at most 1024; `add_step` 0 adds the same row (a bias) to every row.
__device__ __forceinline__ void add_rms_norm(const bf16* add, int64_t add_step, bf16* res, const bf16* w, bf16* out,
                                             int dim, float eps, int rows) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  add += r * add_step;
  res += (int64_t)r * dim;
  out += (int64_t)r * dim;
  float v[4][8];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = (k * 32 + lane) * 8;
    if (c >= dim) break;
    float x[8];
    load8(add + c, x);
    load8(res + c, v[k]);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[k][e] += x[e];
      ss += v[k][e] * v[k][e];
    }
    store8(res + c, v[k]);
  }
  const float rs = rsqrtf(warp_sum(ss) / dim + eps);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = (k * 32 + lane) * 8;
    if (c >= dim) break;
    float g[8];
    load8(w + c, g);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[k][e] *= rs * g[e];
    store8(out + c, v[k]);
  }
}

extern "C" __global__ void codec_add_rms_norm(bf16* x, bf16* res, const bf16* w, int dim, float eps, int rows) {
  add_rms_norm(x, dim, res, w, x, dim, eps, rows);
}

extern "C" __global__ void codec_bias_rms_norm(bf16* res, const bf16* bias, const bf16* w, bf16* out, int dim, float eps,
                                               int rows) {
  add_rms_norm(bias, 0, res, w, out, dim, eps, rows);
}

constexpr int kHead = 64;
constexpr int kWindow = 72;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// qkv rows are [q | k | v], `heads` heads of 64 each; the layer's ring (`kv`)
// is K[72][heads*64] then V, frame `p` at `p % 72`. Per (row, head): RoPE on q
// and k, k and v into the ring, then q against the last min(p + 1, 72) frames,
// softmax in f32. One block of 128 per (row, head), every phase one load deep:
// a frame per thread for the scores, then sixteen 8-lane groups each summing
// every sixteenth frame's V.
extern "C" __global__ void __launch_bounds__(128)
    codec_attention(const bf16* qkv, const int32_t* pos, const int32_t* lines, void* kv, bf16* out, int64_t stride,
                    int heads, float theta) {
  __shared__ float sq[kHead];
  __shared__ float ss[kWindow + 8];
  __shared__ float sacc[4][kHead];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int r = blockIdx.x, head = blockIdx.y;
  const int width = heads * kHead;
  const bf16* row = qkv + (int64_t)r * 3 * width + head * kHead;
  bf16* ring = slot(kv, lines, r, stride);
  const int p = pos[r];
  const int64_t at = (int64_t)(p % kWindow) * width + head * kHead;
  const int64_t vofs = (int64_t)kWindow * width;
  if (w < 2) {
    const float inv_freq = 1.f / powf(theta, (float)(2 * lane) / kHead);
    float s, c;
    sincosf((float)p * inv_freq, &s, &c);
    const bf16* src = row + w * width;
    const float lo = f32(src[lane]), hi = f32(src[lane + 32]);
    const bf16 a = to_bf16(lo * c - hi * s), b = to_bf16(hi * c + lo * s);
    if (w == 0) {
      sq[lane] = f32(a);
      sq[lane + 32] = f32(b);
    } else {
      ring[at + lane] = a;
      ring[at + lane + 32] = b;
    }
  } else if (w == 2) {
    reinterpret_cast<uint32_t*>(ring + vofs + at)[lane] = reinterpret_cast<const uint32_t*>(row + 2 * width)[lane];
  }
  __syncthreads();
  const int n = p + 1 < kWindow ? p + 1 : kWindow;
  const int j0 = p - n + 1;
  if (tid < n) {
    const uint4* k = reinterpret_cast<const uint4*>(ring + (int64_t)((j0 + tid) % kWindow) * width + head * kHead);
    uint4 u[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) u[q] = k[q];
    float dot = 0.f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const bf16* e = reinterpret_cast<const bf16*>(&u[q]);
#pragma unroll
      for (int x = 0; x < 8; ++x) dot += sq[q * 8 + x] * f32(e[x]);
    }
    ss[tid] = dot * rsqrtf((float)kHead);
  }
  __syncthreads();
  float m = -INFINITY;
  for (int j = lane; j < n; j += 32) m = fmaxf(m, ss[j]);
  m = warp_max(m);
  float l = 0.f;
  for (int j = lane; j < n; j += 32) l += __expf(ss[j] - m);
  l = warp_sum(l);
  const int grp = tid >> 3, sub = tid & 7;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const bf16* vbase = ring + vofs + head * kHead + sub * 8;
  uint4 u[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int jj = grp + 16 * i;
    if (jj < n) u[i] = *reinterpret_cast<const uint4*>(vbase + (int64_t)((j0 + jj) % kWindow) * width);
  }
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int jj = grp + 16 * i;
    if (jj >= n) continue;
    const float pj = __expf(ss[jj] - m);
    const bf16* e = reinterpret_cast<const bf16*>(&u[i]);
#pragma unroll
    for (int x = 0; x < 8; ++x) acc[x] += pj * f32(e[x]);
  }
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    acc[x] += __shfl_xor_sync(0xffffffffu, acc[x], 8);
    acc[x] += __shfl_xor_sync(0xffffffffu, acc[x], 16);
  }
  if (lane < 8) {
#pragma unroll
    for (int x = 0; x < 8; ++x) sacc[w][sub * 8 + x] = acc[x];
  }
  __syncthreads();
  if (tid < kHead) {
    const float o = (sacc[0][tid] + sacc[1][tid] + sacc[2][tid] + sacc[3][tid]) / l;
    out[(int64_t)r * width + head * kHead + tid] = to_bf16(o);
  }
}

// A causal conv's input, act(x + bias) with act SnakeBeta or nothing, of every
// row of the frame and of the H = (K - 1) d rows before it (from the history),
// scattered into the dilated im2col [rows, K * C] (tap-major); the last H of
// those rows are the next frame's history. One thread per 8 channels of a row,
// `total` counting them over seqs * (H + T) rows.
template <bool SNAKE>
__device__ __forceinline__ void im2col(const bf16* x, const bf16* bias, const float* a, const float* inv_b, void* state,
                                       const int32_t* pos, const int32_t* lines, bf16* col, int T, int C, int K, int d,
                                       int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int H = (K - 1) * d, C8 = C / 8;
  const int c = i % C8 * 8, g = i / C8;
  const int s = g / (H + T), r = g % (H + T);
  const int p = pos[s];
  bf16* hist = slot(state, lines, s, stride);
  const int64_t half = (int64_t)H * C;
  uint4 u;
  if (r < H) {
    u = *reinterpret_cast<const uint4*>(hist + (p & 1) * half + (int64_t)r * C + c);
  } else {
    float v[8];
    load8(x + ((int64_t)s * T + r - H) * C + c, v);
    if (SNAKE) {
      bias_snake8(v, bias, a, inv_b, c);
    } else {
      float b[8];
      load8(bias + c, b);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] += b[k];
    }
    u = pack8(v);
  }
  const int src = r - H;
  for (int j = 0; j < K; ++j) {
    const int t = src + (K - 1 - j) * d;
    if (t >= 0 && t < T) *reinterpret_cast<uint4*>(col + (((int64_t)s * T + t) * K + j) * C + c) = u;
  }
  if (r >= T) *reinterpret_cast<uint4*>(hist + ((p + 1) & 1) * half + (int64_t)(r - T) * C + c) = u;
}

extern "C" __global__ void codec_im2col(const bf16* x, const bf16* bias, void* state, const int32_t* pos,
                                        const int32_t* lines, bf16* col, int T, int C, int K, int d, int64_t stride,
                                        int total) {
  im2col<false>(x, bias, nullptr, nullptr, state, pos, lines, col, T, C, K, d, stride, total);
}

extern "C" __global__ void codec_im2col_snake(const bf16* x, const bf16* bias, const float* a, const float* inv_b,
                                              void* state, const int32_t* pos, const int32_t* lines, bf16* col, int T,
                                              int C, int K, int d, int64_t stride, int total) {
  im2col<true>(x, bias, a, inv_b, state, pos, lines, col, T, C, K, d, stride, total);
}

// Overlap-add of a transposed conv with kernel 2r, stride r: z rows are
// [L per sequence, 2r * C] tap-major, output row t of a sequence takes tap
// t % r of row t / r and tap t % r + r of the row before, which for the
// sequence's first row is the previous frame's last row, kept in the slot.
// Eight channels per thread.
extern "C" __global__ void codec_col2im(const bf16* z, const bf16* bias, void* state, const int32_t* lines, bf16* out,
                                        int L, int r, int C, int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int C8 = C / 8;
  const int c = i % C8 * 8, g = i / C8;
  const int s = g / (L * r), t = g % (L * r);
  const int l = t / r, j = t % r;
  const int64_t w = (int64_t)2 * r * C;
  const bf16* zs = z + (int64_t)s * L * w;
  float v[8], y[8];
  load8(bias + c, v);
  load8(zs + l * w + j * C + c, y);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += y[k];
  if (l > 0) {
    load8(zs + (l - 1) * w + (j + r) * C + c, y);
  } else {
    bf16* prev = slot(state, lines, s, stride) + j * C + c;
    load8(prev, y);
    copy8(prev, zs + (L - 1) * w + (j + r) * C + c);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += y[k];
  store8(out + (int64_t)i * 8, v);
}

// ConvNeXt's input x = z + zb, z the rows of a transposed conv with kernel =
// stride = r (its GEMM output, already in row order) and zb its bias per
// (row % r, channel), kept in `xin` for the residual; then the depthwise causal
// conv (w [K, C]) and LayerNorm into `out`. One block per row, eight channels
// per thread (C = 8 * blockDim.x); earlier rows from the history (H = K - 1).
extern "C" __global__ void codec_dwconv_ln(const bf16* z, const bf16* zb, const bf16* w, const bf16* b,
                                           const bf16* ln_w, const bf16* ln_b, void* state, const int32_t* pos,
                                           const int32_t* lines, bf16* xin, bf16* out, int T, int r, int C, int K,
                                           int64_t stride, float eps) {
  const int64_t g = blockIdx.x;
  const int s = g / T, t = g % T, c = threadIdx.x * 8, H = K - 1;
  const int p = pos[s];
  bf16* hist = slot(state, lines, s, stride);
  const bf16* old = hist + (p & 1) * (int64_t)H * C;
  bf16* next = hist + ((p + 1) & 1) * (int64_t)H * C;
  float acc[8], cur[8];
  load8(b + c, acc);
  for (int j = 0; j < K; ++j) {
    const int src = t - (K - 1 - j);
    float v[8], wj[8];
    if (src >= 0) {
      float zbv[8];
      load8(z + ((int64_t)s * T + src) * C + c, v);
      load8(zb + (src % r) * C + c, zbv);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = round_bf16(v[k] + zbv[k]);
    } else {
      load8(old + (int64_t)(H + src) * C + c, v);
    }
    load8(w + j * C + c, wj);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      acc[k] += wj[k] * v[k];
      cur[k] = v[k];
    }
  }
  store8(xin + g * C + c, cur);
  if (t + H - T >= 0) store8(next + (int64_t)(t + H - T) * C + c, cur);
  for (int m = t; m < H - T; m += T) copy8(next + (int64_t)m * C + c, old + (int64_t)(m + T) * C + c);
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    acc[k] = round_bf16(acc[k]);
    sum += acc[k];
  }
  const float mean = block_sum(sum) / C;
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) sq += (acc[k] - mean) * (acc[k] - mean);
  const float rstd = rsqrtf(block_sum(sq) / C + eps);
  float lw[8], lb[8];
  load8(ln_w + c, lw);
  load8(ln_b + c, lb);
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = (acc[k] - mean) * rstd * lw[k] + lb[k];
  store8(out + g * C + c, acc);
}

constexpr int kOutTile = 128;
constexpr int kOutC = 128;
constexpr int kOutK = 8;

// The last SnakeBeta (of x + bias), the C -> 1 conv (w [K, C]) and the clamp
// to [-1, 1]. A block per (128 rows, sequence) holds its rows and the K - 1
// before them snaked in shared memory, a thread per output sample. C at most
// 128, K at most 8, T at least K - 1.
extern "C" __global__ void __launch_bounds__(kOutTile)
    codec_conv_out(const bf16* x, const bf16* bias, const float* a, const float* inv_b, const bf16* w, const bf16* wb,
                   void* state, const int32_t* pos, const int32_t* lines, bf16* out, int T, int C, int K,
                   int64_t stride) {
  // Rows C / 2 + 1 words apart, an odd count: a warp reading consecutive rows hits distinct banks.
  __shared__ __nv_bfloat162 sx[(kOutTile + kOutK) * (kOutC / 2 + 1)];
  __shared__ float sw[kOutK * kOutC];
  const int s = blockIdx.y, t0 = blockIdx.x * kOutTile, H = K - 1, tid = threadIdx.x;
  const int ld = C / 2 + 1, C8 = C / 8;
  const int rows = min(kOutTile, T - t0) + H;
  const int p = pos[s];
  bf16* hist = slot(state, lines, s, stride);
  const bf16* old = hist + (p & 1) * (int64_t)H * C;
  bf16* next = hist + ((p + 1) & 1) * (int64_t)H * C;
  for (int i = tid; i < K * C; i += blockDim.x) sw[i] = f32(w[i]);
  for (int i = tid; i < rows * C8; i += blockDim.x) {
    const int row = i / C8, c = i % C8 * 8, src = t0 - H + row;
    float v[8];
    if (src < 0) {
      load8(old + (int64_t)(H + src) * C + c, v);
    } else {
      load8(x + ((int64_t)s * T + src) * C + c, v);
      bias_snake8(v, bias, a, inv_b, c);
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = round_bf16(v[k]);
      if (src >= T - H && row >= H) store8(next + (int64_t)(src - (T - H)) * C + c, v);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) sx[row * ld + c / 2 + k] = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
  }
  __syncthreads();
  if (t0 + tid >= T) return;
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    const __nv_bfloat162* xr = sx + (tid + j) * ld;
    const float* wr = sw + j * C;
    for (int c2 = 0; c2 < C / 2; ++c2) {
      const float2 v = __bfloat1622float2(xr[c2]);
      acc += v.x * wr[2 * c2] + v.y * wr[2 * c2 + 1];
    }
  }
  out[(int64_t)s * T + t0 + tid] = to_bf16(fminf(1.f, fmaxf(-1.f, round_bf16(acc + f32(wb[0])))));
}
