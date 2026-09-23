// The streaming codec decoder's kernels, compiled to one cubin that the kern
// runtime launches from the calls `codec::build` generates.
//
// A call decodes one frame for each of `seqs` sequences. At a stage running
// `T` rows per frame, global row `g` is row `t = g % T` of sequence `s = g / T`,
// whose absolute row index is `pos[s] * T + t`. Everything a causal layer needs
// from earlier frames lives in the sequence's slot of the per-sequence state
// (`state + lines[s] * stride`, plus the layer's offset the call adds):
//
// - a ring of the last `H` input rows of every causal conv, row `q` at `q % H`;
// - the second half of the last GEMM row of every transposed conv;
// - K and V of the last 72 frames of every attention layer, frame `p` at `p % 72`.
//
// Rows before the sequence start read as zero, which is the convs' causal
// padding; a fresh slot is zeroed by its lease.
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }

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

__device__ __forceinline__ float snake(float x, float a, float inv_b) {
  const float s = sinf(a * x);
  return x + inv_b * s * s;
}

__device__ __forceinline__ bf16* slot(void* state, const int32_t* lines, int s, int64_t stride) {
  return reinterpret_cast<bf16*>(static_cast<char*>(state) + (int64_t)lines[s] * stride);
}

#define GRID_STRIDE(i, total) \
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (total); i += (int64_t)gridDim.x * blockDim.x)

// Row r: [codebook 0 | Σ codebooks 1..15], each `half` wide; `books` is
// [16 * rows_per_book, half]. A codebook-0 code past the book (the talker's end
// of speech, whose audio is dropped) reads as 0.
extern "C" __global__ void codec_rvq(const int32_t* codes, const bf16* books, bf16* out, int half, int rows_per_book) {
  const int r = blockIdx.x;
  const int32_t* c = codes + r * 16;
  const int64_t c0 = c[0] < rows_per_book ? c[0] : 0;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    out[(int64_t)r * 2 * half + i] = books[c0 * half + i];
    float acc = 0.f;
    for (int g = 1; g < 16; ++g) acc += f32(books[((int64_t)g * rows_per_book + c[g]) * half + i]);
    out[(int64_t)r * 2 * half + half + i] = to_bf16(acc);
  }
}

template <int ACT, bool RESIDUAL>
__device__ void bias_act(const bf16* x, const bf16* bias, bf16* out, int cols, int64_t total) {
  GRID_STRIDE(i, total) {
    float v = f32(to_bf16(f32(x[i]) + f32(bias[i % cols])));
    if (ACT == 1) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    if (RESIDUAL) v = f32(to_bf16(v)) + f32(out[i]);
    out[i] = to_bf16(v);
  }
}

extern "C" __global__ void codec_bias(bf16* x, const bf16* bias, int cols, int total) {
  bias_act<0, false>(x, bias, x, cols, total);
}

extern "C" __global__ void codec_bias_gelu(bf16* x, const bf16* bias, int cols, int total) {
  bias_act<1, false>(x, bias, x, cols, total);
}

// res += x + bias
extern "C" __global__ void codec_bias_residual(const bf16* x, const bf16* bias, bf16* res, int cols, int total) {
  bias_act<0, true>(x, bias, res, cols, total);
}

// out = snake(x), 8 channels per thread (`cols` a multiple of 8, `total` counts
// the groups).
extern "C" __global__ void codec_snake(const bf16* x, const float* a, const float* inv_b, bf16* out, int cols,
                                       int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = i % (cols / 8) * 8;
  uint4 v = reinterpret_cast<const uint4*>(x)[i];
  bf16* e = reinterpret_cast<bf16*>(&v);
#pragma unroll
  for (int k = 0; k < 8; ++k) e[k] = to_bf16(snake(f32(e[k]), a[c + k], inv_b[c + k]));
  reinterpret_cast<uint4*>(out)[i] = v;
}

// out = snake(x + bias), per channel.
extern "C" __global__ void codec_bias_snake(const bf16* x, const bf16* bias, const float* a, const float* inv_b,
                                            bf16* out, int cols, int total) {
  GRID_STRIDE(i, total) {
    const int c = i % cols;
    out[i] = to_bf16(snake(f32(to_bf16(f32(x[i]) + f32(bias[c]))), a[c], inv_b[c]));
  }
}

// FlashInfer's RMSNorm: f32 statistics, one block per row.
extern "C" __global__ void codec_rms_norm(const bf16* x, const bf16* w, bf16* out, int dim, float eps) {
  const bf16* row = x + (int64_t)blockIdx.x * dim;
  float ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += f32(row[i]) * f32(row[i]);
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[(int64_t)blockIdx.x * dim + i] = to_bf16(f32(row[i]) * r * f32(w[i]));
}

// FlashInfer's FusedAddRMSNorm: res += x; x = rms_norm(res) * w, the norm
// taken over the unrounded sum. `dim` is at most 8 * blockDim.x.
extern "C" __global__ void codec_add_rms_norm(bf16* x, bf16* res, const bf16* w, int dim, float eps) {
  bf16* xr = x + (int64_t)blockIdx.x * dim;
  bf16* rr = res + (int64_t)blockIdx.x * dim;
  float v[8];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    v[k] = 0.f;
    if (i < dim) {
      v[k] = f32(xr[i]) + f32(rr[i]);
      rr[i] = to_bf16(v[k]);
      ss += v[k] * v[k];
    }
  }
  const float r = rsqrtf(block_sum(ss) / dim + eps);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i < dim) xr[i] = to_bf16(v[k] * r * f32(w[i]));
  }
}

constexpr int kHead = 64;
constexpr int kWindow = 72;

// qkv rows are [q | k | v], `heads` heads of 64 each. Q is rotated in place;
// rotated K and V go to the layer's ring (`kv` = K[72][heads*64], V after it)
// at frame `pos % 72`. One warp per (row, head of q, k or v).
extern "C" __global__ void codec_rope_kv(bf16* qkv, const int32_t* pos, const int32_t* lines, void* kv,
                                         int64_t stride, int heads, float theta) {
  const int r = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= 3 * heads) return;
  const int width = heads * kHead;
  bf16* row = qkv + (int64_t)r * 3 * width + head * kHead;
  bf16* ring = slot(kv, lines, r, stride);
  const int p = pos[r];
  const int64_t at = (int64_t)(p % kWindow) * width;
  if (head >= 2 * heads) {
    bf16* dst = ring + (int64_t)kWindow * width + at + (head - 2 * heads) * kHead;
    dst[lane] = row[lane];
    dst[lane + 32] = row[lane + 32];
    return;
  }
  const float lo = f32(row[lane]), hi = f32(row[lane + 32]);
  const float inv_freq = 1.f / powf(theta, (float)(2 * lane) / kHead);
  float s, c;
  sincosf((float)p * inv_freq, &s, &c);
  bf16* dst = head < heads ? row : ring + at + (head - heads) * kHead;
  dst[lane] = to_bf16(lo * c - hi * s);
  dst[lane + 32] = to_bf16(hi * c + lo * s);
}

// Each row's query against the last min(pos + 1, 72) frames of its ring,
// one warp per (row, head), online softmax in f32.
extern "C" __global__ void codec_ring_attention(const bf16* qkv, const int32_t* pos, const int32_t* lines,
                                                const void* kv, bf16* out, int64_t stride, int heads) {
  const int r = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (head >= heads) return;
  const int width = heads * kHead;
  const bf16* q = qkv + (int64_t)r * 3 * width + head * kHead;
  const bf16* ring = slot(const_cast<void*>(kv), lines, r, stride);
  const float q0 = f32(q[lane]), q1 = f32(q[lane + 32]);
  const float scale = rsqrtf((float)kHead);
  const int p = pos[r];
  const int n = p + 1 < kWindow ? p + 1 : kWindow;
  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
  for (int j = p - n + 1; j <= p; ++j) {
    const bf16* k = ring + (int64_t)(j % kWindow) * width + head * kHead;
    const bf16* v = k + (int64_t)kWindow * width;
    const float score = warp_sum(q0 * f32(k[lane]) + q1 * f32(k[lane + 32])) * scale;
    const float m2 = fmaxf(m, score);
    const float corr = __expf(m - m2), e = __expf(score - m2);
    l = l * corr + e;
    a0 = a0 * corr + e * f32(v[lane]);
    a1 = a1 * corr + e * f32(v[lane + 32]);
    m = m2;
  }
  bf16* o = out + (int64_t)r * width + head * kHead;
  o[lane] = to_bf16(a0 / l);
  o[lane + 32] = to_bf16(a1 / l);
}

__device__ __forceinline__ bool causal_row(const bf16* x, const bf16* ring, int64_t g, int t, int src, int p, int T,
                                           int C, int H, int c, float* v) {
  if (src >= 0) {
    *v = f32(x[(g - t + src) * C + c]);
    return true;
  }
  const int64_t q = (int64_t)p * T + src;
  if (q < 0) return false;
  *v = f32(ring[(q % H) * C + c]);
  return true;
}

// The causal, dilated window of every row gathered into [rows, K*C]
// (tap-major), rows before the chunk from the ring. One thread per 8 channels:
// C is a multiple of 8, so every access is 16 bytes and aligned; `total` counts
// those groups.
extern "C" __global__ void codec_im2col(const bf16* x, const void* state, const int32_t* pos, const int32_t* lines,
                                        bf16* col, int T, int C, int K, int d, int H, int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int C8 = C / 8;
  const int c = (i % C8) * 8, gj = i / C8;
  const int j = gj % K, g = gj / K;
  const int s = g / T, t = g % T;
  const int src = t - (K - 1 - j) * d;
  const int64_t q = (int64_t)pos[s] * T + src;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (src >= 0)
    v = *reinterpret_cast<const uint4*>(x + (int64_t)(g - t + src) * C + c);
  else if (q >= 0)
    v = *reinterpret_cast<const uint4*>(slot(const_cast<void*>(state), lines, s, stride) + (q % H) * C + c);
  reinterpret_cast<uint4*>(col)[i] = v;
}

// The last min(T, H) rows of every sequence's chunk into its ring; `total` is
// seqs * W * C with W = min(T, H).
extern "C" __global__ void codec_ring_write(const bf16* x, void* state, const int32_t* pos, const int32_t* lines,
                                            int T, int C, int H, int W, int64_t stride, int total) {
  GRID_STRIDE(i, total) {
    const int c = i % C;
    const int k = (i / C) % W;
    const int s = i / (C * W);
    const int t = T - W + k;
    const int64_t q = (int64_t)pos[s] * T + t;
    slot(state, lines, s, stride)[(q % H) * C + c] = x[((int64_t)s * T + t) * C + c];
  }
}

// Overlap-add of a transposed conv with kernel 2r, stride r: z rows are
// [L per sequence, 2r * C] tap-major, output row t of a sequence takes tap
// t % r of row t / r and tap t % r + r of the row before, which for the
// sequence's first row is the previous chunk's last row, kept in the slot.
extern "C" __global__ void codec_col2im(const bf16* z, const bf16* bias, void* state, const int32_t* lines, bf16* out,
                                        int L, int r, int C, int64_t stride, int total) {
  GRID_STRIDE(i, total) {
    const int c = i % C;
    const int64_t g = i / C;
    const int s = g / ((int64_t)L * r), t = g % ((int64_t)L * r);
    const int l = t / r, j = t % r;
    const bf16* zs = z + (int64_t)s * L * 2 * r * C;
    float v = f32(bias[c]) + f32(zs[(int64_t)l * 2 * r * C + j * C + c]);
    if (l > 0) {
      v += f32(zs[(int64_t)(l - 1) * 2 * r * C + (j + r) * C + c]);
    } else {
      bf16* prev = slot(state, lines, s, stride) + j * C + c;
      v += f32(*prev);
      *prev = zs[(int64_t)(L - 1) * 2 * r * C + (j + r) * C + c];
    }
    out[i] = to_bf16(v);
  }
}

// Transposed conv with kernel = stride = r: no overlap, no state.
extern "C" __global__ void codec_unfold(const bf16* z, const bf16* bias, bf16* out, int C, int total) {
  GRID_STRIDE(i, total) out[i] = to_bf16(f32(bias[i % C]) + f32(z[i]));
}

// ConvNeXt's depthwise causal conv (w [C, K]) and LayerNorm, one block of 256
// per row, C at most 2048; earlier rows from the ring (H = K - 1).
extern "C" __global__ void __launch_bounds__(256)
    codec_dwconv_ln(const bf16* x, const bf16* w, const bf16* b, const bf16* ln_w, const bf16* ln_b,
                    const void* state, const int32_t* pos, const int32_t* lines, bf16* out, int T, int C, int K,
                    int64_t stride, float eps) {
  const int64_t g = blockIdx.x;
  const int s = g / T, t = g % T;
  const bf16* ring = slot(const_cast<void*>(state), lines, s, stride);
  float h[8];
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = threadIdx.x + k * 256;
    h[k] = 0.f;
    if (c >= C) continue;
    float v = f32(b[c]);
    for (int j = 0; j < K; ++j) {
      float xv;
      if (causal_row(x, ring, g, t, t - (K - 1 - j), pos[s], T, C, K - 1, c, &xv)) v += f32(w[c * K + j]) * xv;
    }
    h[k] = f32(to_bf16(v));
    sum += h[k];
  }
  const float mean = block_sum(sum) / C;
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = threadIdx.x + k * 256;
    if (c < C) sq += (h[k] - mean) * (h[k] - mean);
  }
  const float rstd = rsqrtf(block_sum(sq) / C + eps);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = threadIdx.x + k * 256;
    if (c < C) out[g * C + c] = to_bf16((h[k] - mean) * rstd * f32(ln_w[c]) + f32(ln_b[c]));
  }
}

// The last SnakeBeta, the C -> 1 conv (w [K, C]) and the clamp to [-1, 1];
// one warp per sample, earlier rows from the ring (H = K - 1).
extern "C" __global__ void codec_conv_out(const bf16* x, const float* a, const float* inv_b, const bf16* w,
                                          const bf16* bias, const void* state, const int32_t* pos,
                                          const int32_t* lines, bf16* out, int T, int C, int K, int64_t stride,
                                          int rows) {
  const int64_t g = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (g >= rows) return;
  const int s = g / T, t = g % T;
  const bf16* ring = slot(const_cast<void*>(state), lines, s, stride);
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    for (int c = lane; c < C; c += 32) {
      float xv;
      if (!causal_row(x, ring, g, t, t - (K - 1 - j), pos[s], T, C, K - 1, c, &xv)) continue;
      acc += f32(to_bf16(snake(xv, a[c], inv_b[c]))) * f32(w[j * C + c]);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) out[g] = to_bf16(fminf(1.f, fmaxf(-1.f, f32(to_bf16(acc + f32(bias[0]))))));
}

// SiLU(gate) * up over fused [gate | up] rows.
extern "C" __global__ void codec_silu_mul(const bf16* gate_up, bf16* out, int inter, int total) {
  GRID_STRIDE(i, total) {
    const int64_t n = i / inter, c = i % inter;
    const float g = f32(gate_up[n * 2 * inter + c]);
    const float u = f32(gate_up[n * 2 * inter + inter + c]);
    out[i] = to_bf16(f32(to_bf16(g / (1.f + __expf(-g)))) * u);
  }
}
