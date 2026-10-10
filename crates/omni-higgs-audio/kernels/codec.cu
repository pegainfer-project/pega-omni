// The codec decoder's kernels, compiled to one cubin the kern runtime launches from the calls
// `codec.rs` generates.
//
// A call decodes one window of frames into PCM. Activations are `[rows, C]` bf16, row `t` of a
// stage being sample `t` at that stage's rate. Every conv is non-causal with symmetric zero
// padding: a dense conv is `im2col` and a GEMM, a transposed conv a GEMM and an overlap-add, so a
// window's edges see zeros where the whole utterance would see its neighbours.
//
// Elementwise kernels move eight channels (16 bytes) per thread, so channel counts are multiples
// of eight; `total` counts those groups.
#include "common.cuh"

constexpr int kTaps = 7;
constexpr int kBooks = 8;

// Snake: x + sin²(a·x) / a, with 1 / (a + 1e-9) precomputed.
__device__ __forceinline__ float snake(float x, float a, float inv) {
  const float s = sinf(a * x);
  return x + inv * s * s;
}

// Row t = bias + the sum over the eight quantizers of row `codes[t][q]` of quantizer q's table
// (each `[vocab, width]`): the quantizers and `fc2` folded together.
extern "C" __global__ void codec_lookup(const int32_t* codes, const bf16* tables, const bf16* bias, bf16* out,
                                        int width, int vocab, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int per = width / 8;
  const int64_t t = i / per;
  const int c = (i % per) * 8;
  float acc[8], e[8];
  load8(bias + c, acc);
  for (int q = 0; q < kBooks; ++q) {
    load8(tables + ((int64_t)q * vocab + codes[t * kBooks + q]) * width + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  store8(out + t * width + c, acc);
}

// out = snake(x), and with a bias, snake(x + bias).
extern "C" __global__ void codec_snake(const bf16* x, const float* a, const float* inv, bf16* out, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = (i % (cols / 8)) * 8;
  float v[8];
  load8(x + (int64_t)i * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = snake(v[k], a[c + k], inv[c + k]);
  store8(out + (int64_t)i * 8, v);
}

extern "C" __global__ void codec_bias_snake(const bf16* x, const bf16* bias, const float* a, const float* inv,
                                            bf16* out, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = (i % (cols / 8)) * 8;
  float v[8], b[8];
  load8(x + (int64_t)i * 8, v);
  load8(bias + c, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = snake(v[k] + b[k], a[c + k], inv[c + k]);
  store8(out + (int64_t)i * 8, v);
}

// The dilated `[rows, 7 * C]` im2col of a conv's input, tap-major: tap j of row t is row
// t + (j - 3) * dil, zeros past the ends, optionally snaked first (snake(0) = 0, so the padding
// stays zero).
template <bool SNAKE>
__device__ __forceinline__ void im2col(const bf16* x, const float* a, const float* inv, bf16* out, int rows, int cols,
                                       int dil, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int per = cols / 8, row_groups = kTaps * per;
  const int64_t t = i / row_groups;
  const int j = (i % row_groups) / per, c = (i % per) * 8;
  const int64_t r = t + (int64_t)(j - kTaps / 2) * dil;
  float v[8];
  if (r < 0 || r >= rows) {
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = 0.f;
  } else {
    load8(x + r * cols + c, v);
    if (SNAKE) {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[k] = snake(v[k], a[c + k], inv[c + k]);
    }
  }
  store8(out + t * kTaps * cols + j * cols + c, v);
}

extern "C" __global__ void codec_im2col(const bf16* x, bf16* out, int rows, int cols, int dil, int total) {
  im2col<false>(x, nullptr, nullptr, out, rows, cols, dil, total);
}

extern "C" __global__ void codec_im2col_snake(const bf16* x, const float* a, const float* inv, bf16* out, int rows,
                                              int cols, int dil, int total) {
  im2col<true>(x, a, inv, out, rows, cols, dil, total);
}

// Overlap-add of a transposed conv with kernel 2s, stride s and padding ceil(s / 2): the GEMM rows
// `z` are `[rows_in, 2s * C]`, tap j of input row i at `z[i][j * C ..]`. Output row t takes the
// full-length output's row t + pad, which sums tap u % s of input row u / s and tap u % s + s of
// the row before. Plus the conv's bias.
extern "C" __global__ void codec_col2im(const bf16* z, const bf16* bias, bf16* out, int rows_in, int stride_s,
                                        int pad, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int per = cols / 8;
  const int64_t t = i / per;
  const int c = (i % per) * 8;
  const int64_t u = t + pad, i1 = u / stride_s;
  const int j1 = u % stride_s;
  const int64_t row = (int64_t)2 * stride_s * cols;
  float acc[8], e[8];
  load8(bias + c, acc);
  if (i1 < rows_in) {
    load8(z + i1 * row + (int64_t)j1 * cols + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  if (i1 >= 1) {
    load8(z + (i1 - 1) * row + (int64_t)(j1 + stride_s) * cols + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  store8(out + t * cols + c, acc);
}

// A residual unit's end: x += y + bias.
extern "C" __global__ void codec_residual(bf16* x, const bf16* y, const bf16* bias, int cols, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = (i % (cols / 8)) * 8;
  float v[8], e[8], b[8];
  load8(x + (int64_t)i * 8, v);
  load8(y + (int64_t)i * 8, e);
  load8(bias + c, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += e[k] + b[k];
  store8(x + (int64_t)i * 8, v);
}

// The last snake and the C -> 1 conv, kernel 7: pcm[t] = b + Σ_j Σ_c w[j][c] snake(x[t + j - 3][c]),
// f32. One thread per sample.
extern "C" __global__ void codec_conv_out(const bf16* x, const float* a, const float* inv, const float* w,
                                          const float* b, float* pcm, int rows, int cols) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows) return;
  float acc = b[0];
  for (int j = 0; j < kTaps; ++j) {
    const int r = t + j - kTaps / 2;
    if (r < 0 || r >= rows) continue;
    for (int c = 0; c < cols; c += 8) {
      float v[8];
      load8(x + (int64_t)r * cols + c, v);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc += w[j * cols + c + k] * snake(v[k], a[c + k], inv[c + k]);
    }
  }
  pcm[t] = acc;
}
