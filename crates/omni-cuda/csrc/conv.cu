// 1-D convolution for the codec decoder, time-major: activations are [T, C]
// rows so that every dense convolution is a GEMM over T.
//
// - `omni_im2col`: gathers the causal, dilated window of each output step into
//   a [T, K*C] row (tap-major), applying `snake(x + bias)` on the way in. With
//   K = 1 it is the standalone (bias+)SnakeBeta.
// - `omni_col2im`: the overlap-add of a transposed convolution whose GEMM
//   produced [L, K*C_out] rows, plus bias, with the causal right trim.
// - `omni_dwconv_layernorm`: ConvNeXt's depthwise causal conv and LayerNorm.
// - `omni_conv_out`: the last SnakeBeta, the C -> 1 conv and the clamp to
//   [-1, 1], producing f32 samples.
//
// SnakeBeta is `x + inv_b * sin(a * x)^2` with `a = exp(alpha)` and
// `inv_b = 1 / (exp(beta) + 1e-9)` folded at load.
#include "common.cuh"

__device__ __forceinline__ float snake(float x, float a, float inv_b) {
  const float s = sinf(a * x);
  return x + inv_b * s * s;
}

__global__ void im2col_kernel(const bf16* x, const bf16* bias, const float* a, const float* inv_b, bf16* col,
                              uint32_t t_len, uint32_t c_len, uint32_t k_len, uint32_t dilation, size_t total) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t c = i % c_len;
    const uint32_t j = (i / c_len) % k_len;
    const uint32_t t = i / ((size_t)c_len * k_len);
    const int64_t src = (int64_t)t - (int64_t)(k_len - 1 - j) * dilation;
    float v = 0.f;
    if (src >= 0) {
      v = to_f32(x[(size_t)src * c_len + c]);
      if (bias) v = to_f32(to_bf16(v + to_f32(bias[c])));
      if (a) v = snake(v, a[c], inv_b[c]);
    }
    col[i] = to_bf16(v);
  }
}

__global__ void col2im_kernel(const bf16* z, const bf16* bias, bf16* out, uint32_t l_len, uint32_t c_len,
                              uint32_t k_len, uint32_t stride, size_t total) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t c = i % c_len;
    const uint32_t t = i / c_len;
    float v = bias ? to_f32(bias[c]) : 0.f;
    for (uint32_t m = 0; m * stride < k_len; ++m) {
      const int64_t l = (int64_t)(t / stride) - m;
      if (l < 0 || l >= l_len) continue;
      const uint32_t j = t % stride + m * stride;
      v += to_f32(z[(size_t)l * k_len * c_len + (size_t)j * c_len + c]);
    }
    out[i] = to_bf16(v);
  }
}

constexpr int kLnThreads = 256;
constexpr int kLnMaxPerThread = 8;

__global__ void __launch_bounds__(kLnThreads)
    dwconv_layernorm_kernel(const bf16* x, const bf16* w, const bf16* b, const bf16* ln_w, const bf16* ln_b,
                            bf16* out, uint32_t c_len, uint32_t k_len, float eps) {
  const uint32_t t = blockIdx.x;
  float h[kLnMaxPerThread];
  float sum = 0.f;
#pragma unroll
  for (int r = 0; r < kLnMaxPerThread; ++r) {
    const uint32_t c = threadIdx.x + r * kLnThreads;
    h[r] = 0.f;
    if (c >= c_len) continue;
    float v = to_f32(b[c]);
    for (uint32_t j = 0; j < k_len; ++j) {
      const int64_t src = (int64_t)t - (int64_t)(k_len - 1 - j);
      if (src >= 0) v += to_f32(w[c * k_len + j]) * to_f32(x[(size_t)src * c_len + c]);
    }
    h[r] = to_f32(to_bf16(v));
    sum += h[r];
  }
  const float mean = block_sum(sum) / c_len;
  float sq = 0.f;
#pragma unroll
  for (int r = 0; r < kLnMaxPerThread; ++r) {
    const uint32_t c = threadIdx.x + r * kLnThreads;
    if (c < c_len) sq += (h[r] - mean) * (h[r] - mean);
  }
  const float rstd = rsqrtf(block_sum(sq) / c_len + eps);
#pragma unroll
  for (int r = 0; r < kLnMaxPerThread; ++r) {
    const uint32_t c = threadIdx.x + r * kLnThreads;
    if (c < c_len) out[(size_t)t * c_len + c] = to_bf16((h[r] - mean) * rstd * to_f32(ln_w[c]) + to_f32(ln_b[c]));
  }
}

// One warp per output sample.
__global__ void conv_out_kernel(const bf16* x, const float* a, const float* inv_b, const bf16* w, const bf16* bias,
                                float* out, uint32_t t_len, uint32_t c_len, uint32_t k_len) {
  const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const uint32_t lane = threadIdx.x & 31;
  if (t >= t_len) return;
  float acc = 0.f;
  for (uint32_t j = 0; j < k_len; ++j) {
    const int64_t src = (int64_t)t - (int64_t)(k_len - 1 - j);
    if (src < 0) continue;
    for (uint32_t c = lane; c < c_len; c += 32) {
      const float v = to_f32(to_bf16(snake(to_f32(x[(size_t)src * c_len + c]), a[c], inv_b[c])));
      acc += v * to_f32(w[j * c_len + c]);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) out[t] = fminf(1.f, fmaxf(-1.f, to_f32(to_bf16(acc + to_f32(bias[0])))));
}

static unsigned grid_for(size_t total) {
  const size_t blocks = (total + 255) / 256;
  return (unsigned)(blocks < 65535 * 8 ? blocks : 65535 * 8);
}

extern "C" {

int omni_im2col(const bf16* x, const bf16* bias, const float* a, const float* inv_b, bf16* col, uint32_t t_len,
                uint32_t c_len, uint32_t k_len, uint32_t dilation, cudaStream_t stream) {
  const size_t total = (size_t)t_len * k_len * c_len;
  if (total == 0) return 0;
  im2col_kernel<<<grid_for(total), 256, 0, stream>>>(x, bias, a, inv_b, col, t_len, c_len, k_len, dilation, total);
  return (int)cudaGetLastError();
}

// `z` is [l_len, k_len * c_len] (tap-major); `out` is [l_len * stride, c_len].
int omni_col2im(const bf16* z, const bf16* bias, bf16* out, uint32_t l_len, uint32_t c_len, uint32_t k_len,
                uint32_t stride, cudaStream_t stream) {
  const size_t total = (size_t)l_len * stride * c_len;
  if (total == 0) return 0;
  col2im_kernel<<<grid_for(total), 256, 0, stream>>>(z, bias, out, l_len, c_len, k_len, stride, total);
  return (int)cudaGetLastError();
}

// `w` is [c_len, k_len].
int omni_dwconv_layernorm(const bf16* x, const bf16* w, const bf16* b, const bf16* ln_w, const bf16* ln_b, bf16* out,
                          uint32_t t_len, uint32_t c_len, uint32_t k_len, float eps, cudaStream_t stream) {
  if (c_len > kLnThreads * kLnMaxPerThread) return (int)cudaErrorInvalidValue;
  if (t_len == 0) return 0;
  dwconv_layernorm_kernel<<<t_len, kLnThreads, 0, stream>>>(x, w, b, ln_w, ln_b, out, c_len, k_len, eps);
  return (int)cudaGetLastError();
}

// `w` is [k_len, c_len] (tap-major).
int omni_conv_out(const bf16* x, const float* a, const float* inv_b, const bf16* w, const bf16* bias, float* out,
                  uint32_t t_len, uint32_t c_len, uint32_t k_len, cudaStream_t stream) {
  if (t_len == 0) return 0;
  constexpr unsigned warps = 8;
  conv_out_kernel<<<(t_len + warps - 1) / warps, warps * 32, 0, stream>>>(x, a, inv_b, w, bias, out, t_len, c_len,
                                                                          k_len);
  return (int)cudaGetLastError();
}

}  // extern "C"
