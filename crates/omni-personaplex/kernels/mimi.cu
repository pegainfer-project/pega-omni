// Mimi, streamed: every tick encodes one 1920-sample frame of each session's
// caller audio to 8 codes and decodes one frame of the agent's codes to 1920
// samples, the kernels `mimi.rs` generates calls to.
//
// Activations are `[rows, C]` bf16, a session's T rows of a stage
// contiguous; global row g is row t = g % T of session s = g / T. Dense convs
// are im2col (here) and a GEMM, transposed convs a GEMM and an overlap-add.
// Everything a causal layer needs from earlier frames lives in the session's
// slot of the per-session state (`state + lines[s] * stride`, plus the
// layer's offset the call adds):
//
// - a conv's history: the last H = kernel - stride rows of its (activated)
//   input, as two halves `[2][H][C]`; frame p reads half p & 1 and writes the
//   next frame's into the other, so a launch never overwrites what another of
//   its threads still reads;
// - a transposed conv's partial output (the overlap its next frame adds);
// - K and V of the last CONTEXT positions of every transformer layer, a ring.
//
// A fresh slot is zeroed by its lease: rows before the stream starts read as
// zero, the reference's causal padding. The one exception is the encoder's
// downsampling conv, which the reference pads by replicating its first input
// row; frame 0 does that instead of reading its history.
//
// A bias is not always added where it arises: one that only feeds the next
// layer's input is carried there and added on load (the `bias` params), the
// way a residual block's output bias is carried to the next conv.
#include "attend.cuh"

__device__ __forceinline__ float elu(float x) { return x > 0.f ? x : expm1f(x); }

__device__ __forceinline__ void loadf8(const float* p, float* v) {
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = p[k];
}

// The encoder's first conv, 1 -> C channels over raw f32 samples, kernel K:
// out[t][c] = b[c] + Σ_j w[c][j] x[t - (K - 1) + j]. The history is the last
// K - 1 samples (f32). One thread per (session, sample, 8 channels).
extern "C" __global__ void mimi_conv_in(const float* pcm, const float* w, const float* b, void* state,
                                        const int32_t* pos, const int32_t* lines, bf16* out, int T, int C, int K,
                                        int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int C8 = C / 8, H = K - 1;
  const int c = i % C8 * 8, g = i / C8;
  const int s = g / T, t = g % T;
  const int p = pos[s];
  float* hist = slot<float>(state, lines, s, stride);
  const float* old = hist + (p & 1) * H;
  const float* x = pcm + (int64_t)s * T;
  float acc[8];
  loadf8(b + c, acc);
  for (int j = 0; j < K; ++j) {
    const int src = t - H + j;
    const float v = src >= 0 ? x[src] : old[H + src];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += w[(c + k) * K + j] * v;
  }
  store8(out + (int64_t)g * C + c, acc);
  if (c == 0 && t >= T - H) hist[((p + 1) & 1) * H + t - (T - H)] = x[t];
}

// A causal conv's input act(x + bias) (act ELU or nothing) of every row of
// the frame and of the H = K - S rows before it (the history), scattered
// into the im2col `[T / S rows, K * C]` (tap-major) of a conv with kernel K
// and stride S; the last H of those rows are the next frame's history. With
// `replicate`, frame 0's history is its first row (the reference's replicate
// padding). One thread per (session, row, 8 channels) over seqs * (H + T).
extern "C" __global__ void mimi_im2col(const bf16* x, const bf16* bias, void* state, const int32_t* pos,
                                       const int32_t* lines, bf16* col, int T, int C, int K, int S, int act,
                                       int replicate, int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int H = K - S, C8 = C / 8, To = T / S;
  const int c = i % C8 * 8, g = i / C8;
  const int s = g / (H + T), r = g % (H + T);
  const int p = pos[s];
  bf16* hist = slot<bf16>(state, lines, s, stride);
  const int64_t half = (int64_t)H * C;
  auto input = [&](int row) {
    float v[8], b[8];
    load8(x + ((int64_t)s * T + row) * C + c, v);
    load8(bias + c, b);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = act ? elu(round_bf16(v[k] + b[k])) : v[k] + b[k];
    return pack8(v);
  };
  const uint4 u = r >= H ? input(r - H)
                  : (replicate && p == 0) ? input(0)
                                          : *reinterpret_cast<const uint4*>(hist + (p & 1) * half + (int64_t)r * C + c);
  const int lo = max(0, (r - K + S) / S), hi = min(To - 1, r / S);
  for (int t = lo; t <= hi; ++t) {
    const int j = r - t * S;
    if (j >= 0 && j < K) *reinterpret_cast<uint4*>(col + (((int64_t)s * To + t) * K + j) * C + c) = u;
  }
  if (r >= T) *reinterpret_cast<uint4*>(hist + ((p + 1) & 1) * half + (int64_t)(r - T) * C + c) = u;
}

// out = act(x + bias), eight channels per thread (`total` counts the groups).
extern "C" __global__ void mimi_bias_act(const bf16* x, const bf16* bias, bf16* out, int cols, int act, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8], b[8];
  load8(x + (int64_t)i * 8, v);
  load8(bias + i % (cols / 8) * 8, b);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = act ? elu(round_bf16(v[k] + b[k])) : v[k] + b[k];
  store8(out + (int64_t)i * 8, v);
}

// Overlap-add of a transposed conv with kernel 2r, stride r: z rows are
// [L per session, 2r * C] tap-major; output row t of a session takes tap
// t % r of row t / r plus tap t % r + r of the row before, which for the
// first row is the previous frame's last row, kept in the slot. Plus bias.
extern "C" __global__ void mimi_col2im(const bf16* z, const bf16* bias, void* state, const int32_t* lines, bf16* out,
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
    bf16* prev = slot<bf16>(state, lines, s, stride) + j * C + c;
    load8(prev, y);
    copy8(prev, zs + (L - 1) * w + (j + r) * C + c);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] += y[k];
  store8(out + (int64_t)i * 8, v);
}

// The decoder's upsampling: a depthwise transposed conv, kernel 4, stride 2,
// no bias, from one row per session to two: y[t] = x w[:, t] + x_prev w[:, t + 2],
// x_prev (the previous frame's row) kept in the slot. One thread per (session, channel).
extern "C" __global__ void mimi_upsample(const bf16* x, const float* w, void* state, const int32_t* lines, bf16* out,
                                         int C, int64_t stride, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int s = i / C, c = i % C;
  bf16* prev = slot<bf16>(state, lines, s, stride) + c;
  const float now = f32(x[i]), before = f32(*prev);
  out[((int64_t)s * 2) * C + c] = to_bf16(now * w[c * 4] + before * w[c * 4 + 2]);
  out[((int64_t)s * 2 + 1) * C + c] = to_bf16(now * w[c * 4 + 1] + before * w[c * 4 + 3]);
  *prev = x[i];
}

// LayerNorm over `dim` (a multiple of 256, at most 1024) channels, a warp per
// row, with `bias` (per channel, may be all zeros) added to the residual row
// first when `add` is set.
extern "C" __global__ void mimi_layer_norm(bf16* res, const bf16* bias, int add, const bf16* w, const bf16* b,
                                           bf16* out, int dim, float eps, int rows) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  bf16* row = res + (int64_t)r * dim;
  float v[4][8];
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = (k * 32 + lane) * 8;
    if (c >= dim) break;
    load8(row + c, v[k]);
    if (add) {
      float bb[8];
      load8(bias + c, bb);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[k][e] = round_bf16(v[k][e] + bb[e]);
      store8(row + c, v[k]);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) sum += v[k][e];
  }
  const float mean = warp_sum(sum) / dim;
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = (k * 32 + lane) * 8;
    if (c >= dim) break;
#pragma unroll
    for (int e = 0; e < 8; ++e) sq += (v[k][e] - mean) * (v[k][e] - mean);
  }
  const float rstd = rsqrtf(warp_sum(sq) / dim + eps);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = (k * 32 + lane) * 8;
    if (c >= dim) break;
    float g[8], bb[8];
    load8(w + c, g);
    load8(b + c, bb);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[k][e] = (v[k][e] - mean) * rstd * g[e] + bb[e];
    store8(out + (int64_t)r * dim + c, v[k]);
  }
}

// Exact (erf) GELU in place, eight values per thread.
extern "C" __global__ void mimi_gelu(bf16* x, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v[8];
  load8(x + (int64_t)i * 8, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = 0.5f * v[k] * (1.f + erff(v[k] * 0.70710678118654752f));
  store8(x + (int64_t)i * 8, v);
}

constexpr int kMimiHead = 64;

// Transformer rows [q | k | v] (heads of 64), T rows per session at
// positions frame * T + t: rotary embedding of q in place and of k into the
// ring slot position % context, v copied there. A warp per (row, head of q, k or v).
extern "C" __global__ void mimi_rope_kv(bf16* qkv, const int32_t* pos, const int32_t* lines, void* state, int T,
                                        int heads, float coef, int context, int64_t stride) {
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (head >= 3 * heads) return;
  const int s = n / T;
  const int p = pos[s] * T + n % T;
  bf16* src = qkv + ((int64_t)n * 3 * heads + head) * kMimiHead;
  bf16* ring = slot<bf16>(state, lines, s, stride);
  bf16* dst = ring + ((int64_t)(p % context) * 2 * heads + head - heads) * kMimiHead;
  if (head >= 2 * heads) {
    for (int i = threadIdx.x & 31; i < kMimiHead; i += 32) dst[i] = src[i];
    return;
  }
  rope_head<kMimiHead>(src, head < heads ? src : dst, (float)p, coef);
}

struct RingSlots {
  int first, context;
  __device__ int64_t operator()(int t) const { return (first + t) % context; }
};

extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    mimi_attend(const bf16* qkv, const int32_t* pos, const int32_t* lines, const void* state, bf16* out, int T,
                int heads, float scale, int context, int64_t stride) {
  const int n = blockIdx.x, s = n / T;
  const int p = pos[s] * T + n % T;
  const int len = min(p + 1, context);
  const bf16* ring = slot<bf16>(const_cast<void*>(state), lines, s, stride);
  attend_row<kMimiHead>(qkv + (int64_t)n * 3 * heads * kMimiHead, ring, out + (int64_t)n * heads * kMimiHead, heads,
                        len, scale, RingSlots{p - len + 1, context});
}

constexpr int kBookDim = 256;
constexpr int kBookSize = 2048;
constexpr int kQuantThreads = 256;

// Split residual vector quantization of each session's projected latent
// `[semantic 256 | acoustic 256]`: codebook 0 quantizes the semantic half,
// codebooks 1..7 the acoustic half residually. Nearest by euclidean distance
// (‖e‖² − 2 x·e, `norms` holding ‖e‖²), in f32, lowest index on ties.
// `books` is [8][2048][256] f32. One block per session.
extern "C" __global__ void __launch_bounds__(kQuantThreads)
    mimi_quantize(const bf16* latent, const float* books, const float* norms, int32_t* codes, int n_books) {
  const int s = blockIdx.x, tid = threadIdx.x;
  __shared__ float x[kBookDim];
  __shared__ unsigned long long best;
  for (int q = 0; q < n_books; ++q) {
    if (q <= 1) x[tid] = f32(latent[(int64_t)s * 2 * kBookDim + (q == 0 ? 0 : kBookDim) + tid]);
    if (tid == 0) best = ~0ull;
    __syncthreads();
    const float* book = books + (int64_t)q * kBookSize * kBookDim;
    unsigned long long mine = ~0ull;
    for (int e = tid; e < kBookSize; e += kQuantThreads) {
      const float4* row = reinterpret_cast<const float4*>(book + (int64_t)e * kBookDim);
      float dot = 0.f;
#pragma unroll 8
      for (int d = 0; d < kBookDim / 4; ++d) {
        const float4 v = row[d];
        dot += x[4 * d] * v.x + x[4 * d + 1] * v.y + x[4 * d + 2] * v.z + x[4 * d + 3] * v.w;
      }
      const float dist = norms[q * kBookSize + e] - 2.f * dot;
      const uint32_t u = __float_as_uint(dist);
      const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned long long packed = (unsigned long long)key << 32 | (uint32_t)e;
      mine = packed < mine ? packed : mine;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const unsigned long long other = __shfl_xor_sync(0xffffffffu, mine, o);
      mine = other < mine ? other : mine;
    }
    if ((tid & 31) == 0) atomicMin(&best, mine);
    __syncthreads();
    const int code = (int)(best & 0xffffffffu);
    if (tid == 0) codes[s * n_books + q] = code;
    if (q >= 1) x[tid] -= book[(int64_t)code * kBookDim + tid];
    __syncthreads();
  }
}

// The decoder's quantizer lookup: row s is [book 0 at code 0 | Σ books 1..7
// at codes 1..7], codes at `codes[s * ld + 0..8]`; `books` is [8][2048][256].
// Its projection is the GEMM after. One thread per 8 channels of a row.
extern "C" __global__ void mimi_dequantize(const int32_t* codes, int ld, const float* books, bf16* out, int n_books,
                                           int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int per = 2 * kBookDim / 8;
  const int s = i / per, c = i % per * 8;
  const int32_t* code = codes + (int64_t)s * ld;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int lo = c < kBookDim ? 0 : 1, hi = c < kBookDim ? 1 : n_books, d = c % kBookDim;
  for (int q = lo; q < hi; ++q) {
    const float* e = books + ((int64_t)q * kBookSize + code[q]) * kBookDim + d;
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += e[k];
  }
  store8(out + (int64_t)s * 2 * kBookDim + c, acc);
}

// The decoder's last conv, C -> 1 channel, kernel K, over ELU(x + bias):
// pcm[t] = b + Σ_j Σ_c w[j][c] act(x[t - (K - 1) + j][c]), f32. The
// history is the last K - 1 activated rows. One thread per (session, sample).
extern "C" __global__ void mimi_conv_out(const bf16* x, const bf16* bias, const float* w, const float* b, void* state,
                                         const int32_t* pos, const int32_t* lines, float* pcm, int T, int C, int K,
                                         int64_t stride, int total) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total) return;
  const int s = g / T, t = g % T, H = K - 1;
  const int p = pos[s];
  bf16* hist = slot<bf16>(state, lines, s, stride);
  const bf16* old = hist + (p & 1) * (int64_t)H * C;
  bf16* next = hist + ((p + 1) & 1) * (int64_t)H * C;
  float acc = b[0];
  for (int j = 0; j < K; ++j) {
    const int src = t - H + j;
    for (int c = 0; c < C; c += 8) {
      float v[8];
      if (src >= 0) {
        float bb[8];
        load8(x + ((int64_t)s * T + src) * C + c, v);
        load8(bias + c, bb);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = round_bf16(elu(round_bf16(v[k] + bb[k])));
        if (j == K - 1 && t >= T - H) store8(next + (int64_t)(t - (T - H)) * C + c, v);
      } else {
        load8(old + (int64_t)(H + src) * C + c, v);
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) acc += w[j * C + c + k] * v[k];
    }
  }
  pcm[g] = acc;
}
