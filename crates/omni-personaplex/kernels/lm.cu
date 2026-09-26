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
  const int64_t at = (int64_t)blockIdx.x * dim;
  float v[8];
  load8(x + at + threadIdx.x * 8, v);
  rms_norm_row(v, alpha, out + at, dim, eps);
}

// x = bf16(x + y), then out = norm(x).
extern "C" __global__ void lm_add_norm(const bf16* y, bf16* x, const bf16* alpha, bf16* out, int dim, float eps) {
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
  silu_mul(gate_up, out, inter, total);
}

constexpr int kHelium = 128;

// Helium rows [q | k | v] (heads of 128): rotary embedding of q in place and
// of k into the row's slot, v copied there. A warp per (row, head of q, k or v).
extern "C" __global__ void lm_rope(bf16* qkv, const int32_t* pos, const int32_t* slots, void* kv, int heads,
                                   float coef) {
  const int n = blockIdx.x;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (head >= 3 * heads) return;
  bf16* src = qkv + ((int64_t)n * 3 * heads + head) * kHelium;
  bf16* dst = static_cast<bf16*>(kv) + ((int64_t)slots[n] * 2 * heads + head - heads) * kHelium;
  if (head >= 2 * heads) {
    for (int i = threadIdx.x & 31; i < kHelium; i += 32) dst[i] = src[i];
    return;
  }
  rope_head<kHelium>(src, head < heads ? src : dst, (float)pos[n], coef);
}

struct PagedSlots {
  const int32_t* pages;
  int page;
  __device__ int64_t operator()(int t) const { return (int64_t)pages[t / page] * page + t % page; }
};

// Row `n` (of session `ragged ? seq[n] : n`, at `pos[n]`) against its
// session's first min(pos + 1, context) ring slots, pages `pages[indptr[s]..]`.
extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    lm_attend(const bf16* qkv, const void* kv, const int32_t* pos, const int32_t* seq, int ragged,
              const int32_t* indptr, const int32_t* pages, int page, bf16* out, int heads, float scale, int context) {
  const int n = blockIdx.x;
  const int s = ragged ? seq[n] : n;
  const int len = min(pos[n] + 1, context);
  attend_row<kHelium>(qkv + (int64_t)n * 3 * heads * kHelium, static_cast<const bf16*>(kv),
                      out + (int64_t)n * heads * kHelium, heads, len, scale, PagedSlots{pages + indptr[s], page});
}

// --- The depformer ------------------------------------------------------------

constexpr int kDep = 64;
constexpr int kDepSteps = 8;

// Step `k`'s input: its slice of the projected Helium output plus the
// embedding of the token the step before drew (the text token for step 0),
// rounded to bf16. A block of dim / 8 per session.
extern "C" __global__ void lm_dep_embed(const bf16* projected, int k, const bf16* table, const int32_t* drawn,
                                        bf16* out, int dim) {
  const int s = blockIdx.x, c = threadIdx.x * 8;
  const int64_t id = drawn[s * kDrawn + k];
  float v[8], e[8];
  load8(projected + ((int64_t)s * kDepSteps + k) * dim + c, v);
  load8(table + id * dim + c, e);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = round_bf16(v[j] + e[j]);
  store8(out + (int64_t)s * dim + c, v);
}

// Step `step`'s K and V of each session into its workspace slot.
extern "C" __global__ void lm_dep_kv(const bf16* qkv, bf16* ws, int step, int heads) {
  const int s = blockIdx.x, width = heads * kDep;
  const bf16* src = qkv + (int64_t)s * 3 * width + width;
  bf16* dst = ws + ((int64_t)s * kDepSteps + step) * 2 * width;
  for (int i = threadIdx.x * 8; i < 2 * width; i += blockDim.x * 8) copy8(dst + i, src + i);
}

struct DenseSlots {
  int64_t first;
  __device__ int64_t operator()(int t) const { return first + t; }
};

extern "C" __global__ void __launch_bounds__(kAttnWarps * 32)
    lm_dep_attend(const bf16* qkv, const bf16* ws, int step, bf16* out, int heads, float scale) {
  const int s = blockIdx.x;
  attend_row<kDep>(qkv + (int64_t)s * 3 * heads * kDep, ws, out + (int64_t)s * heads * kDep, heads, step + 1, scale,
                   DenseSlots{(int64_t)s * kDepSteps});
}

// --- Sampling -------------------------------------------------------------------
//
// The reference draws from softmax(logits / T) restricted to the top k by an
// exponential race (argmax p / E, E ~ Exp(1)), which is the Gumbel-max trick:
// argmax over the kept tokens of logits / T - log E. One block per row; the
// top-k threshold is found by a radix select over the order-preserving bits
// of logits / T (every value at or above the k-th largest is kept), four
// passes over the row, each re-reading it from global memory so any
// vocabulary fits. E comes from a counter hash of the row's seed, so a draw
// is a pure function of (seed, row, column, token). A temperature of zero
// takes the argmax. The drawn (or forced, when `force` is non-negative) token
// goes to `drawn[s][col]`.

constexpr int kSampleThreads = 1024;

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

extern "C" __global__ void __launch_bounds__(kSampleThreads)
    lm_sample(const bf16* logits, int vocab, int64_t ld, float temperature, int top_k, const int32_t* force,
              const int32_t* seeds, int col, int32_t* drawn) {
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
  const float t = temperature > 0.f ? temperature : 1.f;
  const int k_keep = top_k > 0 && top_k < vocab ? top_k : vocab;
  uint32_t prefix = 0;
  if (temperature > 0.f && k_keep < vocab) {
    uint32_t left = k_keep;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
      __syncthreads();
      const uint32_t high = shift == 24 ? 0u : 0xffffffffu << (shift + 8);
      for (int i = tid; i < vocab; i += blockDim.x) {
        const uint32_t key = order_key(f32(l[i]) / t);
        if (((key ^ prefix) & high) == 0) atomicAdd(&hist[(key >> shift) & 255], 1u);
      }
      __syncthreads();
      if (tid == 0) {
        uint32_t above = 0;
        for (int b = 255; b >= 0; --b) {
          if (above + hist[b] >= left) {
            sel_prefix = prefix | (uint32_t)b << shift;
            sel_left = left - above;
            break;
          }
          above += hist[b];
        }
      }
      __syncthreads();
      prefix = sel_prefix;
      left = sel_left;
    }
  }
  if (tid == 0) best = 0;
  __syncthreads();
  const uint32_t seed = mix((uint32_t)seeds[2 * s] ^ mix((uint32_t)seeds[2 * s + 1] + 0x9e3779b9u * (col + 1)));
  unsigned long long mine = 0;
  for (int i = tid; i < vocab; i += blockDim.x) {
    const float x = f32(l[i]) / t;
    if (order_key(x) < prefix) continue;
    float score = x;
    if (temperature > 0.f) {
      const float u = ((mix(seed ^ mix((uint32_t)i)) >> 8) + 0.5f) * (1.f / 16777216.f);
      score = x - __logf(-__logf(u));
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
  const int s = blockIdx.x;
  int32_t* own = slot<int32_t>(state, lines, s, stride);
  if (threadIdx.x < kStreams + kBooks - 1) own[threadIdx.x] = init[s * (kStreams + kBooks - 1) + threadIdx.x];
}
