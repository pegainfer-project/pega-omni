// Token selection over small vocabularies (codec codebooks, at most 4096
// entries), one block per row, the whole logits processor chain fused:
// repetition penalty, suppression, temperature, top-k and the categorical draw.
//
// The order matches Hugging Face `generate`: penalty and suppression edit the
// raw logits, then temperature, then top-k, then softmax and a draw against
// the host-supplied uniform. A temperature of zero selects the argmax (lowest
// index on ties, like `torch.argmax`).
#include "common.cuh"

#include <cfloat>

constexpr int kMaxVocab = 4096;
constexpr int kThreads = 1024;

__global__ void __launch_bounds__(kThreads)
    sample_kernel(const bf16* logits, uint32_t ld, uint32_t vocab, float temperature, int32_t top_k, float penalty,
                  const uint32_t* seen, uint32_t suppress_lo, uint32_t suppress_hi,
                  int32_t exempt, const uint8_t* block_exempt, const float* uniform, int32_t* out) {
  __shared__ float val[kMaxVocab];
  __shared__ int idx[kMaxVocab];
  const int row = blockIdx.x, tid = threadIdx.x;
  const bf16* l = logits + (size_t)row * ld;
  const uint32_t words = (vocab + 31) / 32;
  const bool exempt_blocked = block_exempt && block_exempt[row];

  for (int i = tid; i < kMaxVocab; i += kThreads) {
    float v = -INFINITY;
    if (i < (int)vocab) {
      v = to_f32(l[i]);
      if (seen && (seen[(size_t)row * words + i / 32] >> (i % 32) & 1u)) {
        v = v > 0.f ? v / penalty : v * penalty;
      }
      const bool in_range = (uint32_t)i >= suppress_lo && (uint32_t)i < suppress_hi;
      if ((in_range && i != exempt) || (i == exempt && exempt_blocked)) v = -INFINITY;
    }
    val[i] = v;
    idx[i] = i;
  }
  __syncthreads();

  if (temperature <= 0.f) {
    for (int stride = kMaxVocab / 2; stride > 0; stride >>= 1) {
      for (int i = tid; i < stride; i += kThreads) {
        const float a = val[i], b = val[i + stride];
        if (b > a || (b == a && idx[i + stride] < idx[i])) {
          val[i] = b;
          idx[i] = idx[i + stride];
        }
      }
      __syncthreads();
    }
    if (tid == 0) out[row] = idx[0];
    return;
  }

  for (int i = tid; i < kMaxVocab; i += kThreads) val[i] /= temperature;
  __syncthreads();
  // Bitonic sort, descending by value.
  for (int size = 2; size <= kMaxVocab; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int i = tid; i < kMaxVocab; i += kThreads) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const bool swap = desc ? val[i] < val[j] : val[i] > val[j];
          if (swap) {
            const float tv = val[i];
            val[i] = val[j];
            val[j] = tv;
            const int ti = idx[i];
            idx[i] = idx[j];
            idx[j] = ti;
          }
        }
      }
      __syncthreads();
    }
  }

  if (tid >= 32) return;
  const int k = top_k > 0 && top_k < (int)vocab ? top_k : (int)vocab;
  const float m = val[0];
  float total = 0.f;
  for (int i = tid; i < k; i += 32) total += __expf(val[i] - m);
  total = warp_sum(total);
  const float target = uniform[row] * total;
  float prefix = 0.f;
  for (int base = 0; base < k; base += 32) {
    const int i = base + tid;
    float x = i < k ? __expf(val[i] - m) : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, x, o);
      if (tid >= o) x += y;
    }
    const unsigned hit = __ballot_sync(0xffffffffu, i < k && prefix + x >= target);
    if (hit) {
      if (tid == 0) out[row] = idx[base + __ffs(hit) - 1];
      return;
    }
    prefix += __shfl_sync(0xffffffffu, x, 31);
  }
  if (tid == 0) out[row] = idx[k - 1];
}

extern "C" {

// `seen` is a per-row bitmap of `ceil(vocab / 32)` words marking tokens under
// the repetition penalty (null: no penalty). Tokens in [suppress_lo,
// suppress_hi) are masked except `exempt`, which is itself masked on rows with
// `block_exempt` set (null: never).
int omni_sample(const bf16* logits, uint32_t ld, uint32_t rows, uint32_t vocab, float temperature, int32_t top_k,
                float penalty, const uint32_t* seen, uint32_t suppress_lo,
                uint32_t suppress_hi, int32_t exempt, const uint8_t* block_exempt, const float* uniform, int32_t* out,
                cudaStream_t stream) {
  if (vocab > kMaxVocab) return (int)cudaErrorInvalidValue;
  if (rows == 0) return 0;
  sample_kernel<<<rows, kThreads, 0, stream>>>(logits, ld, vocab, temperature, top_k, penalty, seen, suppress_lo,
                                               suppress_hi, exempt, block_exempt, uniform, out);
  return (int)cudaGetLastError();
}

}  // extern "C"
