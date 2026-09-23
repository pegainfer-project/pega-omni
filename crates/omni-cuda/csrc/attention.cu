// Attention: FlashInfer FA2's paged batch prefill at head_dim 128, bf16 Q/KV/O,
// NHD layout, for every Qwen3-shaped transformer (talker, code predictor).
// Decode rows are prefill rows of length one, so a step with admissions and
// running rows is one call. The host builds the tile plan (`request_indices`,
// `qo_tile_indices`) for the CTA tile it passes.
#include "common.cuh"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

using namespace flashinfer;

using Full = DefaultAttention</*custom_mask=*/false, /*sliding_window=*/false, /*soft_cap=*/false, /*alibi=*/false>;

extern "C" {

int omni_prefill_cta_tile_q(int64_t avg_packed_qo_len, uint32_t head_dim) {
  return (int)FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);
}

int omni_paged_prefill_hd128(bf16* q, uint32_t q_stride_n, bf16* out, bf16* k_pool, bf16* v_pool,
                             int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len,
                             int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
                             int32_t* kv_tile_indices, int32_t* kv_chunk_size, uint32_t* total_rows,
                             uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t page_size,
                             uint32_t rows, uint32_t batch, uint32_t tiles, uint32_t cta_tile_q,
                             float sm_scale, cudaStream_t stream) {
  constexpr uint32_t D = 128;
  int64_t strides[3] = {(int64_t)page_size * num_kv_heads * D, (int64_t)num_kv_heads * D, (int64_t)D};
  paged_kv_t<bf16, int32_t> kv(num_kv_heads, page_size, D, batch, QKVLayout::kNHD, k_pool, v_pool, strides,
                               page_indices, page_indptr, last_page_len, /*rope_pos_offset=*/nullptr);
  using Params = BatchPrefillPagedParams<bf16, bf16, bf16, int32_t>;
  Params p(q, kv, /*custom_mask=*/nullptr, q_indptr, /*mask_indptr=*/nullptr, /*q_rope_offset=*/nullptr, out,
           /*lse=*/nullptr, /*alibi=*/nullptr, num_qo_heads, q_stride_n, D, /*window_left=*/-1,
           /*soft_cap=*/0.f, sm_scale, /*rope_scale=*/1.f, /*rope_theta=*/1e4f);
  p.request_indices = request_indices;
  p.qo_tile_indices = qo_tile_indices;
  p.kv_tile_indices = kv_tile_indices;
  p.o_indptr = q_indptr;
  p.kv_chunk_size_ptr = kv_chunk_size;
  p.max_total_num_rows = rows;
  p.total_num_rows = total_rows;
  p.padded_batch_size = tiles;
  p.partition_kv = false;
  OMNI_TRY(DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    cudaError_t e = BatchPrefillWithPagedKVCacheDispatched</*SAME_KV_STRIDES=*/true, CTA_TILE_Q, D, D,
                                                            PosEncodingMode::kNone, false, MaskMode::kCausal,
                                                            Full, Params>(p, nullptr, nullptr, false, stream);
    if (e != cudaSuccess) return (int)e;
  }))
}

}  // extern "C"
