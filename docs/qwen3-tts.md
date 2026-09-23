# Qwen3-TTS engine

`pega-omni qwen3-tts --model-path <Qwen3-TTS-12Hz-1.7B-CustomVoice>` serves the
12 Hz CustomVoice checkpoint on one GPU. Crates: `omni-cuda` (the talker's
kernels and GPU layer) and `omni-qwen3-tts` (weights, prompt, talker, codec,
engine). The codec decoder is a [kern](https://github.com/pegainfer-project/kern)
manifest the crate generates at load and `kern-runtime` executes.

## One step

Everything runs serially, in one thread:

1. **Admit** waiting requests FIFO while batch slots, the per-step token
   budget (`--max-step-tokens`) and KV pages allow. A request reserves its
   whole KV at admission, sized by a frame cap proportional to its input, so
   nothing is preempted and a running request never waits for memory.
2. **Talker**: one forward over a ragged batch (new prompts first, then each
   running row's last frame summed over its 16 codebook embeddings), paged KV
   through FlashInfer's batch prefill kernel, then codebook 0 drawn on the GPU
   (repetition penalty, special-token suppression, top-k, temperature).
3. **Code predictor**: fifteen passes filling codebooks 1-15 for every row.
   Its KV holds at most 16 tokens, so it gets one page per batch slot,
   rewritten every step. Codes stay on the device until the frame is complete:
   a step synchronizes with the host once.
4. **Codec**: every frame the step produced is decoded in one batched call,
   one frame per request. The decoder streams: each request holds a slot of
   decoder state (each causal conv's last input rows, each overlapping
   transposed conv's last GEMM row, each attention layer's K/V for the last
   72 frames), so a frame costs the same at any point of an utterance and
   nothing is recomputed.
5. **Emit**: requests with a chunk due (first after `--first-chunk-frames`,
   then every `--chunk-frames`, the rest at the end) send their decoded
   samples as s16le. Chunk sizes only decide when audio leaves.

Overlapping the codec with the next talker step is the obvious next change;
it lands only with an `omni-bench` A/B.

## Kernels

`omni-cuda/csrc` holds the talker's three files: `attention.cu` (FlashInfer
paged prefill for the talker and predictor), `transformer.cu` (RMSNorm via
FlashInfer, fused q/k norm + RoPE + KV write, SiLU-gate, embedding
gather-sum, bias + activation) and `sampling.cu` (the whole draw in one kernel
per row). GEMMs are cuBLAS.

`omni-qwen3-tts/kernels/codec.cu` is the codec's: RVQ lookup, RMSNorm,
attention that applies RoPE and appends to its 72-frame K/V ring, SnakeBeta
(fused into the bias epilogue and into the next conv's im2col), causal im2col
over a double-buffered history, overlap-add carrying the previous frame's row,
depthwise conv + LayerNorm, and the output conv. Residual GEMMs accumulate in
place (cuBLASLt with beta = 1) and their biases ride along to the next
consumer's epilogue, so no kernel exists only to add. Every kernel launches
with programmatic dependent launch and loads its weights before waiting on its
producer. `build.rs` compiles
it to a cubin; `codec.rs` pins its sha256 in the manifest, one `decode`
program over a `seqs` var whose calls are those kernels and kern's cuBLASLt
GEMM. kern verifies the manifest, checks every launch's ABI against the
cubin, allocates the workspaces at their bounds, hands out per-request state
slots, and captures the program as a CUDA graph per batch bucket. Weight
layout transforms (fused QKV, conv taps, folded scales) happen once at load,
on the host, for both halves.

FlashInfer is a git submodule pinned at v0.7.0; `build.rs` initializes it (and
only its CCCL) on first build. `OMNI_CUDA_ARCH` picks the target (default
`103a`, GB300).

## Correctness

`tools/qwen3_tts/golden.py` records the official implementation's run (qwen-tts
at 022e286b, torch 2.11, bf16) and `crates/omni-qwen3-tts/tests/golden.rs`
compares against it, teacher-forced with the recorded codes:

| | ours vs official |
|---|---|
| tokenizer ids | identical |
| prompt embeddings | worst row cosine 0.99999 |
| talker logits, 80 steps | worst cosine 0.99994, argmax agrees 98.8% |
| code-predictor logits, 79 × 15 | worst cosine 0.9993, argmax agrees 94.5% |
| codec, streamed frame by frame | 36.6 dB SNR |
| codec, a stream joining 7 frames late in the same batches | 36.6 dB SNR |

```bash
CUDA_VISIBLE_DEVICES=0 uv run tools/qwen3_tts/golden.py --model <ckpt> --out en-ryan.safetensors
OMNI_QWEN3_TTS_MODEL=<ckpt> OMNI_QWEN3_TTS_GOLDEN=en-ryan.safetensors cargo test -p omni-qwen3-tts --release
```

**Why stream the decoder.** The official `chunked_decode` re-decodes 25
frames of left context with every 300-frame chunk. Streamed in 8-frame chunks,
25 frames of context drops the waveform to 22.5 dB against the whole decode;
it takes the attention window's 72 frames to stay at bf16 noise, and then
every 8-frame chunk decodes 80 frames. Carrying each layer's state instead
decodes each frame once and is the whole-utterance decode exactly, up to
rounding: two streams of the same codes in one batch come out bit-identical.

Served audio through Whisper large-v3-turbo transcribes back to the input for
English, Chinese (with `instructions`, and the Sichuan dialect speaker) and
Japanese, including 8 concurrent requests in one batch.

## Performance

One streamed codec frame costs 0.40 ms of GPU time for one request and 1.02 ms
for 64 (150 and 145 kernels in one graph). The talker and code predictor
(~7 ms) dominate every step, so the step period stays at 7.8-8.3 ms from c=1 to
c=64. The chunk schedule then only decides when audio leaves:

- With the default 2+8, every stream plays without underrun at c=64.
- At c=64 the engine serves ~570 audio-s/s at RTF 0.10, with a 24 ms p50 first
  packet.

The chunked codec this replaced recomputed 72 frames of context per chunk. At
c=64 it reached 137 audio-s/s at RTF 0.44, and a quarter of its streams
underran.

[bench.md](bench.md#qwen3-tts-codec-decoder-kernels) holds the codec's
per-kernel breakdown and the A/B against the chunked codec, both under
`omni-bench` and under vLLM-Omni's benchmark.
[qwen3-tts-vs-vllm-omni.md](qwen3-tts-vs-vllm-omni.md) has the method and
script for comparing against vLLM-Omni. Its results predate the streamed codec,
so they are a lower bound: at c=64 with the same 1+25 schedule the engine now
serves 515 audio-s/s, against vLLM-Omni's 145 there.
