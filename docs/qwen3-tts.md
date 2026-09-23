# Qwen3-TTS engine

`pega-omni qwen3-tts --model-path <Qwen3-TTS-12Hz-1.7B-CustomVoice>` serves the
12 Hz CustomVoice checkpoint on one GPU. Crates: `omni-cuda` (kernels and the
GPU layer) and `omni-qwen3-tts` (weights, prompt, talker, codec, engine).

## One step

Everything runs serially on one CUDA stream, in one thread:

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
4. **Codec**: each row with a chunk due (first after `--first-chunk-frames`,
   then every `--chunk-frames`, the rest at the end) is decoded with
   `--context-frames` of left context and its new samples go out as s16le.

Overlapping the codec with the next talker step (a second stream or green
contexts) is the obvious next change; it lands only with an `omni-bench` A/B.

## Kernels

`omni-cuda/csrc` holds four files: `attention.cu` (FlashInfer paged prefill
for the talker and predictor, FlashInfer single prefill with a sliding window
for the codec transformer), `transformer.cu` (RMSNorm via FlashInfer, fused
q/k norm + RoPE + KV write, SiLU-gate, embedding gather-sum, bias + activation
+ residual), `sampling.cu` (the whole draw in one kernel per row) and
`conv.cu` (causal im2col with a fused SnakeBeta, overlap-add for transposed
convs, depthwise conv + LayerNorm, the output conv). GEMMs are cuBLAS. Weight
layout transforms (fused QKV, conv taps, folded scales) happen once at load.

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
| codec, whole utterance | 37.1 dB SNR |
| codec, streamed 2 + 8 × n with 72 frames of context | 36.8 dB SNR |

```bash
CUDA_VISIBLE_DEVICES=0 uv run tools/qwen3_tts/golden.py --model <ckpt> --out en-ryan.safetensors
OMNI_QWEN3_TTS_MODEL=<ckpt> OMNI_QWEN3_TTS_GOLDEN=en-ryan.safetensors cargo test -p omni-qwen3-tts --release
```

**Decoder context.** The official `chunked_decode` uses 25 frames of left
context over 300-frame chunks. Streamed in 8-frame chunks, 25 frames drops the
waveform to 22.5 dB against the whole decode, falling steadily as the true
context outgrows the window (46 dB per chunk while the whole history fits, 16
dB by frame 74). The codec transformer attends 72 frames back, and with 72
frames of context every chunk stays at bf16 noise (~46 dB against our own
whole decode). The default is 72.

Served audio through Whisper large-v3-turbo transcribes back to the input for
English, Chinese (with `instructions`, and the Sichuan dialect speaker) and
Japanese, including 8 concurrent requests in one batch.

## First numbers

2026-09-23, one GB300, server and client unpinned on the same node, 30 diverse
English sentences (~100 characters, ~6 s of audio), voice `ryan`, PCM, chunks
2 + 8 frames, context 72. A baseline, not a tuned result:

| concurrency | req/s | audio-s/s | TTFP p50 / p99 | RTF p50 | underrun > 1 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.1 | 13.6 | 12 / 12 ms | 0.073 | 0 |
| 8 | 9.9 | 65.9 | 24 / 54 ms | 0.116 | 0 |
| 32 | 18.5 | 119.3 | 51 / 103 ms | 0.259 | 114 of 256 (p99 37 ms) |
