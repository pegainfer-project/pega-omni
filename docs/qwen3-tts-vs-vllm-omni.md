# Qwen3-TTS against vLLM-Omni

**TL;DR** (2026-09-23, one GB300 each, vLLM-Omni's own benchmark): with the
same chunk schedule pega-omni reaches first audio 5-8x sooner and serves
1.3-1.6x the audio per second from c=8 up (238 vs 145 audio-s/s at c=64).
Single-request speed is the same on both (RTF 0.07-0.08). The gap is
structure, not kernels: vLLM-Omni hands frames from a talker process to a
code2wav process that polls every 10 ms, and its codec batches little.

## What makes it fair

- **Their benchmark, their workload.** The client is `vllm bench serve --omni
  --backend openai-audio-speech`, run with the parameters vLLM-Omni's nightly CI
  uses for this checkpoint (`tests/dfx/perf/tests/test_tts.json`, as
  `tests/dfx/perf/scripts/run_benchmark.py` expands it): dataset
  `seed-tts-text` on the bundled `seed_tts_smoke` (en), streamed PCM, request
  rate inf, warmups max(2, c), four points c=1/20, c=8/80, c=16/128, c=64/128
  prompts. `audio_underrun` is added to the metrics; nothing else changes.
- **Same request.** Only the body's naming differs: vLLM-Omni takes
  `{"voice":"Vivian","language":"English","task_type":"CustomVoice"}` at the
  top level, pega-omni `{"voice":"vivian","extra":{"language":"english"}}`.
  Both sample with the checkpoint's `generation_config.json` (temperature 0.9,
  top-k 50, repetition penalty 1.05; sub-talker 0.9 / 50). Check that mean
  audio durations agree (they do, 5.3-5.8 s): an engine that stops early looks fast.
- **Same card, one engine at a time.** Each server runs alone on the same GPU;
  the other is stopped. Server on cores 0-31, client on 36-71 (the GPU's NUMA
  node).
- **Their shipped configuration.** vLLM-Omni's default deploy
  (`vllm_omni/deploy/qwen3_tts.yaml`, one GPU: talker and code2wav as two
  processes on it, CUDA graphs, first chunk 1 frame then 25, left context 72).
- **The same chunk schedule** for the engine comparison: pega-omni with
  `--first-chunk-frames 1 --chunk-frames 25` (context 72 is its default).
  pega-omni's own default (2 then 8) is reported separately; chunking is a
  latency/continuity/throughput trade, not an engine property.

## Reproduce

`tools/qwen3_tts/vs_vllm_omni.sh`:

```bash
export MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice WORK=/path/to/scratch GPU=0
tools/qwen3_tts/vs_vllm_omni.sh setup      # uv venv: vllm 0.30.0, vllm-omni 0.30.0rc1, ninja

tools/qwen3_tts/vs_vllm_omni.sh serve-vllm-omni &          # first start compiles, ~5 min
tools/qwen3_tts/vs_vllm_omni.sh bench vllm-omni vllm-omni  # then stop the server

cargo build --release -p omni-server --features qwen3-tts
tools/qwen3_tts/vs_vllm_omni.sh serve-pega-omni --first-chunk-frames 1 --chunk-frames 25 &
tools/qwen3_tts/vs_vllm_omni.sh bench pega-omni pega-omni-1-25
```

Each point leaves `c<N>.json` and the full `c<N>.log` under `$WORK/results/<label>/`.

## Results

2026-09-23, one GB300 (Grace, 2x72 cores), vLLM-Omni v0.30.0rc1 on vLLM 0.30.0
/ torch 2.13.0+cu130, pega-omni at this branch. Zero failed requests
everywhere. TTFP in ms (median / p99), audio throughput in audio-s/s,
continuity = share of streams whose playback underrun stays within 100 ms
(the benchmark's default budget), underrun p99 in s.

| c | engine | TTFP | RTF | audio-s/s | continuity | underrun |
|---:|---|---:|---:|---:|---:|---:|
| 1 | vLLM-Omni (1+25) | 34 / 41 | 0.08 | 13.0 | 100% | 0.07 |
| 1 | pega-omni (1+25) | 7 / 7 | 0.07 | 13.8 | 100% | 0.06 |
| 1 | pega-omni (2+8) | 13 / 13 | 0.08 | 13.3 | 100% | 0.00 |
| 8 | vLLM-Omni (1+25) | 65 / 109 | 0.12 | 65.0 | 1% | 0.17 |
| 8 | pega-omni (1+25) | 12 / 22 | 0.09 | 86.2 | 83% | 0.12 |
| 8 | pega-omni (2+8) | 24 / 43 | 0.11 | 67.5 | 100% | 0.00 |
| 16 | vLLM-Omni (1+25) | 83 / 152 | 0.16 | 95.3 | 0% | 0.27 |
| 16 | pega-omni (1+25) | 16 / 37 | 0.11 | 138.5 | 13% | 0.15 |
| 16 | pega-omni (2+8) | 31 / 70 | 0.16 | 97.5 | 100% | 0.00 |
| 64 | vLLM-Omni (1+25) | 316 / 775 | 0.42 | 144.9 | 5% | 0.66 |
| 64 | pega-omni (1+25) | 39 / 138 | 0.21 | 237.9 | 0% | 0.48 |
| 64 | pega-omni (2+8) | 71 / 261 | 0.38 | 143.0 | 71% | 0.21 |

For scale, vLLM-Omni's CI baseline on H100 (two GPUs, code2wav on the second):
c=1 47 ms / 0.137; c=8 75 ms / 0.186 / 38.4; c=16 714 ms / 0.315 / 55.4;
c=64 5942 ms / 1.166 / 68.7 (median TTFP / RTF / audio-s/s).

## Reading

- **Per-request compute is a tie.** At c=1 both finish a ~5.5 s utterance in
  0.07-0.08 of real time.
- **First audio.** vLLM-Omni's first packet crosses three processes and a
  shared-memory connector whose reader sleeps 10 ms between polls
  (`connector_get_sleep_s`); pega-omni decodes the first frame in the step
  that produced it.
- **Throughput under load** is where the one-loop design shows: talker, code
  predictor and codec run as one batched step, while vLLM-Omni's code2wav
  decodes small batches (`decode_batch_max_size: 4`) and time-slices the GPU
  with the talker process.
- **Continuity** is the chunk schedule's doing on both engines: after a
  1-frame first chunk the second waits 25 frames (2 s of audio), so under load
  nearly every stream gaps. pega-omni's 2+8 keeps every stream continuous to
  c=16, at the cost of decoding more often (throughput at c=64 falls to
  vLLM-Omni's level).

## Not covered

- Audio quality under this workload (`--seed-tts-wer-eval`: WER, speaker
  similarity, UTMOS). Correctness is covered separately by the golden test
  (docs/qwen3-tts.md).
- vLLM-Omni's two-GPU CI layout and its adaptive chunking
  (`codec_chunk_adaptive`), its answers to the codec bottleneck.
- Each point is a single short run (c=64 finishes 128 requests in ~5 s).
