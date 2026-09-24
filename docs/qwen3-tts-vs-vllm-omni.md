# Qwen3-TTS against vLLM-Omni

**TL;DR** (2026-09-23, one GB300 each, vLLM-Omni's own benchmark, vLLM-Omni at
the best single-GPU setup we found for it): with the same chunk schedule
pega-omni reaches first audio 3.7-6.6x sooner and serves 1.4-2.5x the audio
per second (504 vs 202 audio-s/s at c=64). With its own 2+8 schedule every
stream plays through at c=64, at 549 audio-s/s. The gap is structure: one
batched step turns every running stream's newest frame into PCM, while
vLLM-Omni hands frames from a talker process to a code2wav process that polls
every 10 ms and decodes in small batches.

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
- **Their strongest single-GPU setup.** vLLM-Omni's opt-in throughput profile
  (`vllm_omni/deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml`: talker
  and code2wav as two processes on one GPU, CUDA graphs, B8 codec batches,
  first chunk 1 frame then 25, left context 72), with both processes sharing
  the GPU through an MPS daemon, which the profile suggests but does not
  enable. On this card it beats the default deploy (`qwen3_tts.yaml`) from c=8
  up: 202 against 145 audio-s/s at c=64, first audio 185 against 316 ms.
- **The same chunk schedule** for the engine comparison: pega-omni with
  `--first-chunk-frames 1 --chunk-frames 25`. Its codec is streamed, so it has
  no left context to set: every frame is decoded once, with its full history.
  pega-omni's own default (2 then 8) is reported separately; chunking is a
  latency/continuity/throughput trade, not an engine property.

## Reproduce

`tools/qwen3_tts/vs_vllm_omni.sh all` does the whole run and redraws
`assets/qwen3-tts-vs-vllm-omni.png` (the README chart) from the results:

```bash
cargo build --release -p omni-server --features qwen3-tts
export MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice WORK=/path/to/scratch GPU=0
tools/qwen3_tts/vs_vllm_omni.sh all
```

It sets up a uv venv (vllm 0.30.0, vllm-omni 0.30.0rc1, ninja; the first
vLLM-Omni start compiles for ~5 min), then runs vLLM-Omni, pega-omni at 1+25
and pega-omni at its default 2+8, each alone on the GPU, through the four CI
points. The pieces (`setup`, `serve-vllm-omni`, `serve-pega-omni`, `bench`,
`chart`) run on their own too. Each point leaves `c<N>.json` and the full
`c<N>.log` under `$WORK/results/<label>/`.

## Results

2026-09-23, one GB300 (Grace, 2x72 cores), vLLM-Omni v0.30.0rc1 on vLLM 0.30.0
/ torch 2.13.0+cu130, pega-omni at this branch. Zero failed requests
everywhere. TTFP in ms (median / p99), audio throughput in audio-s/s,
continuity = share of streams whose playback underrun stays within 100 ms
(the benchmark's default budget), underrun p99 in s.

| c | engine | TTFP | RTF | audio-s/s | continuity | underrun |
|---:|---|---:|---:|---:|---:|---:|
| 1 | vLLM-Omni (1+25) | 37 / 37 | 0.08 | 12.3 | 100% | 0.07 |
| 1 | pega-omni (1+25) | 6 / 7 | 0.06 | 17.1 | 100% | 0.03 |
| 1 | pega-omni (2+8) | 10 / 10 | 0.06 | 17.2 | 100% | 0.00 |
| 8 | vLLM-Omni (1+25) | 53 / 78 | 0.11 | 72.3 | 32% | 0.15 |
| 8 | pega-omni (1+25) | 14 / 30 | 0.07 | 111.6 | 100% | 0.07 |
| 8 | pega-omni (2+8) | 19 / 34 | 0.07 | 111.1 | 100% | 0.00 |
| 16 | vLLM-Omni (1+25) | 62 / 123 | 0.13 | 115.6 | 9% | 0.21 |
| 16 | pega-omni (1+25) | 15 / 33 | 0.08 | 201.0 | 100% | 0.09 |
| 16 | pega-omni (2+8) | 22 / 30 | 0.08 | 196.5 | 100% | 0.00 |
| 64 | vLLM-Omni (1+25) | 185 / 344 | 0.27 | 201.5 | 0% | 0.56 |
| 64 | pega-omni (1+25) | 28 / 54 | 0.09 | 503.6 | 59% | 0.16 |
| 64 | pega-omni (2+8) | 46 / 62 | 0.09 | 549.4 | 100% | 0.00 |

Mean audio durations agree (5.4-5.8 s everywhere).

For scale, vLLM-Omni's CI baseline on H100 (two GPUs, code2wav on the second):
c=1 47 ms / 0.137; c=8 75 ms / 0.186 / 38.4; c=16 714 ms / 0.315 / 55.4;
c=64 5942 ms / 1.166 / 68.7 (median TTFP / RTF / audio-s/s).

## Reading

- **Per-request compute.** At c=1 pega-omni finishes a ~5.5 s utterance in
  0.06 of real time against 0.08: its decode step (talker, code predictor,
  sampler and codec) is one CUDA graph launch.
- **First audio.** vLLM-Omni's first packet crosses three processes and a
  shared-memory connector whose reader sleeps 10 ms between polls
  (`connector_get_sleep_s`); pega-omni decodes the first frame in the step
  that produced it.
- **Throughput under load.** pega-omni's step decodes one frame of every
  running stream in one batch, so the codec's cost per step barely grows with
  concurrency (RTF 0.06 → 0.09 from c=1 to c=64). vLLM-Omni's code2wav
  time-slices the GPU with the talker process and decodes at most 8 streams
  at a time (RTF 0.08 → 0.27).
- **Continuity** is the chunk schedule's doing: after a 1-frame first chunk
  the second waits 25 frames (2 s of audio), so under load streams gap on
  both engines. With a streamed codec a smaller chunk costs nothing extra, so
  pega-omni's 2+8 keeps every stream continuous to c=64 without losing
  throughput.

## Not covered

- Audio quality under this workload (`--seed-tts-wer-eval`: WER, speaker
  similarity, UTMOS). Correctness is covered separately by the golden test
  (docs/qwen3-tts.md).
- vLLM-Omni's two-GPU CI layout and its adaptive chunking
  (`codec_chunk_adaptive`), its other answers to the codec bottleneck.
- Each point is a single short run (c=64 finishes 128 requests in ~5 s).
