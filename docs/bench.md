# Front-end benchmark against the simulated engine

**TL;DR** (2026-09-23): with one zero-cost engine loop the server sustains
~180k req/s and ~190k audio-s/s with zero failures up to 4096 concurrent
requests; the limiter is the single engine thread handing events to request
tasks, not the HTTP layer. Against a GPU-shaped paced engine the front end adds
no measurable TTFP over the engine floor and no playback underrun at 2048
concurrent streams. The one real bug found (no `TCP_NODELAY`, +40 ms per
packet) is fixed.

## Setup

- One node, 2x Grace (Neoverse-V2, 144 cores), otherwise idle. Server pinned to
  cores 0-31 with 32 tokio workers; `omni-bench` pinned to cores 36-71; loopback.
- Release build. Every run: warmup of `max(64, c)` requests to open connections,
  `--request-rate inf` unless stated, 0 failures unless stated.
- `tools/bench_sweep.sh <tag> <server args>` reproduces each row
  (env `CONCS`, `N`, `RATE`, `EXTRA`, `FORMAT`, `FRAMING`).

## 1. `TCP_NODELAY`

First run, 50k requests, `extra.frames = 13` (1.04 s audio, chunks 1+4+4+4):

| c | before req/s | before TTFP p50 | after req/s | after TTFP p50 |
|---|---:|---:|---:|---:|
| 64 | 1,564 | 41.0 ms | 173,127 | 0.16 ms |

Every packet waited for the client's delayed ACK (~40 ms) because axum does
not set `TCP_NODELAY` on accepted sockets. `omni_frontend::serve` now sets it.

## 2. Ceiling (zero-cost engine)

1M requests per point, 1.04 s audio each, PCM, raw audio framing:

| c | req/s | audio-s/s | TTFP p50 / p99 (ms) | E2E p50 / p99 (ms) | server cores |
|---:|---:|---:|---:|---:|---:|
| 16 | 161,536 | 167,998 | 0.08 / 0.13 | 0.09 / 0.15 | 6.5 |
| 64 | 179,349 | 186,523 | 0.16 / 0.32 | 0.37 / 0.62 | 10.7 |
| 256 | 178,733 | 185,882 | 0.34 / 1.02 | 1.39 / 3.01 | 12.2 |
| 1024 | 181,217 | 188,466 | 4.35 / 5.25 | 5.62 / 11.29 | 13.0 |
| 4096 | 180,217 | 187,426 | 21.7 / 24.8 | 23.0 / 45.0 | 14.8 |

Throughput is flat from c=64 and latency grows linearly after it (Little's
law): a serial stage saturates. `--no-metrics` in the same session gives
192k-201k req/s, so Prometheus recording costs 5-8% at the ceiling.

**Where the time goes.** Per-thread CPU at c=1024: `omni-sim` 1.01 cores
(saturated), tokio workers 15.5 cores over 32. pprof (`--features cpu-profile`)
on the engine thread: ~90% in atomics of the per-request channel send and task
wake (`compare_exchange`, `ldset`, `cas`, `wake_by_val`), 2.5% in the
scheduling core itself. The ceiling is one thread waking ~1.1M request tasks
per second. A real engine emits at 12.5 frames/s per stream, so this is about
88k concurrent real-time streams per engine loop at one-frame chunks, far past
what one GPU generates. Caching metric handles instead of the `counter!` macro
was tried and measured no different, so it did not land.

## 3. Real-time streams (paced engine)

Step cost `8 ms + 20 us x rows`, one-frame chunks, Poisson arrivals at
300 req/s (~580 streams running, ~20 ms steps), 7.7 s of audio per request,
40,960 requests:

| framing | TTFP p50 / p90 / p99 (ms) | underrun | server cores |
|---|---:|---:|---:|
| pcm, raw audio | 29.7 / 38.0 / 41.2 | 0 | 0.30 |
| wav, SSE | 29.5 / 37.8 / 41.0 | 0 | 0.43 |

The engine floor for a new request is the rest of the current step plus one
step, mean 1.5 steps = ~29 ms: the measured TTFP is the floor, so front-end
overhead is below resolution. No request stalled.

## Findings for the GPU engine

Same paced engine with a per-character prefill cost (`5 us/char`) and closed
loop at c=1024: admissions arrive in bursts, one step carries a burst's whole
prefill (~0.6 s), and every running stream waits for it. With one-frame first
chunks the average request stalls 237 ms; with four-frame first chunks the
320 ms buffer hides it. The GPU scheduler needs a prefill budget per step (or
a separate prefill lane) so admissions never stall running streams; the first
chunk size is a policy of that scheduler, not of the front end.

# Qwen3-TTS codec decoder kernels

**TL;DR** (2026-09-23, one GB300): one streamed codec frame for 64 requests
drops from 2.49 ms to 1.02 ms of GPU time, and from 0.77 ms to 0.40 ms for one
request. Measured at the server, at c=64 the step period falls from 13.1 ms to
8.3 ms and the engine serves 4x main's audio per second with every stream
continuous.

## Method

- **Codec alone.** Measured while the codec was still a kern runtime of its
  own (up to 2c556f1): one frame for B streams, 300 times, through its graph.
  The codec is now the tail of the model's `first` and `decode` graphs; to
  reproduce, run nsys with CUDA graph node tracing
  (`--cuda-graph-trace=node`) on a server under `omni-bench` at c=B and take
  each `decode` call's `codec_*` kernels. That gives the codec span per call
  (first codec kernel start to last kernel end) and each kernel's time.
  Under programmatic dependent launch a kernel's duration includes its wait
  on its producer, so each kernel is charged only its critical-path share:
  the part of its duration that no earlier kernel covers.
- **Server.** nsys on a live server under `omni-bench` load (default 2+8
  chunks) gives the step period: talker start to talker start.
- **A/B.** Same session, one server at a time on GPU 2. Server on cores
  108-125 and clients on 126-143, all on the GPU's NUMA node; the rest of the
  machine was idle apart from one job on another GPU. `main` is the chunked
  codec, which recomputes 72 frames of context per chunk. `opt` is the
  streamed kern codec with this work. Two tools:
  - `tools/qwen3_tts/vs_vllm_omni.sh bench`: vLLM-Omni's benchmark at
    c=1/8/16/64.
  - `omni-bench` at c=1/8/32/64: 4c requests (at least 64), warmup
    max(8, c).

  Zero failed requests everywhere.
- **Correctness.** The golden test streams the recorded codes at 36.57 dB SNR
  (36.58 dB before this work). Two streams of the same codes in one batch stay
  bit-identical, and a stream joining 7 frames late does too.

## Codec per frame

| | B=1 | B=64 |
|---|---:|---:|
| kernels per frame, before / after | 207 / 150 | 202 / 145 |
| span, before / after (µs) | 772 / 398 | 2489 / 1019 |

| step | B=1 span (µs) | B=64 span (µs) |
|---|---:|---:|
| streamed kern codec (start) | 772 | 2489 |
| attention applies RoPE and appends K/V itself; one block per (row, head) | 580 | 2312 |
| elementwise work fused into its neighbours, 8-wide loads; residual GEMMs accumulate in place | 487 | 1168 |
| SnakeBeta's sine: two-step range reduction plus `__sinf` | 478 | 1077 |
| programmatic dependent launch, weights loaded before the wait | 422 | 1045 |
| attention loads its K/V window while RoPE runs | 415 | 1037 |
| output conv and depthwise conv issue every tap's load in one round | 398 | 1019 |

Where B=64 went, in µs per frame:

| kernel(s) | before | after |
|---|---:|---:|
| bias + residual adds | 374 | 0 (folded into GEMM accumulate and epilogues) |
| im2col (+ SnakeBeta) | 313 + 150 | 201 |
| bias + SnakeBeta | 267 | 95 |
| attention (+ RoPE and K/V write) | 225 + 14 | 63 |
| output conv | 198 | 24 |
| overlap-add (col2im) | 174 | 30 |

About three quarters of the remaining B=64 time (755 of 1019 µs) is the four
upsampling blocks: the kernel-7 convs' GEMMs (~270 µs) and the im2col that
feeds them (~200 µs), which writes and the GEMM reads a column buffer seven
times the activation.

## Server

nsys under load, default 2+8 chunks. Before is an earlier capture, with the
same script, of the streamed kern codec without this work:

| c | codec span before / after | step period before / after |
|---:|---:|---:|
| 1 | 0.70 / 0.39 ms | 8.66 / 7.81 ms |
| 64 | 5.37 / 1.00 ms | 13.12 / 8.27 ms |

## A/B against main

The tables use these column conventions:

- TTFP is in ms, as p50 / p99.
- Continuity is vLLM-Omni's (underrun within 100 ms) for `vllm bench`, and the
  count of streams with no underrun over 1 ms for `omni-bench`.
- Underrun p99 is in ms.

`omni-bench`:

| c | build | TTFP | RTF p50 | audio-s/s | continuous | underrun p99 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | main 2+8 | 12.1 / 14.0 | 0.075 | 13.3 | 64/64 | 0 |
| 1 | opt 2+8 | 12.2 / 12.4 | 0.075 | 13.3 | 64/64 | 0 |
| 8 | main 2+8 | 23.8 / 46.0 | 0.116 | 64.8 | 64/64 | 0 |
| 8 | opt 2+8 | 18.4 / 18.8 | 0.078 | 96.4 | 64/64 | 0 |
| 32 | main 2+8 | 43.4 / 134.5 | 0.258 | 116.8 | 67/128 | 43 |
| 32 | opt 2+8 | 20.5 / 20.9 | 0.087 | 320.9 | 128/128 | 0 |
| 64 | main 2+8 | 81.4 / 265.5 | 0.439 | 136.8 | 64/256 | 238 |
| 64 | opt 2+8 | 23.7 / 25.8 | 0.101 | 571.0 | 256/256 | 0 |
| 1 | main 1+25 | 6.7 / 6.8 | 0.072 | 13.9 | 0/64 | 59 |
| 1 | opt 1+25 | 6.3 / 6.4 | 0.075 | 13.3 | 0/64 | 68 |
| 8 | main 1+25 | 12.0 / 23.4 | 0.091 | 84.6 | 0/64 | 118 |
| 8 | opt 1+25 | 12.4 / 12.7 | 0.078 | 95.6 | 0/64 | 73 |
| 32 | main 1+25 | 19.7 / 42.5 | 0.154 | 191.8 | 0/128 | 307 |
| 32 | opt 1+25 | 13.7 / 14.0 | 0.087 | 331.7 | 0/128 | 92 |
| 64 | main 1+25 | 27.7 / 54.6 | 0.238 | 239.0 | 0/256 | 564 |
| 64 | opt 1+25 | 16.0 / 16.8 | 0.101 | 553.4 | 0/256 | 120 |

`vllm bench serve --omni`:

| c | build | TTFP | RTF p50 | audio-s/s | continuity |
|---:|---|---:|---:|---:|---:|
| 1 | main 2+8 | 13 / 13 | 0.075 | 13.3 | 100% |
| 1 | opt 2+8 | 13 / 13 | 0.075 | 13.3 | 100% |
| 8 | main 2+8 | 25 / 43 | 0.112 | 68.6 | 100% |
| 8 | opt 2+8 | 18 / 24 | 0.078 | 96.8 | 100% |
| 16 | main 2+8 | 31 / 62 | 0.157 | 98.1 | 100% |
| 16 | opt 2+8 | 19 / 32 | 0.082 | 179.3 | 100% |
| 64 | main 2+8 | 65 / 241 | 0.379 | 140.5 | 70% |
| 64 | opt 2+8 | 33 / 49 | 0.103 | 515.5 | 100% |
| 1 | main 1+25 | 7 / 7 | 0.072 | 13.8 | 100% |
| 1 | opt 1+25 | 7 / 7 | 0.075 | 13.3 | 100% |
| 8 | main 1+25 | 13 / 25 | 0.090 | 85.1 | 79% |
| 8 | opt 1+25 | 12 / 23 | 0.078 | 97.7 | 100% |
| 16 | main 1+25 | 17 / 35 | 0.111 | 134.2 | 13% |
| 16 | opt 1+25 | 13 / 27 | 0.082 | 186.7 | 100% |
| 64 | main 1+25 | 46 / 207 | 0.210 | 240.1 | 0% |
| 64 | opt 1+25 | 22 / 46 | 0.103 | 515.3 | 0% |

**Reading.**

- **c=1.** Both builds decode a frame well inside the talker's step, so they
  tie. main's 1+25 RTF is slightly lower because it decodes rarely.
- **Load.** Under load the step no longer grows with the codec:
  - At c=64, RTF stays at 0.10, against 0.21-0.44 for main.
  - Throughput is 2.1-4.2x main's.
  - p99 TTFP is 3-10x lower.
- **Underruns.** With 2+8 chunks no stream underruns up to c=64. The
  underruns left under 1+25 are the schedule's: after an 80 ms first chunk
  the next arrives 25 steps later.
- **Against the codec before this work.** An earlier session measured the
  streamed codec at c=64 on the same GPU, with its server on other cores:
  373 audio-s/s, RTF 0.156 and TTFP 37 ms under 2+8. This work lifts that to
  571 audio-s/s, RTF 0.10 and TTFP 24 ms.

## Tried, did not land

- **Fused transformer linear layers.** Tensor-core kernels with RMSNorm
  prologues and residual/SiLU epilogues, replacing the cuBLASLt GEMM plus
  norm/activation pairs. They cut 32 launches, but each kernel sat at about
  10 µs whatever its size (latency-bound), so the frame took 412 / 1145 µs
  against 398 / 1019.
- **Folding the RVQ output projection into the codebooks on the host.** At
  load this is 4.3 G MACs, too slow on the CPU. The GPU GEMM stays.
- **Chunking im2col and its GEMM so the column buffer stays in L2.** This
  needs `min`/`sub` in kern's shape expressions, which kern does not have.
- **An implicit-GEMM conv1 on mma.sync.** The estimate was about 80 µs at
  B=64, only for the last block, so it was not built.

# Qwen3-TTS: one manifest

**TL;DR** (2026-09-23, one GB300): the talker, code predictor, sampler and
streamed codec as one kern manifest, one graph launch per decode step, serve
1.27x main's audio per second at c=1 and 4.24x at c=64, with every stream
continuous at c=64 (main: 25%).

## Method

- main (3c4ad29) against this branch's release binary, one server at a time on
  GPU 2; server on cores 108-125, client (`omni-bench`) on 126-143, both on
  the GPU's NUMA node; order main, new, main, new; the table is the mean of
  the two runs.
- Default chunk schedule (2+8). Points c=1/8/32/64 with max(64, 4c) requests
  and max(8, c) warmups; prompts are 64 varied English and Chinese sentences.

## A/B against main

| c | TTFP p50 / p99 ms | RTF | audio-s/s | streams without underrun |
|---:|---|---|---|---|
| 1 | 12.2 / 14.7 → 9.9 / 16.4 | 0.075 → 0.059 | 13.3 → 16.8 (1.27x) | 128/128 → 128/128 |
| 8 | 23.9 / 46.5 → 19.6 / 29.4 | 0.116 → 0.069 | 65.4 → 111.5 (1.71x) | 128/128 → 128/128 |
| 32 | 42.3 / 147.9 → 21.2 / 29.6 | 0.252 → 0.084 | 116.7 → 327.7 (2.81x) | 158/256 → 256/256 |
| 64 | 80.4 / 278.8 → 22.5 / 32.6 | 0.442 → 0.099 | 133.0 → 563.6 (4.24x) | 128/512 → 512/512 |

Under nsys at c=1 each decode step is one `cuGraphLaunch` followed by two
copies back (codes and PCM); the only other launches are the eager prefills.
