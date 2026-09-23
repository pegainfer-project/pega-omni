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
