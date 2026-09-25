# HiDream-O1 against vLLM-Omni

**TL;DR** (2026-09-25, one sm_89 GPU, vLLM-Omni's own diffusion benchmark, 2048 x 2048, 28 steps, no guidance): pega-omni finishes a picture in 15.2 s against vLLM-Omni's 21.0 to 21.2 s at one request at a time, 28% sooner and 39% more pictures per second, with runs of the same engine within 0.05 s of each other; four concurrent requests queue the same way on both, at 34% more pictures per second. Most of the lead is the GEMM tile pega-omni picks for a card at its power cap and its own attention kernel (`docs/hidream-o1.md`, Performance). What only pega-omni does is run the distilled checkpoint's own sampler: vLLM-Omni ships the undistilled model's UniPC sampler, so its pictures from this checkpoint are sampled differently from the reference's.

## What makes it fair

- **Their benchmark.** The client is vLLM-Omni's diffusion benchmark, `benchmarks/diffusion/diffusion_benchmark_serving.py`, the script its nightly perf runner (`tests/dfx/perf/scripts/run_diffusion_benchmark.py`) drives for its image models, with the parameters that runner uses for them: `--dataset random --task t2i`, request rate inf, one warmup request. There is no HiDream-O1 entry in its perf tests, so the points follow its Qwen-Image ones at the model's own resolution: 2048 x 2048, c=1 with 10 prompts and c=4 with 16.
- **The same work per picture.** Both engines run 28 transformer forwards over the whole sequence per picture, without guidance. vLLM-Omni is asked for `num_inference_steps: 28` and `guidance_scale: 0`; pega-omni's distilled sampler always takes 28. vLLM-Omni ships only the undistilled model's sampler (UniPC, shift 3, no fixed timesteps), so its pictures from the distilled checkpoint come from a different sampler than the reference's; that changes what the picture looks like, not what a step costs.
- **The same request otherwise.** vLLM-Omni's `/v1/images/generations` backend always adds its own `num_inference_steps` and `seed` at the top level, which pega-omni refuses as unknown OpenAI fields. `tools/hidream_o1/bench_client.py` runs the same script with those two keys dropped for pega-omni and nothing else changed: prompts, sizes, concurrency, timing and metrics are the benchmark's own code. vLLM-Omni runs the unmodified script.
- **Same card, one engine at a time.** Each server runs alone on the same GPU; the other is stopped and its memory is released. Server on CPUs 0-9,20-29, client on 10-19,30-39 (one socket, SMT siblings kept together).
- **Their shipped configuration.** `vllm serve <checkpoint> --omni` with no other flags; vLLM-Omni picks `HiDreamO1ImagePipeline` from the checkpoint's signature.
- **Alternating runs.** vLLM-Omni, pega-omni, pega-omni, vLLM-Omni, so a drift in the machine over the session shows up as a spread instead of a bias.

## Reproduce

`tools/hidream_o1/vs_vllm_omni.sh`:

```bash
export MODEL=/path/to/HiDream-O1-Image-Dev-2604 WORK=/path/to/scratch GPU=0
tools/hidream_o1/vs_vllm_omni.sh setup      # uv venv: vllm 0.30.0, vllm-omni 0.30.0rc1, ninja

tools/hidream_o1/vs_vllm_omni.sh serve-vllm-omni &
tools/hidream_o1/vs_vllm_omni.sh bench vllm-omni vllm-omni    # then stop the server

cargo build --release -p omni-server --features hidream-o1
tools/hidream_o1/vs_vllm_omni.sh serve-pega-omni &
tools/hidream_o1/vs_vllm_omni.sh bench pega-omni pega-omni
```

Each point leaves `c<N>.json` and the full `c<N>.log` under `$WORK/results/<label>/`. `POINTS=1:10` limits a run to c=1. On a driver older than CUDA 13, `TORCH_BACKEND=cu130` with a CUDA 13 forward-compatibility driver on the library path runs the same wheels.

## Results

2026-09-25, one GPU (sm_89, 48 GB, x86_64), vLLM-Omni v0.30.0rc1 on vLLM 0.30.0 / torch 2.13.0+cu130 (CUDA 13 through the forward-compatibility driver), pega-omni at this branch. Zero failed requests everywhere. Latency in seconds, per picture, from the client.

c=1, 10 prompts, in the order they ran:

| run | throughput (req/s) | latency mean | median | p99 |
|---|---:|---:|---:|---:|
| vLLM-Omni | 0.0471 | 21.23 | 21.40 | 22.41 |
| pega-omni | 0.0658 | 15.20 | 15.46 | 15.60 |
| pega-omni | 0.0657 | 15.23 | 15.47 | 15.62 |
| vLLM-Omni | 0.0476 | 21.01 | 21.12 | 21.50 |

c=4, 16 prompts, one run each:

| engine | throughput (req/s) | latency mean | median | p99 |
|---|---:|---:|---:|---:|
| vLLM-Omni | 0.0485 | 74.82 | 82.29 | 83.58 |
| pega-omni | 0.0649 | 55.88 | 61.61 | 61.75 |

Throughput at c=4 is that of c=1 on both: pictures are served one after another, and a request waits for the ones ahead of it. The machine's load average stood between 15 and 28 through these runs (another user's CPU work, on CPUs the pinned server and client do not use). vLLM-Omni's runs sit 1 to 2% above the 20.82 s it measured the day before at a load near 12, pega-omni's within 0.05 s of each other; against the quieter number the lead at c=1 is 27%.

Memory and start-up are left out: the benchmark records neither, and the two engines report memory at different points (vLLM-Omni's process figure after load, 14.66 GiB; pega-omni's process while serving, 17.3 GiB, its K/V caches sized for the longest prompt it accepts).
