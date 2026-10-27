# VoiceDesign correctness and performance validation

This runbook describes the matched VoiceDesign comparison. The completed
RTX 5090 measurements and numerical checks are recorded in
[bench.md](bench.md#qwen3-tts-voicedesign-on-rtx-5090). The CustomVoice
measurements in `qwen3-tts-vs-vllm-omni.md` use a different checkpoint.

## Verified device (2026-10-27)

The measured container exposes one RTX 5090 (32607 MiB, 575 W power limit),
a 90 GiB memory limit and a 25-core CPU quota. The host driver is 595.71.05
and nvcc is 13.0.88. Both `codec.cu` and `talker.cu` are compiled to `sm_120`.
The complete CUDA build, workspace checks and real-GPU golden replays pass.

The current repository pins CUDA 13 bindings and the official-reference
recorder uses a CUDA 13 PyTorch build. Use a host driver compatible with
CUDA 13 (R580 or newer) and a CUDA 13 toolkit, including nvcc and cuBLAS.
Installing a toolkit inside a container does not upgrade its host driver.
Set `OMNI_CUDA_ARCH=120`; the build default is GB300's `103a`.

## Correctness before timing

1. Record the source commit, model revision, OS, driver, toolkit, library
   versions, CPU affinity, GPU power limit and exact server commands.
2. Run the workspace formatting, Clippy and CPU tests, and build the CUDA
   engine. Unconfigured golden tests return early; a green workspace test
   run is not evidence that GPU golden tests executed.
3. Record and replay official VoiceDesign goldens in English and Chinese
   as described in `qwen3-tts.md`. Include a longer utterance spanning more
   than 300 frames to exercise codec history wraparound and the reference
   wrapper's chunk boundary. Use the recorder's continuous official decoder,
   which matches the stateful Rust codec; the public Python wrapper instead
   resets context every 300 frames. Also replay a CustomVoice golden to check
   the existing checkpoint family. Each test checks a single stream and then
   three interleaved streams sequentially.
4. Run `tools/openai_sdk_check.py` against the real VoiceDesign server.
   Generate free-running English and Chinese audio, save it, transcribe it
   for content checks and listen for noise, truncation and instruction
   adherence. Teacher forcing does not test free-running sampling quality.

Start the pega server with a small allocation, then increase it only after
checking memory usage and the complete workload's admission requirements:

```bash
export OMNI_CUDA_ARCH=120
cargo build --release --locked -p omni-server --features qwen3-tts
target/release/pega-omni qwen3-tts --model-path "$DESIGN_MODEL" \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign --listen 127.0.0.1:18000 \
  --max-batch 4 --max-step-tokens 1024 --kv-gib 1
```

## Matched workloads

Use the same checkpoint and ordered inputs, instructions and languages.
Keep generation settings and maximum output budgets aligned. Equal numeric
seeds do not make different implementations' random number streams equal.
Do not use shorter or prematurely terminated output to claim a speedup.
The client sends vLLM the same per-request frame cap that pega applies:
`min(50 + 6 * Unicode_codepoints(input), checkpoint.max_new_tokens)`.

`tools/qwen3_tts/bench_voicedesign.py` records the complete workload and its
digest, request bodies, audio durations, errors and transport arrival times.
Run it against each engine separately, with the same workload arguments.
Use its audio-saving option for quality checks. It is a diagnostic client;
the repository's same-session `omni-bench` gate still applies to performance
changes.

For `omni-bench`, pass `--instructions` for VoiceDesign. The pega server's
language is `--extra '{"language":"english"}'`; vLLM-Omni's task and language
are top-level fields supplied through
`--extra-body '{"task_type":"VoiceDesign","language":"English","non_streaming_mode":true,"stream":true}'`.
The `non_streaming_mode` field selects the prompt layout, not HTTP delivery;
it matches pega's non-streaming text layout while audio still streams.
Use the same prompt file, instruction, seed and request counts on both sides.
For example, against the already-running pega server:

```bash
taskset -c "$CLIENT_CPUS" target/release/omni-bench \
  --base-url http://127.0.0.1:18000 \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign --voice alloy \
  --instructions 'A calm, clear male voice speaking at a natural pace.' \
  --extra '{"language":"english"}' --prompts "$PROMPTS_EN" \
  --max-concurrency 4 --num-requests 32 --warmup 4 --seed 42 \
  --out "$RESULTS/pega-a1-c4.json"
```

The existing `vs_vllm_omni.sh` hardcodes CustomVoice; do not run it unchanged
for VoiceDesign. vLLM-Omni's `seed-tts-design` dataset can supply varied
instructions. Its per-row task/language fields require a client-side body
adapter for pega's API; adding another forwarding server would distort
latency. Preserve the original workload and record any adaptation.

## Comparison protocol

- Pin a published baseline, initially `vllm==0.30.0` and
  `vllm-omni==0.30.0`, rather than the older release candidate in the
  CustomVoice script. Record the package versions and deploy configuration.
  Current source revisions may require a different matched vLLM version;
  report them as a separate baseline.
- Start with the stable release's `qwen3_tts.yaml`, then evaluate its
  `qwen3_tts_high_concurrency_mrv2_single_gpu.yaml` profile. The latter uses
  codec graph batches up to eight and 1+25 chunks; it is a candidate, not
  presumed optimal on the 5090. Allow both engines deployment tuning. Any
  MPS experiment must use a private daemon and leave unrelated services alone.
- Run one engine at a time on an otherwise idle GPU. Pin the server and
  client to separate physical CPU cores within the container's CPU quota.
  Exclude setup, initial compilation, readiness waiting and warmup requests
  from timing. Record whether later graph captures occur during measurement:
  pega creates graphs lazily, and warmups do not guarantee coverage of every
  admission/batch transition. Those later captures remain in request timings.
  Startup is process launch to `/health`; compiled caches are retained, and
  readiness does not imply every lazy graph has been captured.
- Compare equivalent chunk schedules first (for example 1+25); report
  pega's 2+8 schedule separately. Chunk policy changes latency and continuity.
- Measure concurrency 1, 4, 8 and 16 as memory permits. Treat larger
  concurrency as a separate capacity experiment. A failed or out-of-memory
  baseline is a configuration problem, not a speed win.
- Alternate at least three A/B and B/A pairs after server restarts and
  warmup. Retain every run, including failures. Report variability rather
  than selecting the fastest repetition.
- Report p50/p99 time to first audio, end-to-end latency, real-time factor,
  audio seconds per wall second, playback underrun, failure rate, output
  duration distributions and peak GPU memory. Review saved audio and content
  quality alongside speed.

Profile the verified baseline before changing kernels. Candidate areas from
code inspection are the predictor's short-context attention, small-M GEMMs
and per-frame device-to-host transfers. These are hypotheses, not measured
bottlenecks. Keep an optimization only after golden regression, free-running
quality checks and a same-session before/after win; record actual results in
`docs/bench.md` with hardware and date.
