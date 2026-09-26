<p align="center">
  <img src="assets/logo.svg" width="160" alt="pega-omni">
</p>

<h1 align="center">pega-omni</h1>

<p align="center">OpenAI-compatible speech serving in Rust, from the <a href="https://github.com/pegainfer-project/pegainfer">pegainfer</a> project.</p>

<p align="center">
  <a href="docs/qwen3-tts-vs-vllm-omni.md"><img src="assets/qwen3-tts-vs-vllm-omni.png" width="800" alt="pega-omni vs vLLM-Omni: 6.6x faster first audio, 2.5x the throughput"></a>
</p>

---

Speech models are not text models with an audio suffix. A codec-frame TTS model
streams a packet every few tens of milliseconds for seconds at a time, and the
number a listener feels is **time to first packet** and **whether playback ever
stalls**, not tokens per second. pega-omni is built around those two numbers:
one process, one engine loop, audio handed to the HTTP layer by reference, no
stage-to-stage IPC.

Full-duplex models are further from text serving still: a session listens and
speaks at once for minutes, one 80 ms frame each way per tick, and every open
session costs a frame per tick whether anyone talks. pega-omni serves them over
OpenAI's GPT-Live WebSocket protocol on a clock that never drifts, with all
sessions in one CUDA graph per tick ([docs/duplex.md](docs/duplex.md)).

The repository holds the serving front end, the engine contracts, CPU-only
simulated engines used to load-test the front end, and two GPU engines:
Qwen3-TTS-12Hz-1.7B-CustomVoice ([docs/qwen3-tts.md](docs/qwen3-tts.md)) for
speech, and NVIDIA PersonaPlex-7B ([docs/personaplex.md](docs/personaplex.md))
for full duplex: **128 concurrent live sessions on one GB300 with no playback
stall**.

## Quick start

```bash
cargo build --release
target/release/pega-omni sim --listen 127.0.0.1:8000

curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"pega-omni-sim","input":"Hello from pega-omni.","voice":"alloy"}' -o hello.wav
```

Qwen3-TTS needs the CUDA toolkit (13.x) to build:

```bash
cargo build --release -p omni-server --features qwen3-tts
target/release/pega-omni qwen3-tts --model-path Qwen3-TTS-12Hz-1.7B-CustomVoice

curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"Qwen3-TTS-12Hz-1.7B-CustomVoice","input":"Hello from pega-omni.","voice":"ryan",
       "extra":{"language":"english"}}' -o hello.wav
```

PersonaPlex serves live sessions and a browser demo (microphone in, the model's
voice and transcript out):

```bash
cargo build --release -p omni-server --features personaplex
target/release/pega-omni personaplex --model-path personaplex-7b-v1
# open http://127.0.0.1:8000/ (tunnel with ssh -L 8000:127.0.0.1:8000 for a remote GPU)
```

Any OpenAI client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="pega-omni-sim", voice="alloy", input="Streaming speech.", response_format="pcm"
) as resp:
    for chunk in resp.iter_bytes():
        ...
```

## API

| Route | |
|---|---|
| `POST /v1/audio/speech` | OpenAI's request: `model`, `input`, `voice` (name or `{"id": ...}`), `instructions`, `response_format`, `speed`, `stream_format` (`audio` or `sse`) |
| `GET /v1/live/sessions` | GPT-Live WebSocket, as the official `openai` SDK's `client.live.connect()` speaks it: `session.start` (PCM16 at 24 or 16 kHz, or G.711), `session.input_audio.append`, mute, `session.close`; `session.output_audio.delta` and `session.output_transcript.delta` on the agent's timeline ([docs/duplex.md](docs/duplex.md#protocol)) |
| `GET /` | the live demo page |
| `GET /live/config`, `GET /live/stats` | what the demo reads: the live engine's defaults, and its load as counters (sessions, ticks, tick compute time, late ticks) |
| `GET /v1/models` | the served model |
| `GET /v1/audio/voices` | the engine's voices |
| `GET /metrics` | Prometheus: requests by outcome, TTFP, end-to-end latency, audio samples, engine queue |
| `GET /health` | liveness |

Every response streams: audio leaves as the engine emits it. Where the server
differs from OpenAI, it says so instead of guessing:

- `response_format` defaults to `wav`; `wav` and `pcm` (s16le mono) are encoded,
  `mp3` / `opus` / `aac` / `flac` are refused by name.
- A streaming `wav` carries `0xFFFFFFFF` in both size fields.
- Model-specific options go in one `extra` object that the engine declares
  (the simulator accepts `{"frames": N}`, Qwen3-TTS `language` and `seed`);
  unknown top-level fields are a `400`.
- `stream` (not OpenAI's; vLLM clients such as `vllm bench serve` send it) is
  accepted and changes nothing: every response streams.
- A full engine queue is an immediate `429`, never a wait in the front end.

## Layout

| Crate | |
|---|---|
| `omni-engine` | the contracts: `Speech`, `Event`, `Handle` / `Inbox`; `live`: sessions, the jitter buffer, the clock and `drive`, the loop every live engine runs |
| `omni-frontend` | axum routes, request parsing, wav / pcm / SSE framing, metrics |
| `omni-sim` | the simulated engines (speech and live): pure cores plus thread shells |
| `omni-kern` | the kern manifest builder, weight loading and CUDA helpers the GPU engines share |
| `omni-qwen3-tts` | Qwen3-TTS as one kern manifest (talker, code predictor, sampler, streaming codec decoder, own CUDA kernels), prompt, engine |
| `omni-personaplex` | PersonaPlex-7B as one kern manifest (Mimi encoder, Helium, depformer, Mimi decoder, own CUDA kernels), prompt, tokenizer, live engine |
| `omni-server` | the `pega-omni` binary |
| `omni-bench` | load generator: TTFP, E2E, RTF, playback underrun; `duplex`: concurrent live sessions |

## Measured

The front end against the zero-cost simulator (one engine loop, 32 worker
threads, 1 s of audio per request, unbounded request rate):
**~180k requests/s and ~190k audio-seconds/s with no failures up to 4096
concurrent requests**, TTFP p99 0.3 ms at 64 concurrent. Against a GPU-shaped
paced engine at 2048 concurrent streams it adds no measurable latency over the
engine's own floor and no playback underrun. Method, numbers and the CPU
profile are in [docs/bench.md](docs/bench.md).

## License

Apache-2.0
