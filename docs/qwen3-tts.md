# Qwen3-TTS engine

`pega-omni qwen3-tts --model-path <checkpoint>` serves a 12 Hz 1.7B CustomVoice
or VoiceDesign checkpoint on one GPU. `config.json`'s `tts_model_type` selects
the prompt layout, independently of the directory or served model name. The
whole synthesis path (talker, code predictor, sampler, codec decoder) is one
[kern](https://github.com/pegainfer-project/kern) manifest that
`omni-qwen3-tts` generates at load and one `kern-runtime` executes.

## VoiceDesign

VoiceDesign uses `Qwen3-TTS-12Hz-1.7B-VoiceDesign` and a non-blank
`instructions` field to describe the desired voice. The OpenAI SDK requires a
`voice` argument: any name (or `{"id": "..."}`) is accepted and ignored by this
checkpoint. `GET /v1/audio/voices` returns `{"voices": []}`. Missing, empty or
whitespace-only instructions return `400` with `param: "instructions"`.
CustomVoice still requires a preset speaker and allows optional instructions.

The prompt starts with the instruction's text-only tokens, followed by the
assistant role and the same dual text / codec tracks as CustomVoice, without
the preset-speaker position. Language tags and the non-streaming text layout
are shared. The generated `prefill`, `first` and `decode` programs do not
change; the VoiceDesign checkpoint supplies their weights.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    voice="alloy",
    input="Hello from pega-omni.",
    instructions="A low, slow, middle-aged male voice.",
    response_format="pcm",
    extra_body={"extra": {"language": "english"}},
) as response:
    for chunk in response.iter_bytes():
        ...
```

### Validation procedure

On a supported CUDA GPU, record a VoiceDesign golden using the pinned official
implementation, then replay it through the existing three-stream test:

```bash
uv run tools/qwen3_tts/golden.py --model "$DESIGN_MODEL" \
  --text 'The quick brown fox jumps over the lazy dog.' --language english \
  --instructions 'A low, slow, middle-aged male voice.' --out design-en.safetensors
OMNI_QWEN3_TTS_MODEL="$DESIGN_MODEL" OMNI_QWEN3_TTS_GOLDEN="$PWD/design-en.safetensors" \
  cargo test --release -p omni-qwen3-tts --test golden -- --nocapture

uv run --with openai tools/openai_sdk_check.py \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instructions 'A low, slow, middle-aged male voice.' --extra '{"language":"english"}'
```

Repeat the golden with Chinese text, `--language chinese` and a Chinese voice
description. The test compares both text and instruction tokenizer ids,
prompt embeddings, teacher-forced talker / predictor logits and waveform SNR.
Also run the CustomVoice golden as a regression check. ASR spot checks of
served English and Chinese audio and a same-session CustomVoice `omni-bench`
before / after comparison are required before marking the model verified.
Reference runs need two separate checkpoint directories; changing the served
model name does not change the checkpoint family.

On 2026-10-27, the CUDA 13 build on one RTX 5090 passed the original
three-stream official-reference replays with default GEMMs. The optional M=1
profile also passed single-stream and three-stream replays for every case
below. All existing tokenizer, embedding, logit and 30 dB waveform gates are
unchanged. The table shows the profiled build's waveform SNR against the
official continuous codec.

| Checkpoint / case | Frames | Single stream, dB | Worst of three streams, dB |
|---|---:|---:|---:|
| VoiceDesign English | 73 | 34.36 | 33.71 |
| VoiceDesign Chinese | 93 | 32.37 | 32.42 |
| VoiceDesign long English | 366 | 33.14 | 33.04 |
| CustomVoice English (`ryan`) | 75 | 36.02 | 36.09 |

The official OpenAI Python SDK 3.37.0 checker also passed against the real
VoiceDesign GPU server. Free-running audio and performance results are recorded
in [bench.md](bench.md#qwen3-tts-voicedesign-on-rtx-5090). The historical
performance tables further below describe CustomVoice on a GB300.

## Programs

A request is `prefill`, then `first`, then `decode` until it draws the end
token:

| program | runs | rows | does |
|---|---|---|---|
| `init` | once, at load | 1 | the text track's `<tts_pad>` embedding every decode row adds |
| `prefill` | eager | `tokens`: the new prompts, ragged | text projection + codec track, 28 talker layers into paged KV, each prompt's last hidden state into `hidden` |
| `first` | CUDA graph per bucket | `seqs` | `hidden` → frame → PCM: codec head, codebook-0 draw, fifteen code-predictor passes, codec decoder |
| `decode` | CUDA graph per bucket | `seqs` | last frame → talker (1 token) → frame → PCM, the same tail as `first` |

`first` exists so a prompt's frame needs no talker row of its own: prefill
already produced the hidden state it is drawn from. A decode step is one
graph launch, then one copy of the frames and PCM back.

Graph calls pad `seqs` to a bucket (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96,
128); padding rows run on a lease of their own.

## State

- **Talker KV**: kern paged state, one state per layer (`[slot][K | V][8
  heads][128]` bf16), pages of 16 tokens. A request leases its whole KV at
  admission, sized by a frame cap proportional to its input, so nothing is
  preempted and a running request never waits for memory.
- **Per-sequence slot**: the last frame's sixteen codes, the bitmap of
  codebook-0 codes under the repetition penalty, and the codec decoder's
  state (each causal conv's last input rows, each overlapping transposed
  conv's last GEMM row, each attention layer's K/V for the last 72 frames).
  The draws write the frame and the bitmap there; `decode` reads its input
  from there. Nothing but the frame and its PCM crosses to the host.
- **Code predictor KV**: a workspace of 16 slots per sequence, rewritten
  every call: the predictor sees each frame on its own.

## One step

The engine thread runs steps back to back: admit waiting requests FIFO while
batch slots, the per-step prompt budget (`--max-step-tokens`) and KV allow;
`prefill` + `first` for the admitted ones; `decode` for the rest; then send
whichever requests have a chunk due (first after `--first-chunk-frames`, then
every `--chunk-frames`, the rest at the end). Chunk sizes only decide when
audio leaves: every call already returns its frames decoded.

## Kernels

`kernels/talker.cu` is the talker's and predictor's: paged attention (one
block per row and KV head, online softmax over 32-key tiles; a ragged prefill
row finds its sequence's pages through `kv_indptr`) and its dense twin for
the predictor, per-head RMSNorm + RoPE + K/V write, fused residual RMSNorm (with
a strided row selection, so the predictor's first pass keeps only its second
row), SiLU-gate, embedding gathers, and the sampler. The sampler keeps the
Hugging Face processor order in one block per row: repetition penalty and
control-token suppression on the raw logits, temperature, top-k keeping ties
(a radix select over the float bits), softmax, and a draw against the
request's own uniform in vocabulary order; a `force` input (−1 = draw) takes a
given code instead, which is how the golden test runs the serving path.

`kernels/codec.cu` is the codec's: RVQ lookup, RMSNorm, attention that
applies RoPE and appends to its 72-frame K/V ring, SnakeBeta (fused into the
bias epilogue and into the next conv's im2col), causal im2col over a
double-buffered history, overlap-add carrying the previous frame's row,
depthwise conv + LayerNorm, and the output conv. Residual GEMMs accumulate in
place (cuBLASLt with beta = 1) and their biases ride along to the next
consumer's epilogue, so no kernel exists only to add. Every codec kernel
launches with programmatic dependent launch and loads its weights before
waiting on its producer.

GEMMs are kern's cuBLASLt built-in. `build.rs` compiles both files to cubins
(`OMNI_CUDA_ARCH`, default `103a`, GB300); the manifest pins their sha256.
kern verifies the manifest, checks every launch's ABI against its cubin,
allocates buffers at their var bounds, hands out KV pages and state slots, and
captures `first` and `decode` per bucket. Weight layout transforms (fused QKV
and gate/up, conv taps, folded scales) happen once at load, on the host.

## Optional GEMM algorithm profiles

The default uses cuBLASLt's heuristic. `--gemm-algos <json>` selects an
algorithm for each listed `(m=1, n, k)` shape, where
`y[m, n] = x[m, k] · w[n, k]ᵀ`. The profile is sparse: omitted shapes and
calls with two or more rows keep the heuristic. Factor-two predictor calls,
codec im2col row multiples and accumulating codec GEMMs also keep their
original operations. The selected algorithm applies across regular talker,
predictor and codec GEMMs with that exact shape.

```bash
target/release/pega-omni qwen3-tts --model-path "$DESIGN_MODEL" \
  --gemm-algos tools/qwen3_tts/profiles/rtx-5090-cublaslt-130101-m1-extended.json
```

The included profile targets an NVIDIA GeForce RTX 5090 with cuBLASLt
130101. Loading rejects a different GPU name or cuBLASLt version. Numerical
validation uses the existing golden tolerances; an alternative algorithm
may round differently even when a random-input probe matched exactly.
Profile `device` and `gemms` fields are required. Additional measurement
annotations are accepted, so the reproducible probe's timing metadata can
remain alongside `m`, `n`, `k` and `algo`. Both the original four-entry file
and the extended nine-entry file pass English, Chinese, long continuous
VoiceDesign and CustomVoice golden checks. The original file is retained
byte-for-byte for reproduction.

The nine-entry profile improves same-binary c1 actual-audio throughput by
4.11% over the original four entries in three observed pairs. c4 is
approximately flat (median -0.66%). Fixed-code single-stream replays also
improve by 1.046–1.049x, with equal frame counts and PCM hashes across the
profiles; three-stream replays are flat. See the separate serving, fixed-work
and free-generation quality evidence in
[the measurement record](bench.md#extended-single-row-gemm-profile).

The engine passes the path explicitly to `Model::load`; serving does not
read a GEMM environment variable. For a golden replay only, set
`OMNI_QWEN3_TTS_GEMM_ALGOS` to the same path:

```bash
OMNI_QWEN3_TTS_MODEL="$DESIGN_MODEL" \
OMNI_QWEN3_TTS_GOLDEN="$PWD/design-en.safetensors" \
OMNI_QWEN3_TTS_GEMM_ALGOS="$PWD/tools/qwen3_tts/profiles/rtx-5090-cublaslt-130101-m1-extended.json" \
  cargo test --release -p omni-qwen3-tts --test golden -- --nocapture
```

## Correctness

`tools/qwen3_tts/golden.py` records the official implementation's run (qwen-tts
at 022e286b, torch 2.11, bf16) and `crates/omni-qwen3-tts/tests/golden.rs`
compares against it through `prefill`, `first` and `decode`, the recorded
codes forced in place of the draws. It first replays a single stream, then
three streams at once (two in lockstep, one joining 7 steps late, so it is
prefilled next to running rows).

The recorder calls the official codec decoder continuously. Its public
`speech_tokenizer.decode` wrapper instead resets the decoder every 300 frames
with only 25 frames of left context. Beyond that boundary it is a different
reference for this engine, which preserves each causal layer's history. The
golden metadata records `codec_decode=continuous_decoder_forward`; older
recordings longer than 300 frames must be regenerated with this recorder.

Earlier CustomVoice results on a GB300:

| | ours vs official |
|---|---|
| tokenizer ids | identical |
| prompt embeddings | worst row cosine 0.99999 |
| talker logits, 80 steps | worst cosine 0.99990, argmax agrees 98.8% |
| code-predictor logits, 79 × 15 | worst cosine 0.9997, argmax agrees 94.1% |
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

Measured 2026-09-23 on one GB300 against main (the chunked codec, the
talker launch by launch), `omni-bench`, default 2+8 chunks, server and client
pinned to separate cores, two runs each; [bench.md](bench.md#qwen3-tts-one-manifest)
has the method.

| c | TTFP p50 / p99 ms | RTF | audio-s/s | streams without underrun |
|---:|---|---|---|---|
| 1 | 12.2 / 14.7 → 9.9 / 16.4 | 0.075 → 0.059 | 13.3 → 16.8 (1.27x) | all → all |
| 8 | 23.9 / 46.5 → 19.6 / 29.4 | 0.116 → 0.069 | 65.4 → 111.5 (1.71x) | all → all |
| 32 | 42.3 / 147.9 → 21.2 / 29.6 | 0.252 → 0.084 | 116.7 → 327.7 (2.81x) | 62% → all |
| 64 | 80.4 / 278.8 → 22.5 / 32.6 | 0.442 → 0.099 | 133.0 → 563.6 (4.24x) | 25% → all |

Where it comes from:

- **The streamed codec** decodes each frame once instead of re-decoding a
  72-frame window per chunk; its kernels bring one frame for 64 streams to
  1.02 ms of GPU time ([bench.md](bench.md#qwen3-tts-codec-decoder-kernels)).
- **One graph per step**: talker, code predictor, sampler and codec replay as
  one launch, so c=1 is no longer bound by launching ~1,100 kernels from the
  CPU.

Against vLLM-Omni on its own benchmark, see
[qwen3-tts-vs-vllm-omni.md](qwen3-tts-vs-vllm-omni.md): with the same chunk
schedule, first audio 3.7-6.6x sooner and 1.4-2.5x the audio per second.

What is next: a decode graph is ~1,200 kernel nodes, most of them the 15
code-predictor passes, so at c=1 fusing those is the lever; a step that admits
requests runs `prefill` + `first` and `decode` as two waited calls, the next
thing to overlap.
