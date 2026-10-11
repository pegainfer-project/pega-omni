# Higgs TTS 3 engine

`pega-omni higgs-audio --model-path <higgs-tts-3-4b>` serves Boson AI's Higgs TTS 3 (4B) on one
GPU. The body, the code draws and the codec decoder are one
[kern](https://github.com/pegainfer-project/kern) manifest that `omni-higgs-audio` generates at
load and one `kern-runtime` executes. The model line started in pegainfer
(`pegainfer-project/pegainfer#408`, `#864`); this is its move here.

The model is a Qwen3-4B body (36 layers, 2560 wide, 32 query and 8 key heads) over text and audio
tokens. A frame is 8 codes from a 1026-code vocabulary at 25 frames per second, drawn by one fused
`[8 × 1026, 2560]` head that is also the audio embedding, and spread over 8 steps by the delay
pattern. A codec in the same checkpoint turns frames into 24 kHz PCM.

Not served: voice cloning from reference audio, which needs the codec's encoder and HuBERT.
Control tags (`<|emotion:elation|>`, `<|sfx:laughter|>`, `<|prosody:pause|>`, see the model card's
`PROMPTING.md`) go inside `input`; `instructions` and `speed` are not used.

## Programs

A request is `prefill`, then `first`, then `decode` until its delay pattern ends, and the codec
runs on the frames as they appear:

| program | runs | rows | does |
|---|---|---|---|
| `prefill` | eager | `tokens`: the new prompts, ragged | text embedding, 36 body layers into paged KV, each prompt's last hidden state into `hidden` |
| `first` | CUDA graph per bucket | `seqs` | `hidden` → fused head → one draw per codebook |
| `decode` | CUDA graph per bucket | `seqs` | last row of codes → fused embedding → body (1 token) → head → draws |
| `codec` | eager | `frames` | a window of frames → PCM |

Graph calls pad `seqs` to a bucket (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128); padding rows run
on a lease of their own.

## State

- **KV**: kern paged state, one per layer (`[slot][K | V][8 heads][128]` bf16), pages of 16 tokens.
  A request leases its whole KV at admission, sized by a row cap proportional to its input, so
  nothing is preempted and a running request never waits for memory.
- **Per-sequence slot**: the last row's eight codes. The draws write them there and `decode` reads
  its input from there; only the codes cross to the host.

## The delay pattern

Every codebook draws independently each step; the delay pattern reaches the kernel as the `force`
input (−1 = draw). `delay.rs` is its state machine, a pure function of the codebook-0 codes drawn so
far: for the first 8 steps codebooks `step + 1…7` read the begin code, and once codebook 0 draws the
end code after that a countdown of 6 steps ends the request. `undelay` puts rows back into frames.
The host owns it, so it is property tested without a GPU; teacher forcing in the golden test is the
same input.

## One step

The engine thread runs steps back to back: admit waiting requests FIFO while batch slots, the
per-step prompt budget (`--max-step-tokens`) and KV allow; `prefill` + `first` for the admitted ones;
`decode` for the rest; then decode whatever chunks are due (first after `--first-chunk-frames`, then
every `--chunk-frames`, the rest at the end).

The codec is not causal: each conv sees both neighbours, so a window's edges see zeros where the
whole utterance would see audio. A chunk is therefore decoded from a window that reaches 8 frames
back into audio already sent and 4 frames past the chunk, and only the chunk's own samples leave;
until a request ends, its last 4 frames wait for that context.

## Kernels

The body is `omni-qwen3`'s `Stack` (36 layers over paged KV), shared with Qwen3-TTS, with its
kernels in `omni-qwen3/kernels/qwen3.cu`. `kernels/higgs.cu` is what surrounds it: the embedding
gather, the fused embedding (eight rows added in f32, rounded once), and the sampler: one block per
(sequence, codebook) with temperature, top-k keeping ties (a radix select over the float bits),
softmax and a draw against the request's own uniform in code order.

`kernels/codec.cu` is the codec's. The quantizers and `fc2` are linear after the lookup, so they are
folded at load into eight `[1024, 256]` tables and one constant. Every dense conv is `im2col`
(Snake fused into the gather) and a cuBLASLt GEMM, every transposed conv a GEMM and an overlap-add,
and a residual unit ends in one add. The weights stay in the checkpoint's bf16 and activations are
bf16 between kernels.

GEMMs are kern's cuBLASLt built-in. `build.rs` compiles both files to cubins (`OMNI_CUDA_ARCH`,
default `103a`, GB300), `omni-qwen3`'s a third; the manifest pins their sha256. None of the
kernels uses programmatic dependent launch, so the engine builds and runs on pre-Hopper cards.

## Correctness

`tools/higgs_audio/golden.py` records the Hugging Face modules' run on the CPU (transformers 5.9,
torch 2.11; the body in bf16, the codec in fp32 like the reference engine's) and
`crates/omni-higgs-audio/tests/golden.rs` compares against it through `prefill`, `first` and
`decode`, the recorded codes forced in place of the draws.

```bash
python tools/higgs_audio/golden.py --model <ckpt> --out golden.safetensors
OMNI_HIGGS_AUDIO_MODEL=<ckpt> OMNI_HIGGS_AUDIO_GOLDEN=golden.safetensors cargo test -p omni-higgs-audio --release
```

The reference engine for serving is SGLang Omni; its prompt (`<|tts|> <|text|> … <|audio|>`),
fused head and sampler are what the golden file follows. The test runs three streams of the same
48-step recording, two in lockstep and one joining 7 steps late, so a prefill runs next to running
rows:

| | ours vs reference |
|---|---|
| prompt ids | identical |
| delay pattern and de-delayed frames, 48 rows | identical |
| body logits, 3 streams × 48 steps | worst relative error 0.0081, argmax differs in 9.7% of 1152 draws |
| the same model on CUDA through torch, against its CPU run | worst 0.0075, argmax differs in 7.6% of 384 draws |
| codec, 40 seeded random frames | 38.4 dB SNR |
| codec, the 41 recorded frames | 42.7 dB SNR |

The relative error is the logits' distance after removing the mean, against the reference's spread;
the torch row is the noise two correct bf16 implementations of this model have between them, which
is what the gates (0.015, 15%) sit above. A plain cosine would be 1.0 for any of these, because the
logits share a large common offset.

Served audio through Whisper large-v3-turbo transcribes back to the input for three English
sentences (a pangram, a tongue twister and a 96 character sentence naming the model); the
transcripts differ from the input only in capitalisation and in how compounds are spelled.

## Performance

Measured 2026-10-10 on one 48 GB sm_89 card that other jobs share (the machine's load average stood
between 25 and 40 throughout), so read the ranges and not single numbers. Both engines were driven by
the same `omni-bench` command (`--response-format pcm --stream-format sse`, four prompts of 80 to
95 characters, 4 warm-up requests, 12 requests at c=1 and 6 per slot above), one engine on the card
at a time. pega-omni ran `--max-batch 16 --kv-gib 4`. SGLang Omni is its `main` at 64d9b96 (sglang
0.5.21, torch 2.13.0+cu130) with the server's defaults, whose first chunk is 20 frames and later
ones 75, so pega-omni is shown at its own 12 / 25 default and at 20 / 75. Each cell is the range
over runs: SGLang Omni three (the first and the others on separate server starts), pega-omni two
per schedule.

| c | | TTFP p50 ms | RTF | audio-s/s | requests with underrun |
|---:|---|---|---|---|---|
| 1 | SGLang Omni | 304 to 312 | 0.277 to 0.296 | 3.4 to 3.6 | 12 / 12 |
| | pega-omni, 12 / 25 | 202 | 0.230 to 0.231 | 4.3 | 0 / 12 |
| | pega-omni, 20 / 75 | 272 | 0.230 to 0.231 | 4.3 to 4.4 | 0 / 12 |
| 8 | SGLang Omni | 331 to 345 | 0.281 to 0.287 | 26.4 to 27.5 | 47 to 48 / 48 |
| | pega-omni, 12 / 25 | 251 to 253 | 0.263 | 29.1 to 29.2 | 0 / 48 |
| | pega-omni, 20 / 75 | 330 to 336 | 0.261 to 0.263 | 28.3 to 29.9 | 0 / 48 |
| 16 | SGLang Omni | 408 to 476 | 0.322 to 0.361 | 21.1 to 48.0 | 96 / 96 |
| | pega-omni, 12 / 25 | 271 to 276 | 0.286 | 53.2 | 0 / 96 |
| | pega-omni, 20 / 75 | 358 to 359 | 0.283 to 0.285 | 51.8 to 53.9 | 21 to 33 / 96 |

At the same chunk schedule first audio is 11 to 13% sooner at c=1, level at c=8 and 12 to 25%
sooner at c=16; the shorter default schedule makes it about a third sooner at c=1. RTF is the time
a request takes over the audio it makes, 17 to 22% lower at c=1. Playback stalls are the larger
gap: under `omni-bench`'s playback model (audio buffered up to the first packet, then played in
real time) every SGLang Omni stream underran, by several hundred ms at c=1, while pega-omni's
streams did not at 12 / 25, nor at 20 / 75 below c=16. I did not profile SGLang Omni, so this says
where the engines differ and not why. Its c=16 throughput spread (21 to 48 audio-s/s) is three runs
on a shared card.

A first chunk waits for its frames, 4 more of codec context and the 7 rows the delay pattern adds:
23 steps of about 9 ms at c=1 with 12 frames; fewer first-chunk frames trade first audio for a
window with less context.
