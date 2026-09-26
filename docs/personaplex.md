# PersonaPlex engine

NVIDIA's PersonaPlex-7B is a Moshi-architecture full-duplex speech model: a
7B temporal transformer (Helium) over 17 token streams at 12.5 Hz (the
agent's text, the agent's eight Mimi codebooks, the caller's eight), a small
depformer that draws the agent's codebooks one after another inside a frame,
and the Mimi codec on both ends. Its persona is a prefix: a voice prompt (51
frames of stored embeddings) and a role prompt (text), prefilled before the
first frame.

`crates/omni-personaplex` serves it on one GPU as one generated kern
manifest, the way `omni-qwen3-tts` serves Qwen3-TTS; the manifest builder,
weight loading and shared kernels live in `crates/omni-kern`.

## Programs

| program | shape | what |
|---|---|---|
| `prefill` | eager, ragged over `tokens` | new sessions' prompt rows through Helium into their KV rings; each session's token state set to where the prompt leaves it |
| `tick` | one CUDA graph per bucket of `seqs` (1, 2, 4, …, 128) | per session: Mimi encodes the caller's 80 ms frame, Helium steps, the depformer draws text and eight codebooks, the token state advances, Mimi decodes the agent's frame |

A tick is 967 kernel calls in one graph launch. Padding rows of a bucket run on
a lease of their own, so padding never touches a live session's state.

## State

A session is a lease on two states:

- `kv{0..31}`: Helium's K and V, a ring of 3000 positions (the model's
  context), position `p` in slot `p % 3000`. It is allocated once when the
  session opens and never grows, so a session can run past the context
  without any cache management: the oldest frames fall out of attention
  exactly as they do in the reference. 1.5 GiB per session.
- `seq`: the token state (the row Helium reads next, the caller's codes still
  owed by the acoustic delay, a first-frame override) and Mimi's streaming
  state: every convolution's history, double-buffered by frame parity; the
  transposed convolutions' pending overlap; both Mimi transformers' KV rings
  (250 positions).

Nothing is on the host between ticks except each session's position and
frame count.

## One tick

1. **Encode** the caller's 1920 samples: SEANet (strided convolutions as
   im2col + GEMM, ELU residual blocks), the 8-layer Mimi transformer,
   downsampling to 12.5 Hz, and split residual vector quantization
   (codebook 0 on the semantic half, 1..7 on the acoustic half).
2. **Helium**: the input row (17 streams, each summed from its embedding
   table; an absent token is a zero vector) through 32 layers of full MHA
   with interleaved RoPE over the session's ring, then the text head and the
   text draw.
3. **Depformer**, eight steps: step `k` reads its projection of Helium's
   output plus the embedding of what step `k − 1` drew, runs six layers of
   step-`k` weights attending over the frame's steps `0..=k`, and draws
   codebook `k`. The reference runs sixteen steps; the last eight predict the
   caller's codes, which are heard rather than drawn, so they are skipped.
4. **Advance** the token state (`lm_advance`): the drawn text and codebooks and
   the caller's codes become the next input row, with the acoustic codebooks
   delayed one frame behind the semantic one, as the model was trained.
5. **Decode** the agent's frame (the codes the delay pattern completes this
   tick): dequantize, upsample, the decoder transformer, and the transposed
   SEANet with col2im overlap carried in the state.

Sampling is Gumbel-max over the top-k (text: temperature 0.7, k 25; audio:
0.8, k 250), with the noise a counter hash of a per-session seed and the frame,
so a draw is a pure function of (seed, frame, codebook, token).

## Engine

`engine.rs` is the clock around the model; see
[duplex.md](duplex.md#the-personaplex-engine) for how it fits the live contract.

## Correctness

`tests/golden.rs` runs the model through the serving path against the
reference implementation's own run (`tools/personaplex/golden.py`, voice
NATF2 with the default role prompt, the reference's test input, 150 frames),
with the reference's draws and caller codes forced so the two runs stay on
one trajectory. Three sessions share every call: two in lockstep (their
outputs must be bit-identical) and one prefilled five ticks later, so batch
composition changes under it.

| check | result |
|---|---|
| Helium input rows, emitted agent frames | exact, every frame |
| text logits | cosine ≥ 0.9984, argmax agrees 99.3% |
| depformer logits | cosine ≥ 0.9996, argmax agrees 98.8% |
| encoder latent, voiced caller frames | cosine ≥ 0.990 |
| caller semantic code, voiced frames | 100% |
| decoded agent audio | SNR 31.8 dB |

Near-silent caller frames (RMS below about −60 dBFS) are excluded from the
encoder checks: their latent is dominated by rounding (per-frame cosine
drops to 0.88 there). Over all frames our semantic codes agree 85%; the
reference's own Mimi run in bf16 agrees with its f32 run on 93%.
`tests/prompt.rs` checks the tokenizer against SentencePiece and the prompt
rows and token bookkeeping against the reference, on the host.

## Performance

One GB300, 128 session slots (`--max-sessions 128`), caller audio streamed
at real time; tick time is the GPU call as the engine sees it:

| sessions | tick p50 | tick p99 |
|---:|---:|---:|
| 1 | 8.1 ms | 10.0 ms |
| 8 | 8.0 ms | 8.5 ms |
| 32 | 8.6 ms | 9.3 ms |
| 64 | 11.0 ms | 11.8 ms |
| 128 | 15.0 ms | 15.9 ms |

A tick has 80 ms, so 128 sessions use a fifth of the GPU's time. The limit
at 128 is memory (1.5 GiB of KV each; 207 GiB of the GPU in use), not compute. Load
results are in [bench.md](bench.md#personaplex-live-sessions).
