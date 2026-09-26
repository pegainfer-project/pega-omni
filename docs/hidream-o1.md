# HiDream-O1 engine

`pega-omni hidream-o1 --model-path <HiDream-O1-Image-Dev-2604>` serves the distilled HiDream-O1-Image checkpoints (`-Dev`, `-Dev-2604`, MIT) as OpenAI's `POST /v1/images/generations` on one GPU. The crate is `omni-hidream-o1`: weights, prompt, the model as one kern manifest over its own kernels (`kernels/hidream.cu`) and cuBLASLt, the sampler and the engine.

## The model

HiDream-O1 is Qwen3-VL-8B's text tower (36 layers, hidden 4096, GQA 32/8, interleaved M-RoPE) trained as a diffusion transformer in pixel space: no VAE and no separate text encoder. A picture is a grid of 32 x 32 RGB patches, each patch one token in the same sequence as the prompt:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|> patch patch ...
```

Text token `k` sits at M-RoPE position `(k, k, k)`, patch `(i, j)` at `(4096, 4096 + i, 4096 + j)`. The text attends causally among itself; the timestep slot (`<|tms_token|>`, embedded by a sinusoid and a two-layer MLP) and the patches attend to everything. The last layer maps each patch back to its 3072 pixel values: the model predicts the clean picture x0 directly.

The model generates at eleven fixed resolutions of about four megapixels (2048 x 2048 is 4096 patches); `size` must name one of them.

## One picture

One picture at a time, as calls into the model's kern programs:

1. **`prefill`** (eager): the prompt's causal rows (every text token but the timestep slot) run once through the tower. They never see the timestep or the patches, so their K/V stays valid for every step and is kept.
2. **`start`**, then 28 × **`predict`** and **`advance`** (both captured as CUDA graphs) at the fixed timesteps of the reference's `--model_type dev` (`999, 987, ..., 8`), no guidance: the timestep slot and the patches run through the tower over the whole K/V (one non-causal attention call), the last layer gives x0, and the next latent is `sigma_next * 7.5 * clip(eps) + (1 - sigma_next) * x0` with the noise clipped to 2.5 of its standard deviation. After the last step the latent is x0.
3. **`rgb`**: patches to 8-bit RGB on the GPU; the front end encodes PNG.

The reference computes every step over the whole sequence and splits attention into a causal pass over the text and a full pass over everything; keeping the text K/V from step 1 gives the same attention without the text rows. What changes from one request or step to the next (the prompt's key count, the timestep, the noise draw, sigma) is read from device inputs, so a grid size's two step graphs are captured once and replayed for every step of every request.

Noise is a counter-based Philox stream keyed by the request's `seed` (`extra.seed`, random when omitted), so a seed reproduces its picture. It is not the reference's torch generator: the same seed gives a different, equally valid picture than the reference script.

## Kernels

`kernels/hidream.cu`: the embedding gather, RMSNorm and residual-add RMSNorm, per-head Q/K norm with Qwen3-VL's interleaved three-axis M-RoPE writing K/V to the caches, SiLU-gated multiply, bias-and-activation, the sampler (Philox + Box-Muller noise, its moments, the Euler step) and patches-to-RGB. Attention is FA2 on `mma.sync`: a block owns 128 packed rows (32 query positions times the 4 query heads of one K/V head), Q stays in registers, K and V stream through shared memory 64 keys at a time, and a `causal` flag serves the text prefill. The GEMMs are kern's `extern:cublaslt_bf16_tn`, the four decoder GEMMs of a step on the algorithms `--gemm-algos` pins (see Performance). The f32 shards are read into bf16 at load, Q|K|V and gate|up fused; the vision tower and `lm_head` are not loaded.

## Correctness

`tools/hidream_o1/golden.py` records the official implementation's run (HiDream-O1-Image at 2c2d29f, transformers 4.57.1, torch 2.11, bf16, its non-flash-attention path) and `crates/omni-hidream-o1/tests/golden.rs` compares against it. Prompt tokens and every M-RoPE position match exactly.

bf16 does not settle to one answer for this model: at the noisiest steps the reference agrees with its own float32 model to a cosine of about 0.98. So the engine is held to the reference's own error, with float32 as the truth, rather than to the reference's exact bytes (2048 x 2048, the reference's prompt, seed and noise replayed):

| against float32 | reference bf16 | ours |
|---|---|---|
| x0 cosine, step 0 (t = 999) | 0.98698 | 0.99433 |
| x0 cosine, step 1 (t = 987) | 0.99130 | 0.99099 |
| x0 cosine, step 13 (t = 764) | 0.99847 | 0.99883 |
| x0 cosine, step 27 (t = 8) | 0.999975 | 0.999974 |
| finished picture, PSNR | 28.65 dB | 26.67 dB |

Each step's cosine must be within 0.002 of the reference's; flattening M-RoPE to one axis drops the step-0 cosine to 0.095.

The finished picture is one trajectory of 28 steps that compound their rounding, and equally accurate numerics land several dB apart on it. The reference's own run with eager attention instead of sdpa lands at 27.68 dB, and its two runs agree with each other only to 26.43 dB; this engine lands at 28.73 dB with cuBLASLt's default GEMM algorithms and at 26.67 dB on the algorithms tuning pins on that card (on a GH200 at 26.81 dB either way), and an earlier build of it on FlashInfer attention between 28.20 and 29.34 dB. A sampler bug falls far below: without the noise clip the picture is at 16.70 dB, with sigma off by one step at 11.63 dB. So the test holds the picture within 3 dB of the reference's, and holds accuracy in the teacher-forced steps.

```bash
uv run tools/hidream_o1/golden.py --repo <HiDream-O1-Image checkout> \
    --model <HiDream-O1-Image-Dev-2604> --out hidream-o1-golden.safetensors
OMNI_HIDREAM_O1_MODEL=<ckpt> OMNI_HIDREAM_O1_GOLDEN=hidream-o1-golden.safetensors \
    cargo test -p omni-hidream-o1 --release -- --nocapture
```

`OMNI_HIDREAM_O1_GEMM_ALGOS=<file>` runs the test on the algorithms a server would pin with that file.

## Performance

Single GPU (sm_89, x86_64, 48 GB), CUDA 13.1, 2026-09-25:

| | |
|---|---|
| 28 steps at 2048 x 2048, noise uploaded from the host (the golden replay) | 13.4 s |
| the same through `/v1/images/generations`, PNG and base64 included (vLLM-Omni's benchmark, c=1, `--gemm-algos` from this card) | 15.2 s |
| device memory while serving (the process, graphs captured) | 17.3 GiB |
| load from the f32 shards | 130 to 170 s |

Loading reads the f32 shards, converts them to bf16 on the host and hands them to kern; the page cache and the host conversion set its time.

By our count a step is about 67 TFLOP. A step keeps the card at its 300 W cap and, over a picture, near 87 °C; the SM clock moves with the power each kernel draws, and bf16 tensor throughput with it (about 138 TFLOPS at 950 MHz). At that limit a kernel is as fast as its energy per multiply.

So the decoder's GEMMs run on algorithms measured on the card. `pega-omni hidream-o1-tune-gemms --model-path <ckpt> --out <file>` (`src/tune.rs`) takes cuBLASLt's candidates without a split reduction for each of the four shapes, keeps the largest group whose outputs are bitwise one (so the winner changes the speed and not the picture), and times them as whole `predict` steps of real pictures: one shape at a time, largest first, each candidate in turn over three rounds, the lowest median wins. Timing the GEMMs alone, or four layers of a step, ranked the candidates otherwise. The file records the GPU and cuBLASLt version and `--gemm-algos <file>` pins its algorithms in the manifest (kern's `algo`); a file from another GPU or cuBLASLt is refused, and without one cuBLASLt's heuristic runs.

On this card tuning takes about 8 minutes and every shape settles on cuBLASLt's 256 x 128 tile (`algo 6 tile 24`), where its first choice is a 128 x 64 or 64 x 128 tile, or split-K for the down projection. Alternating whole pictures in one process, that tile made a picture 8.8% faster on gate|up (4097 x 24576 x 4096), 7.9% on the down projection (4097 x 4096 x 12288), and 2.8% and 2.7% on Q|K|V and the output projection. Through the server, alternating runs (vLLM-Omni's benchmark, c=1, 10 prompts): 15.56 and 15.55 s pinned against 19.65 and 19.67 s on the heuristic. On a GH200 (900 W, CUDA 13.1) the candidates of each shape are all bitwise one and within 2% of each other, and pinning changes nothing: 4.24 and 4.25 s pinned, 4.24 and 4.24 s on the heuristic.

The attention kernel is 4.5% slower than FlashInfer's FA2 at this shape timed alone (1.73 against 1.66 ms for 4097 queries over 4130 keys, the same numerics against an f32 reference) and 2% faster over whole pictures alternated in one process (15.39 against 15.70 s). Through the server, alternating runs of vLLM-Omni's benchmark (c=1, 10 prompts), per picture: 15.84 and 15.87 s on the same GEMM tiles with FlashInfer attention and no graphs, 15.28 and 15.26 s as the kern manifest.

Where a step goes (nsys with the graph's nodes traced, one 2048 x 2048 step): the four decoder GEMMs 77%, attention 18%, the element-wise kernels 5%.

Against vLLM-Omni under its own benchmark: [hidream-o1-vs-vllm-omni.md](hidream-o1-vs-vllm-omni.md).

## Not yet

- The undistilled checkpoint: 50 UniPC steps with classifier-free guidance.
- Image editing and reference images, which run the vision tower.
- More than one picture per forward.
- Streaming partial pictures.
