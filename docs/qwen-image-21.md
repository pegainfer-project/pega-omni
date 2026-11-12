# Qwen-Image-2.1 on pega-omni

Qwen-Image-2.1 (7.1B single-stream DiT, 32 blocks, dim 4096, SwiGLU) served as one
generated kern manifest, with the Qwen3-VL-8B text encoder and the 16× VAE in a
Python sidecar that shares the GPU. OpenAI `POST /v1/images/generations`.

## Run

```bash
OMNI_CUDA_ARCH=89 cargo build --release -p omni-server --features qwen-image-21
pega-omni qwen-image-21 --model-path Qwen-Image-2.1-Turbo --device 0 \
  --python /path/to/venv/bin/python
```

The sidecar loads the text encoder lazily on the first request (avoids a host-RAM
race with the DiT weight upload) and keeps it on the GPU afterwards. The VAE uses
256-px tiling to fit alongside the DiT + text encoder in 46 GB.

## Layout

- `src/model.rs` — the manifest: `prefill` (text, causal, writes KV), `decode`
  (image, full attention over cached KV, CUDA-graph), `start`/`step`/`set_z`.
- `kernels/qwen.cu` — FA2 attention (MHA G=1), fused QK-RMSNorm+RoPE, LayerNorm+mod,
  SwiGLU, residual-gate, Philox noise, Euler step.
- `python/sidecar.py` — Qwen3-VL encode (pre-final-norm hidden states) + VAE decode.
- `src/prompt.rs` — 3-axis RoPE positions, timestep sinusoid (bf16 double-round).

## Correctness notes

- The timestep embedder reproduces the reference's bf16 double rounding
  (`bf16(bf16(t*1000)/1000)*1000`); skipping it decorrelates the conditioning.
- RoPE uses adjacent pairs with concatenated axes (16,56,56); negative spatial
  indices wrap into the table tail.
- Every kernel was validated in isolation against PyTorch (cosine ≥ 0.9999):
  attention, qk_rope, ln_mod, silu_mul, res_gate, zero_rms_norm, gelu_tanh, and the
  cuBLASLt GEMM at 4096 rows.
- The engine's Philox noise differs from the reference's `torch.randn`, so
  pixel-PSNR against a fixed golden is not a correctness gate; use `set_z` with the
  golden's initial latent to replay a reference trajectory.

## Measured (single L20, 1024×1024, Turbo 8 steps, bf16)

Steady-state **6.46 s/image** (p50 6.457, p95 6.488), 0.155 img/s.
Breakdown: encode ~0.06 s, DiT ~5.1 s, VAE ~1.3 s. The DiT is near the L20 bf16
GEMM roofline (~3.9 s of GEMM + attention/elementwise). Diffusers eager on the same
card: 6.28 s/image.

Env knobs: `OMNI_TIMING=1` (per-stage latency), `OMNI_DEBUG_PRED=1` (dump pred),
`OMNI_EAGER=1` (disable CUDA graphs).

## Same-card comparison vs vLLM-Omni (L20, 1024×1024, Turbo 8 steps, seed 42)

Measured on one NVIDIA L20 (46 GB), identical prompt/size/steps, steady-state
mean of 3 warmed runs:

| Engine | s/image | throughput |
| --- | --- | --- |
| **pega-omni (this crate)** | **6.46** | 0.155 img/s |
| vLLM-Omni 0.33 (source) + vLLM 0.33.0 | 6.99 | 0.143 img/s |
| diffusers eager | 6.28 | 0.159 img/s |

pega-omni is ~7.6% faster than vLLM-Omni on the same card. Both engines cache
the text prefix K/V, capture the decode step as a CUDA graph and run bf16 GEMMs
near the L20 roofline, so the win is overhead, not math. The plausible
differences: this engine captures each grid size once and keeps the graphs for
the process's life, while the benchmarked vLLM-Omni tied its graph entries to
the owning request's prefix cache and recaptured per request (a same-day fix
makes the entries persistent); and the loop here is one process calling
encode → DiT → VAE synchronously, with no framework scheduler between stages.

Measured A/B on the same L20 (vLLM-Omni, 4 identical consecutive requests)
confirms the lever is the graph/scheduling layer, not kernel fusion:
persistent decode-graph entries cut warm steady-state from ~6.9 s to ~5.55 s
(~20%), while enabling the sm89 fused QK-norm+RoPE kernels changed e2e by
<0.5% (6.88 s fused vs 6.91 s unfused, within noise). The fused-kernel change
was therefore not submitted upstream; the persistent-graph change was
(vllm-project/vllm-omni#10266, RFC #8586 A2).

Setup notes: vLLM-Omni with Qwen-Image-2.1 needs vLLM ≥0.32 (not on PyPI); use the
frozen-commit wheel
`https://wheels.vllm.ai/4ba787ec…/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl`
with `--torch-backend=cu130`, Python ≥3.11, and vllm-omni installed from source
(`SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0`). Benchmark: `tools/qwen_image_21/bench_vllm.py`.