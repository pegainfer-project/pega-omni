# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.14.1",
#   "torchvision",
#   "transformers==5.20.2",
#   "diffusers @ git+https://github.com/huggingface/diffusers@6256aa7666cedd47443adc8f82da9a10e110b09c",
#   "accelerate",
#   "safetensors",
#   "numpy",
#   "pillow",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# torchvision = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Ground truth for the Qwen-Image-2.1 engine, from the official diffusers pipeline.

Runs one text-to-image generation the way `QwenImage21Pipeline` does on the Turbo
checkpoint (the checkpoint's own 8-step sigma grid, no guidance) and records every
boundary the Rust engine must reproduce:

    encoder_hidden   [T, 4096] bf16      VLM hidden states, final norm bypassed, system tokens dropped
    rope_pos         [3, T + N] i32      the (frame, height, width) position of every joint token
    rope_cos         [T + N, 64] f32     cos of the 3-axis rotary angles, axes concatenated
    rope_sin         [T + N, 64] f32     sin of the same
    noise_0          [N, 64] f16         the starting standard normals, spatially flattened
    z_{k}            [N, 64] bf16        the latent step k reads (recorded steps only)
    pred_{k}         [N, 64] bf16        the transformer's velocity prediction at step k (same steps)
    z_final          [N, 64] bf16        the latent the VAE decodes
    image            [H, W, 3] u8        the finished picture, RGB (the VAE's alpha channel dropped)

T is the prompt length after the system tokens are dropped, N = (H/16) * (W/16) is
the latent token count: the 2.1 VAE compresses 16x spatially into 64 channels and
the transformer consumes latents unpatched, so one token is one 16x16 pixel tile.
At 1024x1024, N = 4096.

Three details decide whether an engine agrees with this file:

* The text encoder's `hidden_states[-1]` must be the last decoder layer's output
  *before* the final RMSNorm. transformers 5 ties that entry to `last_hidden_state`,
  so the pipeline neutralizes the norm with a forward hook; the recording inherits it.
* The model reads the timestep as `sigma` rounded to bf16 (`t/1000` on a bf16 tensor),
  not as the float32 sigma the scheduler steps with.
* RoPE is 3-axis (16, 56, 56) with theta 10000 over *adjacent* head-dim pairs, the
  three axes concatenated in that order (8 + 28 + 28 = 64). Text advances one shared
  position on all three axes; the image block freezes the frame axis and lays its
  tokens on a height/width grid centred on zero, so spatial positions are negative
  on the lower half. `rope_pos` holds those signed indices; the pipeline reaches the
  same angles through a lookup table whose negative rows are the negative indices.

`rope_pos` is recomputed here from the documented layout and checked against the
pipeline's own rotary table, so the file cannot carry positions the model did not use.

The sampler is flow matching: `z_{k+1} = z_k + (sigma_{k+1} - sigma_k) * pred_k` in
float32, cast back to bf16. `sigmas` in the metadata is the full grid including the
terminal 0 the scheduler appends.

`noise_0` is the f16 widening of the bf16 draw, kept small the way the other golden
files keep it. f16 subnormals cannot hold all eight bf16 mantissa bits, so a dozen or
so near-zero elements round by less than 1e-7; `z_0` carries the exact bf16 starting
latent for an engine that wants to start bit-exact.

`--kv-cache` records the prefix-cached variant instead. It is mathematically the
same but not bit-identical in bf16 (the decode step tiles attention differently),
so the flag goes into the metadata: a test must compare against the matching run.

    uv run tools/qwen_image_21/golden.py --model-dir /path/to/Qwen-Image-2.1-Turbo \\
        --out golden-qwen21.safetensors
"""

import argparse
import json
import math
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from safetensors.torch import save_file

RECORDED_STEPS = (0, 1, 4, 7)
PROMPT = "A lighthouse on a cliff at dawn."
AXES_DIM = (16, 56, 56)
THETA = 10000
# One vision-language image slot stands for a 2x2 group of latent tokens.
IMG_TOKENS_PER_SLOT = 4


def rope_positions(img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor) -> torch.Tensor:
    """The `[3, L]` (frame, height, width) positions of the joint text/image sequence.

    Mirrors `QwenImage21Rope.forward`'s index construction, minus the table lookup, so the
    recording can state the positions themselves rather than only their cosines.
    """
    frame, image_height, image_width = [], [], []
    is_image = image_pad_mask.tolist()
    cursor, position = 0, 0

    for _, height, width in img_shapes:
        block_start = is_image.index(True, cursor)
        text_len = block_start - cursor
        frame.extend(range(position, position + text_len))
        position += text_len

        cursor = block_start + height * width
        frame.extend([position] * (height * width))
        position += max(height, width)

        image_height.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
        image_width.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

    if cursor < len(is_image):
        frame.extend(range(position, position + len(is_image) - cursor))

    frame_index = torch.tensor(frame, dtype=torch.long)
    pos = frame_index.repeat(3, 1)
    pos[1][image_pad_mask] = torch.tensor(image_height, dtype=torch.long)
    pos[2][image_pad_mask] = torch.tensor(image_width, dtype=torch.long)
    return pos


def rope_cos_sin(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[L, 64]` cos/sin of the rotary angles, the three axes concatenated in order."""
    cos, sin = [], []
    for axis, dim in enumerate(AXES_DIM):
        inv_freq = 1.0 / torch.pow(THETA, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        angle = torch.outer(pos[axis].to(torch.float32), inv_freq)
        cos.append(torch.cos(angle))
        sin.append(torch.sin(angle))
    return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Qwen-Image-2.1-Turbo checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--size", default="1024x1024", help="WIDTHxHEIGHT in pixels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", default=",".join(map(str, RECORDED_STEPS)), help="denoising steps to record")
    ap.add_argument(
        "--kv-cache",
        action="store_true",
        help="record the prefix-cached pipeline variant (same math, different bf16 rounding)",
    )
    args = ap.parse_args()
    width, height = (int(x) for x in args.size.split("x"))
    recorded = tuple(int(x) for x in args.steps.split(","))

    model_dir = Path(args.model_dir).expanduser()
    # The pinned diffusers predates pipeline-configured sampling sigmas (huggingface/diffusers#14950):
    # it warns about `sample_sigmas` in model_index.json and ignores it, so the grid is passed by hand.
    # Passing it explicitly is also correct on a diffusers that does read the key.
    model_index = json.loads((model_dir / "model_index.json").read_text())
    sigmas = model_index.get("sample_sigmas")
    if not sigmas:
        raise SystemExit(f"{model_dir}/model_index.json carries no sample_sigmas; this recorder needs the Turbo grid")
    sigmas = [float(s) for s in sigmas]

    from diffusers import QwenImage21Pipeline

    # device_map puts each component on the GPU as it loads. The checkpoint is 32 GB, so materializing it in
    # host RAM first (the plain `from_pretrained(...).to("cuda")` path) does not fit a 32 GiB machine.
    pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16, device_map="cuda")

    tensors: dict[str, torch.Tensor] = {}
    # The transformer's own rotary table, captured once: it is rebuilt from the same layout every step.
    table: dict[str, torch.Tensor] = {}
    layout: dict[str, object] = {}
    last: dict[str, torch.Tensor] = {}

    def grab_freqs(_module, _args, output):
        table.setdefault("freqs", output.detach().to("cpu", torch.complex64))

    freqs_hook = pipe.transformer.pos_embed.register_forward_hook(grab_freqs)

    held = {"forward": pipe.transformer.forward}
    step = [0]
    num_channels = pipe.transformer.config.in_channels

    def recording_forward(*a, **kw):
        out = held["forward"](*a, **kw)
        k = step[0]
        # Condition-image latents, when present, are prepended; the target image is the tail.
        num_target = math.prod(kw["img_shapes"][0][-1])
        target = kw["hidden_states"][0, -num_target:]
        if k == 0:
            tensors["encoder_hidden"] = kw["encoder_hidden_states"][0].to(torch.bfloat16).cpu().contiguous()
            tensors["noise_0"] = target.to(torch.float16).cpu().contiguous()
            layout["img_shapes"] = list(kw["img_shapes"][0])
            layout["img_mask"] = kw["img_mask"][0].detach().cpu()
            layout["num_target"] = num_target
        if k in recorded:
            pred = out[0] if isinstance(out, tuple) else out.sample
            tensors[f"z_{k}"] = target.to(torch.bfloat16).cpu().contiguous()
            tensors[f"pred_{k}"] = pred[0, -num_target:].to(torch.bfloat16).cpu().contiguous()
        step[0] += 1
        return out

    pipe.transformer.forward = recording_forward

    def after_step(_pipe, _i, _t, kw):
        last["latents"] = kw["latents"]
        return kw

    try:
        result = pipe(
            prompt=args.prompt,
            height=height,
            width=width,
            # retrieve_timesteps checks the grid length against num_inference_steps, so both must be passed.
            num_inference_steps=len(sigmas),
            sigmas=sigmas,
            true_cfg_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            use_kv_cache=args.kv_cache,
            output_type="pil",
            callback_on_step_end=after_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    finally:
        freqs_hook.remove()
        pipe.transformer.forward = held["forward"]
        held.clear()

    assert step[0] == len(sigmas), (step[0], len(sigmas))
    assert all(f"z_{k}" in tensors and f"pred_{k}" in tensors for k in recorded), sorted(tensors)

    # The joint sequence: the VLM mask plus one slot per 2x2 group of target latents, each slot expanded four-fold.
    img_mask = layout["img_mask"]
    repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)
    image_pad_mask = torch.repeat_interleave(img_mask, repeats)
    num_target = layout["num_target"]
    text_len = int(image_pad_mask.shape[0]) - num_target
    assert text_len == tensors["encoder_hidden"].shape[0], (text_len, tensors["encoder_hidden"].shape)
    # The pipeline rounds the request down to a multiple of 2 * vae_scale_factor before laying out latents.
    multiple_of = pipe.vae_scale_factor * 2
    assert num_target == (2 * (height // multiple_of)) * (2 * (width // multiple_of)), num_target

    pos = rope_positions(layout["img_shapes"], image_pad_mask)
    cos, sin = rope_cos_sin(pos)
    freqs = table["freqs"]
    assert freqs.shape == (image_pad_mask.shape[0], sum(AXES_DIM) // 2), freqs.shape
    # The signed positions must reproduce the pipeline's table exactly, or the layout above is wrong.
    err = max((cos - freqs.real).abs().max().item(), (sin - freqs.imag).abs().max().item())
    assert err < 1e-6, f"rope_pos disagrees with the pipeline's rotary table by {err}"
    tensors["rope_pos"] = pos.to(torch.int32).contiguous()
    tensors["rope_cos"] = freqs.real.contiguous()
    tensors["rope_sin"] = freqs.imag.contiguous()

    tensors["z_final"] = last["latents"][0].to(torch.bfloat16).cpu().contiguous()
    # The VAE emits RGBA; the picture the engine serves is the RGB channels, alpha dropped.
    tensors["image"] = torch.from_numpy(np.array(result.images[0].convert("RGB"), dtype=np.uint8)).contiguous()
    assert tuple(tensors["image"].shape) == (height, width, 3), tensors["image"].shape

    metadata = {
        "prompt": args.prompt,
        "size": args.size,
        "seed": str(args.seed),
        "sigmas": ",".join(repr(s) for s in sigmas + [0.0]),
        "text_len": str(text_len),
        "steps": ",".join(map(str, recorded)),
        "num_steps": str(len(sigmas)),
        "kv_cache": str(args.kv_cache),
        "axes_dim": ",".join(map(str, AXES_DIM)),
        "theta": str(THETA),
        "system_tokens_dropped": str(pipe._drop_idx),
        "latent_channels": str(num_channels),
        "vae_scale_factor": str(pipe.vae_scale_factor),
        "dtype": "bfloat16",
        "model": model_dir.name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
    }
    save_file(tensors, args.out, metadata=metadata)
    print(f"wrote {args.out}: {len(tensors)} tensors, text_len {text_len}, image {result.images[0].size}")
    print(f"  sigmas {sigmas + [0.0]}")
    print(f"  rope check max abs err {err:.3g}, kv_cache {args.kv_cache}")


if __name__ == "__main__":
    main()