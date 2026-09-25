# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "torchvision",
#   "transformers==4.57.1",
#   "diffusers",
#   "accelerate",
#   "einops",
#   "numpy",
#   "pillow",
#   "safetensors",
#   "scipy",
#   "tqdm",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# torchvision = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Ground truth for the HiDream-O1 engine, from the official implementation.

Runs one text-to-image generation the way the official `inference.py
--model_type dev` does (28 steps, no guidance, flash scheduler) and records
every boundary the Rust engine must reproduce:

    input_ids        [L]              prompt tokens, the timestep slot last
    position_ids     [3, L + N]       M-RoPE (t, h, w) of every token
    noise_0          [N, 3072] f16    the starting standard normals, patch layout
    noise_{k}        [N, 3072] f16    the normals drawn after step k-1, k = 1..27
    z_{k}            [N, 3072] bf16   the latent step k reads (recorded steps only)
    x0_{k}           [N, 3072] bf16   the model's x0 prediction at step k (same steps)
    x0_fp32_{k}      [N, 3072] f32    the same prediction by the model in float32
    image            [H, W, 3] u8     the finished picture
    image_fp32       [H, W, 3] u8     the picture the float32 model samples on the same noise

N is the patch count. The official decoder runs through its
non-flash-attention path (one 4D mask), which is the same attention as its
two-pass flash path.

bf16 does not settle to one answer here: at the noisiest steps the reference
agrees with its own float32 model to a cosine of about 0.98. So the test holds
the engine to the reference's own error, not to its exact output: float32 is
the truth, both for each recorded step and for the finished picture. The
official pipeline computes in bf16 whatever the weights (it hard-codes the
dtype and an autocast), so `image_fp32` comes from the flash scheduler's step
written out below in float32. Noise is stored as float16 to keep the file near
1.1 GB at 2048 x 2048; the engine test widens it back to f32.

    uv run tools/hidream_o1/golden.py --repo <HiDream-O1-Image checkout> \\
        --model <HiDream-O1-Image-Dev-2604> --out hidream-o1-golden.safetensors
"""

import argparse
import gc
import sys

import einops
import numpy as np
import torch
from safetensors.torch import save_file

RECORDED_STEPS = (0, 1, 13, 27)
PROMPT = (
    "A red fox sitting in fresh snow at the edge of a pine forest at dawn, warm light on its fur, "
    "a small wooden sign beside it that reads HELLO."
)


def to_patches(x):
    """`[1, 3, H, W]` pixels to `[H/32 * W/32, 3072]` patch rows, channel-major inside a patch."""
    return einops.rearrange(x, "B C (H p1) (W p2) -> (B H W) (C p1 p2)", p1=32, p2=32)


def forward_fp32(model, sample, z, t):
    """The float32 model's x0 prediction for patch rows `z` at model time `t`."""
    mask = sample["vinput_mask"][0].cuda()
    out = model(
        input_ids=sample["input_ids"].cuda(),
        position_ids=sample["position_ids"].cuda(),
        vinputs=z.to("cuda", torch.float32).unsqueeze(0),
        timestep=t,
        token_types=sample["token_types"].cuda(),
        use_flash_attn=False,
    )
    return out.x_pred[0, mask].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of github.com/HiDream-ai/HiDream-O1-Image")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--size", default="2048x2048")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    width, height = (int(x) for x in args.size.split("x"))
    sys.path.insert(0, args.repo)

    from models import flash_scheduler, pipeline
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.tms_token = "<|tms_token|>"
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    tensors: dict[str, torch.Tensor] = {}
    sample = pipeline.build_t2i_text_sample(args.prompt, height, width, tokenizer, processor, model.config)
    tensors["input_ids"] = sample["input_ids"][0].to(torch.int64).contiguous()
    tensors["position_ids"] = sample["position_ids"][:, 0].to(torch.int64).contiguous()

    start = torch.randn((1, 3, height, width), generator=torch.Generator("cpu").manual_seed(args.seed + 1))
    tensors["noise_0"] = to_patches(start).to(torch.float16).contiguous()

    draws = []
    original_randn = flash_scheduler.hack_randn_tensor

    def recording_randn(*a, **kw):
        value = original_randn(*a, **kw)
        draws.append(value.cpu())
        return value

    flash_scheduler.hack_randn_tensor = recording_randn

    step = [0]
    held = {"forward": model.forward}

    def recording_forward(**kw):
        kw["use_flash_attn"] = False
        out = held["forward"](**kw)
        k = step[0]
        if k in RECORDED_STEPS and f"x0_{k}" not in tensors:
            mask = sample["vinput_mask"][0].to(out.x_pred.device)
            tensors[f"z_{k}"] = kw["vinputs"][0].to(torch.bfloat16).cpu().contiguous()
            tensors[f"x0_{k}"] = out.x_pred[0, mask].to(torch.bfloat16).cpu().contiguous()
        step[0] += 1
        return out

    model.forward = recording_forward

    def generate():
        return pipeline.generate_image(
            model=model,
            processor=processor,
            prompt=args.prompt,
            height=height,
            width=width,
            num_inference_steps=28,
            guidance_scale=0.0,
            shift=1.0,
            timesteps_list=pipeline.DEFAULT_TIMESTEPS,
            scheduler_name="flash",
            seed=args.seed,
            noise_scale_start=7.5,
            noise_scale_end=7.5,
            noise_clip_std=2.5,
        )

    image = generate()
    assert step[0] == 28 and len(draws) == 28, (step[0], len(draws))
    for k, d in enumerate(draws[:-1], start=1):
        tensors[f"noise_{k}"] = d[0].to(torch.float16).contiguous()
    tensors["image"] = torch.from_numpy(np.array(image, dtype=np.uint8)).contiguous()

    # The recording wrapper holds the bf16 model too; both must go before float32 fits.
    held.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, device_map="cuda").eval()
    for k in RECORDED_STEPS:
        t = torch.tensor([1.0 - pipeline.DEFAULT_TIMESTEPS[k] / 1000.0], device="cuda")
        with torch.no_grad():
            out = forward_fp32(model, sample, tensors[f"z_{k}"], t)
        tensors[f"x0_fp32_{k}"] = out.contiguous()

    # FlashFlowMatchEulerDiscreteScheduler.step with the dev settings, in float32.
    timesteps = pipeline.DEFAULT_TIMESTEPS
    z = 7.5 * to_patches(start)
    for k, step_t in enumerate(timesteps):
        t = torch.tensor([1.0 - step_t / 1000.0], device="cuda")
        with torch.no_grad():
            x0 = forward_fp32(model, sample, z, t)
        if k + 1 < len(timesteps):
            sigma_next = timesteps[k + 1] / 1000.0
            eps = draws[k][0].float()
            clip = 2.5 * eps.std().item()
            z = sigma_next * eps.clamp(-clip, clip) * 7.5 + (1.0 - sigma_next) * x0
        else:
            z = x0
    pixels = einops.rearrange((z + 1) / 2, "(H W) (C p1 p2) -> (H p1) (W p2) C", H=height // 32, p1=32, p2=32)
    tensors["image_fp32"] = torch.round(torch.clamp(pixels * 255, 0, 255)).to(torch.uint8).contiguous()
    metadata = {
        "prompt": args.prompt,
        "size": args.size,
        "seed": str(args.seed),
        "steps": ",".join(map(str, RECORDED_STEPS)),
    }
    save_file(tensors, args.out, metadata=metadata)
    print(f"wrote {args.out}: {len(tensors)} tensors, image {image.size}")


if __name__ == "__main__":
    main()
