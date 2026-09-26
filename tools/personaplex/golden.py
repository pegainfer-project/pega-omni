# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "moshi @ git+https://github.com/NVIDIA/personaplex@3428dfd95309a7f3c84fd93259ded0f810d1ff91#subdirectory=moshi",
#   "safetensors",
#   "numpy",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Ground truth for the PersonaPlex engine, from NVIDIA's reference package.

Runs the reference's offline conversation (`moshi.offline`: voice prompt,
silence, role prompt, silence, then the caller's audio one 80 ms frame at a
time) with sampling on, and records every boundary the Rust engine must
reproduce:

    prompt_inputs  [P, 17]        the LM's input tokens per prompt step after the voice
                                  prompt (text, 8 agent, 8 caller; -1 = none)
    inputs         [T, 17]        the LM's input tokens per conversation step
    text_logits    [T, 32000]     raw text logits per conversation step
    audio_logits   [T, 8, 2048]   raw depformer logits of the agent codebooks
    sampled        [T, 9]         the draws: text token, 8 agent codes
    caller_pcm     [T, 1920]      the caller's audio frames, float in [-1, 1]
    caller_latent  [T, 512]       Mimi's unquantized latent of them (the quantizer's input)
    caller_codes   [T, 8]         Mimi's streaming encoding of them
    agent_codes    [T', 8]        the agent frames the model emitted (one step late)
    agent_pcm      [T', 1920]     Mimi's streaming decoding of them

Logits are raw (before temperature and top-k), so a teacher-forced replay of
`sampled` must match them regardless of sampling.

    uv run tools/personaplex/golden.py --model /path/to/personaplex-7b-v1 \\
        --voice NATF2 --input input_assistant.wav --seconds 12 --out golden.safetensors
"""

import argparse
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# The reference imports sphn (Opus and resampling) at module scope; the
# offline path here needs neither.
sys.modules.setdefault("sphn", types.ModuleType("sphn"))

from moshi.models import loaders  # noqa: E402
from moshi.models.lm import LMGen  # noqa: E402

import sentencepiece  # noqa: E402



def wrap_with_system_tags(text: str) -> str:
    """`moshi.server.wrap_with_system_tags` (importing the server parses argv)."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


ASSISTANT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


def read_wav(path: Path, seconds: float) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    frames = int(seconds * 12.5)
    pcm = pcm[: frames * 1920]
    return np.pad(pcm, (0, frames * 1920 - len(pcm)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--voice", default="NATF2")
    ap.add_argument("--prompt", default=ASSISTANT)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42424242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    mimi = loaders.get_mimi(args.model / loaders.MIMI_NAME, device)
    lm = loaders.get_moshi_lm(args.model / loaders.MOSHI_NAME, device=device)
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.model / loaders.TEXT_TOKENIZER_NAME))
    gen = LMGen(
        lm,
        device=device,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        frame_rate=mimi.frame_rate,
        return_logits=True,
    )
    mimi.streaming_forever(1)
    gen.streaming_forever(1)

    seen: list[torch.Tensor] = []
    prepare = gen.prepare_step_input

    def record(*a, **kw):
        out = prepare(*a, **kw)
        if out is not None:
            seen.append(out[0][0, :, 0].cpu())
        return out

    gen.prepare_step_input = record

    gen.load_voice_prompt_embeddings(str(args.model / "voices" / f"{args.voice}.pt"))
    prompt_ids = tokenizer.encode(wrap_with_system_tags(args.prompt))
    gen.text_prompt_tokens = prompt_ids
    mimi.reset_streaming()
    gen.reset_streaming()
    gen.step_system_prompts(mimi)
    mimi.reset_streaming()
    voice_steps = gen.voice_prompt_embeddings.shape[0]
    prompt_inputs = torch.stack(seen[voice_steps:])
    seen.clear()

    caller = read_wav(args.input, args.seconds)
    caller_pcm = torch.from_numpy(caller).view(-1, 1920)
    caller_latent, caller_codes, text_logits, audio_logits, sampled, agent_codes, agent_pcm = [], [], [], [], [], [], []
    for frame in caller_pcm:
        latent = mimi._encode_to_unquantized_latent(frame.to(device)[None, None])
        codes = mimi.quantizer.encode(latent)
        caller_latent.append(latent[0, :, 0].float().cpu())
        caller_codes.append(codes[0, :, 0].cpu())
        tokens, (tl, al) = gen.step(codes)
        text_logits.append(tl[0, 0, 0].float().cpu())
        audio_logits.append(al[0, :8].float().cpu())
        sampled.append(gen._streaming_state.cache[0, :9, (gen._streaming_state.offset - 1) % 4].cpu())
        if tokens is not None:
            agent_codes.append(tokens[0, 1:9, 0].cpu())
            agent_pcm.append(mimi.decode(tokens[:, 1:9])[0, 0].float().cpu())

    tensors = {
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int64),
        "prompt_inputs": prompt_inputs.to(torch.int64).contiguous(),
        "inputs": torch.stack(seen).to(torch.int64).contiguous(),
        "text_logits": torch.stack(text_logits).contiguous(),
        "audio_logits": torch.stack(audio_logits).contiguous(),
        "sampled": torch.stack(sampled).to(torch.int64).contiguous(),
        "caller_pcm": caller_pcm.contiguous(),
        "caller_latent": torch.stack(caller_latent).contiguous(),
        "caller_codes": torch.stack(caller_codes).to(torch.int64).contiguous(),
        "agent_codes": torch.stack(agent_codes).to(torch.int64).contiguous(),
        "agent_pcm": torch.stack(agent_pcm).contiguous(),
    }
    meta = {"voice": args.voice, "prompt": args.prompt, "voice_steps": str(voice_steps), "seed": str(args.seed)}
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({**meta, **{k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
