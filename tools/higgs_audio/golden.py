# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers>=5.5",
#   "tokenizers",
#   "safetensors",
# ]
# ///
"""Ground truth for the Higgs TTS 3 engine, from the Hugging Face modules.

Runs on the CPU, so it needs no GPU: the Qwen3 body in bf16 and the codec in
fp32, like the reference's codec. Records every boundary the Rust engine must
reproduce:

    prompt_ids     [L]          zero-shot prompt: <|tts|> <|text|> text <|audio|>
    hidden         [S, 2560]    final-norm hidden per step: the prompt's last row, then one per frame
    logits         [S, 8, 1026] fused-head logits per step, bf16 values widened to f32
    codes          [S, 8]       greedy codes of each step, delay-pattern BOC applied
    frames         [T, 8]       the delayed codes put back in step
    frames_wav     [T * 960]    the codec's output for `frames`
    codec_codes    [Tc, 8]      seeded random codes
    codec_wav      [Tc * 960]   the codec's output for `codec_codes`

The decode loop is the reference sampler at temperature zero: codebook 0
reaching the end token starts a countdown of N - 2 more steps. `logits[s]` is
what a teacher-forced replay of `codes[:s]` must reproduce.

    python tools/higgs_audio/golden.py --model /path/to/higgs-tts-3-4b --out golden.safetensors
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

CODEC = "tied.embedding.modality_embeddings.0.model."
FUSED = "tied.embedding.modality_embeddings.0.embedding.weight"
TEXT = "tied.embedding.text_embedding.weight"
BOC, EOC = 1024, 1025
SAMPLES_PER_FRAME = 960


def read(path: Path, keep) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys() if keep(k)}


def load_body(path: Path, cfg: dict) -> tuple[Qwen3Model, torch.Tensor, torch.Tensor]:
    config = Qwen3Config(**cfg["text_config"])
    state = read(path, lambda k: k.startswith("body.") or k in (TEXT, FUSED))
    fused, text = state.pop(FUSED), state.pop(TEXT)
    state = {k.removeprefix("body."): v for k, v in state.items()}
    state["embed_tokens.weight"] = text
    with torch.device("meta"):
        body = Qwen3Model(config)
    body.load_state_dict(state, assign=True, strict=True)
    body.rotary_emb = Qwen3RotaryEmbedding(config)
    return body.eval(), text, fused


def load_codec(path: Path) -> HiggsAudioV2TokenizerModel:
    config = HiggsAudioV2TokenizerConfig.from_json_file(Path(__file__).with_name("codec_config.json"))
    codec = HiggsAudioV2TokenizerModel(config).eval()
    state = read(path, lambda k: k.startswith(CODEC))
    state = {k.removeprefix(CODEC): v.float() for k, v in state.items()}
    missing, _ = codec.load_state_dict(state, strict=False)
    decoding = [k for k in missing if k.startswith(("acoustic_decoder.", "quantizer.", "fc2."))]
    assert not decoding, f"decoder weights missing: {decoding[:8]}"
    return codec


def prompt_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    special = {t: tokenizer.token_to_id(t) for t in ("<|tts|>", "<|text|>", "<|audio|>")}
    assert all(i is not None for i in special.values()), special
    body = tokenizer.encode(text, add_special_tokens=False).ids
    return [special["<|tts|>"], special["<|text|>"], *body, special["<|audio|>"]]


def embed_codes(fused: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(codes.shape[-1]) * 1026
    return F.embedding(codes + offsets, fused).sum(dim=-2)


def generate(body, text, fused, ids: list[int], steps: int):
    n = fused.shape[0] // 1026
    cache = DynamicCache()
    embeds = F.embedding(torch.tensor([ids]), text)
    delay, countdown, done = 0, None, False
    hidden, logits, codes = [], [], []
    for _ in range(steps):
        out = body(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        last = out.last_hidden_state[0, -1]
        step_logits = F.linear(last, fused).reshape(n, 1026)
        step = step_logits.float().argmax(dim=-1)
        if delay < n:
            if delay + 1 < n:
                step[delay + 1 :] = BOC
            delay += 1
        elif countdown is not None:
            countdown -= 1
            done = countdown <= 0
        elif step[0].item() == EOC:
            done = n <= 2
            countdown = n - 2
        hidden.append(last.float())
        logits.append(step_logits.float())
        codes.append(step)
        if done:
            break
        embeds = embed_codes(fused, step)[None, None]
    return torch.stack(hidden), torch.stack(logits), torch.stack(codes)


def undelay(codes: torch.Tensor) -> torch.Tensor:
    length, n = codes.shape
    rows = length - (n - 1)
    return torch.stack([codes[c : c + rows, c] for c in range(n)], dim=1)


def decode(codec: HiggsAudioV2TokenizerModel, codes: torch.Tensor) -> torch.Tensor:
    out = codec.decode(codes.t()[None].long())
    audio = getattr(out, "audio_values", out)
    return audio.reshape(-1).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--codec-frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = Path(args.model)
    cfg = json.loads((model / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))

    torch.manual_seed(args.seed)
    ids = prompt_ids(tokenizer, args.text)
    with torch.no_grad():
        body, text, fused = load_body(model / "model.safetensors", cfg)
        hidden, logits, codes = generate(body, text, fused, ids, args.steps)
        del body
        codec = load_codec(model / "model.safetensors")
        frames = undelay(codes)
        random = torch.randint(0, 1024, (args.codec_frames, 8))
        frames_wav, codec_wav = decode(codec, frames), decode(codec, random)

    tensors = {
        "prompt_ids": torch.tensor(ids, dtype=torch.int32),
        "hidden": hidden,
        "logits": logits,
        "codes": codes.to(torch.int32),
        "frames": frames.to(torch.int32),
        "frames_wav": frames_wav,
        "codec_codes": random.to(torch.int32),
        "codec_wav": codec_wav,
    }
    meta = {"text": args.text, "seed": str(args.seed), "steps": str(len(codes)), "model": model.name}
    save_file({k: v.contiguous() for k, v in tensors.items()}, args.out, metadata=meta)
    print(json.dumps({**meta, **{k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
