# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "torchaudio==2.11.0",
#   "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS@022e286b98fbec7e1e916cb940cdf532cd9f488e",
#   "safetensors",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# torchaudio = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Ground truth for the Qwen3-TTS CustomVoice engine, from the official package.

Runs one `generate_custom_voice` call (non-streaming text track, the official
default) and records every boundary the Rust engine must reproduce:

    text_ids         [n]          prompt token ids (tokenizer check)
    prefill_embeds   [L, 2048]    talker prefill input (prompt assembly check)
    talker_logits    [T+1, 3072]  raw codebook-0 logits per step, the last one emits EOS
    predictor_logits [T, 15, 2048] raw code-predictor logits per frame
    codes            [T, 16]      the sampled frames (teacher-forcing input)
    wav              [T*1920]     codec decoder output, float in [-1, 1]

Logits are pre-processor (no penalty, suppression or temperature), so a
teacher-forced replay of `codes` must match them regardless of sampling.

    uv run tools/qwen3_tts/golden.py --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --out golden.safetensors
"""

import argparse
import json

import torch
from qwen_tts import Qwen3TTSModel
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = tts.model
    seen: dict[str, list] = {"prefill": [], "talker": [], "predictor": []}

    talker_generate = model.talker.generate

    def record_talker(**kw):
        seen["prefill"].append(kw["inputs_embeds"][0].float().cpu())
        out = talker_generate(**kw, output_logits=True)
        seen["talker"] = [l[0].float().cpu() for l in out.logits]
        return out

    predictor_generate = model.talker.code_predictor.generate

    def record_predictor(**kw):
        out = predictor_generate(**kw, output_logits=True)
        seen["predictor"].append(torch.stack([l[0].float().cpu() for l in out.logits]))
        return out

    model.talker.generate = record_talker
    model.talker.code_predictor.generate = record_predictor

    torch.manual_seed(args.seed)
    text_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0][0].cpu()
    codes, _ = model.generate(
        input_ids=[text_ids[None].cuda()],
        instruct_ids=[None],
        languages=[args.language],
        speakers=[args.speaker],
        non_streaming_mode=True,
        **tts._merge_generate_kwargs(),
    )
    codes = codes[0].cpu()
    wavs, rate = model.speech_tokenizer.decode([{"audio_codes": codes.cuda()}])
    frames = codes.shape[0]

    tensors = {
        "text_ids": text_ids.to(torch.int64),
        "prefill_embeds": seen["prefill"][0].contiguous(),
        "talker_logits": torch.stack(seen["talker"][: frames + 1]).contiguous(),
        "predictor_logits": torch.stack(seen["predictor"][:frames]).contiguous(),
        "codes": codes.to(torch.int64).contiguous(),
        "wav": torch.from_numpy(wavs[0]).float().contiguous(),
    }
    meta = {"text": args.text, "speaker": args.speaker, "language": args.language, "seed": str(args.seed), "sample_rate": str(rate)}
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({**meta, **{k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
