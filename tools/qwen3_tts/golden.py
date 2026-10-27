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
"""Ground truth for Qwen3-TTS CustomVoice and VoiceDesign, from the official package.

Runs the official generation path with the non-streaming text track and
records every boundary the Rust engine must reproduce:

    text_ids         [n]          prompt token ids (tokenizer check)
    instruct_ids     [m]          instruction token ids (empty if absent)
    prefill_embeds   [L, 2048]    talker prefill input (prompt assembly check)
    talker_logits    [T+1, 3072]  raw codebook-0 logits per step, the last one emits EOS
    predictor_logits [T, 15, 2048] raw code-predictor logits per frame
    codes            [T, 16]      the sampled frames (teacher-forcing input)
    wav              [T*1920]     continuous codec decoder output, float in [-1, 1]

Logits are pre-processor (no penalty, suppression or temperature), so a
teacher-forced replay of `codes` must match them regardless of sampling.

The public speech-tokenizer decode API splits at 300 frames and retains only
25 frames of left context. The Rust codec preserves causal history throughout
the utterance, so its oracle calls the official decoder directly in one pass.

    uv run tools/qwen3_tts/golden.py --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --out golden.safetensors
    uv run tools/qwen3_tts/golden.py --model /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instructions "A low, slow male voice." --out design.safetensors
"""

import argparse
import json

import torch


def decode_continuous(speech_tokenizer, codes):
    """Decode [frames, codebooks] without the public API's context resets."""
    if speech_tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
        raise ValueError("continuous golden decoding requires the 12 Hz tokenizer")
    with torch.inference_mode():
        wav = speech_tokenizer.model.decoder(
            codes.to(speech_tokenizer.device).transpose(0, 1).unsqueeze(0)
        )[0, 0].float().cpu().contiguous()
    expected_samples = codes.shape[0] * speech_tokenizer.model.decode_upsample_rate
    if wav.numel() != expected_samples:
        raise ValueError(f"decoder returned {wav.numel()} samples; expected {expected_samples}")
    return wav, int(speech_tokenizer.model.get_output_sample_rate())


def main() -> None:
    from qwen_tts import Qwen3TTSModel
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.")
    ap.add_argument("--speaker", help="CustomVoice speaker (default: ryan); not used by VoiceDesign")
    ap.add_argument("--instructions", help="voice description (required for VoiceDesign)")
    ap.add_argument("--language", default="english")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = tts.model
    kind = model.tts_model_type
    if kind not in {"custom_voice", "voice_design"}:
        ap.error(f"unsupported tts_model_type: {kind}")
    if kind == "voice_design":
        if not args.instructions or not args.instructions.strip():
            ap.error("VoiceDesign requires non-blank --instructions")
        if args.speaker is not None:
            ap.error("VoiceDesign has no preset --speaker")
    speaker = (args.speaker or "ryan") if kind == "custom_voice" else None
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
    instruct_ids = (
        tts._tokenize_texts([tts._build_instruct_text(args.instructions)])[0][0].cpu()
        if args.instructions
        else torch.empty(0, dtype=torch.int64)
    )
    codes, _ = model.generate(
        input_ids=[text_ids[None].cuda()],
        instruct_ids=[instruct_ids[None].cuda() if instruct_ids.numel() else None],
        languages=[args.language],
        speakers=[speaker],
        non_streaming_mode=True,
        **tts._merge_generate_kwargs(),
    )
    codes = codes[0].cpu()
    wav, rate = decode_continuous(model.speech_tokenizer, codes)
    frames = codes.shape[0]

    tensors = {
        "text_ids": text_ids.to(torch.int64),
        "instruct_ids": instruct_ids.to(torch.int64),
        "prefill_embeds": seen["prefill"][0].contiguous(),
        "talker_logits": torch.stack(seen["talker"][: frames + 1]).contiguous(),
        "predictor_logits": torch.stack(seen["predictor"][:frames]).contiguous(),
        "codes": codes.to(torch.int64).contiguous(),
        "wav": wav,
    }
    meta = {
        "text": args.text, "speaker": speaker or "", "instructions": args.instructions or "",
        "tts_model_type": kind, "language": args.language, "seed": str(args.seed), "sample_rate": str(rate),
        "codec_decode": "continuous_decoder_forward",
    }
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({**meta, **{k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
