# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "torchaudio==2.11.0",
#   "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS@022e286b98fbec7e1e916cb940cdf532cd9f488e",
#   "safetensors",
#   "soundfile",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# torchaudio = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Ground truth for Qwen3-TTS CustomVoice and Base, from the official package.

Records one sample with the non-streaming text layout unless explicitly changed:

    text_ids         [n]          prompt token ids (tokenizer check)
    prefill_embeds   [L, 2048]    talker prefill input (prompt assembly check)
    talker_logits    [T+1, 3072]  raw codebook-0 logits per step, the last one emits EOS
    predictor_logits [T, 15, 2048] raw code-predictor logits per frame
    codes            [T, 16]      the sampled frames (teacher-forcing input)
    wav              [T*1920]     generated audio, float in [-1, 1]

Base additionally records the prepared reference PCM, FP32 speaker mel and its
BF16 encoder input, speaker embedding, reference text IDs and ICL codes. Both
layouts record trailing_text_hidden and tts_pad_embed for decode replay.
The Base waveform is decoded continuously with the complete ICL reference
prefix, saved as wav_full, and cropped at the exact reference-frame boundary.
The public decoder's 300-frame chunking is not a continuous-decoding oracle.

Logits are pre-processor (no penalty, suppression or temperature), so a
teacher-forced replay of `codes` must match them regardless of sampling.

    uv run tools/qwen3_tts/golden.py --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --out golden.safetensors
    uv run tools/qwen3_tts/golden.py --model /path/to/Qwen3-TTS-12Hz-1.7B-Base --mode icl \
        --ref-audio reference.wav --ref-text "The reference transcript." --out base.safetensors

Reference WAVs must already be mono 24 kHz: preparation and resampling are
separate from recording, so future implementations receive identical samples.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import modeling_qwen3_tts as official
from safetensors.torch import save_file


def cpu(tensor: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--text", default="The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun."
    )
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["customvoice", "xvector", "icl"], default="customvoice")
    ap.add_argument("--ref-audio", type=Path)
    ap.add_argument("--ref-text")
    ap.add_argument("--non-streaming-mode", choices=["true", "false"], default="true")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument(
        "--min-total-frames", type=int, default=0, help="Reject cases that miss the requested codec boundary."
    )
    ap.add_argument(
        "--min-trailing-tokens", type=int, default=0, help="Reject cases without the requested trailing text."
    )
    args = ap.parse_args()
    if args.mode != "customvoice" and args.ref_audio is None:
        ap.error("Base modes require --ref-audio")
    if args.mode == "icl" and not args.ref_text:
        ap.error("ICL requires --ref-text")
    if args.mode != "icl" and args.ref_text is not None:
        ap.error("--ref-text is only used by ICL")
    if args.mode == "customvoice" and args.ref_audio is not None:
        ap.error("CustomVoice does not use --ref-audio")
    reference = None
    if args.ref_audio is not None:
        reference, sample_rate = sf.read(args.ref_audio, dtype="float32")
        if sample_rate != 24000 or reference.ndim != 1 or reference.size == 0:
            ap.error("--ref-audio must contain nonempty mono 24 kHz PCM")

    tts = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = tts.model
    expected = "custom_voice" if args.mode == "customvoice" else "base"
    if model.tts_model_type != expected:
        ap.error(f"--mode {args.mode} requires model type {expected}, got {model.tts_model_type}")
    tensors = {}
    seen = {"talker": [], "predictor": []}

    talker_generate = model.talker.generate

    def record_talker(**kw):
        tensors["prefill_embeds"] = cpu(kw["inputs_embeds"][0])
        tensors["trailing_text_hidden"] = cpu(kw["trailing_text_hidden"][0])
        tensors["tts_pad_embed"] = cpu(kw["tts_pad_embed"][0])
        tensors["attention_mask"] = cpu(kw["attention_mask"][0], torch.int64)
        seen["talker_kwargs"] = {key: value for key, value in kw.items() if not isinstance(value, torch.Tensor)}
        out = talker_generate(**kw, output_logits=True)
        seen["talker"] = [cpu(logits[0]) for logits in out.logits]
        seen["terminal_token"] = int(out.sequences[0, -1])
        return out

    predictor_generate = model.talker.code_predictor.generate

    def record_predictor(**kw):
        seen["predictor_kwargs"] = {key: value for key, value in kw.items() if not isinstance(value, torch.Tensor)}
        out = predictor_generate(**kw, output_logits=True)
        seen["predictor"].append(torch.stack([cpu(logits[0]) for logits in out.logits]))
        return out

    model.talker.generate = record_talker
    model.talker.code_predictor.generate = record_predictor

    torch.manual_seed(args.seed)
    text_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0][0].cpu()
    ref_ids = None
    prompt = None
    ref_codes = None
    if reference is not None:
        tensors["ref_pcm"] = torch.from_numpy(reference).contiguous()
        mel_spectrogram = official.mel_spectrogram

        def record_mel(*positional, **keywords):
            mel = mel_spectrogram(*positional, **keywords)
            tensors["speaker_mel"] = cpu(mel[0].transpose(0, 1))
            return mel

        def record_speaker_input(_module, inputs):
            tensors["speaker_mel_input"] = cpu(inputs[0][0])

        hook = model.speaker_encoder.register_forward_pre_hook(record_speaker_input)
        try:
            with patch.object(official, "mel_spectrogram", record_mel):
                items = tts.create_voice_clone_prompt(
                    ref_audio=(reference, 24000), ref_text=args.ref_text, x_vector_only_mode=args.mode == "xvector"
                )
        finally:
            hook.remove()
        tensors["speaker_embedding"] = cpu(items[0].ref_spk_embedding)
        prompt = tts._prompt_items_to_voice_clone_prompt(items)
        ref_codes = items[0].ref_code
        if ref_codes is not None:
            tensors["ref_codes"] = cpu(ref_codes, torch.int64)
            ref_ids = tts._tokenize_texts([tts._build_ref_text(args.ref_text)])[0]
            tensors["ref_text_ids"] = cpu(ref_ids[0], torch.int64)

    generate_kwargs = tts._merge_generate_kwargs(max_new_tokens=args.max_new_tokens)
    mode_kwargs = (
        {"instruct_ids": [None], "speakers": [args.speaker]}
        if args.mode == "customvoice"
        else {"ref_ids": [ref_ids], "voice_clone_prompt": prompt}
    )
    codes, _ = model.generate(
        input_ids=[text_ids[None].cuda()],
        languages=[args.language],
        non_streaming_mode=args.non_streaming_mode == "true",
        **mode_kwargs,
        **generate_kwargs,
    )
    codes = codes[0].cpu()
    frames = codes.shape[0]
    if seen["terminal_token"] != model.config.talker_config.codec_eos_token_id:
        raise ValueError("Generation did not reach EOS; increase --max-new-tokens before recording a golden")
    if frames == 0 or len(seen["talker"]) != frames + 1 or len(seen["predictor"]) != frames:
        raise ValueError("Expected one talker EOS step after the generated audio frames")
    groups = model.config.talker_config.num_code_groups
    if codes.shape != (frames, groups) or any(logits.shape[0] != groups - 1 for logits in seen["predictor"]):
        raise ValueError("Expected one talker code and one predictor logit vector per remaining code group")
    ref_frames = 0 if ref_codes is None else ref_codes.shape[0]
    total_frames = ref_frames + frames
    trailing_tokens = tensors["trailing_text_hidden"].shape[0]
    if total_frames < args.min_total_frames:
        raise ValueError(f"Expected at least {args.min_total_frames} total codec frames, got {total_frames}")
    if trailing_tokens < args.min_trailing_tokens:
        raise ValueError(f"Expected at least {args.min_trailing_tokens} trailing text tokens, got {trailing_tokens}")

    tokenizer = model.speech_tokenizer
    rate = tokenizer.model.config.output_sample_rate
    samples_per_frame = tokenizer.model.decode_upsample_rate
    crop_start = ref_frames * samples_per_frame
    full_samples = total_frames * samples_per_frame
    if args.mode == "customvoice":
        wavs, rate = tokenizer.decode([{"audio_codes": codes.cuda()}])
        wav = torch.from_numpy(wavs[0])
    else:
        decode_codes = codes if ref_codes is None else torch.cat([ref_codes.cpu(), codes])
        decoded = tokenizer.model.decoder(decode_codes.T[None].to(tokenizer.device))[0, 0]
        if decoded.numel() != full_samples:
            raise ValueError("Continuous decoder length does not match the complete reference and generated codes")
        tensors["wav_full"] = cpu(decoded)
        wav = decoded[crop_start:full_samples]
    if wav.numel() != frames * samples_per_frame:
        raise ValueError("Generated waveform length does not match the code-frame boundary")

    tensors.update(
        text_ids=cpu(text_ids, torch.int64),
        talker_logits=torch.stack(seen["talker"]).contiguous(),
        predictor_logits=torch.stack(seen["predictor"]).contiguous(),
        codes=cpu(codes, torch.int64),
        wav=cpu(wav),
    )
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {name}")
    meta = {
        "text": args.text,
        "speaker": args.speaker if args.mode == "customvoice" else "",
        "language": args.language,
        "seed": str(args.seed),
        "sample_rate": str(rate),
        "mode": args.mode,
        "non_streaming_mode": args.non_streaming_mode,
        "generate_kwargs": json.dumps(generate_kwargs, sort_keys=True),
        "talker_generate_kwargs": json.dumps(seen["talker_kwargs"], sort_keys=True),
        "predictor_generate_kwargs": json.dumps(seen["predictor_kwargs"], sort_keys=True),
        "waveform_decoder": "public" if args.mode == "customvoice" else "continuous",
        "codec_boundaries": json.dumps(
            {
                "reference_frames": ref_frames,
                "generated_frames": frames,
                "total_frames": total_frames,
                "samples_per_frame": samples_per_frame,
                "full_samples": full_samples,
                "crop_start": crop_start,
                "crop_end": full_samples,
                "reference_input_samples": 0 if reference is None else reference.size,
            },
            sort_keys=True,
        ),
        "text_track": json.dumps(
            {
                "target_tokens": text_ids.numel(),
                "reference_tokens": 0 if ref_ids is None else ref_ids.numel(),
                "prefill_tokens": tensors["prefill_embeds"].shape[0],
                "trailing_tokens": trailing_tokens,
            },
            sort_keys=True,
        ),
    }
    if args.ref_audio is not None:
        meta["ref_text"] = args.ref_text or ""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({**meta, **{k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
