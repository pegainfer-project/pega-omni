# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "transformers==4.57.3",
#   "numpy==2.5.4",
#   "scipy==1.18.1",
#   "soundfile==0.14.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Small paired Base-layout comparison, independent of the Qwen speaker encoder.

The JSON manifest has `records` with case_id, language (en/zh), mode,
non_streaming_mode, text, and waveform; `controls` has one entry per language
with reference, heldout (same speaker, different content), and different
(different speaker). Audio paths are relative to the manifest. Runs are
limited to 30 seconds for the Whisper short-form path.

Whisper transcripts and simple English WER / Chinese CER are diagnostics,
not a quality gate. WavLM is VoxCeleb-finetuned and its model card is tagged
English: report Chinese cosine scores with the controls, without a universal
speaker threshold. The controls are context, not a calibrated population.

    uv run tools/qwen3_tts/quality.py --manifest quality-input.json \\
      --asr-model /path/to/whisper-small --speaker-model /path/to/wavlm-base-plus-sv \\
      --device cuda:0 --out quality-results.json
"""

import argparse
import gc
import json
import math
import re
import unicodedata
from pathlib import Path


def normalized(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    if language == "en":
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)
    return [character for character in text if unicodedata.category(character)[0] in "LN"]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, got in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != got)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import (
        Wav2Vec2FeatureExtractor,
        WavLMForXVector,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    if not records:
        raise ValueError("expected at least one paired case")
    controls = {item["language"]: item for item in manifest["controls"]}
    if len(controls) != len(manifest["controls"]):
        raise ValueError("expected one control group per language")
    paired = {}
    for record in records:
        if record["language"] not in {"en", "zh"} or record["language"] not in controls:
            raise ValueError(f"unsupported language or missing controls: {record['language']}")
        if type(record["non_streaming_mode"]) is not bool:
            raise ValueError("non_streaming_mode must be a JSON boolean")
        key = (record["case_id"], record["mode"])
        entries = paired.setdefault(key, {})
        layout = record["non_streaming_mode"]
        if layout in entries:
            raise ValueError(f"duplicate layout for {key}")
        entries[layout] = (record["text"], record["language"])
    if any(set(entries) != {True, False} or entries[True] != entries[False] for entries in paired.values()):
        raise ValueError("each case/mode needs True and False layouts with identical text and language")

    root = args.manifest.resolve().parent

    def audio(relative: str) -> np.ndarray:
        path = root / relative
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if samples.shape[1] != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError(f"expected finite, nonempty mono audio: {path}")
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            samples = resample_poly(samples[:, 0], 16000 // divisor, rate // divisor, window=("kaiser", 5.0))
        else:
            samples = samples[:, 0]
        return samples.astype(np.float32)

    device = torch.device(args.device)
    torch.set_num_threads(4)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.asr_model, local_files_only=True)
    asr = (
        WhisperForConditionalGeneration.from_pretrained(args.asr_model, dtype=dtype, local_files_only=True)
        .eval()
        .to(device)
    )
    asr_parameters = {
        "task": "transcribe",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
    }
    results = []
    with torch.inference_mode():
        for record in records:
            samples = audio(record["waveform"])
            if len(samples) > 30 * 16000:
                raise ValueError(f"short-form evaluation requires <=30 seconds: {record['waveform']}")
            batch = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            features = batch.input_features.to(device, dtype)
            output = asr.generate(
                features,
                attention_mask=batch.attention_mask.to(device),
                language=record["language"],
                return_dict_in_generate=True,
                return_timestamps=False,
                **asr_parameters,
            )
            # The default Whisper tensor path strips EOS during segment assembly.
            tokens = output.sequences
            eos = asr.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos)
            if int(tokens[0, -1]) not in eos_ids:
                raise ValueError(f"ASR did not reach EOS within its token budget: {record['waveform']}")
            transcription = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            reference = normalized(record["text"], record["language"])
            hypothesis = normalized(transcription, record["language"])
            if not reference:
                raise ValueError("expected nonempty normalized reference text")
            edits = edit_distance(reference, hypothesis)
            result = {
                **record,
                "transcription": transcription,
                "metric": "wer" if record["language"] == "en" else "cer",
                "edit_distance": edits,
                "reference_units": len(reference),
                "error_rate": edits / len(reference),
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "case_id": record["case_id"],
                        "mode": record["mode"],
                        "layout": record["non_streaming_mode"],
                        "transcription": transcription,
                        "error_rate": result["error_rate"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    del asr, processor, features, tokens, output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.speaker_model, local_files_only=True)
    speaker = WavLMForXVector.from_pretrained(args.speaker_model, local_files_only=True).eval().to(device)
    embeddings = {}

    def embedding(path: str) -> torch.Tensor:
        if path not in embeddings:
            samples = audio(path)
            inputs = extractor(samples, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            value = speaker(**inputs).embeddings
            embeddings[path] = torch.nn.functional.normalize(value, dim=-1).cpu()[0]
        return embeddings[path]

    def similarity(left: str, right: str) -> float:
        return float(torch.dot(embedding(left), embedding(right)))

    baselines = []
    with torch.inference_mode():
        for language, group in controls.items():
            baselines.append(
                {
                    "language": language,
                    **{key: Path(group[key]).name for key in ("reference", "heldout", "different")},
                    "same_speaker_cosine": similarity(group["reference"], group["heldout"]),
                    "different_speaker_cosine": similarity(group["reference"], group["different"]),
                }
            )
        for result in results:
            group = controls[result["language"]]
            result["speaker_cosine"] = similarity(group["reference"], result["waveform"])
            result["heldout_speaker_cosine"] = similarity(group["heldout"], result["waveform"])
            result["different_speaker_cosine"] = similarity(group["different"], result["waveform"])
    report = {
        "models": {"asr": args.asr_model.name, "speaker": args.speaker_model.name},
        "asr_parameters": asr_parameters,
        "controls": baselines,
        "records": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
