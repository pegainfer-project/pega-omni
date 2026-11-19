# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "safetensors"]
# ///
"""Record the Base case matrix and check its codec/text boundaries end to end.

Four cases times two clone modes use the True layout. --diagnostics also
records both short references with the False layout. Each recorder process
releases its CUDA allocations before the next case starts. Select a GPU with
CUDA_VISIBLE_DEVICES; this does not require an idle machine.

    uv run tools/qwen3_tts/record_cases.py --manifest /data/base-inputs/manifest.json \
        --model /models/Qwen3-TTS-12Hz-1.7B-Base --out /data/base-fixtures --diagnostics

--check-only checks existing fixtures and rewrites the summary and listening
WAVs without loading models. Fixtures and exported WAVs remain outside Git.
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", help="Base checkpoint directory (required when recording)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--case", action="append", choices=["en-short", "en-long", "cn-short", "cn-long"])
    args = ap.parse_args()
    if not args.check_only and args.model is None:
        ap.error("--model is required when recording")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    recorder = Path(__file__).with_name("golden.py")
    summary = []
    for case in manifest["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        reference = root / case["wav"]
        layouts = ["true", "false"] if args.diagnostics and case["id"].endswith("short") else ["true"]
        for layout in layouts:
            for mode in ["xvector", "icl"]:
                name = f"{case['id']}-{mode}-{layout}"
                fixture = args.out / f"{name}.safetensors"
                if not args.check_only:
                    command = [
                        "uv",
                        "run",
                        str(recorder),
                        "--model",
                        args.model,
                        "--out",
                        str(fixture),
                        "--mode",
                        mode,
                        "--text",
                        case["text"],
                        "--language",
                        case["language"],
                        "--ref-audio",
                        str(reference),
                        "--non-streaming-mode",
                        layout,
                        "--seed",
                        "0",
                    ]
                    if mode == "icl":
                        command += ["--ref-text", case["ref_text"]]
                    if mode == "icl" and case["id"].endswith("long"):
                        command += ["--min-total-frames", "301"]
                    if layout == "false":
                        command += ["--min-trailing-tokens", "2"]
                    print(f"Recording {name}", flush=True)
                    (args.out / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
                    with (args.out / f"{name}.log").open("w", encoding="utf-8") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                with safe_open(fixture, framework="numpy") as saved:
                    meta = saved.metadata()
                    expected = {
                        "mode": mode,
                        "non_streaming_mode": layout,
                        "text": case["text"],
                        "language": case["language"],
                        "seed": "0",
                    }
                    if mode == "icl":
                        expected["ref_text"] = case["ref_text"]
                    if any(meta.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"{fixture}: metadata does not match this recording request")
                    boundaries = json.loads(meta["codec_boundaries"])
                    track = json.loads(meta["text_track"])
                    shapes = {key: saved.get_slice(key).get_shape() for key in saved.keys()}  # noqa: SIM118
                    if "ref_codec_latent" in shapes:
                        latent = saved.get_tensor("ref_codec_latent")
                        code_shape = shapes.get("ref_codes")
                        if (
                            mode != "icl"
                            or not code_shape
                            or latent.shape != (code_shape[0], 512)
                            or latent.dtype != np.float32
                            or not np.isfinite(latent).all()
                        ):
                            raise ValueError(f"{fixture}: expected finite FP32 codec latents aligned to reference codes")
                    if mode == "icl" and case["id"].endswith("long") and boundaries["total_frames"] <= 300:
                        raise ValueError(f"{fixture}: long ICL did not cross 300 frames")
                    if layout == "false" and track["trailing_tokens"] <= 1:
                        raise ValueError(f"{fixture}: False layout did not exercise trailing text")
                    wav = saved.get_tensor("wav")
                    full = saved.get_tensor("wav_full")
                    if not np.array_equal(wav, full[boundaries["crop_start"] : boundaries["crop_end"]]):
                        raise ValueError(f"{fixture}: generated waveform differs from the recorded full decode crop")
                summary.append(
                    {
                        "case_id": case["id"],
                        "mode": mode,
                        "non_streaming_mode": layout == "true",
                        "fixture": fixture.name,
                        "shapes": shapes,
                        "codec_boundaries": boundaries,
                        "text_track": track,
                    }
                )
                if case["id"].endswith("short"):
                    path = args.out / f"{name}.wav"
                    pcm = np.rint(np.clip(wav, -1, 32767 / 32768) * 32768).astype("<i2")
                    with wave.open(str(path), "wb") as audio:
                        audio.setparams((1, 2, int(meta["sample_rate"]), 0, "NONE", "not compressed"))
                        audio.writeframes(pcm.tobytes())
                print(f"Checked {name}: {boundaries}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
