# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "soundfile==0.14.0"]
# ///
"""Prepare the pinned Base reference recordings without changing their sources.

SoundFile reads both FLAC and WAV; SciPy supplies the polyphase resampler.
The corpus roots contain the relative paths in cases.json.

    uv run tools/qwen3_tts/prepare.py --en-root /data/EN_speech --cn-root /data/CN_speech --out /data/base-inputs
"""

import argparse
import json
import math
import platform
import wave
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly


def prepare(case: dict, roots: dict[str, Path], audio: dict, out: Path) -> dict:
    parts = []
    source_rate = case["sources"][0]["sample_rate"]
    for source in case["sources"]:
        root = roots[source["dataset"]]
        path = root / source["path"]
        transcript = source["transcript"]
        label_path = root / transcript["path"]
        lines = label_path.read_text(encoding="utf-8").splitlines()
        labels = dict(line.split(maxsplit=1) for line in lines if line.strip())
        label = labels[transcript["key"]]
        if transcript["format"] == "aishell3":
            label = "".join(label.split()[::2])
        if label != source["text"]:
            raise ValueError(f"{path}: transcript differs from the pinned dataset label")
        pcm, rate = sf.read(path, dtype="float64", always_2d=True)
        if rate != source_rate or rate != source["sample_rate"] or pcm.shape != (source["samples"], 1):
            raise ValueError(f"{path}: unexpected source rate or shape: {rate}, {pcm.shape}")
        parts.append(pcm[:, 0])
    if "".join(source["text"] for source in case["sources"]) != case["ref_text"]:
        raise ValueError(f"{case['id']}: reference text differs from its source labels")
    # Concatenate before resampling, without inserted silence or boundary fades.
    pcm = np.concatenate(parts)
    divisor = math.gcd(source_rate, audio["sample_rate"])
    converted = resample_poly(
        pcm,
        audio["sample_rate"] // divisor,
        source_rate // divisor,
        window=tuple(audio["window"]),
        padtype=audio["padtype"],
    )
    converted = np.rint(np.clip(converted, -1.0, 32767 / 32768) * 32768).astype("<i2")
    path = out / f"{case['id']}.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(audio["channels"])
        wav.setsampwidth(2)
        wav.setframerate(audio["sample_rate"])
        wav.writeframes(converted.tobytes())
    return {
        **case,
        "wav": path.name,
        "sample_rate": audio["sample_rate"],
        "samples": len(converted),
        "duration_seconds": len(converted) / audio["sample_rate"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    ap.add_argument("--en-root", type=Path, required=True)
    ap.add_argument("--cn-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    prepared = [
        prepare(case, {"en": args.en_root, "cn": args.cn_root}, cases["audio"], args.out) for case in cases["cases"]
    ]
    controls = [
        prepare(case, {"en": args.en_root, "cn": args.cn_root}, cases["audio"], args.out)
        for case in cases["evaluation_audio"]
    ]
    manifest = {
        **cases,
        "conversion_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
        },
        "cases": prepared,
        "evaluation_audio": controls,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({case["id"]: {key: case[key] for key in ("samples", "duration_seconds")} for case in prepared}))


if __name__ == "__main__":
    main()
