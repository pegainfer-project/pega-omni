# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp>=3.10,<4"]
# ///
"""Matched VoiceDesign HTTP traces; run each engine alone on the same GPU.

    uv run tools/qwen3_tts/bench_voicedesign.py --backend pega-omni \
      --base-url http://127.0.0.1:8000 --model Qwen3-TTS-12Hz-1.7B-VoiceDesign \
      --concurrency 4 --num-requests 32 --warmup 4 --seed 42 --output pega-a1.json

Repeat with --backend vllm-omni and a separate output. Keep checkpoint, server
sampling, chunk policy, CPU affinity and every workload flag the same. Alternate
A/B then B/A after restarting/warming the servers, never run both on the GPU.
vLLM receives non_streaming_mode=true to match pega-omni's text prompt layout;
this is independent of HTTP streaming. For a fixed 1+25 comparison, set vLLM's
server codec chunk to 25 and pass --vllm-initial-codec-chunk-frames 1; configure
pega-omni's first/chunk frame sizes on its server command as well.
--metadata-file records hardware, server revisions/commands and configuration;
--chunk-note and --sampling-note document settings that this client cannot check.
--workload-file accepts a JSON array of {id,input,instructions,language} objects
(language: english or chinese). Both built-in and custom workloads are recorded
in full, with a SHA256 covering the exact ordered inputs and generation seeds.
--workload-jsonl accepts the same fields with optional id; task_type, when
present, must be VoiceDesign. Other generation overrides are rejected so an
official dataset cannot silently change the comparison's sampling or limits.
Each row records pega-omni's frame cap: min(50 + 6 * Unicode characters,
checkpoint max_new_tokens). vLLM receives the same max_new_tokens explicitly.
--checkpoint-max-new-tokens defaults to the verified 1.7B checkpoint's 8192;
set it to the loaded checkpoint's generation_config.json value on both runs.

Raw output is assumed to be mono 24 kHz signed 16-bit little-endian PCM. HTTP
reads are transport chunks, not codec frames. TTFB is first nonempty audio, not
HTTP headers. Playback underrun starts at that first audio (no initial buffer).
RTF is E2E/audio duration. Failed/partial streams never count toward throughput.
HTTP success and duration do not establish speech correctness: compare per-case
durations, saved audio and ASR/golden results before accepting a speed claim.
Identical numeric seeds do not imply identical RNG streams across engines.

This is a workload/stream diagnostic, supplementing the repository's required
same-session omni-bench performance gate; it does not replace that gate.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

BYTES_PER_SECOND = 24_000 * 2
DEFAULT_WORKLOAD = [
    {
        "id": "en-short",
        "language": "english",
        "input": "Hello, welcome to our project.",
        "instructions": "A calm, clear, middle-aged male voice, speaking at a natural pace.",
    },
    {
        "id": "en-medium",
        "language": "english",
        "input": "Before we begin, please check that your microphone is connected. We will walk through each step together, and you can pause whenever you need.",
        "instructions": "A warm, clear female voice, speaking at a natural pace.",
    },
    {
        "id": "en-long",
        "language": "english",
        "input": "This morning, the research team gathered to review the latest experiment. They compared the results carefully, discussed several possible explanations, and agreed to repeat the measurement under the same conditions. After lunch, they prepared a short report describing what they had learned and which questions still needed more evidence.",
        "instructions": "A calm, clear, middle-aged male voice, speaking at a natural pace.",
    },
    {
        "id": "zh-short",
        "language": "chinese",
        "input": "你好，欢迎一起学习这个项目。",
        "instructions": "沉稳清晰的中年男声，用自然的语速说普通话。",
    },
    {
        "id": "zh-medium",
        "language": "chinese",
        "input": "开始之前，请先确认设备已经连接。我们会一步一步介绍操作过程，如果遇到不明白的地方，可以暂停下来，仔细检查后再继续。",
        "instructions": "温暖清晰的女声，用自然的语速说普通话。",
    },
    {
        "id": "zh-long",
        "language": "chinese",
        "input": "今天上午，研究小组一起讨论了最新的实验结果。他们认真比较各项数据，分析了几种可能的原因，并决定在相同条件下重新进行一次测量。午饭后，大家整理了一份简短的报告，记录已经确认的结论，也列出仍然需要更多证据的问题。只有把这些细节记录清楚，其他人才能重复实验，并理解结果之间的差别。",
        "instructions": "沉稳清晰的中年男声，用自然的语速说普通话。",
    },
]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_jsonl(path):
    cases = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL line {line_number} must be an object")
        unknown = row.keys() - {"id", "input", "instructions", "language", "task_type"}
        if unknown:
            raise ValueError(
                f"JSONL line {line_number}: unsupported fields {sorted(unknown)}; normalize explicitly"
            )
        if row.get("task_type", "VoiceDesign") != "VoiceDesign":
            raise ValueError(
                f"JSONL line {line_number}: expected task_type VoiceDesign"
            )
        cases.append(
            {
                "id": row.get("id", f"case-{line_number:05d}"),
                "input": row.get("input"),
                "instructions": row.get("instructions"),
                "language": row.get("language", "").lower(),
            }
        )
    return cases


def workload(cases, count, seed, checkpoint_cap=8192):
    if not isinstance(cases, list) or not cases:
        raise ValueError("workload must be a nonempty JSON array")
    if not isinstance(checkpoint_cap, int) or checkpoint_cap < 1:
        raise ValueError("checkpoint generation cap must be a positive integer")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "input",
            "instructions",
            "language",
        }:
            raise ValueError(
                "each case needs exactly id, input, instructions, language"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in case.values()
        ):
            raise ValueError("all case fields must be nonblank strings")
        if case["language"] not in ("english", "chinese") or case["id"] in ids:
            raise ValueError("use unique case ids and language english or chinese")
        ids.add(case["id"])
    rows = [
        {
            **cases[i % len(cases)],
            "generation_seed": seed + i,
            "expected_max_new_tokens": min(
                50 + 6 * len(cases[i % len(cases)]["input"]), checkpoint_cap
            ),
        }
        for i in range(count)
    ]
    random.Random(seed).shuffle(rows)
    return [{**row, "request_id": i} for i, row in enumerate(rows)]


def request_body(backend, model, row, initial_codec_chunk_frames=None):
    body = {
        "model": model,
        "input": row["input"],
        "instructions": row["instructions"],
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }
    if backend == "pega-omni":
        # A placeholder voice is required by the front end and ignored by VoiceDesign.
        body.update(
            voice="alloy",
            extra={"language": row["language"], "seed": row["generation_seed"]},
        )
    elif backend == "vllm-omni":
        body.update(
            task_type="VoiceDesign",
            language=row["language"].capitalize(),
            seed=row["generation_seed"],
            non_streaming_mode=True,
            max_new_tokens=row["expected_max_new_tokens"],
        )
        if initial_codec_chunk_frames is not None:
            body["initial_codec_chunk_frames"] = initial_codec_chunk_frames
    else:
        raise ValueError(f"unknown backend {backend}")
    return body


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(q):
        at = (len(values) - 1) * q
        lo = math.floor(at)
        hi = math.ceil(at)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)

    return {
        "n": len(values),
        "min": values[0],
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "max": values[-1],
    }


def stream_metrics(arrivals, e2e):
    """Arrival times are request-relative seconds, lengths are raw PCM bytes."""
    total = sum(chunk["bytes"] for chunk in arrivals)
    first = arrivals[0]["at_s"] if arrivals else None
    dry = None
    stall = 0.0
    for chunk in arrivals:
        at = chunk["at_s"]
        if dry is not None:
            stall += max(0.0, at - dry)
        dry = (
            max(at, dry if dry is not None else at) + chunk["bytes"] / BYTES_PER_SECOND
        )
    audio = total / BYTES_PER_SECOND
    return {
        "audio_bytes": total,
        "audio_s": audio,
        "ttfb_s": first,
        "e2e_s": e2e,
        "rtf": e2e / audio if audio else None,
        "underrun_s": stall,
    }


def summarize(records, wall):
    good = [row for row in records if row["ok"]]
    total = sum(row["audio_s"] for row in good)
    summary = {
        "requests": len(records),
        "successful": len(good),
        "failed": len(records) - len(good),
        "wall_s": wall,
        "successful_audio_s": total,
        "audio_s_per_wall_s": total / wall if wall else None,
        "successful_requests_per_s": len(good) / wall if wall else None,
    }
    for metric in ("ttfb_s", "e2e_s", "audio_s", "rtf", "underrun_s"):
        summary[metric] = distribution([row[metric] for row in good])
    summary["duration_by_case"] = {
        case: {
            "successful": sum(r["ok"] for r in rows),
            "failed": sum(not r["ok"] for r in rows),
            "audio_s": distribution([r["audio_s"] for r in rows if r["ok"]]),
        }
        for case in sorted({r["case_id"] for r in records})
        for rows in [[r for r in records if r["case_id"] == case]]
    }
    return summary


async def fire(
    session,
    url,
    backend,
    model,
    row,
    phase_start,
    retain_audio,
    min_audio_s,
    initial_codec_chunk_frames=None,
):
    from aiohttp import ClientError

    start = time.perf_counter()
    result = {
        "request_id": row["request_id"],
        "case_id": row["id"],
        "language": row["language"],
        "expected_max_new_tokens": row["expected_max_new_tokens"],
        "request_body": request_body(backend, model, row, initial_codec_chunk_frames),
        "started_at_s": start - phase_start,
        "ok": False,
        "status": None,
        "error": None,
        "arrivals": [],
        "response_headers": {},
    }
    pcm = bytearray() if retain_audio else None
    sha = hashlib.sha256()
    prefix = bytearray()
    try:
        async with session.post(url, json=result["request_body"]) as response:
            result["status"] = response.status
            result["headers_at_s"] = time.perf_counter() - start
            result["response_headers"] = {
                k: v
                for k, v in response.headers.items()
                if k.lower()
                in (
                    "content-type",
                    "content-length",
                    "transfer-encoding",
                    "x-audio-sample-rate",
                    "x-audio-channels",
                )
            }
            if response.status != 200:
                raise ValueError(
                    f"HTTP {response.status}: {(await response.content.read(4096)).decode(errors='replace')}"
                )
            media = response.headers.get("Content-Type", "").split(";")[0].lower()
            if media not in ("audio/pcm", "audio/x-pcm", "application/octet-stream"):
                raise ValueError(f"expected raw PCM content type, got {media!r}")
            for header, expected in (
                ("X-Audio-Sample-Rate", 24000),
                ("X-Audio-Channels", 1),
            ):
                actual = response.headers.get(header)
                if actual is not None and int(actual) != expected:
                    raise ValueError(f"{header}: expected {expected}, got {actual}")
            async for chunk in response.content.iter_any():
                if not chunk:
                    continue
                at = time.perf_counter() - start
                result["arrivals"].append({"at_s": at, "bytes": len(chunk)})
                sha.update(chunk)
                if len(prefix) < 12:
                    prefix.extend(chunk[: 12 - len(prefix)])
                if pcm is not None:
                    pcm.extend(chunk)
            total = sum(c["bytes"] for c in result["arrivals"])
            if not total or total % 2:
                raise ValueError(f"empty or unaligned s16le stream: {total} bytes")
            if prefix[:4] == b"RIFF" and prefix[8:12] == b"WAVE":
                raise ValueError("server returned WAV instead of raw PCM")
            if total / BYTES_PER_SECOND < min_audio_s:
                raise ValueError(
                    f"audio shorter than --min-audio-seconds={min_audio_s}"
                )
            result["ok"] = True
    except (ClientError, OSError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(stream_metrics(result["arrivals"], time.perf_counter() - start))
    result["audio_sha256"] = sha.hexdigest()
    if pcm is not None:
        result["_pcm"] = pcm
    return result


async def phase(session, url, args, rows, retain_audio=False):
    # Workers preserve the seeded dispatch order without creating N queued HTTP tasks.
    pending = iter(rows)
    records = []
    start = time.perf_counter()

    async def worker():
        for row in pending:
            records.append(
                await fire(
                    session,
                    url,
                    args.backend,
                    args.model,
                    row,
                    start,
                    retain_audio,
                    args.min_audio_seconds,
                    getattr(args, "vllm_initial_codec_chunk_frames", None),
                )
            )

    await asyncio.gather(*(worker() for _ in range(min(args.concurrency, len(rows)))))
    return sorted(records, key=lambda r: r["request_id"]), time.perf_counter() - start


async def run(args, report):
    import aiohttp

    report["environment"]["aiohttp"] = aiohttp.__version__
    root = args.base_url.rstrip("/")
    url = root + ("/audio/speech" if root.endswith("/v1") else "/v1/audio/speech")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(
        limit=args.concurrency, limit_per_host=args.concurrency
    )
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
        trust_env=False,
        headers={"Accept-Encoding": "identity"},
    ) as session:
        warm, warm_wall = await phase(session, url, args, report["warmup_workload"])
        report["warmup"] = {"summary": summarize(warm, warm_wall), "records": warm}
        if any(not row["ok"] for row in warm):
            report["aborted"] = "warmup failed; measured phase was not started"
            return
        records, wall = await phase(
            session, url, args, report["workload"], args.audio_dir is not None
        )
    report["measurement"] = {"summary": summarize(records, wall), "records": records}
    # Disk writes occur after the measurement window. Request timings exclude them.
    if args.audio_dir:
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        for row in records:
            path = args.audio_dir / f"request-{row['request_id']:05d}.pcm"
            path.write_bytes(row.pop("_pcm"))
            row["audio_file"] = str(path.resolve())


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", required=True, choices=("pega-omni", "vllm-omni"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="workload shuffle and per-request generation seeds",
    )
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument(
        "--checkpoint-max-new-tokens",
        type=int,
        default=8192,
        help="loaded checkpoint generation_config.json max_new_tokens, for matched caps",
    )
    ap.add_argument(
        "--min-audio-seconds",
        type=float,
        default=0,
        help="optional application-specific truncation floor",
    )
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--workload-file", type=Path)
    source.add_argument("--workload-jsonl", type=Path)
    ap.add_argument(
        "--metadata-file",
        type=Path,
        help="JSON object: hardware, commands, revisions, sampling, etc.",
    )
    ap.add_argument(
        "--chunk-note", default="unrecorded: specify actual server chunk settings"
    )
    ap.add_argument(
        "--vllm-initial-codec-chunk-frames",
        type=int,
        help="vLLM-only per-request first chunk override; set 1 for fixed 1+25",
    )
    ap.add_argument(
        "--sampling-note",
        default="unrecorded: verify both checkpoint sampling defaults",
    )
    ap.add_argument("--run-label", default="")
    ap.add_argument(
        "--audio-dir",
        type=Path,
        help="optional saved PCM for listening/ASR, outside timed window",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write the exact workload and request bodies without HTTP",
    )
    args = ap.parse_args()
    if (
        args.concurrency < 1
        or args.num_requests < 1
        or args.warmup < 0
        or args.seed < 0
    ):
        ap.error(
            "concurrency/num-requests must be positive; warmup/seed must be nonnegative"
        )
    if args.seed + args.num_requests + args.warmup >= 2**63:
        ap.error("generation seeds must fit signed 64-bit integers")
    if not 1 <= args.checkpoint_max_new_tokens < 2**63:
        ap.error("checkpoint-max-new-tokens must be a positive signed 64-bit integer")
    if (
        args.vllm_initial_codec_chunk_frames is not None
        and args.vllm_initial_codec_chunk_frames < 1
    ):
        ap.error("vllm-initial-codec-chunk-frames must be positive")
    if args.backend != "vllm-omni" and args.vllm_initial_codec_chunk_frames is not None:
        ap.error(
            "vllm-initial-codec-chunk-frames applies only to vllm-omni; configure pega on the server"
        )
    if (
        not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.min_audio_seconds)
        or args.min_audio_seconds < 0
    ):
        ap.error(
            "timeout must be finite and positive; min-audio-seconds finite and nonnegative"
        )
    if args.output.exists():
        ap.error(
            "output already exists; choose a new path to preserve previous A/B results"
        )
    if (
        args.audio_dir is not None
        and args.audio_dir.exists()
        and any(args.audio_dir.iterdir())
    ):
        ap.error("audio-dir must be new or empty")
    return args


def main():
    args = parse_args()
    cases = (
        read_jsonl(args.workload_jsonl)
        if args.workload_jsonl
        else json.loads(args.workload_file.read_text())
        if args.workload_file
        else DEFAULT_WORKLOAD
    )
    metadata = json.loads(args.metadata_file.read_text()) if args.metadata_file else {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata-file must contain a JSON object")
    rows = workload(cases, args.num_requests, args.seed, args.checkpoint_max_new_tokens)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pcm": {"sample_rate": 24000, "channels": 1, "sample_format": "s16le"},
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "metadata": metadata,
        "workload": rows,
        "workload_sha256": digest(rows),
        "warmup_workload": workload(
            cases,
            args.warmup,
            args.seed + args.num_requests,
            args.checkpoint_max_new_tokens,
        ),
        "planned_request_bodies": [
            request_body(
                args.backend, args.model, row, args.vllm_initial_codec_chunk_frames
            )
            for row in rows
        ],
    }
    if not args.dry_run:
        asyncio.run(run(args, report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second invocation from overwriting this run.
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
    summary = report.get("measurement", report.get("warmup", {})).get("summary")
    print(
        json.dumps(
            summary or {"dry_run": True, "workload_sha256": report["workload_sha256"]},
            indent=2,
        )
    )
    if report.get("aborted"):
        print(report["aborted"], file=sys.stderr)
        return 2
    return 2 if summary and summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
