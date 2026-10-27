# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp>=3.10,<4"]
# ///
"""CPU-only benchmark contract checks: uv run tools/qwen3_tts/test_bench_voicedesign.py."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import bench_voicedesign as bench
from aiohttp import web


class MetricsTest(unittest.TestCase):
    def test_underrun_excludes_initial_wait_and_counts_playback_gaps(self):
        result = bench.stream_metrics(
            [
                {"at_s": 1.0, "bytes": 4800},
                {"at_s": 1.05, "bytes": 4800},
                {"at_s": 1.5, "bytes": 9600},
            ],
            1.6,
        )
        self.assertEqual(result["ttfb_s"], 1.0)
        self.assertAlmostEqual(result["audio_s"], 0.4)
        self.assertAlmostEqual(result["underrun_s"], 0.3)
        self.assertAlmostEqual(result["rtf"], 4.0)

    def test_failed_partial_audio_cannot_inflate_throughput(self):
        good = {
            "ok": True,
            "case_id": "short",
            **bench.stream_metrics([{"at_s": 0.1, "bytes": 48000}], 0.5),
        }
        failed = {
            "ok": False,
            "case_id": "long",
            **bench.stream_metrics([{"at_s": 0.2, "bytes": 480000}], 2.0),
        }
        result = bench.summarize([good, failed], 2.0)
        self.assertEqual((result["successful"], result["failed"]), (1, 1))
        self.assertEqual(result["successful_audio_s"], 1.0)
        self.assertEqual(result["audio_s_per_wall_s"], 0.5)
        self.assertEqual(result["audio_s"]["n"], 1)
        self.assertIsNone(result["duration_by_case"]["long"]["audio_s"])

    def test_empty_results_and_percentiles(self):
        self.assertIsNone(bench.summarize([], 0)["ttfb_s"])
        self.assertEqual(bench.distribution([1.0, 3.0])["p50"], 2.0)

    def test_exact_replay_workload_is_backend_independent(self):
        rows = bench.workload(bench.DEFAULT_WORKLOAD, 12, 42)
        self.assertEqual(rows, bench.workload(bench.DEFAULT_WORKLOAD, 12, 42))
        self.assertNotEqual(rows, bench.workload(bench.DEFAULT_WORKLOAD, 12, 43))
        for row in rows:
            pega = bench.request_body("pega-omni", "same-checkpoint", row)
            vllm = bench.request_body("vllm-omni", "same-checkpoint", row)
            for field in ("input", "instructions", "response_format", "stream_format"):
                self.assertEqual(pega[field], vllm[field])
            self.assertEqual(pega["extra"]["seed"], vllm["seed"])
            self.assertEqual(pega["extra"]["language"], vllm["language"].lower())
            self.assertNotIn("extra", vllm)
            self.assertNotIn("task_type", pega)
            self.assertNotIn("non_streaming_mode", pega)
            self.assertNotIn("max_new_tokens", pega)
            self.assertTrue(vllm["non_streaming_mode"])
            self.assertEqual(vllm["max_new_tokens"], row["expected_max_new_tokens"])
            self.assertEqual(vllm["task_type"], "VoiceDesign")
            self.assertEqual(
                bench.request_body("vllm-omni", "same-checkpoint", row, 1)[
                    "initial_codec_chunk_frames"
                ],
                1,
            )
            self.assertNotIn("initial_codec_chunk_frames", vllm)

    def test_generation_cap_counts_unicode_characters_and_checkpoint_limit(self):
        case = {
            "id": "unicode",
            "input": "你好，世界。😀",
            "instructions": "自然清晰的普通话男声。",
            "language": "chinese",
        }
        row = bench.workload([case], 1, 42)[0]
        self.assertEqual(row["expected_max_new_tokens"], 50 + 6 * 7)
        self.assertNotEqual(
            row["expected_max_new_tokens"], 50 + 6 * len(case["input"].encode())
        )
        capped = bench.workload([case], 1, 42, checkpoint_cap=64)[0]
        self.assertEqual(capped["expected_max_new_tokens"], 64)
        self.assertNotEqual(bench.digest([row]), bench.digest([capped]))
        self.assertEqual(
            bench.request_body("vllm-omni", "design", capped)["max_new_tokens"],
            64,
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            bench.workload([case], 1, 42, checkpoint_cap=0)

    def test_jsonl_requires_explicit_normalization_of_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.jsonl"
            path.write_text(
                '{"input":"hello","instructions":"calm","language":"English","task_type":"VoiceDesign"}\n'
            )
            rows = bench.read_jsonl(path)
            self.assertEqual(rows[0]["language"], "english")
            self.assertEqual(len(bench.workload(rows, 1, 0)), 1)
            path.write_text(
                '{"input":"hello","instructions":"calm","language":"English","max_new_tokens":10}\n'
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                bench.read_jsonl(path)


class StreamingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.active = 0
        self.peak = 0
        self.bodies = []

        async def handler(request):
            body = await request.json()
            self.bodies.append(body)
            if body["input"] == "error":
                return web.json_response({"error": "bad instructions"}, status=400)
            if body["input"] == "json":
                return web.json_response({"error": "not audio"})
            response = web.StreamResponse(
                headers={
                    "Content-Type": "audio/pcm",
                    "X-Audio-Sample-Rate": "24000",
                    "X-Audio-Channels": "1",
                }
            )
            await response.prepare(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            if body["input"] == "odd":
                await response.write(b"x")
            elif body["input"] == "wav":
                await response.write(b"RIFF\x00\x00\x00\x00WAVE")
            elif body["input"] != "empty":
                await response.write(b"\x00\x00" * 1200)
                await asyncio.sleep(0.01)
                await response.write(b"\x00\x00" * 1200)
            self.active -= 1
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}/v1/audio/speech"
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2))

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def test_real_streaming_client_retains_pcm_and_bounds_concurrency(self):
        args = SimpleNamespace(
            backend="pega-omni", model="design", concurrency=2, min_audio_seconds=0.0
        )
        rows = bench.workload([bench.DEFAULT_WORKLOAD[0]], 5, 42)
        records, wall = await bench.phase(self.session, self.url, args, rows, True)
        self.assertTrue(all(row["ok"] for row in records))
        self.assertEqual(self.peak, 2)
        self.assertGreater(wall, 0.02)
        for row in records:
            self.assertEqual(
                row["expected_max_new_tokens"], rows[0]["expected_max_new_tokens"]
            )
            self.assertEqual(row["audio_bytes"], 4800)
            self.assertEqual(len(row["_pcm"]), 4800)
            self.assertGreaterEqual(row["ttfb_s"], 0.009)
            self.assertGreater(row["e2e_s"], row["ttfb_s"])
            self.assertEqual(row["audio_s"], 0.1)

    async def test_http_non_audio_empty_unaligned_and_wav_are_failures(self):
        row = bench.workload([bench.DEFAULT_WORKLOAD[0]], 1, 42)[0]
        for text in ("error", "json", "empty", "odd", "wav"):
            result = await bench.fire(
                self.session,
                self.url,
                "vllm-omni",
                "design",
                {**row, "input": text},
                0.0,
                False,
                0.0,
            )
            self.assertFalse(result["ok"], text)
            self.assertIsNotNone(result["error"], text)
        self.assertEqual(self.bodies[0]["task_type"], "VoiceDesign")

    async def test_duration_floor_is_a_failure_not_a_fast_success(self):
        row = bench.workload([bench.DEFAULT_WORKLOAD[0]], 1, 42)[0]
        result = await bench.fire(
            self.session, self.url, "pega-omni", "design", row, 0.0, False, 0.5
        )
        self.assertFalse(result["ok"])
        self.assertIn("shorter", result["error"])
        self.assertEqual(result["audio_s"], 0.1)

    async def test_failed_warmup_aborts_measured_phase(self):
        row = bench.workload([bench.DEFAULT_WORKLOAD[0]], 1, 42)[0]
        report = {
            "environment": {},
            "warmup_workload": [{**row, "input": "error"}],
            "workload": [row],
        }
        args = SimpleNamespace(
            base_url=self.url.removesuffix("/audio/speech"),
            timeout=2,
            concurrency=1,
            backend="pega-omni",
            model="design",
            min_audio_seconds=0.0,
            audio_dir=None,
        )
        await bench.run(args, report)
        self.assertIn("aborted", report)
        self.assertNotIn("measurement", report)
        self.assertEqual(len(self.bodies), 1)


if __name__ == "__main__":
    unittest.main()
