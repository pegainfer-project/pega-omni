#!/usr/bin/env python3
"""Benchmark pega-omni qwen-image-21 throughput on L20.

Usage: python bench.py [--url http://127.0.0.1:8000] [--n 5] [--size 1024x1024]
Reports per-request latency (p50/p95) and images/second.
"""
import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def generate(url, model, prompt, size, seed):
    body = json.dumps({"model": model, "prompt": prompt, "size": size, "extra": {"seed": seed}}).encode()
    req = urllib.request.Request(f"{url}/v1/images/generations", data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    img_bytes = base64.b64decode(data["data"][0]["b64_json"])
    return elapsed, len(img_bytes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="model-local")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--prompt", default="A lighthouse on a cliff at dawn.")
    ap.add_argument("--concurrency", type=int, default=1, help="in-flight requests (1 = sequential)")
    args = ap.parse_args()

    # Warmup
    print("Warming up...")
    generate(args.url, args.model, args.prompt, args.size, 0)

    def one(i):
        dt, nbytes = generate(args.url, args.model, args.prompt, args.size, i + 1)
        print(f"  request {i+1}: {dt:.3f}s ({nbytes} bytes)")
        return dt

    t0 = time.perf_counter()
    if args.concurrency <= 1:
        times = [one(i) for i in range(args.n)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            times = list(ex.map(one, range(args.n)))
    wall = time.perf_counter() - t0

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mean = sum(times) / len(times)
    print(f"\nResults ({args.n} requests, {args.size}, c={args.concurrency}):")
    print(f"  mean: {mean:.3f}s")
    print(f"  p50:  {p50:.3f}s")
    print(f"  p95:  {p95:.3f}s")
    print(f"  throughput: {args.n / wall:.3f} images/s")
    print(f"  DiT steps: 8 (Turbo)")

if __name__ == "__main__":
    main()