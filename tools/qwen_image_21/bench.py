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
    args = ap.parse_args()

    # Warmup
    print("Warming up...")
    generate(args.url, args.model, args.prompt, args.size, 0)

    times = []
    for i in range(args.n):
        dt, nbytes = generate(args.url, args.model, args.prompt, args.size, i + 1)
        times.append(dt)
        print(f"  request {i+1}: {dt:.3f}s ({nbytes} bytes)")

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mean = sum(times) / len(times)
    print(f"\nResults ({args.n} requests, {args.size}):")
    print(f"  mean: {mean:.3f}s")
    print(f"  p50:  {p50:.3f}s")
    print(f"  p95:  {p95:.3f}s")
    print(f"  throughput: {1/mean:.3f} images/s")
    print(f"  DiT steps: 8 (Turbo)")

if __name__ == "__main__":
    main()