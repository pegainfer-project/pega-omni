# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "httpx"]
# ///
"""Checks concurrent speech and client cancellation against a running server.

Use an otherwise idle speech server with metrics enabled. Models and voices
are discovered through the API; model-specific options go through `--extra`.
The existing `openai_sdk_check.py` covers audio formats and protocol errors.

    uv run tools/speech_e2e_check.py --base-url http://127.0.0.1:8000/v1

For a CPU run, start `pega-omni sim --max-batch 2 --step-base-us 1000`.
The cancellation request must still be streaming after its first PCM chunk:
a zero-cost simulator or a short `--cancel-input` can finish before disconnect.

After closing that stream, poll until `omni_requests_total{outcome="cancelled"}`
increases and both `omni_engine_waiting` and `omni_engine_running` are zero.
These are the frontend's observed cancellation and the engine's occupancy;
the check then requires a fresh request to finish and occupancy to return to
zero again. No latency threshold decides success.
"""

import argparse
import asyncio
import json
from urllib.parse import urlsplit, urlunsplit

import httpx
import openai


async def metrics(http, url):
    response = await http.get(url)
    response.raise_for_status()
    names = {
        "omni_engine_waiting": "waiting",
        "omni_engine_running": "running",
        'omni_requests_total{outcome="cancelled"}': "cancelled",
    }
    state = {"cancelled": 0.0}
    for line in response.text.splitlines():
        sample, _, value = line.partition(" ")
        if sample in names:
            state[names[sample]] = float(value.split()[0])
    assert "waiting" in state and "running" in state, f"missing engine occupancy metrics at {url}"
    return state


async def idle(http, url, timeout, cancelled=None):
    state = None
    try:
        async with asyncio.timeout(timeout):
            while True:
                state = await metrics(http, url)
                if state["waiting"] == state["running"] == 0 and (cancelled is None or state["cancelled"] > cancelled):
                    return state
                await asyncio.sleep(0.05)
    except TimeoutError as error:
        raise AssertionError(
            f"engine did not become idle with the expected cancellation within {timeout}s: {state}"
        ) from error


async def complete(client, request, text):
    async with client.audio.speech.with_streaming_response.create(input=text, **request) as response:
        size = 0
        async for chunk in response.iter_bytes():
            size += len(chunk)
    assert size > 0 and size % 2 == 0, f"expected nonempty mono s16le PCM, got {size} bytes"
    return size


async def cancel(client, request, text):
    async with client.audio.speech.with_streaming_response.create(input=text, **request) as response:
        first = await anext(response.iter_bytes(), b"")
        assert first, "cancellation request ended without a PCM chunk"
    return len(first)


async def check(args):
    base = urlsplit(args.base_url)
    metrics_url = args.metrics_url or urlunsplit((base.scheme, base.netloc, "/metrics", "", ""))
    async with (
        openai.AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout, max_retries=0) as client,
        httpx.AsyncClient(timeout=5.0) as http,
    ):
        models = [model.id async for model in client.models.list()]
        assert models, "server advertises no models"
        model = args.model or models[0]
        voices = (await client.get("/audio/voices", cast_to=object))["voices"]
        assert voices, "server advertises no voices"
        request = {"model": model, "voice": voices[0], "response_format": "pcm", "extra_body": {"extra": args.extra}}
        await idle(http, metrics_url, args.release_timeout)

        sizes = await asyncio.wait_for(
            asyncio.gather(complete(client, request, args.input), complete(client, request, args.input)), args.timeout
        )
        before = await idle(http, metrics_url, args.release_timeout)
        print(f"model={model} voice={voices[0]}: two concurrent PCM requests completed ({sizes} bytes)")

        first = await asyncio.wait_for(cancel(client, request, args.cancel_input), args.timeout)
        after = await idle(http, metrics_url, args.release_timeout, cancelled=before["cancelled"])
        size = await asyncio.wait_for(complete(client, request, args.input), args.timeout)
        await idle(http, metrics_url, args.release_timeout)
        print(
            f"closed after {first} PCM bytes; cancelled={after['cancelled']:g}, occupancy=0; next request completed ({size} bytes)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--metrics-url", help="defaults to /metrics on the same server")
    ap.add_argument("--model", help="defaults to the first advertised model")
    ap.add_argument("--extra", type=json.loads, default={}, help="the request's `extra` object, as JSON")
    ap.add_argument("--input", default="A short sentence for a streaming speech service check.")
    ap.add_argument(
        "--cancel-input",
        default=(
            "This longer recording checks that closing a streaming response releases the request, so another listener can use the speech service. "
            * 4
        ).strip(),
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="deadline per speech check, in seconds")
    ap.add_argument(
        "--release-timeout", type=float, default=30.0, help="deadline for cancellation and idle metrics, in seconds"
    )
    args = ap.parse_args()
    if args.timeout <= 0 or args.release_timeout <= 0:
        ap.error("timeouts must be positive")
    if not isinstance(args.extra, dict):
        ap.error("--extra must be a JSON object")
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
