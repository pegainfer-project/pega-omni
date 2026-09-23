"""Drives a running pega-omni server with the official `openai` Python SDK.

The SDK is the executable form of OpenAI's API reference: if its stock calls
work unchanged against the server, a client written for OpenAI works too.

    uv run --with openai tools/openai_sdk_check.py --base-url http://127.0.0.1:8000/v1 --model pega-omni-sim
"""

import argparse
import base64
import json
import struct

import openai


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pega-omni-sim")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    frames = {"extra": {"frames": 10}}
    frame_bytes = 1920 * 2

    wav = client.audio.speech.create(model=args.model, voice="alloy", input="Hello from the SDK.", extra_body=frames)
    data = wav.read()
    riff, _, wave = struct.unpack("<4sI4s", data[:12])
    assert (riff, wave, len(data)) == (b"RIFF", b"WAVE", 44 + 10 * frame_bytes), (riff, wave, len(data))

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice="nova", input="Streaming.", response_format="pcm", extra_body=frames
    ) as resp:
        chunks = [c for c in resp.iter_bytes() if c]
    assert sum(map(len, chunks)) == 10 * frame_bytes, sum(map(len, chunks))

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice="coral", input="Events.", response_format="pcm", stream_format="sse", extra_body=frames
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    audio = sum(len(base64.b64decode(e["audio"])) for e in events if e["type"] == "speech.audio.delta")
    assert (audio, events[-1]["type"], events[-1]["usage"]["output_tokens"]) == (10 * frame_bytes, "speech.audio.done", 10)

    try:
        client.audio.speech.create(model=args.model, voice="nobody", input="x")
        raise AssertionError("unknown voice accepted")
    except openai.BadRequestError as e:
        assert e.body["param"] == "voice", e.body

    try:
        client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="x")
        raise AssertionError("unknown model accepted")
    except openai.NotFoundError as e:
        assert e.body["code"] == "model_not_found", e.body

    assert [m.id for m in client.models.list()] == [args.model]
    print(f"openai {openai.__version__}: wav, pcm stream, sse stream, errors and models all conform")


if __name__ == "__main__":
    main()
