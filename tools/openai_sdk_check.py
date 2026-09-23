"""Drives a running pega-omni server with the official `openai` Python SDK.

The SDK is the executable form of OpenAI's API reference: if its stock calls
work unchanged against the server, a client written for OpenAI works too.

    uv run --with openai tools/openai_sdk_check.py --base-url http://127.0.0.1:8000/v1 --model pega-omni-sim

Any engine: voices come from `GET /v1/audio/voices`, `--extra` is passed through.
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
    ap.add_argument("--extra", type=json.loads, default={}, help="the request's `extra` object, as JSON")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    voice = client.get("/audio/voices", cast_to=object)["voices"][0]
    extra = {"extra": args.extra}

    wav = client.audio.speech.create(model=args.model, voice=voice, input="Hello from the SDK.", extra_body=extra)
    data = wav.read()
    riff, _, wave = struct.unpack("<4sI4s", data[:12])
    assert (riff, wave) == (b"RIFF", b"WAVE") and len(data) > 44, (riff, wave, len(data))

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Streaming.", response_format="pcm", extra_body=extra
    ) as resp:
        pcm = sum(len(c) for c in resp.iter_bytes())
    assert pcm > 0 and pcm % 2 == 0, pcm

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Events.", response_format="pcm", stream_format="sse", extra_body=extra
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    audio = sum(len(base64.b64decode(e["audio"])) for e in events if e["type"] == "speech.audio.delta")
    done = events[-1]
    assert audio > 0 and done["type"] == "speech.audio.done" and done["usage"]["output_tokens"] > 0, (audio, done)

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
