"""Drives a running pega-omni server with the official `openai` Python SDK.

The SDK is the executable form of OpenAI's API reference: if its stock calls
work unchanged against the server, a client written for OpenAI works too.

    uv run --with openai tools/openai_sdk_check.py --base-url http://127.0.0.1:8000/v1 --model pega-omni-sim

Any engine: voices come from `GET /v1/audio/voices`, `--extra` is passed through.
Pass `--reference-audio` to check a server started with `sim --reference-audio`.
"""

import argparse
import base64
import io
import json
import math
import struct
import wave

import openai


def reference_wav_uri() -> str:
    sample_rate = 16_000
    samples = [round(8_000 * math.sin(math.tau * 440 * i / sample_rate)) for i in range(sample_rate)]
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return "data:audio/wav;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def assert_bad_request(client: openai.OpenAI, model: str, voice: str, param: str, extra_body: dict | None = None) -> None:
    try:
        client.audio.speech.create(model=model, voice=voice, input="x", extra_body=extra_body)
        raise AssertionError(f"invalid {param} accepted")
    except openai.BadRequestError as e:
        assert e.body["param"] == param, e.body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pega-omni-sim")
    ap.add_argument("--extra", type=json.loads, default={}, help="the request's `extra` object, as JSON")
    ap.add_argument("--reference-audio", action="store_true", help="exercise reference conditioning on sim")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    voices = client.get("/audio/voices", cast_to=object)["voices"]
    extra = {"extra": args.extra}
    if args.reference_audio:
        assert voices == [], voices
        voice = "alloy"
        reference = {**extra, "ref_audio": reference_wav_uri()}
        audio_only = {**reference, "x_vector_only_mode": True}
        icl = {**reference, "ref_text": "A reference transcript."}
    else:
        voice = voices[0]
        audio_only = icl = extra

    wav = client.audio.speech.create(model=args.model, voice=voice, input="Hello from the SDK.", extra_body=audio_only)
    data = wav.read()
    riff, _, wave_tag = struct.unpack("<4sI4s", data[:12])
    assert (riff, wave_tag) == (b"RIFF", b"WAVE") and len(data) > 44, (riff, wave_tag, len(data))

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Streaming.", response_format="pcm", extra_body=icl
    ) as resp:
        pcm = sum(len(c) for c in resp.iter_bytes())
    assert pcm > 0 and pcm % 2 == 0, pcm

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Events.", response_format="pcm", stream_format="sse", extra_body=icl
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    audio = sum(len(base64.b64decode(e["audio"])) for e in events if e["type"] == "speech.audio.delta")
    done = events[-1]
    assert audio > 0 and done["type"] == "speech.audio.done" and done["usage"]["output_tokens"] > 0, (audio, done)

    if args.reference_audio:
        assert_bad_request(client, args.model, voice, "ref_text", reference)
        malformed = {**audio_only, "ref_audio": "data:audio/wav;base64"}
        assert_bad_request(client, args.model, voice, "ref_audio", malformed)
    else:
        assert_bad_request(client, args.model, "nobody", "voice")

    try:
        client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="x")
        raise AssertionError("unknown model accepted")
    except openai.NotFoundError as e:
        assert e.body["code"] == "model_not_found", e.body

    assert [m.id for m in client.models.list()] == [args.model]
    reference_result = ", reference audio" if args.reference_audio else ""
    print(f"openai {openai.__version__}: wav, pcm stream, sse stream, errors and models{reference_result} all conform")


if __name__ == "__main__":
    main()
