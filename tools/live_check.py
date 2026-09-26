# /// script
# requires-python = ">=3.10"
# dependencies = ["openai[realtime]>=3.19.2"]
# ///
"""Checks a server's GPT-Live WebSocket with the official openai SDK's live client.

Connects with `client.live.connect()` and sends through the SDK's own command
helpers, so the URL, the handshake and the client events are the SDK's. Every
server event is validated, strictly, against the SDK's model for its type
(`openai.types.live.ServerEvent`), and a field the model does not declare is
an error too, so a schema drift fails the check. Per wire rate (PCM16 at
24 kHz and 16 kHz by default) it asserts what a GPT-Live client relies on:

- a `session.start` with an unknown field is refused by an `error` naming it
  and echoing its `event_id` as `error.client_event_id`, and the socket stays
  usable;
- `session.start` gets `session.started` echoing its `event_id`, with the
  model, the chosen `audio.format` and a voice;
- appended audio (paced at real time) is answered by
  `session.output_audio.delta` events whose `start_ms` never goes back,
  whose `end_ms - start_ms` matches the decoded audio at the session's rate,
  and which keep pace with the wall clock;
- `session.input_audio.mute` / `unmute` are acknowledged, echoing their ids;
- `session.close` ends with `session.closed` (`close_requested`, the same
  session snapshot, usage) and the server closing the socket;
- every server event has a unique `event_id`.

    uv run tools/live_check.py --base-url http://127.0.0.1:8000/v1
"""

import argparse
import asyncio
import base64
import json
import math
import sys
import time
import typing

import openai
import websockets
from openai.types.live import ServerEvent
from pydantic import BaseModel

MODELS = {
    typing.get_args(cls.model_fields["type"].annotation)[0]: cls
    for cls in typing.get_args(typing.get_args(ServerEvent)[0])
}
# The SDK's primary-WebSocket audio delta declares no `event_id`; this server
# stamps one on every event, which the SDK tolerates.
UNDECLARED = {("session.output_audio.delta", "event_id")}


def undeclared(model, at=""):
    """Paths of fields `model` carries that its class does not declare."""
    found = [f"{at}{k}" for k in (model.model_extra or {})]
    for name in type(model).model_fields:
        value = getattr(model, name)
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, BaseModel):
                found += undeclared(v, f"{at}{name}.")
    return found


def parse(raw):
    kind = json.loads(raw).get("type")
    assert kind in MODELS, f"server event type {kind!r} is not in the SDK: {raw[:200]!r}"
    event = MODELS[kind].model_validate_json(raw, strict=True)
    extra = [p for p in undeclared(event) if (kind, p) not in UNDECLARED]
    assert not extra, f"{kind} carries fields the SDK does not declare: {extra}"
    return event


class Session:
    def __init__(self, conn):
        self.conn = conn
        self.ids = set()

    async def recv(self, timeout=10.0):
        """The next event other than a usage update, or None once the socket closed."""
        while True:
            try:
                raw = await asyncio.wait_for(self.conn.recv_bytes(), timeout)
            except websockets.ConnectionClosed:
                return None
            event = parse(raw)
            assert event.event_id is not None and event.event_id not in self.ids, f"missing or repeated event_id: {event}"
            self.ids.add(event.event_id)
            if event.type != "session.usage.updated":
                return event


def tone(seconds, rate):
    n = int(seconds * rate)
    samples = (int(6000 * math.sin(2 * math.pi * 220 * i / rate) * (0.5 + 0.5 * math.sin(i / (rate / 8)))) for i in range(n))
    return b"".join(s.to_bytes(2, "little", signed=True) for s in samples)


async def check(client, model, rate, seconds):
    async with client.live.connect() as conn:
        s = Session(conn)

        await conn.send_raw(json.dumps({"type": "session.start", "event_id": "bad", "session": {"model": model, "temprature": 0.7}}))
        e = await s.recv()
        assert e.type == "error", e
        assert e.error.param == "session.temprature" and e.error.client_event_id == "bad", e

        await conn.session.start(session={"model": model, "audio": {"format": {"type": "audio/pcm", "rate": rate}}}, event_id="start")
        started = await s.recv(timeout=60)
        assert started.type == "session.started" and started.client_event_id == "start", started
        sess = started.session
        assert sess.model == model and sess.audio.format.rate == rate and sess.audio.output.voice, started
        print(f"started {sess.id} model={sess.model} voice={sess.audio.output.voice} format={sess.audio.format.type}@{rate}")

        pcm = tone(seconds, rate)
        chunk = rate // 50 * 2
        t0 = time.monotonic()
        audio_events, last_start, first = 0, -1, None
        lateness = []

        async def feed():
            for k, at in enumerate(range(0, len(pcm), chunk)):
                await asyncio.sleep(max(0.0, t0 + k * 0.02 - time.monotonic()))
                await conn.session.input_audio.append(audio=base64.b64encode(pcm[at : at + chunk]).decode())
            await conn.session.input_audio.mute(event_id="m")
            await conn.session.input_audio.unmute(event_id="u")

        feeder = asyncio.create_task(feed())
        acks = []
        while not feeder.done() or len(acks) < 2:
            e = await s.recv()
            assert e is not None, "socket closed mid-session"
            now = time.monotonic()
            if e.type == "session.output_audio.delta":
                n = len(base64.b64decode(e.delta)) // 2
                assert e.start_ms is not None and e.end_ms is not None, "audio deltas carry their timeline"
                assert e.start_ms > last_start, f"start_ms went back: {e.start_ms} after {last_start}"
                assert e.end_ms - e.start_ms == n * 1000 // rate, f"{e.start_ms}..{e.end_ms} ms for {n} samples"
                last_start = e.start_ms
                first = first or (now, e.start_ms)
                lateness.append((now - first[0]) * 1000 - (e.start_ms - first[1]))
                audio_events += 1
            elif e.type in ("session.input_audio.muted", "session.input_audio.unmuted"):
                acks.append((e.type, e.client_event_id))
            elif e.type == "error":
                raise AssertionError(f"unexpected error: {e}")
        await feeder
        assert acks == [("session.input_audio.muted", "m"), ("session.input_audio.unmuted", "u")], acks
        expected = seconds * 12.5
        assert audio_events >= 0.7 * expected, f"{audio_events} audio deltas for {seconds} s of audio"
        worst = max(lateness)
        assert worst < 1000, f"output fell {worst:.0f} ms behind its timeline"
        print(f"{audio_events} audio deltas, timeline lateness max {worst:.1f} ms")

        await conn.session.close(event_id="bye")
        while True:
            e = await s.recv()
            assert e is not None, "socket closed before session.closed"
            if e.type == "session.closed":
                break
        assert e.reason == "close_requested" and e.client_event_id == "bye" and e.usage.seconds > 0, e
        assert e.session == sess, f"session.closed reports {e.session}, session.started {sess}"
        assert await s.recv() is None, "socket still open after session.closed"
        print(f"closed: {e.reason}, {e.usage.seconds} s billed")


async def main_async(args):
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="unused")
    model = args.model or (await client.models.list()).data[0].id
    for rate in args.rate or [24_000, 16_000]:
        await check(client, model, rate, args.seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", help="session.model; defaults to the first model /v1/models lists")
    ap.add_argument("--rate", type=int, action="append", choices=[16_000, 24_000], help="wire PCM rate; repeatable (default: both)")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
