# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets>=13"]
# ///
"""Checks a server's GPT-Live WebSocket against the protocol, as an outside client.

Runs one session against `/v1/live/sessions` and asserts what a GPT-Live
client relies on:

- a strict `session.start` refuses an unknown field with an `error` naming it,
  echoing `client_event_id`, and leaves the socket usable;
- `session.start` gets `session.started` with an id, the model and the voice;
- appended audio (paced at real time) is answered by
  `session.output_audio.delta` events whose `start_ms` never goes back,
  whose `end_ms - start_ms` matches the decoded PCM, and which keep pace with
  the wall clock;
- `session.input_audio.mute` / `unmute` are acknowledged;
- `session.close` ends with `session.closed` (`close_requested`, usage) and
  the server closing the socket;
- every server event carries `type` and a unique `event_id`.

    uv run tools/live_check.py --url ws://127.0.0.1:8000/v1/live/sessions
"""

import argparse
import asyncio
import base64
import json
import math
import sys
import time

import websockets

RATE = 24_000


class Session:
    def __init__(self, ws):
        self.ws = ws
        self.ids = set()

    async def send(self, event):
        await self.ws.send(json.dumps(event))

    async def recv(self, timeout=10.0):
        """The next event other than a usage update, or None once the socket closed."""
        while True:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout)
            except websockets.ConnectionClosed:
                return None
            event = json.loads(raw)
            assert isinstance(event.get("type"), str), event
            assert event.get("event_id") not in self.ids, f"repeated event_id: {event}"
            self.ids.add(event["event_id"])
            if event["type"] != "session.usage.updated":
                return event


def tone(seconds):
    n = int(seconds * RATE)
    samples = (int(6000 * math.sin(2 * math.pi * 220 * i / RATE) * (0.5 + 0.5 * math.sin(i / 3000))) for i in range(n))
    return b"".join(s.to_bytes(2, "little", signed=True) for s in samples)


async def check(url, seconds):
    async with websockets.connect(url, max_size=None) as ws:
        s = Session(ws)

        await s.send({"type": "session.start", "client_event_id": "bad", "session": {"temprature": 0.7}})
        e = await s.recv()
        assert e["type"] == "error", e
        assert e["error"]["param"] == "session.temprature" and e["error"]["client_event_id"] == "bad", e

        await s.send({"type": "session.start", "session": {}})
        started = await s.recv(timeout=60)
        assert started["type"] == "session.started", started
        sess = started["session"]
        assert sess["id"] and sess["model"] and sess["audio"]["output"]["voice"], started
        print(f"started {sess['id']} model={sess['model']} voice={sess['audio']['output']['voice']}")

        pcm = tone(seconds)
        chunk = RATE // 50 * 2
        t0 = time.monotonic()
        audio_events, last_start, first = 0, -1, None
        lateness = []

        async def feed():
            for k, at in enumerate(range(0, len(pcm), chunk)):
                await asyncio.sleep(max(0.0, t0 + k * 0.02 - time.monotonic()))
                await s.send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm[at : at + chunk]).decode()})
            await s.send({"type": "session.input_audio.mute", "client_event_id": "m"})
            await s.send({"type": "session.input_audio.unmute", "client_event_id": "u"})

        feeder = asyncio.create_task(feed())
        acks = []
        while not feeder.done() or len(acks) < 2:
            e = await s.recv()
            assert e is not None, "socket closed mid-session"
            now = time.monotonic()
            if e["type"] == "session.output_audio.delta":
                n = len(base64.b64decode(e["delta"])) // 2
                assert e["start_ms"] >= last_start + 1 or last_start < 0, f"start_ms went back: {e['start_ms']} after {last_start}"
                assert e["end_ms"] - e["start_ms"] == n * 1000 // RATE, e | {"delta": f"<{n} samples>"}
                last_start = e["start_ms"]
                first = first or (now, e["start_ms"])
                lateness.append((now - first[0]) * 1000 - (e["start_ms"] - first[1]))
                audio_events += 1
            elif e["type"] in ("session.input_audio.muted", "session.input_audio.unmuted"):
                acks.append((e["type"], e.get("client_event_id")))
            elif e["type"] == "error":
                raise AssertionError(f"unexpected error: {e}")
        await feeder
        assert acks == [("session.input_audio.muted", "m"), ("session.input_audio.unmuted", "u")], acks
        expected = seconds * 12.5
        assert audio_events >= 0.7 * expected, f"{audio_events} audio deltas for {seconds} s of audio"
        worst = max(lateness)
        assert worst < 1000, f"output fell {worst:.0f} ms behind its timeline"
        print(f"{audio_events} audio deltas, timeline lateness max {worst:.1f} ms")

        await s.send({"type": "session.close"})
        while True:
            e = await s.recv()
            assert e is not None, "socket closed before session.closed"
            if e["type"] == "session.closed":
                break
        assert e["reason"] == "close_requested" and e["usage"]["seconds"] > 0, e
        assert await s.recv() is None, "socket still open after session.closed"
        print(f"closed: {e['reason']}, {e['usage']['seconds']} s billed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/live/sessions")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    asyncio.run(check(args.url, args.seconds))
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
