# Full duplex

A full-duplex model listens and speaks at the same time. Every 80 ms it takes
a frame of the caller's audio and emits a frame of its own, whether anyone is
talking or not; interrupting, back-channelling ("mm-hm") and turn taking are
things the model does, not things the server detects. Serving it is a
different problem from request-shaped TTS:

| | Speech (`/v1/audio/speech`) | Live (`/v1/live/sessions`) |
|---|---|---|
| unit of work | a request, done when the audio is | a session, minutes long |
| transport | HTTP request, streamed response | one WebSocket, both directions |
| scheduling | throughput first, batch what is waiting | a clock: every session advances one frame per tick |
| idle cost | none | every open session costs a frame per tick |
| capacity metric | requests/s, time to first packet | sessions that stay on time |

This document covers how pega-omni serves live sessions: the contract, the
protocol, the clock and the jitter buffer, the engine, how to try it and how
to measure it.

## The contract

`omni_engine::live` is the whole interface between the front end and a live
engine, the same way `omni_engine` is for speech: a channel. On the engine's
side it also holds what every live engine shares (`drive`, below).

- `LiveInfo` is what the engine serves: model, sample rate, samples per
  frame, voices and their defaults, `max_sessions`, `max_frames` (session
  length). `LiveInfo::check` turns the client's `SessionDraft` into a
  `Session`, so the engine never re-validates a voice or the instructions.
- `LiveHandle::start` hands the engine a `LiveSubmission`: the session's id,
  the session, a receiver of the caller's audio (s16le mono, chunks of any
  size) and a sink for `Output`.
- The engine answers `Output::Started` once the prompt is in, then one
  `Output::Audio { frame, pcm }` per tick and `Output::Text { frame, delta }`
  whenever the model's text stream says something, and at most one
  `Output::Closed { reason, frames, underruns, dropped }`.
- Ending is ownership. The front end drops the audio sender when the caller
  leaves or asks to close; the engine retires the session and, if the sink
  is still read, acknowledges with `Closed { reason: Hangup }` carrying its
  final counts. The engine ends a session itself with `Expired` (it reached
  `max_frames`), `Busy` (it was full) or `Aborted`.

## Protocol

The wire protocol is the primary-WebSocket form of OpenAI's GPT-Live API
(`gpt-live-1`, `/v1/live/sessions`), the one OpenAI API designed for
continuous, turnless audio. The Realtime API (`/v1/realtime`) is shaped around
responses (`response.create`, `commit`, VAD, `truncate`) that a native
full-duplex model does not have, and fitting one into it means inventing
response boundaries out of silence. GPT-Live has none of that: audio streams
both ways, and output is placed on a session timeline.

The schema is the official `openai` Python SDK's (`openai.types.live`), not
our reading of the docs: `tools/live_check.py` drives the server with the
SDK's own live client (`client.live.connect()`) and validates every server
event strictly against the SDK's model for it.

A socket carries one session.

Client events, each with an optional `event_id`:

| event | fields | notes |
|---|---|---|
| `session.start` | `session: {model, instructions?, audio?: {format?, output?: {voice?}}, delegation?, input?, store?}` | `model` is required. Strict: an unknown field is an `error` naming it (`param: "session.temprature"`). Blank `instructions` mean the default. `voice` is a name or `{"id": name}`. `delegation` may be `null` or `{"type": "client"}`; `input` may be absent or `[]`; `store` may be `false` |
| `session.input_audio.append` | `audio`: base64, in the session's format | never empty, whole samples, any chunk size; not acknowledged |
| `session.input_audio.mute` / `unmute` | | acknowledged with `session.input_audio.muted` / `unmuted`; the front end replaces muted audio with silence of the same length, so the engine hears silence and its buffer does not run dry |
| `session.close` | | answered by `session.closed` |

`audio.format` is one of:

| format | |
|---|---|
| `{"type": "audio/pcm", "rate": 24000}` | PCM16 little-endian mono; the default |
| `{"type": "audio/pcm", "rate": 16000}` | PCM16 little-endian mono |
| `{"type": "audio/pcmu", "rate": 8000}` | G.711 μ-law |
| `{"type": "audio/pcma", "rate": 8000}` | G.711 A-law |

It applies both ways. The front end converts between it and the engine's own
rate (`omni_frontend::live_audio`): G.711 through the ITU tables, rates
through a band-limited FFT resampler (rubato) with fixed 10 ms chunks, so the
engine hears the same samples however the network split them, and each
direction adds a constant 5 ms. A format at the engine's rate passes through
untouched. The engine never sees anything but its own rate.

Server events, each with `type` and a unique `event_id`. A server event that
answers a command echoes the command's `event_id` as `client_event_id`:

| event | fields |
|---|---|
| `session.started` | `session` (below), `client_event_id` of the `session.start` |
| `session.output_audio.delta` | `delta` (base64, in the session's format), `start_ms`, `end_ms` |
| `session.output_transcript.delta` | `delta`, `start_ms`, `end_ms` |
| `session.input_audio.muted` / `unmuted` | `client_event_id` |
| `session.usage.updated` | `usage: {seconds}`, every 5 s |
| `session.closed` | `session` (the same snapshot), `reason` (`close_requested`, `expired`, `connection_lost`), `usage: {seconds}`, `client_event_id` of the `session.close` |
| `error` | `error: {type, code, message, param?, client_event_id?}` |

The session (the SDK's `SessionResource`) is `{id, status: "active",
expires_at, model, instructions, audio: {format, output: {voice}},
delegation: {"type": "client"}}`, resolved: the defaults the engine filled in
are spelled out. `expires_at` (Unix seconds) is when a session that started
at `session.start` reaches `max_frames`. `session.usage.updated` carries no
`context_window`: the engines' context is a ring that never fills, so no
usage ratio would mean what the field promises.

**Timeline.** The agent's audio is one contiguous stream from
`session.started`: frame `n` is the `n`th frame the engine ran for the
session, covering `[80 n, 80 (n + 1))` ms of it. The server never sends a
done event: audio simply continues. The transcript is the model's own text
stream, placed on the same timeline as the audio it was spoken with. The SDK
makes `start_ms` / `end_ms` optional on primary-WebSocket audio deltas; this
server always sends them, since the timeline is what a client measures
lateness against.

**Refused, by name.** What the engine cannot honour is an `error`
(`code: "unsupported"`, `param` naming the event type or field), never
silently ignored:

- `session.update`: a full-duplex model's prompt (voice, instructions) is its
  prefix, prefilled before the first frame; changing it means a new session.
- `session.instructions.append`, `session.thinking.append`,
  `session.commentary.append`, `response.item.create`, `response.create`:
  context injection and responses need a model that takes text mid-session.
- `session.delegation` of `type: "responses"` (tools, a reasoning backend):
  same. Client delegation is accepted: the model never delegates, so the
  client never has anything to handle.
- A non-empty `session.input`: the model takes no text history.
- `session.store: true`: nothing is kept after a session closes.
- `session.client`: it configures a WebRTC data channel.
- Input transcription (`session.input_transcript.delta`) is not produced; the
  model does not transcribe the caller.

Other errors: `session_not_started` (audio before `session.start`),
`session_already_started`, `missing_required_parameter` (e.g.
`param: "session.model"`), `model_not_found` (`param: "session.model"`),
`invalid_value` (a voice, format or rate the server does not serve),
`invalid_audio` (`param: "audio"`: empty, not base64, or not whole samples),
`server_busy` (`type: server_error`; the engine is at `max_sessions` or its
queue is full; the socket stays open for another `session.start`).

## Jitter and the clock

The network delivers the caller's audio in bursts; the engine consumes exactly
one frame per tick. Between them sits `omni_engine::live::Jitter`, per
session:

- playout starts once `prebuffer_frames` frames are buffered (default 2,
  160 ms), so arrival jitter up to that much is absorbed instead of cutting
  silence into the caller's speech;
- a tick takes a frame if one is buffered, else it gets silence, the partial
  frame stays, and playout waits to refill the prebuffer (an **underrun**,
  counted and reported in `Closed`);
- audio beyond `jitter_frames` frames (default 4, 320 ms) is late by more
  than the buffer is meant to absorb, and the oldest is **dropped**, so a
  burst after a network stall cannot add permanent latency.

The clock is `omni_engine::live::Clock`: tick `k` is due at `k · 80 ms` after
the engine's epoch. It never drifts (ticks are not scheduled relative to the
previous one) and never bursts: a tick that overran its slot runs the latest
due tick and skips the ones it missed, counting them in
`omni_engine_late_ticks_total`. A skipped tick is not run for anyone: each
session's jitter buffer discards the caller's frames of it (not counted as
underruns), so the caller's latency stays bounded, and the agent's stream,
which only counts frames that ran, arrives that much later than the wall
clock; the client's playout buffer absorbs it or stalls.

Both are pure and property-tested (`crates/omni-engine/tests/live.rs`):
samples are conserved (pushed = played + skipped + dropped + buffered), the
buffer is bounded, each dry spell is one underrun, byte splits do not matter,
and the clock's decision always lands in the right slot.

`omni_engine::live::drive` is the loop every live engine runs: the clock,
admission (a session past `max_sessions` is answered `Busy` before the engine
sees it), and the `Pulse` counters behind `/live/stats` and `/metrics`. An
engine implements `Ticker` (`take` a session, run a `tick`) and keeps each
session as a `Line`, which owns its jitter buffer, its frame count, expiry at
`max_frames` and its `Output`s.

## The PersonaPlex engine

`pega-omni personaplex` serves NVIDIA's PersonaPlex-7B
([personaplex.md](personaplex.md) has the model side). One thread owns the
GPU and runs the clock:

- **Admission.** A `session.start` becomes a session if fewer than
  `--max-sessions` are open; otherwise the client gets `server_busy`.
  The session's prompt (voice + role prompt, ~100 rows) is built and its KV
  ring leased on arrival; the next tick prefills it, batched with any other
  new sessions, sends `session.started` and runs its frame 0.
  Connect-to-`session.started` measures ~60 ms p50 and ~110 ms p99 from 8 to
  128 sessions, and joining sessions do not make running ones late.
- **The tick.** Every 80 ms: drain each session's audio into its jitter
  buffer, pop one frame each (silence on an underrun), run one CUDA graph over
  all sessions, send each its agent frame (s16le) and whatever text its text
  token completed.
- **Lateness.** A tick that overran the clock drops the caller frames of the
  ticks it skipped, so the caller's latency stays bounded; the agent's
  stream counts only frames it ran.
- **Ending.** Hang-up (socket closed, `session.close`), `--max-seconds`
  (`expired`), a sink nobody reads, or a model fault, which ends every
  session and stops the engine.
- **Every session is independent.** Each has its own KV ring, codec state and
  sampling seed. Sessions join and leave between any two ticks, and a
  session's audio does not depend on who else is connected.

The transcript is the model's own text stream (its "inner monologue"), one
token per frame, so it is timestamped to the frame and costs nothing extra.
It is what the model meant to say, not a transcription of its audio, and it
occasionally skips a piece the audio has (for example the `t` of `don't`).

### Options

Server (`pega-omni personaplex --help`):

| flag | default | |
|---|---|---|
| `--model-path` | | the checkpoint directory (`model.safetensors`, the Mimi and tokenizer files, `voices/`) |
| `--max-sessions` | 16 | session slots; each holds 1.5 GiB of KV, allocated at start |
| `--max-seconds` | 240 | session length before `expired`; the context is a ring, so this is a policy, not a model limit |
| `--jitter-frames` | 4 | caller audio buffered (80 ms frames) before the oldest is dropped |
| `--prebuffer-frames` | 2 | caller audio buffered before the engine consumes it, and again after it ran dry; each frame adds 80 ms of latency and absorbs 80 ms of network jitter |
| `--device` | 0 | GPU |
| `--queue` | 65536 | sessions waiting for the engine thread to take them |

Per session (`session.start`):

| field | |
|---|---|
| `audio.output.voice` | one of the 18 voice prompts (`NATF0-3`, `NATM0-3`, `VARF0-4`, `VARM0-4`); default `NATF2` |
| `instructions` | the role prompt, e.g. `You work for a bakery called Sunrise; take the caller's order.`; default is a general helpful-assistant persona; at most as many characters as always fit the context even as byte-fallback tokens (711 with the shipped voices) |
| `audio.format` | PCM16 at 24 kHz (the model's rate) or 16 kHz, or G.711 μ-law / A-law at 8 kHz, both ways |

Voice and instructions are fixed for a session: they are its prefix.

### Design choices

- **GPT-Live, not Realtime.** Realtime is built around responses and
  commits; a native full-duplex model has neither, and a translation layer
  has to invent response boundaries out of silence. GPT-Live streams
  audio both ways on a timeline, which is what the model does. A Realtime
  adapter can sit on top later if a client ecosystem needs it.
- **Fixed slots, not paged KV.** A session's memory is known when it opens
  and never grows (the context is a 3000-frame ring), so allocation is one
  lease at admission and nothing after. Out-of-memory mid-session cannot
  happen, and neither can the long-session crashes that come with growing
  caches.
- **One graph per tick, all sessions batched.** The per-tick cost is one
  launch regardless of session count; at 128 sessions a tick takes 15 ms
  out of 80.
- **The clock never catches up.** A late tick skips the ticks it missed
  rather than bursting, so one hiccup cannot turn into seconds of added
  latency.

## Trying it

The demo page is served by any server with a live engine, at `/`:

    pega-omni sim-live --listen 127.0.0.1:8000
    # open http://127.0.0.1:8000/

It asks for the microphone (echo cancellation, noise suppression and gain
control on; pick the device in the page, and the speaker too where the
browser can route audio, and switch either mid-session; device names appear
once the page has been granted the microphone), streams it to `/v1/live/sessions` in 20 ms chunks from an
AudioWorklet, and plays the agent through a second worklet with a 120 ms
start-up buffer. It
shows both levels, the playback buffer, gaps and frame lateness, the
engine's live load (sessions in use, the tick's compute time against its
80 ms budget, the share of time the engine is busy, late ticks; polled from
`GET /live/stats`, so it moves while other clients load the server), and the
conversation as chat bubbles: the agent's from its text stream, yours (opt-in,
"Show what I say") from the browser's own speech recognition, which is only a
display aid; the model hears your audio, not that text. Use headphones: echo cancellation helps, but a model that hears
itself answers itself.

`sim-live` is the CPU engine: it echoes you 480 ms late and says a word every
half second it hears you, enough to see the whole loop work. For the real
model (CUDA toolkit 13.x to build, one GPU, ~20 GiB plus 1.5 GiB per session
slot):

    cargo build --release -p omni-server --features personaplex
    target/release/pega-omni personaplex --model-path personaplex-7b-v1 --listen 127.0.0.1:8000
    # open http://127.0.0.1:8000/, pick a voice, edit the role prompt, press Start

Any GPT-Live client works the same way; `tools/live_check.py` is a minimal
one, on the official `openai` SDK's live client.

Browsers only grant the microphone to a secure context: `http://localhost`
qualifies, a remote `http://` address does not. For a server on another
machine, tunnel it:

    ssh -L 8000:127.0.0.1:8000 gpu-box
    # then open http://127.0.0.1:8000/ locally

or put it behind https.

## Benchmark

`omni-bench duplex` opens N concurrent sessions over the WebSocket, streams
each one's caller audio at real time in 20 ms chunks (PCM16 at `--rate`,
24 kHz by default or 16 kHz; a wav at that rate with `--input`, looped, or a synthetic speech-like signal, different per session),
closes them after `--seconds`, and reports per level:

- **started ms**: connect to `session.started` (prompt prefill);
- **late ms**: how far each output frame arrived behind its place on the
  session timeline, relative to the first frame; a real-time server keeps it
  flat however long the session runs;
- **underrun ms**: stall time of a player that starts 120 ms
  (`--playout-ms`) after the first frame and plays the agent's stream in
  real time, and the number of sessions that stalled at all.

A level is clean when every session ran and none stalled. `--ramp 8,16,32,…`
runs levels in order and stops at the first unclean one; the largest clean
level is the server's live capacity.

    omni-bench duplex --base-url http://127.0.0.1:8000 --ramp 8,16,24,32,48,64 --seconds 30 --out ramp.json

Against `sim-live --tick-base-us 2000 --tick-per-session-us 3000` (a tick
costs 2 ms + 3 ms per session, so 26 sessions fit in 80 ms) the ramp finds
24 clean and 32 not, with late p99 at 66 ms and 155 ms respectively.

`tools/live_check.py` is the protocol oracle, the official SDK's client
checking a live server against the rules above at 24 and 16 kHz; CI runs it and a short ramp against
`sim-live`.
