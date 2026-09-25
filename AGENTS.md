# pega-omni

OpenAI-compatible speech and image serving. The front end knows no model: it
parses a request against the engine's `EngineInfo` (`ImageInfo` for an image
engine), hands a checked `Speech` (`Generation`) through a channel, and sends
back what the engine emits: PCM as it streams, pictures once they are done.
Anything that knows a model's name, codec, frame rate or voices belongs in an
engine crate, never in `omni-frontend`.

`docs/` is the design and measurement record; code, comments and commit
messages are English. Only judgment lives in this file; anything a machine can
check belongs in CI.

## Orientation

- `crates/omni-engine`: the contract. An engine is whatever drains an
  `Inbox`; cancellation is the dropped event receiver. Keep it tiny.
  `image.rs` is the image contract, the same shape; pictures cross it as raw
  RGB, PNG is the front end's.
- `crates/omni-frontend`: routes, protocol parsing (`protocol.rs`), response
  framing (`audio.rs`). Serve through `omni_frontend::serve` so accepted
  sockets get `TCP_NODELAY` (without it every packet waits ~40 ms).
- `crates/omni-sim`: `Sim` is a pure state machine (admission, frames,
  chunks, step cost); `spawn` is the thread shell. Decisions go in `Sim`,
  where property tests reach them.
- `crates/omni-qwen3-tts`: Qwen3-TTS as one generated kern manifest.
  `talker.rs`, `stack.rs` and `codec.rs` emit the calls (through
  `omni-kern`'s builder), `kernels/*.cu` the kernels they launch; `model.rs` is the runtime
  shell, `engine.rs` the scheduler. `tests/golden.rs` is the correctness
  oracle.
- `crates/omni-kern`: what both GPU engines share: the manifest builder
  (`Gen`), weight loading, `common.cuh`.
- `crates/omni-personaplex`: PersonaPlex-7B (full duplex) as one manifest;
  `model.rs` the runtime shell, `engine.rs` the clock, `tests/golden.rs` the
  oracle.
- `crates/omni-hidream-o1`: HiDream-O1-Image as one generated kern manifest.
  `model.rs` emits the programs and is the runtime shell, `kernels/hidream.cu`
  the kernels they launch, `sampler.rs` the schedule, `engine.rs` the loop.
  `tests/golden.rs` is the correctness oracle.
- `crates/omni-server`: the `pega-omni` binary; one subcommand per engine.
- `crates/omni-bench`: the load generator; `playback.rs` is the underrun model.
- `tools/openai_sdk_check.py` (`openai_images_check.py` for images): the
  official SDK against a live server; it is the compatibility oracle, not our
  reading of the docs.
- `tools/live_check.py`: the GPT-Live oracle, like the SDK check for speech.
- `tools/personaplex/golden.py`: records the reference run
  `omni-personaplex`'s golden test compares against.
- `tools/qwen3_tts/`: `golden.py` records the official run the golden test
  compares against; `vs_vllm_omni.sh` and `chart.py` reproduce the README
  comparison.
- `tools/hidream_o1/`: `golden.py` records the official run the golden test
  compares against.

## What we prefer

**Functional core, imperative shell.** Logic is `fn(data) -> data`; the clock,
channels, sockets and files sit in a thin layer that does not branch.

**Parse, don't validate.** A type exists because something was checked when
it was built (`Speech` only comes out of `EngineInfo::check`). Downstream code
does not re-check.

**Errors say who acts.** API errors are OpenAI-shaped, name the `param` at
fault, and state expected versus got.

**Dependencies are decisions.** Prefer a mature crate over hand-written
infrastructure; name a new crate and why it earns its weight in the commit
message. No crate for something a dozen lines do (the wav header).

**Small.** No trait until there are two implementations; no compatibility
shims; `BTreeMap` over `HashMap` (clippy enforces it) so iteration order is
deterministic.

**Comments explain the contract and the why.** The module doc is the
module's design doc. No comment that restates the code.

## Gates

- `cargo fmt --check`, `cargo clippy --all-targets --all-features -D warnings`,
  `cargo test` — CI.
- Tests are integration tests in each crate's `tests/`, through the public API;
  enumerable behaviour (chunking, admission) gets a property test.
- A protocol change passes `tools/openai_sdk_check.py` against a live server
  (CI runs it).
- A performance claim is a same-session A/B with `omni-bench`, server and
  client pinned to separate cores on an otherwise idle machine, recorded in
  `docs/bench.md` with date and hardware. A change without a measured win does
  not land as a performance change.

## Commits

`<area>: <what is true after the commit, lowercase, no period>`, area being the
crate or feature (`frontend:`, `sim:`, `bench:`, `ci:`, `docs:`). Every commit
carries `Signed-off-by` (`git commit -s`); CI checks it.
