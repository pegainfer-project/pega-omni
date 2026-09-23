#!/usr/bin/env bash
# Qwen3-TTS CustomVoice, pega-omni against vLLM-Omni under vLLM-Omni's own
# benchmark. docs/qwen3-tts-vs-vllm-omni.md is the method; this script is its
# executable form.
#
#   vs_vllm_omni.sh all                       setup, both servers in turn, both benches, the README chart
#   vs_vllm_omni.sh setup                     uv venv with vLLM-Omni + its repo (dataset)
#   vs_vllm_omni.sh serve-vllm-omni [args]    vLLM-Omni, single-GPU throughput profile under MPS
#   vs_vllm_omni.sh serve-pega-omni [args]    pega-omni qwen3-tts
#   vs_vllm_omni.sh bench <engine> <label>    vLLM-Omni's CI points against a running server
#   vs_vllm_omni.sh chart <vllm> <pega> <png> the README chart from two bench labels
#
# Environment: MODEL (checkpoint directory, required), WORK (venv, repo and
# results; default ./vs-vllm-omni), GPU (default 0), PORT (default 18000),
# SERVER_CPUS / CLIENT_CPUS (taskset lists, default 0-31 / 36-71),
# PEGA_OMNI (binary, default target/release/pega-omni).
set -euo pipefail

VLLM=0.30.0
VLLM_OMNI=0.30.0rc1
NAME=Qwen3-TTS-12Hz-1.7B-CustomVoice
HERE=$(cd "$(dirname "$0")" && pwd)

: "${MODEL:?set MODEL to the Qwen3-TTS-12Hz-1.7B-CustomVoice checkpoint directory}"
WORK=$(realpath -m "${WORK:-vs-vllm-omni}")
GPU=${GPU:-0}
PORT=${PORT:-18000}
SERVER_CPUS=${SERVER_CPUS:-0-31}
CLIENT_CPUS=${CLIENT_CPUS:-36-71}
PEGA_OMNI=${PEGA_OMNI:-target/release/pega-omni}
export PATH=$WORK/.venv/bin:$PATH

setup() {
  mkdir -p "$WORK"
  [ -d "$WORK/vllm-omni" ] ||
    git clone -q --depth 1 --branch "v$VLLM_OMNI" https://github.com/vllm-project/vllm-omni.git "$WORK/vllm-omni"
  [ -x "$WORK/.venv/bin/vllm" ] && return
  uv venv -q --python 3.12 "$WORK/.venv"
  # ninja: vLLM JIT-compiles FlashInfer kernels at startup and dies without it.
  VIRTUAL_ENV=$WORK/.venv uv pip install "vllm==$VLLM" "vllm-omni==$VLLM_OMNI" ninja --torch-backend=auto
}

# vLLM-Omni's strongest single-GPU setup we found: its opt-in throughput
# profile (talker and code2wav on one GPU, B8 codec graphs) with both stage
# processes sharing the GPU through a private MPS daemon, which the profile
# suggests but does not enable. It beats the default deploy on TTFP and
# throughput from c=8 up.
serve_vllm_omni() {
  local mps=$WORK/mps
  mkdir -p "$mps/pipe" "$mps/log"
  export CUDA_MPS_PIPE_DIRECTORY=$mps/pipe CUDA_MPS_LOG_DIRECTORY=$mps/log
  CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  # find_spec locates the package without importing it (importing logs to stdout).
  local deploy
  deploy=$(python -c 'import importlib.util as u, pathlib as p; print(p.Path(u.find_spec("vllm_omni").origin).parent / "deploy")')
  CUDA_VISIBLE_DEVICES=$GPU taskset -c "$SERVER_CPUS" vllm serve "$MODEL" --omni --port "$PORT" \
    --served-model-name "$NAME" --trust-remote-code \
    --deploy-config "$deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml" "$@" &
  server=$!
  trap 'kill $server 2>/dev/null; wait $server; echo quit | nvidia-cuda-mps-control' EXIT
  trap 'exit 143' TERM INT
  wait $server
}

serve_pega_omni() {
  CUDA_VISIBLE_DEVICES=$GPU exec taskset -c "$SERVER_CPUS" "$PEGA_OMNI" qwen3-tts --model-path "$MODEL" \
    --model "$NAME" --listen "127.0.0.1:$PORT" "$@"
}

# vLLM-Omni's nightly CI for this checkpoint (tests/dfx/perf/tests/test_tts.json,
# expanded by tests/dfx/perf/scripts/run_benchmark.py): seed_tts_smoke (en),
# request rate inf, warmups max(2, c), audio_underrun added to the metrics.
bench() {
  local engine=${1:?engine: vllm-omni or pega-omni} label=${2:?label} body
  case $engine in
    vllm-omni) body='{"voice":"Vivian","language":"English","task_type":"CustomVoice"}' ;;
    pega-omni) body='{"voice":"vivian","extra":{"language":"english"}}' ;;
    *) echo "unknown engine $engine" >&2; exit 2 ;;
  esac
  local out=$WORK/results/$label
  mkdir -p "$out"
  for point in 1:20 8:80 16:128 64:128; do
    local c=${point%:*} n=${point#*:}
    taskset -c "$CLIENT_CPUS" vllm bench serve --omni --host 127.0.0.1 --port "$PORT" \
      --model "$NAME" --tokenizer "$MODEL" --trust-remote-code \
      --backend openai-audio-speech --endpoint /v1/audio/speech \
      --dataset-name seed-tts-text --dataset-path "$WORK/vllm-omni/benchmarks/build_dataset/seed_tts_smoke" \
      --seed-tts-locale en --extra-body "$body" \
      --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
      --max-concurrency "$c" --num-prompts "$n" --num-warmups $((c > 2 ? c : 2)) --request-rate inf \
      --save-result --result-dir "$out" --result-filename "c$c.json" > "$out/c$c.log" 2>&1
    grep -E "Failed requests|Median AUDIO_TTFP|Median AUDIO_RTF|Audio throughput|continuity|Mean AUDIO_DURATION" "$out/c$c.log" |
      sed "s/^/c$c  /"
  done
}

chart() {
  uv run -q --with cairosvg python "$HERE/chart.py" "$WORK/results/${1:?vllm label}" "$WORK/results/${2:?pega label}" "${3:?png}"
}

# One engine at a time on the GPU: start, wait for health, bench, stop.
session() {
  local serve=$1 engine=$2 label=$3; shift 3
  mkdir -p "$WORK/results/$label"
  "$0" "$serve" "$@" > "$WORK/results/$label/server.log" 2>&1 &
  local pid=$!
  until curl -sf "127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "$label: server died, see $WORK/results/$label/server.log" >&2; exit 1; }
    sleep 2
  done
  echo "== $label"
  bench "$engine" "$label"
  kill $pid
  wait $pid 2>/dev/null || true
  while nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 1; done
}

all() {
  setup
  session serve-vllm-omni vllm-omni vllm-omni
  session serve-pega-omni pega-omni pega-omni-1-25 --first-chunk-frames 1 --chunk-frames 25
  session serve-pega-omni pega-omni pega-omni
  chart vllm-omni pega-omni-1-25 "$HERE/../../assets/qwen3-tts-vs-vllm-omni.png"
}

cmd=${1:-}
shift || true
case $cmd in
  all) all ;;
  setup) setup ;;
  serve-vllm-omni) serve_vllm_omni "$@" ;;
  serve-pega-omni) serve_pega_omni "$@" ;;
  bench) bench "$@" ;;
  chart) chart "$@" ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
