#!/usr/bin/env bash
# Qwen3-TTS CustomVoice, pega-omni against vLLM-Omni under vLLM-Omni's own
# benchmark. docs/qwen3-tts-vs-vllm-omni.md is the method; this script is its
# executable form.
#
#   vs_vllm_omni.sh setup                     uv venv with vLLM-Omni + its repo (dataset)
#   vs_vllm_omni.sh serve-vllm-omni [args]    vLLM-Omni, default deploy, one GPU
#   vs_vllm_omni.sh serve-pega-omni [args]    pega-omni qwen3-tts
#   vs_vllm_omni.sh bench <engine> <label>    vLLM-Omni's CI points against a running server
#
# Environment: MODEL (checkpoint directory, required), WORK (venv, repo and
# results; default ./vs-vllm-omni), GPU (default 0), PORT (default 18000),
# SERVER_CPUS / CLIENT_CPUS (taskset lists, default 0-31 / 36-71),
# PEGA_OMNI (binary, default target/release/pega-omni).
set -euo pipefail

VLLM=0.30.0
VLLM_OMNI=0.30.0rc1
NAME=Qwen3-TTS-12Hz-1.7B-CustomVoice

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
  uv venv -q --python 3.12 "$WORK/.venv"
  # ninja: vLLM JIT-compiles FlashInfer kernels at startup and dies without it.
  VIRTUAL_ENV=$WORK/.venv uv pip install "vllm==$VLLM" "vllm-omni==$VLLM_OMNI" ninja --torch-backend=auto
}

serve_vllm_omni() {
  CUDA_VISIBLE_DEVICES=$GPU exec taskset -c "$SERVER_CPUS" vllm serve "$MODEL" --omni --port "$PORT" \
    --served-model-name "$NAME" --trust-remote-code "$@"
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

cmd=${1:-}
shift || true
case $cmd in
  setup) setup ;;
  serve-vllm-omni) serve_vllm_omni "$@" ;;
  serve-pega-omni) serve_pega_omni "$@" ;;
  bench) bench "$@" ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
