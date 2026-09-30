#!/usr/bin/env bash
# HiDream-O1-Image-Dev, pega-omni against vLLM-Omni under vLLM-Omni's own
# diffusion benchmark. docs/hidream-o1-vs-vllm-omni.md is the method; this
# script is its executable form.
#
#   vs_vllm_omni.sh setup                     uv venv with vLLM-Omni + its repo (the benchmark)
#   vs_vllm_omni.sh serve-vllm-omni [args]    vLLM-Omni, default deploy, one GPU
#   vs_vllm_omni.sh serve-pega-omni [args]    pega-omni hidream-o1
#   vs_vllm_omni.sh bench <engine> <label>    the benchmark points against a running server
#
# Environment: MODEL (checkpoint directory, required), WORK (venv, repo and
# results; default ./vs-vllm-omni), GPU (default 0), PORT (default 18000),
# SERVER_CPUS / CLIENT_CPUS (taskset lists, default 0-9,20-29 / 10-19,30-39),
# PEGA_OMNI (binary, default target/release/pega-omni), TORCH_BACKEND (uv's
# torch index, default auto), POINTS (concurrency:prompts pairs, default
# "1:10 4:16").
set -euo pipefail

VLLM=0.30.0
VLLM_OMNI=0.30.0rc1
NAME=HiDream-O1-Image-Dev-2604
SIZE=2048
STEPS=28

: "${MODEL:?set MODEL to the HiDream-O1-Image-Dev-2604 checkpoint directory}"
WORK=$(realpath -m "${WORK:-vs-vllm-omni}")
GPU=${GPU:-0}
PORT=${PORT:-18000}
SERVER_CPUS=${SERVER_CPUS:-0-9,20-29}
CLIENT_CPUS=${CLIENT_CPUS:-10-19,30-39}
PEGA_OMNI=${PEGA_OMNI:-target/release/pega-omni}
HERE=$(dirname "$(realpath "$0")")
export PATH=$WORK/.venv/bin:$PATH

setup() {
  mkdir -p "$WORK"
  [ -d "$WORK/vllm-omni" ] ||
    git clone -q --depth 1 --branch "v$VLLM_OMNI" https://github.com/vllm-project/vllm-omni.git "$WORK/vllm-omni"
  uv venv -q --python 3.12 "$WORK/.venv"
  # ninja: vLLM JIT-compiles FlashInfer kernels at startup and dies without it.
  VIRTUAL_ENV=$WORK/.venv uv pip install "vllm==$VLLM" "vllm-omni==$VLLM_OMNI" ninja \
    --torch-backend="${TORCH_BACKEND:-auto}"
}

serve_vllm_omni() {
  CUDA_VISIBLE_DEVICES=$GPU exec taskset -c "$SERVER_CPUS" vllm serve "$MODEL" --omni --port "$PORT" \
    --served-model-name "$NAME" "$@"
}

serve_pega_omni() {
  CUDA_VISIBLE_DEVICES=$GPU exec taskset -c "$SERVER_CPUS" "$PEGA_OMNI" hidream-o1 --model-path "$MODEL" \
    --model "$NAME" --listen "127.0.0.1:$PORT" "$@"
}

# vLLM-Omni's diffusion benchmark (benchmarks/diffusion/diffusion_benchmark_serving.py,
# as tests/dfx/perf/scripts/run_diffusion_benchmark.py drives it for its image
# models): random prompts, text to image, request rate inf, one warmup. Both
# engines run 28 transformer forwards per picture without guidance: vLLM-Omni is
# asked for 28 steps and guidance 0, pega-omni's distilled sampler always takes 28.
bench() {
  local engine=${1:?engine: vllm-omni or pega-omni} label=${2:?label}
  local client=("$WORK/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py") args=()
  case $engine in
    vllm-omni) args=(--num-inference-steps "$STEPS" --extra-body '{"guidance_scale":0}') ;;
    pega-omni) client=("$HERE/bench_client.py" "$WORK/vllm-omni") ;;
    *) echo "unknown engine $engine" >&2; exit 2 ;;
  esac
  local out=$WORK/results/$label
  mkdir -p "$out"
  for point in ${POINTS:-1:10 4:16}; do
    local c=${point%:*} n=${point#*:}
    taskset -c "$CLIENT_CPUS" python "${client[@]}" --host 127.0.0.1 --port "$PORT" --model "$NAME" \
      --endpoint /v1/images/generations --dataset random --task t2i --width "$SIZE" --height "$SIZE" \
      --num-prompts "$n" --max-concurrency "$c" --request-rate inf --warmup-requests 1 "${args[@]}" \
      --output-file "$out/c$c.json" > "$out/c$c.log" 2>&1
    grep -E "Successful|Failed|throughput|Latency (Mean|Median|P99)" "$out/c$c.log" | sed "s/^/c$c  /"
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
