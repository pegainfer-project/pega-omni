#!/usr/bin/env bash
# Usage: tools/bench_sweep.sh <tag> <server-args...>   (env: CONCS, N, EXTRA, SRV_CPUS, CLI_CPUS, WORKERS, FORMAT, FRAMING, RATE)
set -euo pipefail
tag=$1; shift
B=${B:-target/release}
R=${R:-results}
PORT=${PORT:-18000}
SRV_CPUS=${SRV_CPUS:-0-31}; CLI_CPUS=${CLI_CPUS:-36-71}; WORKERS=${WORKERS:-32}
CONCS=${CONCS:-"64 256 1024 4096"}; N=${N:-200000}; RATE=${RATE:-inf}
EXTRA=${EXTRA:-'{"frames":13}'}; FORMAT=${FORMAT:-pcm}; FRAMING=${FRAMING:-audio}
mkdir -p "$R/$tag"
taskset -c "$SRV_CPUS" "$B/pega-omni" sim --listen 127.0.0.1:$PORT --workers "$WORKERS" "$@" > "$R/$tag/server.log" 2>&1 &
srv=$!
trap 'kill -INT $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for _ in $(seq 50); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 0.1; done
for c in $CONCS; do
  n=$(( N > c * 20 ? N : c * 20 ))
  echo "== $tag c=$c n=$n"
  # server CPU in cores per second (utime+stime ticks / 100) and RSS KiB
  ( prev=$(awk '{print $14+$15}' /proc/$srv/stat); while sleep 1 && kill -0 $srv 2>/dev/null; do
      t=$(awk '{print $14+$15}' /proc/$srv/stat); echo "$(( t - prev )) $(awk '/VmRSS/{print $2}' /proc/$srv/status)"; prev=$t; done ) > "$R/$tag/server-cpu-c$c.txt" &
  mon=$!
  taskset -c "$CLI_CPUS" "$B/omni-bench" --base-url http://127.0.0.1:$PORT --num-requests $n --max-concurrency $c \
    --request-rate "$RATE" --extra "$EXTRA" --response-format "$FORMAT" --stream-format "$FRAMING" \
    --warmup $(( c > 64 ? c : 64 )) --label "$tag-c$c" --out "$R/$tag/c$c.json" | tee "$R/$tag/c$c.txt" || true
  kill $mon 2>/dev/null || true
done
curl -s localhost:$PORT/metrics > "$R/$tag/metrics.txt" || true
