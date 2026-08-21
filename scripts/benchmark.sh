#!/bin/bash
# Usage:
#   scripts/benchmark.sh [args...]              # foreground, single run
#   scripts/benchmark.sh --background [args...] # background, logs to results/
#
# Uses scripts/.venv if present (pip install -r requirements.txt there),
# falls back to system python3.
cd "$(dirname "$0")"
PYTHON="./.venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

if [ "$1" = "--background" ]; then
  shift
  mkdir -p results
  nohup "$PYTHON" -u benchmark_memory_usage.py "$@" \
    > results/benchmark.log \
    2> results/benchmark.err &
  echo "Benchmark running in background (PID $!)."
  echo "Progress:  tail -f scripts/results/benchmark.log"
  echo "Errors:    tail -f scripts/results/benchmark.err"
  echo "Results:   scripts/results/tier_benchmarks.json"
else
  "$PYTHON" benchmark_memory_usage.py "$@"
fi
