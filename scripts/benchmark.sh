#!/bin/bash
# Usage:
#   scripts/benchmark.sh [args...]              # foreground, single run
#   scripts/benchmark.sh --background [args...] # background, logs to results/
cd "$(dirname "$0")"

if [ "$1" = "--background" ]; then
  shift
  mkdir -p results
  nohup python3 -u benchmark_memory_usage.py "$@" \
    > results/benchmark.log \
    2> results/benchmark.err &
  echo "Benchmark running in background (PID $!)."
  echo "Progress:  tail -f scripts/results/benchmark.log"
  echo "Errors:    tail -f scripts/results/benchmark.err"
  echo "Results:   scripts/results/tier_benchmarks.json"
else
  python3 benchmark_memory_usage.py "$@"
fi
