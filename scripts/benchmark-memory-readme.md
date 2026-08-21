# Model Tiering Benchmark

## Setup
```bash
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install -r scripts/requirements.txt
```
Also needs `LITELLM_MASTER_KEY`, `LITELLM_BASE_URL`, `LLAMA_SWAP_API_KEY`,
`LLAMA_SWAP_BASE_URL` in your shell environment (e.g. `set -a && source
.env && set +a` from the repo root) — see `.env.example`.

## Overview
This benchmark measures time-to-first-token, decode/prompt throughput, and
GPU memory footprint for every text model registered in llama-swap, driven
through LiteLLM. Results are used to tier models for llama-swap: which ones
are cheap enough to keep preloaded, and which should be left to
load-on-demand and get evicted under memory pressure.

Text models are discovered dynamically from LiteLLM's `/v1/model/info`,
filtering out anything with `mode == "image_generation"` (the Stable
Diffusion / Kawai image models) — no hardcoded exclude list to keep in sync
by hand.

## GPU memory on this hardware
The DGX Spark's GB10 is a unified-memory device. `nvidia-smi`'s standard
`memory.used`/`memory.total` fields report `[N/A]` here, and even
`--query-compute-apps=used_memory` reports the wrong number for llama.cpp
models (ATS unified memory means their weight/KV-cache allocations never go
through an explicit `cudaMalloc` nvidia-smi's driver-level accounting can
see — see the script's module docstring for the full story). The benchmark
instead reads real host RSS via `docker inspect`'s container PID +
`/proc/<pid>/status`, diffed against a startup baseline of always-on infra
containers — accurate for both llama.cpp and vLLM.

## How to Run
Foreground, single run:
```bash
scripts/benchmark.sh
```

Background (logs to `scripts/results/benchmark.log`, writes as it goes):
```bash
scripts/benchmark.sh --background
```

Test specific models only:
```bash
python3 scripts/benchmark_memory_usage.py --only Phi-4 Qwen2-7B
```

Cap the large-context test prompt size (default 8000 tokens — some models
here are configured up to 262144 context, and testing all the way to that
would take a very long time per model):
```bash
python3 scripts/benchmark_memory_usage.py --max-ctx-test-tokens 4000
```

Best-effort disable/reduce reasoning on models that support it (otherwise
`decode_tokens_per_sec` mixes real generation with thinking tokens,
understating throughput for the model's actual answers):
```bash
python3 scripts/benchmark_memory_usage.py --no-reasoning
```

## What each model test does
1. **Cold request** — a short "count to 100" prompt, streamed. Captures
   `cold_ttft_s` (time to first token, including llama-swap's
   load-on-demand time if the model wasn't already resident) and
   `decode_tokens_per_sec` (from inter-token timing across the ~100 output
   tokens).
2. **GPU memory read** — resident memory right after the cold request
   settles: `memory_loaded_mb`.
3. **Warm request** — a large filler prompt sized to
   `min(configured_context, --max-ctx-test-tokens)`, sent to the
   now-already-loaded model. Captures `warm_ttft_large_ctx_s` (pure
   prompt-processing time, since load time is no longer in the mix) and
   `prompt_tokens_per_sec`.
4. **GPU memory read again** — `memory_at_large_ctx_mb`, the KV-cache-grown
   footprint at that context length.

`configured_context` is read straight from `llama-swap/config.yaml`'s
`capabilities.context` per model — the real per-model max already audited
into that file — not guessed or pulled from LiteLLM (which doesn't expose
it).

## Output
Results are written to `scripts/results/tier_benchmarks.json`:
```json
{
  "models": {
    "model-name": {
      "loaded": true,
      "configured_context": 8192,
      "cold_ttft_s": 0.14,
      "cold_total_s": 3.29,
      "cold_completion_tokens": 100,
      "decode_tokens_per_sec": 31.5,
      "memory_loaded_mb": 374,
      "large_ctx_test_tokens": 2000,
      "warm_ttft_large_ctx_s": 0.16,
      "prompt_tokens_per_sec": 12818.7,
      "memory_at_large_ctx_mb": 374
    }
  }
}
```
A model that failed to load has `"loaded": false` and an `"error"` field
instead of the fields above.

## Using results for tiers
- **`memory_loaded_mb` / `memory_at_large_ctx_mb`** — the actual GPU
  footprint cost of keeping a model resident. Low-footprint models are
  cheap to preload; high-footprint ones (the 70B/120B-class entries) are
  the ones you want evicted whenever they're not actively in use.
- **`cold_ttft_s`** — how painful a cold load is. High cold TTFT + frequent
  use is the strongest argument for preloading a model rather than letting
  llama-swap evict it.
- **`decode_tokens_per_sec` / `prompt_tokens_per_sec`** — real throughput,
  also directly reusable as input to `scripts/estimate_model_cost.py` (see
  `scripts/cost-model.md`) for the `metadata.input_cost_per_token` /
  `output_cost_per_token` fields in `llama-swap/config.yaml`.

## Notes
- Each request uses a 1200s timeout, matching llama-swap's
  `healthCheckTimeout`.
- Models are tested sequentially, one at a time — llama-swap's exclusive
  swap group means only one is resident at once anyway, so this also keeps
  GPU memory readings unambiguous.
- The image-generation models (Stable Diffusion 1.5/3.5, Kawai) are
  automatically skipped via LiteLLM's `mode` field, not a hardcoded list.
