#!/usr/bin/env python3
"""
Tiering benchmark for llama-swap models, driven through LiteLLM.

For each text model (found via LiteLLM's /v1/model/info, filtered to
mode != "image_generation" so image-gen entries like the Stable Diffusion
/ Kawai models are skipped automatically):

  1. Send a small cold request -> measures time-to-first-token (TTFT)
     including llama-swap's load-on-demand time, and decode tok/s from
     the streamed token deltas.
  2. Read the model container's real host RSS immediately after -> this is
     the resident memory footprint for this model at small context.
  3. Send a second, large-prompt request (model already warm) -> measures
     prompt-processing tok/s and TTFT at large context, isolated from
     load time.
  4. Read RSS again -> "at max(-ish) context" footprint.

Memory measurement note: the DGX Spark (GB10) is a unified-memory device
where nvidia-smi's `memory.used`/`memory.total` report "Not Supported" —
verified directly. `--query-compute-apps=pid,used_memory` DOES return a
number, but it's the wrong one for llama.cpp models here: every GGUF
entry (see the `llamacpp_base` macro in llama-swap/config.yaml) sets
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, and this hardware runs in ATS
addressing mode (confirmed via `nvidia-smi
-q`) — CPU and GPU share one page table, so llama.cpp's weight/KV-cache
memory is plain host malloc/mmap that the GPU accesses directly, never
going through an explicit cudaMalloc call nvidia-smi's driver-level
per-process accounting can see. Verified live against a 70B model: nvidia-smi
reported 170 MiB while `docker stats` and /proc/<pid>/status VmRSS both
independently agreed on ~26-27 GiB. So this version reads real host RSS
instead, via the container's host PID (`docker inspect .State.Pid` +
/proc/<pid>/status VmRSS) — accurate for both llama.cpp (unified memory)
and vLLM (regular cudaMalloc, which is also real host-visible memory on
this coherent-memory system) alike. The model's container is identified by
diffing `docker ps` against a startup baseline snapshot of the
always-running infra containers (litellm, proxy, llama-swap, etc.), since
llama-swap's default exclusive swap group means only one model container
is ever new/extra at a time.

Context length per model comes from its `models/<name>.yaml`'s
`capabilities.context` (the real per-model max already audited into that
file).

TTFT vs. eviction wait: cold_ttft_s is measured from request-sent, which
includes however long llama-swap took to evict whatever model was
previously loaded and boot this one -- not a property of this model, but of
test ordering. llama-swap v250 exposes `GET /running` (auth: apiKeys entry
in llama-swap/config.yaml, verified live against the running stack), which
reports each model's real process state (stopped/starting/ready/stopping/
shutdown, from internal/process/process.go). This script polls that
endpoint on a background thread during the cold request to timestamp the
"starting" -> "ready" transition, splitting the result into
`time_to_ready_s` (eviction + load wait) and `true_ttft_s` (first-token
latency once actually serving, comparable across models regardless of
what was loaded before them in the run).

Run in background: scripts/benchmark.sh --background
Output: scripts/results/tier_benchmarks.json (cumulative, merged across runs)
    and scripts/results/tier_benchmarks_<UTC timestamp>.json (this run only)

--no-reasoning best-effort disables/reduces thinking on reasoning-tuned
models (see stream_completion()) -- without it, decode_tokens_per_sec mixes
real generation with thinking tokens, understating throughput for the
model's actual answers.

Reads credentials/URLs from the environment (see .env / .env.example):
  LITELLM_MASTER_KEY, LITELLM_BASE_URL, LLAMA_SWAP_API_KEY, LLAMA_SWAP_BASE_URL
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import requests
import yaml


def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: {name} is not set (see .env.example) — this script needs it to reach your stack")
    return value


API_KEY = require_env("LITELLM_MASTER_KEY")
BASE_URL = require_env("LITELLM_BASE_URL")
LLAMA_SWAP_API_KEY = require_env("LLAMA_SWAP_API_KEY")
LLAMA_SWAP_URL = require_env("LLAMA_SWAP_BASE_URL")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results", "tier_benchmarks.json")

REQUEST_TIMEOUT = 1200  # matches llama-swap's healthCheckTimeout
SETTLE_SECONDS = 2  # let nvidia-smi/driver bookkeeping catch up after a request
STATE_POLL_INTERVAL_S = 0.15


def get_text_models():
    """Fetch models from LiteLLM's /v1/model/info, keep only non-image-gen ones."""
    resp = requests.get(
        f"{BASE_URL}/v1/model/info",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [m["model_name"] for m in data if m.get("model_info", {}).get("mode") != "image_generation"]


def _iter_model_entries():
    """Yield (alias, entry) for every model across models/*.yaml."""
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.yaml"))):
        with open(path) as f:
            doc = yaml.safe_load(f)
        for alias, entry in (doc.get("models") or {}).items():
            yield alias, entry


def get_configured_contexts():
    """Read capabilities.context per model alias from models/*.yaml."""
    contexts = {}
    for alias, entry in _iter_model_entries():
        ctx = (entry.get("capabilities") or {}).get("context")
        if ctx:
            contexts[alias] = ctx
    return contexts


def get_model_engines():
    """Classify each model alias as 'vllm' or 'llama.cpp' by reading its cmd."""
    engines = {}
    for alias, entry in _iter_model_entries():
        cmd = entry.get("cmd", "")
        if "vllm_base" in cmd or "vllm-spark:local" in cmd:
            engines[alias] = "vllm"
        elif "llamacpp_base" in cmd or "llama.cpp" in cmd:
            engines[alias] = "llama.cpp"
        else:
            engines[alias] = "other"
    return engines


def get_running_container_names():
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return set()
    return set(line.strip() for line in result.stdout.strip().splitlines() if line.strip())


def get_container_rss_mb(container_name):
    """Real host RSS (MB) for a container's main process, via /proc -- see module
    docstring for why this replaces nvidia-smi on this hardware."""
    try:
        pid_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True, text=True, timeout=15,
        )
        if pid_result.returncode != 0:
            return None
        pid = pid_result.stdout.strip()
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except Exception as e:
        print(f"  Error reading RSS for {container_name}: {e}", file=sys.stderr)
    return None


def get_active_model_memory_mb(baseline_containers):
    """Sum RSS (MB) across any container(s) not present in the startup baseline.

    llama-swap's default exclusive swap group means only the currently
    loaded model's container should be "extra" at any point in time.
    """
    current = get_running_container_names()
    new_containers = current - baseline_containers
    if not new_containers:
        return None
    total = 0.0
    found_any = False
    for name in new_containers:
        rss = get_container_rss_mb(name)
        if rss is not None:
            total += rss
            found_any = True
    return total if found_any else None


def get_model_state(model):
    """Read this model's current process state from llama-swap's /running
    (stopped/starting/ready/stopping/shutdown -- internal/process/process.go).
    Returns None if the model has no active process (fully stopped/never started)."""
    try:
        resp = requests.get(
            f"{LLAMA_SWAP_URL}/running",
            headers={"Authorization": f"Bearer {LLAMA_SWAP_API_KEY}"},
            timeout=5,
        )
        resp.raise_for_status()
        for entry in resp.json().get("running", []):
            if entry.get("model") == model:
                return entry.get("state")
    except Exception:
        return None
    return None


def watch_for_ready(model, start_time, result, stop_event):
    """Background-thread poll loop: records the wall-clock offset (from
    start_time) at which `model` first reaches llama-swap's "ready" state.

    This separates "how long until llama-swap finished evicting whatever was
    previously loaded and booted this model" from the model's own true
    time-to-first-token once actually serving -- a cold_ttft_s measured
    against total elapsed time conflates the two, and which model happened to
    be loaded right before this one in the test sequence would otherwise
    skew results per-run.
    """
    saw_starting = False
    while not stop_event.is_set():
        state = get_model_state(model)
        if state == "starting":
            saw_starting = True
        elif state == "ready":
            result["time_to_ready_s"] = time.time() - start_time
            result["was_cold_load"] = saw_starting
            return
        time.sleep(STATE_POLL_INTERVAL_S)


def get_actual_boot_context(model):
    """Query llama.cpp's own /props (proxied through llama-swap's /upstream/<model>/
    passthrough) for the real n_ctx and slot count this boot actually got.

    Matters because config.yaml's capabilities.context is the GGUF's theoretical
    max, but llama.cpp's --fit-ctx auto-shrinks per-slot ctx to fit VRAM given
    --parallel N -- verified live on Laguna-S-2.1-Q4_K_M: capabilities.context
    says 262144, but /props reported n_ctx: 131072 with total_slots: 2 (the
    auto-fit halved it to fit 2 KV-cache copies in budget). So configured_context
    in results is NOT reliably what's actually resident; this is.

    Only call this for the model that is currently loaded/ready -- hitting
    /upstream/<model>/... for a different, not-yet-running model can itself
    trigger a cold load through llama-swap's normal request routing.
    """
    try:
        resp = requests.get(
            f"{LLAMA_SWAP_URL}/upstream/{model}/props",
            headers={"Authorization": f"Bearer {LLAMA_SWAP_API_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "actual_boot_n_ctx": data.get("default_generation_settings", {}).get("n_ctx"),
            "total_slots": data.get("total_slots"),
        }
    except Exception as e:
        print(f"  Error reading /props for {model}: {e}", file=sys.stderr)
        return {"actual_boot_n_ctx": None, "total_slots": None}


def stream_completion(model, prompt, max_tokens, no_reasoning=False):
    """POST a streaming chat completion, return (ttft_s, total_s, completion_tokens).

    no_reasoning=True adds the request-body hints reasoning-tuned models
    (Qwen3.5 thinking, Nemotron reasoning, etc.) respect to skip their
    thinking phase -- best-effort, not every model honors every hint.
    Servers ignore fields they don't recognize, so it's safe to always send
    both rather than needing to know per-model which one applies.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if no_reasoning:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["reasoning_effort"] = "low"
        # LiteLLM's openai/ adapter (our model: openai/<name> convention)
        # rejects reasoning_effort as an unrecognized param by default --
        # confirmed via a real 400: "litellm.UnsupportedParamsError: openai
        # does not support parameters: ['reasoning_effort']". Its own error
        # message names this per-request escape hatch.
        body["allowed_openai_params"] = ["reasoning_effort"]
    start = time.time()
    ttft = None
    completion_tokens = 0
    with requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=body,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            # Reasoning-tuned models (Qwen3.5 thinking, gpt-oss, Nemotron
            # reasoning, etc.) stream thinking tokens in a separate
            # `reasoning_content` field via llama.cpp's OpenAI-compat server,
            # not `content` -- checked only `content` originally, which left
            # TTFT/token-count null for any model still "thinking" when
            # max_tokens ran out. Both are real generated tokens with real
            # per-token cost, so both count for timing purposes here.
            text = delta.get("content") or delta.get("reasoning_content")
            if text:
                if ttft is None:
                    ttft = time.time() - start
                completion_tokens += 1
    total = time.time() - start
    return ttft, total, completion_tokens


def build_large_prompt(target_tokens):
    """Repeat filler text to approximate a target token count (~1.3 tokens/word)."""
    words_needed = max(1, int(target_tokens / 1.3))
    filler = "The quick brown fox jumps over the lazy dog. "
    words_per_filler = len(filler.split())
    repeats = max(1, words_needed // words_per_filler)
    return "Summarize the following in one word.\n\n" + (filler * repeats)


def test_model(model, configured_context, max_ctx_test_tokens, baseline_containers, no_reasoning=False):
    print(f"  Cold request (small prompt)...")
    mem_before = get_active_model_memory_mb(baseline_containers)

    # Poll llama-swap's /running in the background to separate "waited for
    # the previous model to be evicted + this one to load" from this model's
    # own true first-token latency -- see watch_for_ready() docstring.
    state_result = {}
    stop_event = threading.Event()
    cold_start = time.time()
    watcher = threading.Thread(target=watch_for_ready, args=(model, cold_start, state_result, stop_event), daemon=True)
    watcher.start()
    try:
        # A prompt that reliably produces enough tokens for a stable decode-rate
        # sample -- "Say OK." often stops after 1-2 tokens, which makes
        # decode_tokens_per_sec pure noise.
        cold_ttft, cold_total, cold_tokens = stream_completion(
            model, "Count from 1 to 100, one number per line.", max_tokens=100,
            no_reasoning=no_reasoning,
        )
    except Exception as e:
        stop_event.set()
        print(f"  FAILED: {model}: {e}", file=sys.stderr)
        return {"loaded": False, "error": str(e)}
    finally:
        stop_event.set()
        watcher.join(timeout=STATE_POLL_INTERVAL_S * 2)

    time.sleep(SETTLE_SECONDS)
    mem_loaded_mb = get_active_model_memory_mb(baseline_containers)
    boot_ctx_info = get_actual_boot_context(model)

    decode_tps = None
    if cold_ttft is not None and cold_tokens > 1 and (cold_total - cold_ttft) > 0:
        decode_tps = (cold_tokens - 1) / (cold_total - cold_ttft)

    time_to_ready_s = state_result.get("time_to_ready_s")
    was_cold_load = state_result.get("was_cold_load", False)
    # True TTFT once the model was actually serving, isolated from any
    # eviction/load wait baked into cold_ttft_s. Falls back to cold_ttft_s
    # when the model was already ready before this request (was_cold_load
    # False), or when the watcher missed the transition (rare, small window).
    true_ttft_s = None
    if cold_ttft is not None:
        if time_to_ready_s is not None and time_to_ready_s < cold_ttft:
            true_ttft_s = cold_ttft - time_to_ready_s
        else:
            true_ttft_s = cold_ttft

    # Cap the throughput-test prompt against the REAL booted n_ctx, not the
    # config.yaml theoretical max -- sending more tokens than the model
    # actually has room for would just error/truncate. Falls back to
    # configured_context if /props couldn't be read.
    actual_ctx = boot_ctx_info.get("actual_boot_n_ctx")
    ctx_ceiling = actual_ctx or configured_context or max_ctx_test_tokens
    large_ctx_target = min(ctx_ceiling, max_ctx_test_tokens)
    large_prompt = build_large_prompt(large_ctx_target)
    print(f"  Warm request (~{large_ctx_target} token prompt)...")
    try:
        warm_ttft, warm_total, warm_tokens = stream_completion(
            model, large_prompt, max_tokens=10, no_reasoning=no_reasoning,
        )
    except Exception as e:
        print(f"  Large-context request FAILED: {model}: {e}", file=sys.stderr)
        warm_ttft = warm_total = warm_tokens = None

    time.sleep(SETTLE_SECONDS)
    # NOT expected to differ meaningfully from memory_loaded_mb -- llama.cpp
    # allocates the full KV cache (n_ctx x total_slots) at boot, before any
    # request arrives, not incrementally as context fills. Kept mainly as a
    # sanity check that nothing unexpected grew.
    mem_large_ctx_mb = get_active_model_memory_mb(baseline_containers)

    prompt_tps = None
    if warm_ttft and warm_ttft > 0:
        prompt_tps = large_ctx_target / warm_ttft

    return {
        "loaded": True,
        "memory_before_mb": mem_before,
        "configured_context": configured_context,
        "actual_boot_n_ctx": actual_ctx,
        "total_slots": boot_ctx_info.get("total_slots"),
        "cold_ttft_s": cold_ttft,
        "was_cold_load": was_cold_load,
        "time_to_ready_s": time_to_ready_s,
        "true_ttft_s": true_ttft_s,
        "cold_total_s": cold_total,
        "cold_completion_tokens": cold_tokens,
        "decode_tokens_per_sec": decode_tps,
        "memory_loaded_mb": mem_loaded_mb,
        "large_ctx_test_tokens": large_ctx_target,
        "warm_ttft_large_ctx_s": warm_ttft,
        "prompt_tokens_per_sec": prompt_tps,
        "memory_at_large_ctx_mb": mem_large_ctx_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Tiering benchmark for llama-swap text models")
    parser.add_argument("--status", action="store_true", help="Print progress as it runs (default: on)")
    parser.add_argument("--max-ctx-test-tokens", type=int, default=8000,
                         help="Cap on the large-context test prompt size, to keep runtime sane for 200k+ ctx models (default: 8000)")
    parser.add_argument("--only", nargs="*", default=None, help="Only test these model names")
    parser.add_argument("--skip-vllm", action="store_true",
                         help="Skip vLLM-engine models (identified via config.yaml cmd) -- they're much slower to cold-load and dominate total run time")
    parser.add_argument("--no-reasoning", action="store_true",
                         help="Best-effort disable/reduce reasoning on models that support it (chat_template_kwargs.enable_thinking=false, reasoning_effort=low) -- decode tok/s otherwise mixes real generation with thinking tokens, understating throughput for the model's actual answers")
    args = parser.parse_args()

    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)

    models = get_text_models()
    if args.only:
        models = [m for m in models if m in args.only]
    engines = get_model_engines()
    if args.skip_vllm:
        skipped = [m for m in models if engines.get(m) == "vllm"]
        models = [m for m in models if engines.get(m) != "vllm"]
        if skipped:
            print(f"Skipping {len(skipped)} vLLM models: {', '.join(skipped)}")
    contexts = get_configured_contexts()
    print(f"Found {len(models)} text models to benchmark")

    # Snapshot of always-on infra containers (litellm, proxy, llama-swap, the
    # standalone llama.cpp router, etc.) so each test can tell which container
    # is the one llama-swap just spawned for the model under test.
    baseline_containers = get_running_container_names()

    # Load any existing results and merge into them, rather than starting
    # from empty -- a targeted --only run (e.g. re-verifying one model after
    # a config fix) previously clobbered every other model's results in the
    # same file, since this started as {} unconditionally. Confirmed this
    # destroyed a full 12-model run's data during a real --only re-test.
    results = {}
    start_time = datetime.now(timezone.utc)
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
            results = existing.get("models", {})
            if existing.get("start"):
                start_time = datetime.fromisoformat(existing["start"])
        except Exception as e:
            print(f"  Could not load existing results to merge: {e}", file=sys.stderr)

    # Per-run snapshot with a unique, datestamped filename, written alongside
    # the cumulative tier_benchmarks.json -- so each run's results survive
    # independently of later runs re-merging into the shared file, instead of
    # only ever having "whatever tier_benchmarks.json currently looks like".
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_results_path = os.path.join(SCRIPT_DIR, "results", f"tier_benchmarks_{run_stamp}.json")

    def save():
        output = {
            "start": start_time.isoformat(),
            "end": datetime.now(timezone.utc).isoformat(),
            "models": results,
        }
        for path in (RESULTS_PATH, run_results_path):
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # survive as much as possible of a hard crash (e.g. driver/kernel panic on model load), not just a clean exit

    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Testing: {model}", flush=True)
        # Placeholder written before the test starts: if the host dies mid-load
        # (kernel panic, OOM-killed driver, etc.) this is what's left on disk,
        # so you know exactly which model was in flight instead of just missing data.
        results[model] = {"loaded": None, "status": "in_progress"}
        save()
        results[model] = test_model(
            model, contexts.get(model), args.max_ctx_test_tokens, baseline_containers,
            no_reasoning=args.no_reasoning,
        )
        save()  # write after every model so a killed/crashed run still leaves partial results
        time.sleep(2)

    save()
    print(f"Results saved to {RESULTS_PATH}")
    print(f"Per-run snapshot saved to {run_results_path}")


if __name__ == "__main__":
    main()
