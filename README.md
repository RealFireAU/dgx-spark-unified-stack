# DGX Spark Unified Inference Stack

One box, one entry point: every engine (llama.cpp/GGUF, vLLM/safetensors,
ComfyUI/stable-diffusion.cpp for diffusion) sits behind llama-swap
(load/evict/GPU-sharing) and LiteLLM (the single OpenAI-compatible API +
admin UI).

## Storage

Two flat pools, both under `LLM_ROOT_PATH`:

- `LLM_ROOT_PATH/gguf/<org>/<repo>/*.gguf` — every GGUF model, read by the
  llama.cpp containers llama-swap spawns.
- `LLM_ROOT_PATH/safetensors/<org>/<repo>/` — every safetensors checkpoint,
  read by vLLM containers.
- `LLM_ROOT_PATH/diffusion/` — a third pool, structured differently on
  purpose: ComfyUI organizes by model *role* (`checkpoints/`, `loras/`,
  `vae/`, `controlnet/`, …), not by source repo, and creates those
  subfolders itself on first launch.

## What's here

- **llama-swap** — spawns/evicts model containers on demand (public
  non-root image, no local Dockerfile). Web UI at `/ui` on whatever
  hostname you route to it. Guarded by `apiKeys` (`LLAMA_SWAP_API_KEY`),
  the same key LiteLLM sends.
- **litellm** + **litellm-db** — the single OpenAI-compatible API + admin
  UI (public non-root image, no local Dockerfile).
- **vllm**, **comfyui**, **sd-cpp** — build targets only (`profiles:
  [tools]`, never run directly). llama-swap spawns ephemeral containers
  from these images on demand; `docker compose build <name>` produces each
  one. All three need a real GB10/CUDA13 arm64 build, so they're built
  locally rather than pulled.
- **proxy** (Caddy) — **the only service with host ports** (80/443) and so
  the only way into the stack. Real certs via Cloudflare DNS-01 (see
  `proxy/Caddyfile.sample`) — copy it to `proxy/Caddyfile` (gitignored) and
  fill in your own hostnames and use your own ACME condif HTTP-01/DNS-01.

### Networking

Services are segmented across five Docker networks rather than one flat one,
so an engine is unreachable from the LAN by construction, not by auth:

| network | joins | purpose |
|---|---|---|
| `edge` | proxy ↔ litellm | public ingress → the API gateway |
| `ui` | proxy ↔ llama-swap | llama-swap web UI + `/comfyui/*` |
| `inference` | litellm ↔ llama-swap | the only inference path |
| `data` | litellm ↔ litellm-db | `internal: true` — no egress at all |
| `engines` | vllm/comfyui/sd-cpp build targets | isolated; nothing routes here |

No engine publishes a host port. Caddy cannot reach Postgres, vLLM, or
llama.cpp directly; llama-swap cannot reach Postgres.

Automatic1111-style clients are served by a LiteLLM **passthrough** at
`/sdapi/*` (`general_settings.pass_through_endpoints`), which forwards
verbatim to llama-swap with its API key injected — so A1111 tooling gets a
route that still goes through LiteLLM. It's a passthrough, not a
translation: callers pass `"model": "<registered diffusion model>"` in the
JSON body for llama-swap to dispatch on. Covers all three routes llama-swap
exposes (`POST /sdapi/v1/txt2img`, `POST /sdapi/v1/img2img`,
`GET /sdapi/v1/loras`).

Your Caddyfile can also block inference paths (`/v1/*`, `/sdapi/*`) on the
llama-swap admin hostname so inference only ever arrives via LiteLLM, where
logging and cost tracking happen — see the `@swap_inference` matcher in
`proxy/Caddyfile.sample`.

One caveat worth knowing: Docker networks are **bidirectional**. Sharing
`inference` means llama-swap can also reach litellm, which cannot be
prevented at the network layer.

## Bring it up

```bash
cp .env.example .env      # fill in real values, including LLM_ROOT_PATH
cp proxy/Caddyfile.sample proxy/Caddyfile   # fill in your own hostnames

# Create these yourself first, as your own user -- Docker auto-creates a
# missing bind-mount directory as root:root on first `up`, which then
# blocks you from writing your own files into it afterward.
set -a && source .env && set +a
mkdir -p models "$LLM_ROOT_PATH"/{gguf,safetensors,diffusion}

# vllm first, on its own -- comfyui's Dockerfile is FROM vllm-spark:local,
# and `docker compose build` can parallelize a multi-service build, which
# would break if comfyui starts before that image exists.
docker compose build vllm
docker compose build comfyui sd-cpp

docker compose up -d
```

Nothing is reachable on a host port except Caddy, so reach the stack
through the hostnames you configured, not `localhost:<port>`.

## Registering models

llama-swap's model list is driven entirely by `models/` (gitignored — real
quants/paths are specific to what you've downloaded, not portable). See
`models.example/README.md` for the schema and two worked examples.

- **Add a model**: drop a file in `models/`. llama-swap picks it up live
  via `-config-dir` + `-watch-config` — no restart.
- **Remove a model**: delete its file. Same, no restart.
- **LiteLLM side**: `docker compose restart litellm` after any change —
  LiteLLM has no hot-reload, unlike llama-swap. `LiteLLM/litellm_startup_hook.py`
  runs as a [worker startup
  hook](https://docs.litellm.ai/docs/proxy/worker_startup_hooks) inside the
  litellm container itself, so every start/restart regenerates the model
  list from `models/*.yaml` automatically — no separate step needed.

## Known gaps

- **vllm/comfyui/sd-cpp** are build-only targets with no default running
  instance — llama-swap spawns them on demand per your `models/` entries.
  If you want one running standalone for testing, use
  `docker compose run --rm vllm ...` (or `comfyui`/`sd-cpp`) directly.
- Some vLLM checkpoints are architecture/quant edge cases (custom MoE
  variants, third-party resharded formats) that may not load cleanly with
  the generic `vllm-spark:local` image — worth a real boot test before
  relying on any specific one, not just registering it.
- A real login/session gate (vs. this stack's own `apiKeys`) in front of
  the llama-swap admin UI and any other exposed hostname is worth adding
  before exposing this stack beyond your LAN — Cloudflare Access or
  Authelia both work well as a forward-auth layer in front of Caddy.

---

# With thanks to:

[mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama](https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama) which I took heavy inspiration from.
