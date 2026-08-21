# models/ — per-model registration

`models/` itself is gitignored (real quants/paths are specific to what your
box has downloaded, not portable to a different machine). This directory
shows the schema with worked examples — copy one into your own
`models/<name>.yaml` and adjust:

- `example-llamacpp-model.yaml` — a GGUF text model via llama.cpp
- `example-vllm-model.yaml` — a safetensors text model via vLLM, with cost
  `metadata`
- `example-comfyui-model.yaml` — the special `comfyui_auto` entry (no
  `capabilities:` block, `unlisted: true`) that gives llama-swap's native
  ComfyUI integration a container to spawn
- `example-sd-cpp-model.yaml` — an image-generation model via
  stable-diffusion.cpp (`capabilities.out: [image]`), natively OpenAI- and
  Automatic1111-compatible, no shim needed

llama-swap runs with `-config llama-swap/config.yaml -config-dir models/
-watch-config`, which merges every `*.yaml` under `models/` into the config
and polls for changes every 2s — add a file to register a model, delete one
to unregister it, no restart needed. Each file is real llama-swap
partial-config: exactly one `models:` key, containing exactly one model.

After adding/removing a model:

```bash
docker compose restart litellm   # LiteLLM has no hot-reload, unlike llama-swap
```

LiteLLM's model list regenerates automatically on every start/restart — see
`LiteLLM/litellm_startup_hook.py`.

## Fields

- `capabilities.context` — the model's real max context length (from the
  GGUF header or `config.json`'s `max_position_embeddings`), not a guess —
  used to derive LiteLLM's `max_input_tokens`/`max_output_tokens`.
- `capabilities.tools` — whether *this server invocation* actually passes
  the flags for tool-calling (e.g. `--enable-auto-tool-choice` for vLLM),
  not just whether the underlying model supports it upstream.
- `metadata` — optional, arbitrary key/value data exposed through
  llama-swap's own API. Cost fields (`input_cost_per_token`,
  `output_cost_per_token`, generated via `scripts/estimate_model_cost.py`)
  live here and get copied into LiteLLM's `model_info` automatically by the
  generator. Any other key you set here also gets merged into
  `model_info`, e.g. `supports_reasoning: true` if you know a model
  supports it (the generator can't infer that on its own).
- `cmd`/`cmdStop` — use the shared `${llamacpp_base}`/`${vllm_base}` macros
  (defined in `llama-swap/config.yaml`) to avoid repeating the docker-run
  boilerplate. `${MODEL_ID}` is a real llama-swap macro (this file's model
  key), `${PORT}`/`${host}` are assigned per-boot. `${env.VAR}` pulls from
  the container's environment (e.g. `${env.LLM_ROOT_PATH}`,
  `${env.REPO_CONFIG_PATH}` — both already set on the `llama-swap` service
  in `docker-compose.yml`) so a model file never needs a hardcoded host path.
- `unlisted: true` — keeps a model out of `/v1/models` without removing it
  (see `example-comfyui-model.yaml`) — for entries that aren't a normal
  single-capability chat/completion model.
