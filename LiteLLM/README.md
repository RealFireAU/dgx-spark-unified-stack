# LiteLLM model list generation

LiteLLM's `model_list` is driven by `models/*.yaml` (llama-swap's own
per-model config) rather than being hand-written into `config.yaml`. That
generation happens inside the litellm container itself, as a [worker
startup hook](https://docs.litellm.ai/docs/proxy/worker_startup_hooks)
(`LITELLM_WORKER_STARTUP_HOOKS`), configured in `docker-compose.yml`'s
`litellm` service — not a separate compose service or a manual script run.

## Files

- `config.yaml` — static, hand-maintained. Has no `include:` line. Add
  non-llama-swap providers directly here.
- `litellm_startup_hook.py` — generates the model list from `models/*.yaml`
  and wires it into the running config via `include:`.

## Why this isn't a single, direct generate-then-include step

litellm's CLI (`proxy_cli.py`, `run_server`) parses `--config` twice:

1. Once synchronously, before the FastAPI app (and therefore before any
   worker startup hook) exists, to pull out `litellm_settings`/
   `general_settings` (e.g. `json_logs`, `database_url`). This parse
   resolves `include:` if present, and raises `FileNotFoundError` if the
   included file doesn't exist yet.
2. Again inside `proxy_startup_event` (the FastAPI lifespan), which is
   what actually builds `llm_router`/`model_list`. `LITELLM_WORKER_STARTUP_HOOKS`
   runs immediately before this second parse, not the first.

Both parses read the same file path (whatever was passed to `--config`).
Because of (1), that path can't reference `llama-swap-models.yaml` at
container start — the file doesn't exist yet at that point. So:

- `config.yaml` (this directory, host-mounted read-only into the container
  at `/app/config.static.yaml`) never has an `include:` line, and parses
  fine on its own.
- The litellm service's `command:` copies that file to `/app/config.yaml`
  (container-local, writable) before `litellm` is exec'd — this has to
  happen before any Python/hook code runs, since parse (1) happens first.
- The startup hook then generates `llama-swap-models.yaml` and appends
  `include: [llama-swap-models.yaml]` to that `/app/config.yaml` copy,
  before parse (2) reads it.

Neither generated file (`/app/config.yaml`, `/app/llama-swap-models.yaml`)
is bind-mounted to the host — both exist only inside the container's own
filesystem, regenerated fresh on every start/restart.

## Regenerating the model list

`docker compose restart litellm` — LiteLLM has no hot-reload, so a change
to `models/*.yaml` needs a restart regardless of how the list is generated.
