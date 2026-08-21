"""LiteLLM worker startup hook (LITELLM_WORKER_STARTUP_HOOKS=litellm_startup_hook:generate_models).

Generates the model list from models/*.yaml and wires it into config.yaml
via `include:`. See README.md in this directory for why.
"""
import fcntl
import glob
import os

import yaml

CONFIG_PATH = "/app/config.yaml"
MODELS_FILENAME = "llama-swap-models.yaml"
LOCK_PATH = "/app/.litellm_startup_hook.lock"

ANCHOR_HEADER = (
    "# GENERATED FILE — edit models/<name>.yaml then restart litellm.\n"
    "# Do not hand-edit as it will be discarded.\n"
)


def mode_for(caps):
    return "image_generation" if "image" in (caps.get("out") or []) else "chat"


def model_info_for(block):
    caps = block.get("capabilities")
    info = {}

    if caps:
        info["mode"] = mode_for(caps)
        ctx = caps.get("context")
        if ctx:
            info["max_input_tokens"] = ctx
            info["max_output_tokens"] = ctx
            info["max_tokens"] = ctx
        if "image" in (caps.get("in") or []):
            info["supports_vision"] = True
        if caps.get("tools"):
            info["supports_function_calling"] = True
            info["supports_tool_choice"] = True
            info["supports_parallel_function_calling"] = True
        info["supports_system_messages"] = True
        info["supports_native_streaming"] = True

    metadata = block.get("metadata")
    if metadata:
        # metadata may carry cost fields AND/OR explicit model_info
        # overrides (e.g. supports_reasoning) - merge verbatim, explicit
        # wins over anything derived above.
        info.update(metadata)

    return info


def indent(text, spaces):
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def generate(models_dir, out_path):
    entries = []
    for path in sorted(glob.glob(os.path.join(models_dir, "*.yaml"))):
        with open(path) as f:
            doc = yaml.safe_load(f)
        for model_id, block in (doc.get("models") or {}).items():
            entries.append((model_id, model_info_for(block)))

    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        f.write(ANCHOR_HEADER)
        f.write("\n")
        f.write("x-llama-swap-defaults: &llama_swap_defaults\n")
        f.write("  api_base: os.environ/LLAMA_SWAP_API_BASE\n")
        f.write("  api_key: os.environ/LLAMA_SWAP_API_KEY\n")
        f.write("\n")
        f.write("model_list:\n")
        for model_id, info in entries:
            f.write(f"  - model_name: {model_id}\n")
            f.write("    litellm_params:\n")
            f.write("      <<: *llama_swap_defaults\n")
            f.write(f"      model: openai/{model_id}\n")
            if info:
                f.write("    model_info:\n")
                dumped = yaml.dump(info, default_flow_style=False, sort_keys=False)
                f.write(indent(dumped, 6))
                f.write("\n")
            f.write("\n")
    # Atomic on POSIX: readers never observe a partial/truncated file,
    # and concurrent workers racing this rename just clobber, not interleave.
    os.replace(tmp_path, out_path)

    return len(entries)


def generate_models():
    # Serialize across worker processes: each worker runs this hook once at
    # startup, and without a lock, concurrent runs can interleave writes to
    # the shared models file / config.yaml (see LiteLLM/README.md).
    with open(LOCK_PATH, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            _generate_models_locked()
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _generate_models_locked():
    models_dir = os.environ.get("LITELLM_MODELS_DIR", "/app/models")
    out_path = os.environ.get("LITELLM_MODELS_OUT", f"/app/{MODELS_FILENAME}")
    count = generate(models_dir, out_path)
    print(f"[litellm_startup_hook] Wrote {count} models to {out_path}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    # Parse (not substring-match) so this can't false-positive on the
    # word "llama-swap-models.yaml" appearing in config.yaml's own
    # header comment -- comments are stripped by the YAML parser.
    if MODELS_FILENAME not in (config.get("include") or []):
        with open(CONFIG_PATH, "a") as f:
            f.write(f"\ninclude:\n  - {MODELS_FILENAME}\n")
        print(f"[litellm_startup_hook] Injected include: {MODELS_FILENAME} into {CONFIG_PATH}")
