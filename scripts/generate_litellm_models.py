#!/usr/bin/env python3
"""Generate LiteLLM/llama-swap-models.yaml (model_list only) from every
models/*.yaml (real llama-swap partial-config). Included via `include:` in
LiteLLM/config.yaml, which stays static/hand-maintained.

Needs pyyaml: pip install -r scripts/requirements.txt (already present in
the config-generator compose service's own container -- only needed on the
host if you run this script directly instead).

Usage: python3 scripts/generate_litellm_models.py [--models-dir DIR] [--out FILE]
"""
import argparse
import glob
import os

import yaml

ANCHOR_HEADER = (
    "# GENERATED FILE — edit models/<name>.yaml then re-run\n"
    "# scripts/generate_litellm_models.py. Do not hand-edit.\n"
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


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--models-dir", default=os.path.join(root, "models"))
    ap.add_argument("--out", default=os.path.join(root, "LiteLLM", "llama-swap-models.yaml"))
    args = ap.parse_args()

    entries = []
    for path in sorted(glob.glob(os.path.join(args.models_dir, "*.yaml"))):
        with open(path) as f:
            doc = yaml.safe_load(f)
        for model_id, block in (doc.get("models") or {}).items():
            entries.append((model_id, model_info_for(block)))

    with open(args.out, "w") as f:
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

    print(f"Wrote {len(entries)} models to {args.out}")


if __name__ == "__main__":
    main()
