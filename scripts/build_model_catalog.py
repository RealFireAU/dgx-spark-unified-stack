#!/usr/bin/env python3
"""
Build a metadata catalog for every model file on disk, so real GGUF-header
facts (architecture, real context length, quant type, vision capability,
tool-calling support) are looked up once and recorded, instead of manually
re-reading headers with the `gguf` package each time a model gets audited.

GGUF metadata comes straight from the file's own KV header via GGUFReader
(no tensor data loaded, fast even for 80GB+ files). Safetensors metadata
comes from each model directory's config.json/tokenizer_config.json.

Usage:
    python3 scripts/build_model_catalog.py [--gguf-root PATH] [--safetensors-root PATH]

Defaults to $LLM_ROOT_PATH/gguf and $LLM_ROOT_PATH/safetensors (see .env).
Output: scripts/results/model_catalog.json
"""
import argparse
import json
import os
import sys

from gguf import GGUFReader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ROOT_PATH = os.environ.get("LLM_ROOT_PATH", os.path.expanduser("~/LLMs"))
DEFAULT_GGUF_ROOT = os.path.join(LLM_ROOT_PATH, "gguf")
DEFAULT_SAFETENSORS_ROOT = os.path.join(LLM_ROOT_PATH, "safetensors")
CATALOG_PATH = os.path.join(SCRIPT_DIR, "results", "model_catalog.json")


def field_value(reader, key):
    f = reader.get_field(key)
    if f is None:
        return None
    try:
        return f.contents()
    except Exception:
        return None


def read_gguf_metadata(path):
    try:
        r = GGUFReader(path)
    except Exception as e:
        return {"error": str(e)}

    arch = field_value(r, "general.architecture")
    is_vision_proj = field_value(r, "clip.has_vision_encoder") is not None or "mmproj" in os.path.basename(path).lower()

    meta = {
        "architecture": arch,
        "is_vision_projector": bool(is_vision_proj),
        "name": field_value(r, "general.name"),
        "size_label": field_value(r, "general.size_label"),
        "file_type": field_value(r, "general.file_type"),
        "quantization_version": field_value(r, "general.quantization_version"),
        "license": field_value(r, "general.license"),
        "base_model_name": field_value(r, "general.base_model.0.name"),
        "base_model_org": field_value(r, "general.base_model.0.organization"),
        "finetune": field_value(r, "general.finetune"),
    }

    if is_vision_proj:
        meta["vision_projector_type"] = field_value(r, "clip.projector_type")
        meta["vision_image_size"] = field_value(r, "clip.vision.image_size")
    elif arch:
        # Context length and attention shape live under an arch-prefixed key,
        # e.g. "qwen2.context_length" -- not a fixed key name across archs.
        meta["context_length"] = field_value(r, f"{arch}.context_length")
        meta["embedding_length"] = field_value(r, f"{arch}.embedding_length")
        meta["block_count"] = field_value(r, f"{arch}.block_count")
        meta["head_count"] = field_value(r, f"{arch}.attention.head_count")
        meta["head_count_kv"] = field_value(r, f"{arch}.attention.head_count_kv")
        meta["expert_count"] = field_value(r, f"{arch}.expert_count")
        meta["expert_used_count"] = field_value(r, f"{arch}.expert_used_count")

        chat_template = field_value(r, "tokenizer.chat_template")
        if chat_template:
            template_lower = chat_template.lower()
            meta["has_chat_template"] = True
            # Heuristic, not a guarantee: presence of tool_call markup in the
            # embedded chat template is the closest signal available from the
            # GGUF header alone for whether this model was trained/templated
            # for tool calling.
            meta["chat_template_mentions_tool_calls"] = "tool_call" in template_lower or "<tools>" in template_lower
        else:
            meta["has_chat_template"] = False

    return meta


def read_safetensors_metadata(model_dir):
    meta = {}
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            meta["architecture"] = (cfg.get("architectures") or [None])[0]
            meta["model_type"] = cfg.get("model_type")
            meta["context_length"] = cfg.get("max_position_embeddings")
            meta["quantization_config"] = cfg.get("quantization_config", {}).get("quant_method") if cfg.get("quantization_config") else None
            meta["vision_config_present"] = "vision_config" in cfg
        except Exception as e:
            meta["config_error"] = str(e)

    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        try:
            with open(tok_cfg_path) as f:
                tok_cfg = json.load(f)
            chat_template = tok_cfg.get("chat_template")
            if isinstance(chat_template, str):
                template_lower = chat_template.lower()
                meta["has_chat_template"] = True
                meta["chat_template_mentions_tool_calls"] = "tool_call" in template_lower or "<tools>" in template_lower
            else:
                meta["has_chat_template"] = bool(chat_template)
        except Exception as e:
            meta["tokenizer_config_error"] = str(e)

    return meta


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def scan_gguf(root):
    catalog = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".gguf"):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, root)
            print(f"  {rel_path}")
            entry = read_gguf_metadata(full_path)
            entry["size_bytes"] = os.path.getsize(full_path)
            entry["size_human"] = human_size(entry["size_bytes"])
            entry["path"] = full_path
            catalog[rel_path] = entry
    return catalog


def scan_safetensors(root):
    catalog = {}
    if not os.path.isdir(root):
        return catalog
    for org in os.listdir(root):
        org_path = os.path.join(root, org)
        if not os.path.isdir(org_path):
            continue
        for model_name in os.listdir(org_path):
            model_dir = os.path.join(org_path, model_name)
            if not os.path.isdir(model_dir):
                continue
            rel_path = os.path.join(org, model_name)
            print(f"  {rel_path}")
            entry = read_safetensors_metadata(model_dir)
            total_bytes = 0
            for f in os.listdir(model_dir):
                if f.endswith(".safetensors"):
                    total_bytes += os.path.getsize(os.path.join(model_dir, f))
            entry["size_bytes"] = total_bytes
            entry["size_human"] = human_size(total_bytes)
            entry["path"] = model_dir
            catalog[rel_path] = entry
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf-root", default=DEFAULT_GGUF_ROOT)
    parser.add_argument("--safetensors-root", default=DEFAULT_SAFETENSORS_ROOT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)

    print(f"Scanning GGUF files under {args.gguf_root} ...")
    gguf_catalog = scan_gguf(args.gguf_root)

    print(f"Scanning safetensors models under {args.safetensors_root} ...")
    safetensors_catalog = scan_safetensors(args.safetensors_root)

    output = {
        "gguf": gguf_catalog,
        "safetensors": safetensors_catalog,
    }
    with open(CATALOG_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCatalog written to {CATALOG_PATH}")
    print(f"  {len(gguf_catalog)} GGUF files, {len(safetensors_catalog)} safetensors models")


if __name__ == "__main__":
    main()
