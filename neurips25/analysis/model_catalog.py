"""
Hancock benchmark model metadata (parameter counts and modality).
Sizes from official model cards / papers; see MODEL_SOURCES in notebook.
"""

from __future__ import annotations

from pathlib import Path

# log_dir_name -> metadata
HANCOCK_MODEL_CATALOG: dict[str, dict] = {
    "meditron3-8b": {
        "display_name": "Meditron3-8B",
        "params_b": 8,
        "modality": "text",
        "hf_id": "OpenMeditron/Meditron3-8B",
        "family": "Llama-3.1 medical",
    },
    "llama31-8b": {
        "display_name": "Llama-3.1 8B Instruct (Unsloth)",
        "params_b": 8,
        "modality": "text",
        "hf_id": "unsloth/Llama-3.1-8B-Instruct",
        "family": "Llama-3.1",
    },
    "qwen25-7b": {
        "display_name": "Qwen2.5-VL-7B",
        "params_b": 7,
        "modality": "vision+text",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "Qwen2.5-VL",
    },
    "qwen25-7b-with-tools": {
        "display_name": "Qwen2.5-VL-7B (+tools)",
        "params_b": 7,
        "modality": "vision+text",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "Qwen2.5-VL",
    },
    "qwen25-32b": {
        "display_name": "Qwen2.5-VL-32B",
        "params_b": 32,
        "modality": "vision+text",
        "hf_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "family": "Qwen2.5-VL",
    },
    "qwen25-32b-with-tools": {
        "display_name": "Qwen2.5-VL-32B (+tools)",
        "params_b": 32,
        "modality": "vision+text",
        "hf_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "family": "Qwen2.5-VL",
    },
    "gemma3-12b": {
        "display_name": "Gemma-3-12B",
        "params_b": 12,
        "modality": "vision+text",
        "hf_id": "google/gemma-3-12b-it",
        "family": "Gemma 3",
    },
    "gemma3-12b-with-tools": {
        "display_name": "Gemma-3-12B (+tools)",
        "params_b": 12,
        "modality": "vision+text",
        "hf_id": "google/gemma-3-12b-it",
        "family": "Gemma 3",
    },
    "gemma3-27b": {
        "display_name": "Gemma-3-27B",
        "params_b": 27,
        "modality": "vision+text",
        "hf_id": "google/gemma-3-27b-it",
        "family": "Gemma 3",
    },
    "gemma3-27b-with-tools": {
        "display_name": "Gemma-3-27B (+tools)",
        "params_b": 27,
        "modality": "vision+text",
        "hf_id": "google/gemma-3-27b-it",
        "family": "Gemma 3",
    },
    "internvl3-38b": {
        "display_name": "InternVL3-38B",
        "params_b": 38,
        "modality": "vision+text",
        "hf_id": "OpenGVLab/InternVL3-38B",
        "family": "InternVL3",
    },
    "internvl3-38b-with-tools": {
        "display_name": "InternVL3-38B (+tools)",
        "params_b": 38,
        "modality": "vision+text",
        "hf_id": "OpenGVLab/InternVL3-38B",
        "family": "InternVL3",
    },
    "internvl3-78b": {
        "display_name": "InternVL3-78B",
        "params_b": 78,
        "modality": "vision+text",
        "hf_id": "OpenGVLab/InternVL3-78B",
        "family": "InternVL3",
    },
    "llama90b": {
        "display_name": "Llama-3.2-90B-Vision",
        "params_b": 90,
        "modality": "vision+text",
        "hf_id": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "family": "Llama 3.2",
    },
    "mistralsmall": {
        "display_name": "Mistral-Small-3.1-24B",
        "params_b": 24,
        "modality": "vision+text",
        "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "family": "Mistral",
    },
    "gpt4o": {
        "display_name": "GPT-4o",
        "params_b": None,
        "modality": "vision+text",
        "hf_id": "OpenAI API",
        "family": "GPT-4o",
    },
    "o4-mini": {
        "display_name": "o4-mini",
        "params_b": None,
        "modality": "vision+text",
        "hf_id": "OpenAI API",
        "family": "o-series",
    },
}

MEDITRON_LOG_KEY = "meditron3-8b"


def catalog_row(log_dir_name: str) -> dict:
    meta = HANCOCK_MODEL_CATALOG.get(log_dir_name, {})
    return {
        "log_dir": log_dir_name,
        "display_name": meta.get("display_name", log_dir_name),
        "params_b": meta.get("params_b"),
        "modality": meta.get("modality", "unknown"),
        "hf_id": meta.get("hf_id", ""),
        "family": meta.get("family", ""),
    }


def discover_log_dirs(log_root: str | Path) -> list[Path]:
    log_root = Path(log_root)
    dirs = []
    for d in sorted(log_root.iterdir()):
        if d.is_dir() and list(d.glob("*_chatlog_*.json")):
            dirs.append(d)
    return dirs
