"""Hancock extension log directory names under agent_logs_hancock/."""

from __future__ import annotations

# Default runs (no SEER/TCGA injection): plain model id.
LOG_DIRS_NO_POPULATION: dict[str, str] = {
    "Llama-3.1 8B": "llama31-8b",
    "Meditron 8B": "meditron3-8b",
}

# Runs with staging → TCGA/SEER lookup → cohort rule in the prompt.
POPULATION_SUFFIX = "_with-population"
LOG_DIRS_WITH_POPULATION: dict[str, str] = {
    label: f"{subdir}{POPULATION_SUFFIX}"
    for label, subdir in LOG_DIRS_NO_POPULATION.items()
}

ALL_EXTENSION_DIRS = set(LOG_DIRS_NO_POPULATION.values()) | set(LOG_DIRS_WITH_POPULATION.values())
