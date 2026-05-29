"""
Utilities for loading and classifying Hancock agent benchmark logs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from neurips25.utils.question_utils import detect_outcome_question_type

IMAGE_PATTERNS = [
    r"\bH&E\b",
    r"\bIHC\b",
    r"\bhistologic type\b",
    r"\bhistopatholog",
    r"\bslides?\b",
    r"\bimages?\b",
    r"\bWSI",
    r"\bCD3\b",
    r"\bCD8\b",
    r"\bCD163\b",
    r"\bmacrophage",
    r"\binfiltration front\b",
    r"\btumor center\b",
]

BLOOD_PATTERNS = [
    r"\bblood test",
    r"\bcoagulation\b",
    r"\brenal function\b",
    r"\bhematolog",
    r"\bpreoperative blood\b",
    r"\bplatelet\b",
    r"\bcreatinine\b",
    r"\binr\b",
    r"\bhemoglobin\b",
    r"\belectrolyte",
    r"\berythrocyte",
    r"\bleukocyte",
    r"\bwhite blood cell",
    r"\bgranulocyte",
    r"\bglucose level",
    r"\bglucose\b",
    r"\bpotassium\b",
    r"\bsodium\b",
    r"\bcalcium\b",
    r"\bchloride\b",
    r"\bhyponatremia\b",
    r"\bhyperkalemia\b",
    r"\beosinophil",
    r"\bperioperative\b",
]


def classify_question(question: str) -> str:
    """Return 'image', 'blood', 'prognosis', or 'other'."""
    outcome = detect_outcome_question_type(question)
    if outcome:
        return "prognosis"
    q_lower = question.lower()
    for pattern in IMAGE_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return "image"
    for pattern in BLOOD_PATTERNS:
        if re.search(pattern, q_lower):
            return "blood"
    return "other"


def is_text_only_category(category: str) -> bool:
    return category in ("blood", "prognosis")


def load_benchmark_registry(bench_path: str | Path) -> pd.DataFrame:
    """Build a registry of all benchmark questions with categories."""
    bench_path = Path(bench_path)
    with open(bench_path) as f:
        bench = json.loads(f.readline())

    rows = []
    for case_id, case_data in bench.items():
        q_idx = 0
        for item in case_data:
            if "question" not in item:
                continue
            question = item["question"]
            category = classify_question(question)
            rows.append({
                "case_id": str(case_id),
                "question_idx": q_idx,
                "question": question,
                "gold_answer": item["answer"],
                "category": category,
                "text_only": is_text_only_category(category),
            })
            q_idx += 1
    return pd.DataFrame(rows)


def load_chatlog(log_path: str | Path, model_name: str) -> list[dict[str, Any]]:
    """Load a single chatlog JSON and return question records (exclude conversation)."""
    with open(log_path) as f:
        data = json.load(f)
    records = []
    case_id = Path(log_path).stem.split("_chatlog_")[0]
    for i, entry in enumerate(data):
        if "question" not in entry:
            continue
        category = classify_question(entry["question"])
        pop_raw = entry.get("population_informed_answer")
        if isinstance(pop_raw, dict):
            pop_answer = pop_raw
        else:
            pop_answer = {}
        records.append({
            "case_id": case_id,
            "log_index": i,
            "model": model_name,
            "question": entry["question"],
            "gold_answer": entry.get("answer"),
            "response": entry.get("response"),
            "correct": entry.get("correct"),
            "question_time": entry.get("question_time"),
            "category": category,
            "text_only": is_text_only_category(category),
            "outcome_question_type": entry.get("outcome_question_type"),
            "confidence_self_reported": entry.get("confidence_self_reported"),
            "confidence_selected_prob": entry.get("confidence_selected_prob"),
            "confidence_choice_probs": entry.get("confidence_choice_probs"),
            "choices": entry.get("choices"),
            "patient_staging": entry.get("patient_staging"),
            "population_stats": entry.get("population_stats"),
            "population_informed_answer": pop_answer.get("answer") if pop_answer else None,
            "population_answer_letter": pop_answer.get("answer_letter") if pop_answer else None,
            "population_rationale": pop_answer.get("rationale") if pop_answer else None,
            "population_answer_matches_model": entry.get("population_answer_matches_model"),
            "files_accessed": entry.get("files_accessed", []),
        })
    return records


def load_model_logs(log_dir: str | Path, model_name: str) -> pd.DataFrame:
    """Load all chatlogs from a directory into a DataFrame."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return pd.DataFrame()

    all_records = []
    for log_path in sorted(log_dir.glob("*_chatlog_*.json")):
        all_records.extend(load_chatlog(log_path, model_name))
    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame(all_records)


def merge_with_registry(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Merge log DataFrame with benchmark registry on case_id + question text."""
    if df.empty:
        return df
    merged = df.merge(
        registry[["case_id", "question", "question_idx", "category", "text_only"]],
        on=["case_id", "question"],
        how="left",
        suffixes=("", "_bench"),
    )
    merged["category"] = merged["category"].fillna(merged.get("category_bench"))
    merged["text_only"] = merged["text_only"].fillna(merged.get("text_only_bench"))
    if "category_bench" in merged.columns:
        merged = merged.drop(columns=["category_bench", "text_only_bench"], errors="ignore")
    return merged


def population_informed_correct(row: pd.Series) -> bool | None:
    """Whether population-informed answer matches gold."""
    if pd.isna(row.get("population_answer_letter")) or pd.isna(row.get("gold_answer")):
        return None
    gold = str(row["gold_answer"]).strip()[0].upper()
    return row["population_answer_letter"].upper() == gold


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analysis columns."""
    if df.empty:
        return df
    out = df.copy()
    out["population_correct"] = out.apply(population_informed_correct, axis=1)
    out["model_follows_population"] = out["population_answer_matches_model"]
    return out
