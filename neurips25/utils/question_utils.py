import json
import os
import re
from pathlib import Path
from typing import Any


SURVIVAL_PATTERNS = [
    r"alive in \d+ years?",
    r"will (?:the patient )?be alive",
    r"survive.*years?",
]
RECURRENCE_PATTERNS = [
    r"recurrence in \d+ years?",
    r"will.*recurrence",
    r"cancer will have a recurrence",
]


def detect_outcome_question_type(question: str) -> str | None:
    """Return 'survival', 'recurrence', or None."""
    q_lower = question.lower()
    for pattern in SURVIVAL_PATTERNS:
        if re.search(pattern, q_lower):
            return "survival"
    for pattern in RECURRENCE_PATTERNS:
        if re.search(pattern, q_lower):
            return "recurrence"
    return None


def load_hancock_patient_data(case_id: str, cases_path: str = "data/hancock/cases") -> dict[str, Any]:
    """Load pathological and clinical JSON for a Hancock case."""
    case_dir = Path(cases_path) / str(case_id)
    result: dict[str, Any] = {}
    for filename, key in [
        ("patient_pathological_data.json", "pathological"),
        ("patient_clinical_data.json", "clinical"),
    ]:
        filepath = case_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                result[key] = json.load(f)
    return result
