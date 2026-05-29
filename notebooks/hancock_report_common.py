"""Shared helpers for Hancock Llama/Meditron report notebook and execute script."""

from __future__ import annotations

import re

import numpy as np

from neurips25.utils.question_utils import detect_outcome_question_type

BLOOD_SUB_PATTERNS = [
    ("coagulation", [
        r"coagulation", r"\binr\b", r"\bapt+t\b", r"thromboplastin",
        r"prothrombin", r"thrombin", r"bleeding", r"clotting", r"platelet",
    ]),
    ("renal", [r"renal", r"creatinine", r"\bgfr\b", r"glomerular", r"\burea\b"]),
    ("electrolytes", [
        r"electrolyte", r"\bpotassium\b", r"\bsodium\b", r"\bcalcium\b",
        r"\bchloride\b", r"\bmagnesium\b", r"hyponatremia", r"hyperkalemia",
    ]),
    ("hematology", [
        r"hemoglobin", r"hematocrit", r"anemia", r"erythrocyte", r"leukocyte",
        r"white blood", r"granulocyte", r"lymphocyte", r"eosinophil",
        r"hematolog", r"thrombocytopenia", r"thromboembolic", r"hematological",
    ]),
    ("glucose", [r"\bglucose\b", r"hyperglycemia", r"hypoglycemia"]),
]

# Display order for accuracy bar plot (image excluded).
SUBCATEGORIES = [
    "coagulation", "renal", "hematology", "electrolytes", "glucose",
    "survival", "recurrence",
]

SUBCAT_LABELS = {
    "coagulation": "Coagulation",
    "renal": "Renal function",
    "hematology": "Hematology",
    "electrolytes": "Electrolytes",
    "glucose": "Glucose",
    "survival": "Survival (2y)",
    "recurrence": "Recurrence (2y)",
}


def detect_blood_subcategory(question: str) -> str:
    q_lower = question.lower()
    for name, patterns in BLOOD_SUB_PATTERNS:
        for pat in patterns:
            if re.search(pat, q_lower):
                return name
    return "glucose"  # fallback for rare blood questions


def assign_question_subcategory(row) -> str | None:
    """Fine-grained subcategory; None for image (excluded from plots)."""
    cat = row["category"]
    if cat == "image":
        return None
    if cat == "prognosis":
        oqt = row.get("outcome_question_type")
        if oqt and str(oqt) not in ("nan", "None"):
            return str(oqt)
        return detect_outcome_question_type(row["question"]) or "prognosis"
    if cat == "blood":
        return detect_blood_subcategory(row["question"])
    return "other"


def annotate_overconfident(ax, grid, curve, perfect, text="Overconfident", prefer_high_conf=True):
    """Label overconfidence; by default arrow points to the high-confidence region (near 1)."""
    gap = perfect - curve
    over_mask = gap > 0.02
    if not over_mask.any():
        return
    candidates = np.where(over_mask)[0]
    if prefer_high_conf:
        i = int(candidates[np.argmax(grid[candidates])])
    else:
        i = int(np.argmax(np.where(over_mask, gap, -1.0)))
    x_pt = float(grid[i])
    y_mid = float((curve[i] + perfect[i]) / 2)
    ax.annotate(
        text,
        xy=(x_pt, y_mid),
        xytext=(max(0.05, x_pt - 0.32), max(0.08, y_mid - 0.18)),
        arrowprops=dict(arrowstyle="->", linewidth=0.9, color="#8b0000"),
        fontsize=9,
        color="#8b0000",
    )
