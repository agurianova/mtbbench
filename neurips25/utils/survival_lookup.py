import json
from pathlib import Path
from typing import Any

from neurips25.utils.staging import _normalize_hpv, _normalize_site


REFERENCE_PATH = Path(__file__).resolve().parent.parent / "data" / "hnc_survival_reference.json"


def _load_reference() -> dict[str, Any]:
    with open(REFERENCE_PATH, "r") as f:
        return json.load(f)


def _get_cohort_stats(reference: dict, source: str, site: str, hpv: str, stage: str) -> dict[str, float]:
    source_data = reference.get(source, {})
    site_data = source_data.get(site) or source_data.get("other", {})
    hpv_key = hpv if hpv in site_data else "hpv_unknown"
    cohort = site_data.get(hpv_key) or site_data.get("hpv_unknown", {})
    return cohort.get(stage, cohort.get("stage_III", {"5_year_survival": 0.50, "2_year_recurrence": 0.30}))


def lookup_population_stats(staging: dict[str, Any]) -> dict[str, Any]:
    """
    Look up SEER and TCGA survival/recurrence rates for a staged patient cohort.
    """
    reference = _load_reference()
    site = _normalize_site(staging.get("primary_tumor_site", ""))
    hpv = staging.get("hpv_status") or _normalize_hpv(staging.get("hpv_association_p16", ""))
    stage = staging.get("ajcc_stage_group", "stage_III")

    seer_stats = _get_cohort_stats(reference, "seer", site, hpv, stage)
    tcga_stats = _get_cohort_stats(reference, "tcga", site, hpv, stage)

    survival_threshold = reference.get("metadata", {}).get("survival_threshold", 0.60)
    recurrence_threshold = reference.get("metadata", {}).get("recurrence_threshold", 0.60)

    avg_survival = (seer_stats["5_year_survival"] + tcga_stats["5_year_survival"]) / 2
    avg_recurrence = (seer_stats["2_year_recurrence"] + tcga_stats["2_year_recurrence"]) / 2

    return {
        "cohort_key": {"site": site, "hpv_status": hpv, "ajcc_stage_group": stage},
        "seer": {
            "5_year_survival": seer_stats["5_year_survival"],
            "2_year_recurrence": seer_stats["2_year_recurrence"],
        },
        "tcga": {
            "5_year_survival": tcga_stats["5_year_survival"],
            "2_year_recurrence": tcga_stats["2_year_recurrence"],
        },
        "expected_5_year_survival": avg_survival,
        "expected_2_year_recurrence": avg_recurrence,
        "survival_threshold": survival_threshold,
        "recurrence_threshold": recurrence_threshold,
    }


def population_informed_answer(question_type: str, population_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Derive a population-statistics-informed answer for survival or recurrence questions.

    Survival: if expected 5-year survival >= 60%, answer Yes (A).
    Recurrence: if expected 2-year recurrence rate >= 60%, answer Yes (A); otherwise No (B).
    """
    if question_type == "survival":
        rate = population_stats["expected_5_year_survival"]
        threshold = population_stats["survival_threshold"]
        answer_letter = "A" if rate >= threshold else "B"
        answer_text = "A) Yes" if answer_letter == "A" else "B) No"
        outcome = "patient likely alive" if answer_letter == "A" else "patient unlikely to survive at 5 years"
        rationale = (
            f"Expected 5-year survival {rate:.1%} is "
            f"{'>=' if rate >= threshold else '<'} {threshold:.0%} threshold → {outcome}"
        )
    elif question_type == "recurrence":
        rate = population_stats["expected_2_year_recurrence"]
        threshold = population_stats["recurrence_threshold"]
        if rate >= threshold:
            answer_letter = "A"
            answer_text = "A) Yes"
            rationale = f"Expected 2-year recurrence {rate:.1%} >= {threshold:.0%} → recurrence expected"
        else:
            answer_letter = "B"
            answer_text = "B) No"
            rationale = f"Expected 2-year recurrence {rate:.1%} < {threshold:.0%} → recurrence not expected"
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    return {
        "question_type": question_type,
        "answer_letter": answer_letter,
        "answer": answer_text,
        "rationale": rationale,
    }


def format_population_context(staging: dict[str, Any], population_stats: dict[str, Any], question_type: str) -> str:
    """
    Format population statistics as context to inject before survival/recurrence questions.
    """
    informed = population_informed_answer(question_type, population_stats)
    return (
        f"Population reference data for this patient's cohort "
        f"({staging.get('primary_tumor_site')}, {staging.get('hpv_status')}, {staging.get('ajcc_stage_group')}):\n"
        f"- Patient staging: pT={staging.get('pT_stage')}, pN={staging.get('pN_stage')}, "
        f"HPV={staging.get('hpv_status')}, AJCC group={staging.get('ajcc_stage_group')}\n"
        f"- SEER expected 5-year survival: {population_stats['seer']['5_year_survival']:.1%}, "
        f"2-year recurrence: {population_stats['seer']['2_year_recurrence']:.1%}\n"
        f"- TCGA expected 5-year survival: {population_stats['tcga']['5_year_survival']:.1%}, "
        f"2-year recurrence: {population_stats['tcga']['2_year_recurrence']:.1%}\n"
        f"- Combined expected 5-year survival: {population_stats['expected_5_year_survival']:.1%}\n"
        f"- Combined expected 2-year recurrence: {population_stats['expected_2_year_recurrence']:.1%}\n"
        f"- Population-informed recommendation: {informed['answer']} ({informed['rationale']})\n"
        f"Use these population statistics together with individual patient data when answering.\n"
    )
