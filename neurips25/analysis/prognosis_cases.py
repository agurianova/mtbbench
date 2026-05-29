"""
Prognosis case-study helpers for Meditron analysis.
"""

from __future__ import annotations

import pandas as pd


def classify_prognosis_outcomes(df_prog: pd.DataFrame) -> pd.DataFrame:
    """Add outcome category labels for prognosis rows."""
    out = df_prog.copy()
    mc = out["correct"].astype(bool)
    pc = out["population_correct"].astype(bool)
    follows = out["population_answer_matches_model"].astype(bool)

    def label(row):
        mc, pc, f = bool(row["correct"]), bool(row["population_correct"]), bool(row["population_answer_matches_model"])
        if mc and pc and f:
            return "success_aligned"
        if mc and not pc and not f:
            return "success_ignored_population"
        if not mc and pc and not f:
            return "missed_population_signal"
        if not mc and not pc and f:
            return "misled_by_population"
        if mc and pc and not f:
            return "success_despite_divergence"
        if not mc and pc and f:
            return "followed_population_still_wrong"
        return "other"

    out["outcome_category"] = out.apply(label, axis=1)
    return out


def summarize_prognosis_impact(df_prog: pd.DataFrame) -> pd.DataFrame:
    """Count prognosis outcome categories."""
    labeled = classify_prognosis_outcomes(df_prog)
    summary = (
        labeled.groupby("outcome_category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["pct"] = (summary["count"] / len(labeled) * 100).round(1)
    return summary


def format_case_study(row: pd.Series) -> str:
    """Format one prognosis row as readable case study text."""
    from neurips25.utils.survival_lookup import population_informed_answer

    staging = row.get("patient_staging") or {}
    stats = row.get("population_stats") or {}
    seer = stats.get("seer", {})
    tcga = stats.get("tcga", {})
    qtype = row.get("outcome_question_type")
    rationale = row.get("population_rationale", "")
    if stats and qtype:
        rationale = population_informed_answer(qtype, stats).get("rationale", rationale)

    lines = [
        f"**Case {row['case_id']} — {row.get('outcome_question_type', 'prognosis')}**",
        f"- Staging: {staging.get('primary_tumor_site')}, pT={staging.get('pT_stage')}, "
        f"pN={staging.get('pN_stage')}, HPV={staging.get('hpv_status')}, AJCC={staging.get('ajcc_stage_group')}",
        f"- SEER: 5y survival {seer.get('5_year_survival', 0):.1%}, 2y recurrence {seer.get('2_year_recurrence', 0):.1%}",
        f"- TCGA: 5y survival {tcga.get('5_year_survival', 0):.1%}, 2y recurrence {tcga.get('2_year_recurrence', 0):.1%}",
        f"- Population recommendation: {row.get('population_informed_answer')} — {rationale}",
        f"- Meditron answer: {row.get('response')} ({'correct' if row.get('correct') else 'incorrect'})",
        f"- Gold standard: {row.get('gold_answer')}",
        f"- Followed population stats: {row.get('population_answer_matches_model')}",
    ]
    return "\n".join(lines)


CASE_STUDY_KEYS = {
    "success_aligned": ("116", "survival"),
    "success_ignored_population": ("104", "recurrence"),
    "missed_population_signal": ("104", "survival"),
    "misled_by_population": ("296", "survival"),
}


def pick_case_studies(df_prog: pd.DataFrame) -> dict[str, pd.Series]:
    """Return named case studies for notebook/report."""
    studies = {}
    for name, (case_id, qtype) in CASE_STUDY_KEYS.items():
        match = df_prog[
            (df_prog["case_id"].astype(str) == str(case_id))
            & (df_prog["outcome_question_type"] == qtype)
        ]
        if not match.empty:
            studies[name] = match.iloc[0]
    return studies
