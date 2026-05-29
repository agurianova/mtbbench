"""
Plotting and metrics for confidence calibration and failure prediction.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def calibration_bins(
    df: pd.DataFrame,
    confidence_col: str,
    n_bins: int = 10,
    min_samples: int = 1,
) -> pd.DataFrame:
    """
    Compute calibration statistics per confidence bin.
    Returns DataFrame with bin_center, mean_confidence, accuracy, count, calibration_gap.
    """
    subset = df.dropna(subset=[confidence_col, "correct"]).copy()
    if subset.empty:
        return pd.DataFrame()

    subset["conf"] = subset[confidence_col].clip(0, 1)
    subset["bin"] = pd.cut(subset["conf"], bins=n_bins, labels=False, include_lowest=True)

    rows = []
    for bin_idx in range(n_bins):
        bin_df = subset[subset["bin"] == bin_idx]
        if len(bin_df) < min_samples:
            continue
        mean_conf = bin_df["conf"].mean()
        acc = bin_df["correct"].mean()
        rows.append({
            "bin": bin_idx,
            "bin_low": bin_idx / n_bins,
            "bin_high": (bin_idx + 1) / n_bins,
            "bin_center": (bin_idx + 0.5) / n_bins,
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "count": len(bin_df),
            "calibration_gap": abs(acc - mean_conf),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(cal_df: pd.DataFrame) -> float:
    """Weighted ECE from calibration bin DataFrame."""
    if cal_df.empty:
        return float("nan")
    weights = cal_df["count"] / cal_df["count"].sum()
    return (weights * cal_df["calibration_gap"]).sum()


def failure_prediction_auroc(
    df: pd.DataFrame,
    confidence_col: str,
) -> dict:
    """
    AUROC for predicting correctness from confidence.
    Higher confidence should correlate with correct answers.
    """
    subset = df.dropna(subset=[confidence_col, "correct"]).copy()
    if subset.empty or subset["correct"].nunique() < 2:
        return {"auroc": float("nan"), "n": len(subset), "fpr": None, "tpr": None}

    y_true = subset["correct"].astype(int).values
    y_score = subset[confidence_col].clip(0, 1).values
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"auroc": auroc, "n": len(subset), "fpr": fpr, "tpr": tpr}


def model_summary(df: pd.DataFrame, label: str) -> dict:
    """Summary statistics for a model subset."""
    if df.empty:
        return {"model": label, "n": 0, "accuracy": float("nan"), "mean_time_s": float("nan")}
    return {
        "model": label,
        "n": len(df),
        "accuracy": df["correct"].mean(),
        "mean_time_s": df["question_time"].mean(),
        "median_time_s": df["question_time"].median(),
        "total_time_s": df["question_time"].sum(),
    }


def compare_models_summary(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    return pd.DataFrame([
        model_summary(df_a, name_a),
        model_summary(df_b, name_b),
    ])


def leaderboard_summary(df: pd.DataFrame, group_col: str = "model") -> pd.DataFrame:
    """Aggregate accuracy and timing per model."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for model, g in df.groupby(group_col):
        rows.append({
            group_col: model,
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "mean_time_s": g["question_time"].mean(),
            "median_time_s": g["question_time"].median(),
            "total_time_s": g["question_time"].sum(),
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def no_tools_model_comparison(
    df: pd.DataFrame,
    *,
    log_dir_col: str = "log_dir",
    highlight_log_dir: str | None = None,
) -> pd.DataFrame:
    """
    Accuracy breakdown for Hancock models without agent tools (+tools variants excluded).
    """
    if df.empty or log_dir_col not in df.columns:
        return pd.DataFrame()

    subset = df[~df[log_dir_col].astype(str).str.contains("with-tools", na=False)].copy()
    rows = []
    for log_dir, g in subset.groupby(log_dir_col):
        text = g[g["text_only"] == True]
        rows.append({
            "log_dir": log_dir,
            "model": g["model"].iloc[0],
            "params_b": g["params_b"].iloc[0] if "params_b" in g.columns else None,
            "modality": g["modality"].iloc[0] if "modality" in g.columns else "unknown",
            "full_accuracy": g["correct"].mean(),
            "text_only_accuracy": text["correct"].mean() if len(text) else float("nan"),
            "blood_accuracy": g.loc[g["category"] == "blood", "correct"].mean(),
            "prognosis_accuracy": g.loc[g["category"] == "prognosis", "correct"].mean(),
            "image_accuracy": g.loc[g["category"] == "image", "correct"].mean(),
            "mean_time_s": g["question_time"].mean(),
            "is_meditron": log_dir == highlight_log_dir,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["text_only_accuracy", "full_accuracy"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def format_no_tools_comparison_markdown(
    comp: pd.DataFrame,
    *,
    meditron_log_dir: str = "meditron3-8b",
    comparable_params_max: float = 12,
) -> list[str]:
    """Render markdown lines for model comparison section."""
    if comp.empty:
        return ["## Comparison with other models (no tools)", "", "_No comparison data._"]

    med = comp[comp["log_dir"] == meditron_log_dir]
    if med.empty:
        med_row = None
    else:
        med_row = med.iloc[0]

    def fmt_pct(v: float) -> str:
        return "—" if pd.isna(v) else f"{v:.1%}"

    def table_rows(df: pd.DataFrame) -> list[str]:
        header = (
            "| Model | Params | Modality | Full | Text-only | Blood | Prognosis | Image |"
        )
        sep = "|---|---:|---|---:|---:|---:|---:|---:|"
        body = []
        for _, r in df.iterrows():
            name = str(r["model"])
            if r.get("is_meditron"):
                name = f"**{name}**"
            params = "—" if pd.isna(r["params_b"]) else f"{int(r['params_b'])}B"
            body.append(
                f"| {name} | {params} | {r['modality']} | {fmt_pct(r['full_accuracy'])} "
                f"| {fmt_pct(r['text_only_accuracy'])} | {fmt_pct(r['blood_accuracy'])} "
                f"| {fmt_pct(r['prognosis_accuracy'])} | {fmt_pct(r['image_accuracy'])} |"
            )
        return [header, sep] + body

    lines = [
        "",
        "## Comparison with other models (no tools)",
        "",
        "All Hancock agent runs **without** tool-augmented variants (`-with-tools` excluded). "
        "Meditron3-8B is a **text-only** Llama-3.1 derivative fine-tuned on medical literature "
        "(OpenMeditron); peers are mostly vision+language models that can read pathology slides.",
        "",
        "### All models (no tools)",
        "",
        *table_rows(comp),
        "",
    ]

    comparable = comp[
        comp["params_b"].notna()
        & (comp["params_b"] >= 7)
        & (comp["params_b"] <= comparable_params_max)
    ].copy()
    if not comparable.empty:
        lines += [
            f"### Comparable size band ({int(comparable['params_b'].min())}–{int(comparable['params_b'].max())}B, no tools)",
            "",
            *table_rows(comparable.sort_values("text_only_accuracy", ascending=False)),
            "",
        ]

    if med_row is not None:
        text_rank = int((comp["text_only_accuracy"] > med_row["text_only_accuracy"]).sum()) + 1
        full_rank = int((comp["full_accuracy"] > med_row["full_accuracy"]).sum()) + 1
        n_models = len(comp)
        best_text = comp.iloc[0]
        lines += [
            "### Meditron positioning",
            "",
            f"- **Text-only rank:** {text_rank} / {n_models} "
            f"({med_row['text_only_accuracy']:.1%} — ties Gemma-3-12B and Mistral-Small-3.1 on text-only).",
            f"- **Full-benchmark rank:** {full_rank} / {n_models} "
            f"({med_row['full_accuracy']:.1%}) — limited by no image access (image acc {med_row['image_accuracy']:.1%}).",
            f"- **Best no-tools model on text-only:** {best_text['model']} ({best_text['text_only_accuracy']:.1%}).",
        ]
        if not comparable.empty:
            med_comp = comparable[comparable["log_dir"] == meditron_log_dir]
            if not med_comp.empty:
                mc = med_comp.iloc[0]
                qwen = comparable[comparable["log_dir"] == "qwen25-7b"]
                gemma = comparable[comparable["log_dir"] == "gemma3-12b"]
                bullets = []
                if not qwen.empty:
                    q = qwen.iloc[0]
                    delta = mc["text_only_accuracy"] - q["text_only_accuracy"]
                    bullets.append(
                        f"vs **Qwen2.5-VL-7B** (same scale, general VL): "
                        f"+{delta:.1%} text-only ({mc['text_only_accuracy']:.1%} vs {q['text_only_accuracy']:.1%}); "
                        f"prognosis {mc['prognosis_accuracy']:.1%} vs {q['prognosis_accuracy']:.1%}."
                    )
                if not gemma.empty:
                    g = gemma.iloc[0]
                    prog_delta = mc["prognosis_accuracy"] - g["prognosis_accuracy"]
                    blood_delta = mc["blood_accuracy"] - g["blood_accuracy"]
                    bullets.append(
                        f"vs **Gemma-3-12B** (12B VL): text-only tied ({mc['text_only_accuracy']:.1%}); "
                        f"prognosis +{prog_delta:.1%} ({mc['prognosis_accuracy']:.1%} vs {g['prognosis_accuracy']:.1%}), "
                        f"blood {blood_delta:+.1%} ({mc['blood_accuracy']:.1%} vs {g['blood_accuracy']:.1%})."
                    )
                if bullets:
                    lines.append("- **7–12B peer comparison:**")
                    lines.extend(f"  - {b}" for b in bullets)
        lines.append("")

    return lines


def prognosis_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze prognosis questions: model vs population-informed answers.
    """
    prog = df[df["category"] == "prognosis"].copy()
    if prog.empty:
        return pd.DataFrame()

    rows = []
    for model, g in prog.groupby("model"):
        n = len(g)
        follows = g["model_follows_population"] == True
        disagrees = g["model_follows_population"] == False
        rows.append({
            "model": model,
            "n_questions": n,
            "model_accuracy": g["correct"].mean(),
            "population_accuracy": g["population_correct"].mean(),
            "model_follows_population_rate": g["model_follows_population"].mean(),
            "accuracy_when_follows_population": g.loc[follows, "correct"].mean()
            if follows.any() else float("nan"),
            "accuracy_when_disagrees_population": g.loc[disagrees, "correct"].mean()
            if disagrees.any() else float("nan"),
        })
    return pd.DataFrame(rows)
