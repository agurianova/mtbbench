#!/usr/bin/env python3
"""Generate Hancock extension figures for the research proposal."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from hancock_report_common import SUBCATEGORIES, SUBCAT_LABELS, assign_question_subcategory
from log_paths import LOG_DIRS_NO_POPULATION, LOG_DIRS_WITH_POPULATION, POPULATION_SUFFIX
from neurips25.analysis.log_loader import (
    add_derived_columns,
    load_benchmark_registry,
    load_model_logs,
    merge_with_registry,
)
from neurips25.analysis.metrics import (
    calibration_bins,
    expected_calibration_error,
    failure_prediction_auroc,
    prognosis_analysis,
)
from neurips25.analysis.model_catalog import catalog_row, discover_log_dirs

FIG = ROOT / "notebooks" / "figures"
MODELS = list(LOG_DIRS_NO_POPULATION.keys())
MODEL_SHORT = {"Llama-3.1 8B": "Llama 3.1", "Meditron 8B": "Meditron"}
MODEL_COLORS = {"Llama-3.1 8B": "#3D5A80", "Meditron 8B": "#B85042"}
HIGHLIGHT_COLORS = MODEL_COLORS
BASELINE_FILL = "#E6E6E6"
ERR_COLOR = "#6B6B6B"
LOGPROBS_COL = "confidence_selected_prob"
MIN_SUBCAT_N = 10
AUROC_TRIGGER = 0.7
BLOOD_SUBCATS = ["coagulation", "renal", "hematology", "electrolytes"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save_fig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_image(img: Image.Image, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    img.save(FIG / f"{name}.png")


def load_text_only(log_dirs: list[tuple[str, str]]) -> pd.DataFrame:
    registry = load_benchmark_registry(ROOT / "data/questions_hancock_bench.json")
    frames = [
        load_model_logs(ROOT / "agent_logs_hancock" / subdir, label)
        for label, subdir in log_dirs
    ]
    df = add_derived_columns(merge_with_registry(pd.concat(frames, ignore_index=True), registry))
    df["subcategory"] = df.apply(assign_question_subcategory, axis=1)
    return df[df["text_only"]].copy()


def load_prognosis(log_dirs: list[tuple[str, str]], *, require_population_fields: bool = True) -> pd.DataFrame:
    registry = load_benchmark_registry(ROOT / "data/questions_hancock_bench.json")
    frames = [
        load_model_logs(ROOT / "agent_logs_hancock" / subdir, label)
        for label, subdir in log_dirs
    ]
    df = add_derived_columns(merge_with_registry(pd.concat(frames, ignore_index=True), registry))
    prog = df[df["category"] == "prognosis"].copy()
    if require_population_fields:
        prog = prog.dropna(subset=["population_answer_letter"])
    return prog


def metrics_for_subset(sub: pd.DataFrame, conf_col: str) -> dict:
    s = sub.dropna(subset=[conf_col, "correct"])
    if s.empty:
        return {"ece": np.nan, "cal": pd.DataFrame()}
    cal = calibration_bins(s, conf_col)
    return {
        "ece": expected_calibration_error(cal) if not cal.empty else np.nan,
        "cal": cal,
    }


def bootstrap_accuracy_ci(correct, n_boot: int = 2000, seed: int = 42):
    v = np.asarray(correct, dtype=float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, n, replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


# --- Experiment 1 ---

def build_model_list(log_root: Path) -> list[dict]:
    seen_labels = set()
    entries = []
    for label, log_dir in LOG_DIRS_NO_POPULATION.items():
        entries.append({"label": label, "log_dir": log_dir, "params_b": 8, "highlight": True})
        seen_labels.add(label)
    skip = set(LOG_DIRS_WITH_POPULATION.values())
    for path in discover_log_dirs(log_root):
        name = path.name
        if name in skip or name.endswith(POPULATION_SUFFIX) or "with-tools" in name:
            continue
        meta = catalog_row(name)
        label = meta["display_name"]
        if label in seen_labels:
            continue
        params = meta["params_b"]
        entries.append({
            "label": label,
            "log_dir": name,
            "params_b": params if params is not None else 999,
            "highlight": False,
        })
        seen_labels.add(label)
    return entries


def short_label(name: str) -> str:
    shortcuts = {
        "Qwen2.5-VL-7B": "Qwen2.5-VL 7B",
        "Qwen2.5-VL-32B": "Qwen2.5-VL 32B",
        "Llama-3.2-90B-Vision": "Llama 3.2 90B",
        "InternVL3-38B": "InternVL3 38B",
        "InternVL3-78B": "InternVL3 78B",
        "Gemma-3-12B": "Gemma 3 12B",
        "Gemma-3-27B": "Gemma 3 27B",
        "Mistral-Small-3.1-24B": "Mistral Small 24B",
    }
    return shortcuts.get(name, name)


def fig_medical_llm_overall() -> None:
    registry = load_benchmark_registry(ROOT / "data/questions_hancock_bench.json")
    log_root = ROOT / "agent_logs_hancock"
    model_list = build_model_list(log_root)
    frames = []
    for ent in model_list:
        df_m = load_model_logs(log_root / ent["log_dir"], ent["label"])
        if not df_m.empty:
            frames.append(df_m)
    df = add_derived_columns(merge_with_registry(pd.concat(frames, ignore_index=True), registry))
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df_text = df[df["text_only"]].dropna(subset=["correct"]).copy()
    rows = []
    for ent in model_list:
        sub = df_text[df_text["model"] == ent["label"]]
        if sub.empty:
            continue
        acc, lo, hi, _ = bootstrap_accuracy_ci(sub["correct"].values)
        rows.append({
            "model": ent["label"],
            "params_b": ent["params_b"],
            "params_label": f"{ent['params_b']}B" if ent["params_b"] < 100 else "API",
            "highlight": ent["highlight"],
            "accuracy": acc,
            "ci_low": lo,
            "ci_high": hi,
        })
    acc_df = pd.DataFrame(rows).sort_values(["params_b", "model"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(acc_df))
    colors = [HIGHLIGHT_COLORS[r["model"]] if r["highlight"] else BASELINE_FILL for _, r in acc_df.iterrows()]
    ax.bar(
        x, acc_df["accuracy"], width=0.62, color=colors, edgecolor="none", zorder=3,
        yerr=[acc_df["accuracy"] - acc_df["ci_low"], acc_df["ci_high"] - acc_df["accuracy"]],
        capsize=2.5, error_kw={"elinewidth": 0.8, "ecolor": ERR_COLOR, "capthick": 0.8},
    )
    for i, (_, r) in enumerate(acc_df.iterrows()):
        if r["highlight"]:
            ax.text(i, min(r["ci_high"] + 0.02, 0.98), f"{r['accuracy']:.0%}",
                    ha="center", va="bottom", fontsize=8, color="#2A2A2A")
    ix_llama = acc_df.index[acc_df["model"] == "Llama-3.1 8B"].tolist()
    ix_med = acc_df.index[acc_df["model"] == "Meditron 8B"].tolist()
    if ix_llama and ix_med:
        i0, i1 = sorted([ix_llama[0], ix_med[0]])
        dpp = (acc_df.loc[ix_med[0], "accuracy"] - acc_df.loc[ix_llama[0], "accuracy"]) * 100
        y_top = acc_df.loc[[i0, i1], "ci_high"].max() + 0.11
        ax.annotate("", xy=(i1, y_top), xytext=(i0, y_top),
                    arrowprops=dict(arrowstyle="-", color="#404040", lw=0.9))
        ax.text((i0 + i1) / 2, y_top + 0.015, f"{dpp:+.1f} pp", ha="center", va="bottom", fontsize=8, color="#404040")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{short_label(r['model'])}\n({r['params_label']})" for _, r in acc_df.iterrows()],
        rotation=40, ha="right", fontsize=7.5,
    )
    ax.set_ylabel("Text-only accuracy")
    ax.set_ylim(0, min(1.0, acc_df["ci_high"].max() + 0.16))
    ax.set_xlim(-0.6, len(acc_df) - 0.4)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#CCCCCC", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=HIGHLIGHT_COLORS["Llama-3.1 8B"], label="Llama 3.1 8B"),
            mpatches.Patch(facecolor=HIGHLIGHT_COLORS["Meditron 8B"], label="Meditron 8B"),
            mpatches.Patch(facecolor=BASELINE_FILL, label="Other models"),
        ],
        loc="upper left", frameon=False, fontsize=7.5,
    )
    ax.set_title("Hancock — text-only questions (95% CI; no cohort stats in prompt)", loc="left", fontsize=9, pad=10)
    fig.tight_layout()
    save_fig(fig, "medical_llm_text_only_overall")


# --- Experiment 2: calibration ---

def annotate_overconfident(ax, grid, curve, perfect, n_arrows: int = 2) -> None:
    gap = np.where(curve < perfect, perfect - curve, 0.0)
    if gap.max() <= 0:
        return
    for k, idx in enumerate(np.argsort(gap)[-n_arrows:][::-1]):
        if gap[idx] <= 0:
            continue
        x, y_lo, y_hi = grid[idx], curve[idx], perfect[idx]
        dy = (y_hi - y_lo) * 0.35
        ax.annotate(
            "Overconfident" if k == 0 else "",
            xy=(x, y_lo + dy),
            xytext=(x + 0.14 - 0.06 * k, y_lo - 0.16 - 0.05 * k),
            fontsize=7,
            color="#6B3A36",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#6B3A36", lw=1.1, shrinkA=0, shrinkB=2),
        )


def fig_calibration_reliability() -> None:
    df_text = load_text_only(list(LOG_DIRS_NO_POPULATION.items()))

    def plot_reliability(ax, sub, color, title):
        m = metrics_for_subset(sub, LOGPROBS_COL)
        cal = m["cal"]
        if len(cal) < 3:
            return
        x_pts = cal["mean_confidence"].values
        y_pts = cal["accuracy"].values
        order = np.argsort(x_pts)
        x_pts, y_pts = x_pts[order], y_pts[order]
        grid = np.linspace(max(0.02, x_pts.min()), min(0.98, x_pts.max()), 200)
        curve = PchipInterpolator(x_pts, y_pts, extrapolate=False)(grid)
        perfect = grid
        ax.fill_between(grid, curve, perfect, where=curve < perfect, color="#E8C4C0", alpha=0.5)
        ax.plot(grid, perfect, "--", color="#888", linewidth=1)
        ax.plot(grid, curve, color=color, linewidth=2)
        ax.scatter(x_pts, y_pts, s=22, c=color, edgecolors="white", linewidths=0.4)
        annotate_overconfident(ax, grid, curve, perfect)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{title}\nECE = {m['ece']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, model in zip(axes, MODELS):
        plot_reliability(ax, df_text[df_text["model"] == model], MODEL_COLORS[model],
                         model.replace("Llama-3.1 8B", "Llama 3.1"))
    fig.suptitle("Reliability — log-probability confidence", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "cal_reliability_logprobs")


def fig_calibration_auroc() -> None:
    df_text = load_text_only(list(LOG_DIRS_NO_POPULATION.items()))
    llama_rows = []
    for sc in SUBCATEGORIES:
        if sc == "glucose":
            continue
        sub = df_text[(df_text["model"] == "Llama-3.1 8B") & (df_text["subcategory"] == sc)]
        s = sub.dropna(subset=[LOGPROBS_COL, "correct"])
        if len(s) < MIN_SUBCAT_N or s["correct"].nunique() < 2:
            continue
        auroc = failure_prediction_auroc(s, LOGPROBS_COL)["auroc"]
        llama_rows.append({"subcategory": SUBCAT_LABELS[sc], "key": sc, "auroc": auroc})
    llama_sub = pd.DataFrame(llama_rows).sort_values("auroc", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ":", color="#AAA", linewidth=1, label="Random (0.5)")
    for model in MODELS:
        sub = df_text[df_text["model"] == model].dropna(subset=[LOGPROBS_COL, "correct"])
        res = failure_prediction_auroc(sub, LOGPROBS_COL)
        ax.plot(
            res["fpr"], res["tpr"], color=MODEL_COLORS[model], lw=2,
            label=f"{model.replace('Llama-3.1 8B', 'Llama 3.1')} ({res['auroc']:.2f})",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — log-probability", loc="left", fontsize=9, pad=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    y = np.arange(len(llama_sub))
    colors = [
        MODEL_COLORS["Llama-3.1 8B"] if a >= AUROC_TRIGGER and k in BLOOD_SUBCATS else "#C8C8C8"
        for a, k in zip(llama_sub["auroc"], llama_sub["key"])
    ]
    ax.barh(y, llama_sub["auroc"], color=colors, height=0.55)
    ax.axvline(AUROC_TRIGGER, color="#404040", linestyle="--", linewidth=1, label=f"Trigger ({AUROC_TRIGGER})")
    ax.set_yticks(y)
    ax.set_yticklabels(llama_sub["subcategory"], fontsize=8)
    ax.set_xlabel("AUROC (log-probability)")
    ax.set_xlim(0.35, 0.95)
    ax.set_title("Llama 3.1 — by subcategory", loc="left", fontsize=9, pad=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.xaxis.grid(True, linestyle="-", linewidth=0.35, color="#CCC")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Failure prediction from confidence (text-only)", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, "cal_auroc_panel")


# --- Experiment 3: accuracy when following vs disagreeing with cohort rule ---

def fig_cohort_usage() -> None:
    prog = load_prognosis(list(LOG_DIRS_WITH_POPULATION.items()))
    summary = prognosis_analysis(prog)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    x = np.arange(len(MODELS))
    w = 0.28
    acc_follow = [
        summary.loc[summary["model"] == m, "accuracy_when_follows_population"].iloc[0]
        for m in MODELS
    ]
    acc_disagree = [
        summary.loc[summary["model"] == m, "accuracy_when_disagrees_population"].iloc[0]
        for m in MODELS
    ]
    bars_follow = ax.bar(
        x - w / 2, acc_follow, w, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85,
        label="When follows cohort",
    )
    bars_disagree = ax.bar(
        x + w / 2, acc_disagree, w, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.35,
        label="When disagrees",
    )
    for b, v in zip(bars_follow, acc_follow):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars_disagree, acc_disagree):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy conditional on follow vs disagree", loc="left", fontsize=9, pad=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.35, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "cohort_usage_panel")


# --- Case 116 (PIL) ---

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def fig_case_example() -> None:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    def pil_font(size, bold=False, symbol=False):
        path = SANS_BOLD if (symbol and bold) else SANS if symbol else SERIF_BOLD if bold else SERIF
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    def centered_text(text, box, fnt, fill="black", line_spacing=8):
        x1, y1, x2, y2 = box
        lines = text.split("\n")
        boxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        total_h = sum(heights) + line_spacing * (len(lines) - 1)
        y = y1 + (y2 - y1 - total_h) / 2
        for line, w, h in zip(lines, widths, heights):
            draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=fnt, fill=fill)
            y += h + line_spacing

    def rounded_box(x1, y1, x2, y2, outline, width=3, fill="white", radius=14):
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

    def arrow(x1, y1, x2, y2, color="gray", width=4):
        draw.line([x1, y1, x2, y2], fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 20
        p1 = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
        p2 = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
        draw.polygon([p1, p2, (x2, y2)], fill=color)

    def numbered_circle(cx, cy, n, outline="#dddddd", fill="#f4f4f4"):
        draw.ellipse([cx - 29, cy - 29, cx + 29, cy + 29], fill=fill, outline=outline, width=2)
        centered_text(str(n), (cx - 29, cy - 29, cx + 29, cy + 29), pil_font(24, True))

    red = "#c94736"
    blue = "#1f4f7f"
    dark_blue = "#35475b"
    gray = "#d0d0d0"
    light_red_bg = "#fff8f5"

    draw.text((300, 25), "Case 116 · 5-year survival ·", font=pil_font(48, True), fill="black")
    draw.text((1010, 25), "Meditron 8B", font=pil_font(48, True), fill=red)

    top_y1, top_y2 = 120, 365
    box_w = 335
    x_positions = [35, 430, 825, 1220]
    titles = ["Question", "Staging", "TCGA / SEER", "Rule"]
    bodies = [
        "Alive in 5 years?\nA) Yes    B) No",
        "52 y, M · oropharynx\npT2 pN0 · HPV− · II",
        "SEER 62% · TCGA 58%\nMean survival 60%",
        "Rate > 50% + 10 pp\n→ A) Yes",
    ]

    for i, x in enumerate(x_positions):
        outline = "#e6bfae" if i == 3 else gray
        fill = light_red_bg if i == 3 else "white"
        rounded_box(x, top_y1, x + box_w, top_y2, outline=outline, fill=fill, width=2)
        numbered_circle(
            x + box_w // 2, top_y1 + 48, i + 1,
            outline="#e7cfc5" if i == 3 else "#dddddd",
            fill="#fff1ec" if i == 3 else "#f4f4f4",
        )
        centered_text(titles[i], (x, top_y1 + 85, x + box_w, top_y1 + 135), pil_font(31, True))
        centered_text(bodies[i], (x, top_y1 + 145, x + box_w, top_y2 - 20), pil_font(26), fill="#222222")

    for i in range(3):
        arrow(x_positions[i] + box_w + 10, 245, x_positions[i + 1] - 10, 245)

    left_x1, left_y1, left_x2, left_y2 = 30, 450, 770, 805
    rounded_box(left_x1, left_y1, left_x2, left_y2, outline="#26384a", width=3)
    draw.rounded_rectangle([left_x1, left_y1, left_x2, left_y1 + 80], radius=12, fill=dark_blue)
    draw.rectangle([left_x1, left_y1 + 45, left_x2, left_y1 + 80], fill=dark_blue)
    centered_text("Without population stats", (left_x1, left_y1, left_x2, left_y1 + 80), pil_font(32, True), fill="white")
    centered_text("Clinical context only", (left_x1, left_y1 + 105, left_x2, left_y1 + 165), pil_font(30))
    draw.text((205, 612), "✗", font=pil_font(50, True, symbol=True), fill=red)
    draw.text((260, 620), "B) No — incorrect", font=pil_font(38, True), fill=red)
    centered_text("gold: A) Yes\nPessimistic on surgery", (left_x1, 675, left_x2, 755), pil_font(28), fill="#222222")

    right_x1, right_y1, right_x2, right_y2 = 825, 450, 1565, 805
    rounded_box(right_x1, right_y1, right_x2, right_y2, outline=red, width=3)
    draw.rounded_rectangle([right_x1, right_y1, right_x2, right_y1 + 80], radius=12, fill=red)
    draw.rectangle([right_x1, right_y1 + 45, right_x2, right_y1 + 80], fill=red)
    centered_text("With TCGA / SEER in prompt", (right_x1, right_y1, right_x2, right_y1 + 80), pil_font(32, True), fill="white")
    centered_text("Cohort 60% → A) Yes", (right_x1, right_y1 + 110, right_x2, right_y1 + 170), pil_font(30))
    draw.text((980, 633), "✓", font=pil_font(50, True, symbol=True), fill=blue)
    draw.text((1040, 640), "A) Yes — correct", font=pil_font(38, True), fill=blue)
    centered_text("Matches rule & gold", (right_x1, 700, right_x2, 750), pil_font(28), fill="#222222")

    arrow(600, 365, 390, 450, color="#666666", width=4)
    arrow(1385, 365, 1210, 450, color=red, width=4)

    save_image(img, "cohort_case_example_116")


def generate_all() -> Path:
    """Write proposal figures to notebooks/figures/ (PNG only)."""
    fig_medical_llm_overall()
    fig_calibration_reliability()
    fig_calibration_auroc()
    fig_cohort_usage()
    fig_case_example()
    return FIG


if __name__ == "__main__":
    out = generate_all()
    print("Figures saved to", out)
