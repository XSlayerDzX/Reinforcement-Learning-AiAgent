"""Figure 4 — Illegal action rate: masking ablation.

Key figure for the masking ablation section of the paper.
Shows that action masking is necessary — unmasked models waste a large
fraction of steps on illegal actions.

Two sub-figures:
  A) Bar chart: mean illegal_action_rate per model (baselines + PPO eval)
     with error bars (±1 std). Models are colour-coded by type.
  B) Line plot: illegal_action_rate per game during training for
     BCPPO_s1 vs BCPPO_NoMask_s1 (smoothed), showing whether masking
     helps the agent *learn* to avoid illegal actions over time.

Reads:
  - eval_{policy}_aggregate.csv  (illegal_action_rate_mean / _std)
  - training_log.csv             (illegal_action_rate per game, PPO only)

Saves to:
  figures/fig_illegal_rate_bar.png
  figures/fig_illegal_rate_training.png

Run:
    python -m analytics.plot_illegal_rate
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Ai.models.run_config import (
    PPO_RUN_IDS,
    BASELINE_POLICY_NAMES,
    baseline_eval_dir,
    ppo_eval_dir,
    ppo_log_dir,
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 10

# Display order and labels for bar chart
ALL_POLICIES_ORDERED = [
    "random", "heuristic", "bc_only",
    "PPOScratch_s1", "PPOScratch_s2",
    "BCPPO_s1", "BCPPO_s2",
    "BCPPO_NoMask_s1",
]

DISPLAY_NAMES = {
    "random":          "Random",
    "heuristic":       "Heuristic",
    "bc_only":         "BC-only",
    "PPOScratch_s1":   "PPO-Scratch (s1)",
    "PPOScratch_s2":   "PPO-Scratch (s2)",
    "BCPPO_s1":        "BC→PPO (s1)",
    "BCPPO_s2":        "BC→PPO (s2)",
    "BCPPO_NoMask_s1": "BC→PPO\nNo-Mask (s1)",
}

# Colour by group
GROUP_COLOR = {
    "random":          "#9E9E9E",
    "heuristic":       "#4CAF50",
    "bc_only":         "#9C27B0",
    "PPOScratch_s1":   "#EF9A9A",
    "PPOScratch_s2":   "#EF9A9A",
    "BCPPO_s1":        "#2196F3",
    "BCPPO_s2":        "#64B5F6",
    "BCPPO_NoMask_s1": "#FF9800",
}

# Training comparison: masked vs unmasked
COMPARE_RUNS = {
    "BCPPO_s1":        {"color": "#2196F3", "linestyle": "-",  "label": "BC→PPO (masked)"},
    "BCPPO_NoMask_s1": {"color": "#FF9800", "linestyle": "-.", "label": "BC→PPO (no mask)"},
}


def _load_agg(policy_id: str) -> dict | None:
    if policy_id in BASELINE_POLICY_NAMES:
        path = baseline_eval_dir(policy_id) / f"eval_{policy_id}_aggregate.csv"
    else:
        path = ppo_eval_dir(policy_id) / f"eval_{policy_id}_aggregate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else None


def _load_training_log(run_id: str) -> pd.DataFrame | None:
    path = ppo_log_dir(run_id) / "training_log.csv"
    if not path.exists():
        print(f"[WARN] training_log.csv not found for {run_id}")
        return None
    df = pd.read_csv(path)
    if "illegal_action_rate" not in df.columns or "game_id" not in df.columns:
        print(f"[WARN] illegal_action_rate column missing in {path}")
        return None
    return df


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


# ── Figure A: bar chart ──────────────────────────────────────────────────

def plot_illegal_bar():
    names, means, stds, colors = [], [], [], []

    for pid in ALL_POLICIES_ORDERED:
        agg = _load_agg(pid)
        if agg is None:
            print(f"[SKIP] No aggregate CSV for {pid}")
            continue
        ill_mean = float(agg.get("illegal_action_rate_mean", 0.0))
        ill_std  = float(agg.get("illegal_action_rate_std",  0.0))
        names.append(DISPLAY_NAMES.get(pid, pid))
        means.append(ill_mean)
        stds.append(ill_std)
        colors.append(GROUP_COLOR.get(pid, "#BDBDBD"))

    if not names:
        print("[ERROR] No aggregate CSVs found. Run eval scripts first.")
        return

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.6,
                  error_kw=dict(elinewidth=1.5, ecolor="#333333"))

    # Annotate each bar with the % value
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.1%}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Illegal Action Rate (mean ± std)", fontsize=12)
    ax.set_ylim(0, min(max(means) * 1.4 + 0.05, 1.05))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title(
        "Illegal Action Rate per Model\n"
        "(fraction of steps violating availability mask or elixir rule)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Vertical separator lines between groups
    ax.axvline(x=2.5, color="#BDBDBD", linestyle="--", linewidth=1)
    ax.axvline(x=4.5, color="#BDBDBD", linestyle="--", linewidth=1)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_illegal_rate_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out}")


# ── Figure B: training comparison masked vs unmasked ────────────────────

def plot_illegal_training():
    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False

    for run_id, style in COMPARE_RUNS.items():
        df = _load_training_log(run_id)
        if df is None:
            continue
        ax.plot(
            df["game_id"],
            _smooth(df["illegal_action_rate"], SMOOTH_WINDOW),
            color=style["color"], linestyle=style["linestyle"],
            linewidth=2.5, label=style["label"], alpha=0.9,
        )
        any_plotted = True

    if not any_plotted:
        print("[WARN] No training data for masked/unmasked comparison.")
        return

    ax.set_xlabel("Training Game", fontsize=12)
    ax.set_ylabel(f"Illegal Action Rate (smooth={SMOOTH_WINDOW})", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title(
        "Illegal Action Rate During Training: Masked vs Unmasked",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=1)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_illegal_rate_training.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out}")


if __name__ == "__main__":
    plot_illegal_bar()
    plot_illegal_training()
