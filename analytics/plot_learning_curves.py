"""Figure 1 — Learning curves (win rate + illegal action rate).

Two-panel figure per PPO run:
  Top panel   : win_rate_last_20 vs game_id
  Bottom panel: illegal_action_rate (smoothed) vs game_id

Baseline win rates (flat horizontal lines) are overlaid on the top panel
for reference.

Saves to figures/fig_learning_curves.png.

Run:
    python -m analytics.plot_learning_curves
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
    ppo_log_dir,
    baseline_eval_dir,
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 10   # rolling window for illegal_action_rate smoothing

# Visual style per PPO run
RUN_STYLE = {
    "BCPPO_s1":        {"color": "#2196F3", "linestyle": "-",  "label": "BC→PPO (s1)"},
    "BCPPO_s2":        {"color": "#64B5F6", "linestyle": "--", "label": "BC→PPO (s2)"},
    "PPOScratch_s1":   {"color": "#F44336", "linestyle": "-",  "label": "PPO-Scratch (s1)"},
    "PPOScratch_s2":   {"color": "#EF9A9A", "linestyle": "--", "label": "PPO-Scratch (s2)"},
    "BCPPO_NoMask_s1": {"color": "#FF9800", "linestyle": "-.", "label": "BC→PPO No-Mask (s1)"},
}

# Baseline display names and colours for reference lines
BASELINE_STYLE = {
    "random":    {"color": "#9E9E9E", "linestyle": ":",  "label": "Random"},
    "heuristic": {"color": "#4CAF50", "linestyle": ":",  "label": "Heuristic"},
    "bc_only":   {"color": "#9C27B0", "linestyle": ":",  "label": "BC-only"},
}


def _load_ppo_log(run_id: str) -> pd.DataFrame | None:
    path = ppo_log_dir(run_id) / "training_log.csv"
    if not path.exists():
        print(f"[WARN] training_log.csv not found for {run_id}")
        return None
    df = pd.read_csv(path)
    needed = {"game_id", "win_rate_last_20", "illegal_action_rate"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[WARN] Missing columns {missing} in {path}")
        return None
    return df


def _load_baseline_winrate(policy_id: str) -> float | None:
    path = baseline_eval_dir(policy_id) / f"eval_{policy_id}_aggregate.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[0].get("win_rate_mean", 0.0))


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_learning_curves():
    fig, (ax_win, ax_ill) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    any_plotted = False

    # ── PPO runs ──────────────────────────────────────────────────────────
    for run_id in PPO_RUN_IDS:
        df = _load_ppo_log(run_id)
        if df is None:
            continue
        style = RUN_STYLE.get(run_id, {"color": "gray", "linestyle": "-", "label": run_id})

        ax_win.plot(
            df["game_id"], df["win_rate_last_20"],
            color=style["color"], linestyle=style["linestyle"],
            linewidth=2, label=style["label"], alpha=0.9,
        )
        ax_ill.plot(
            df["game_id"], _smooth(df["illegal_action_rate"], SMOOTH_WINDOW),
            color=style["color"], linestyle=style["linestyle"],
            linewidth=2, label=style["label"], alpha=0.9,
        )
        any_plotted = True

    if not any_plotted:
        print("[ERROR] No training logs found. Run training first.")
        return

    # ── Baseline reference lines on win rate panel ────────────────────────
    for policy_id, bstyle in BASELINE_STYLE.items():
        wr = _load_baseline_winrate(policy_id)
        if wr is not None:
            ax_win.axhline(
                y=wr, color=bstyle["color"],
                linestyle=bstyle["linestyle"], linewidth=1.5,
                label=bstyle["label"], alpha=0.7,
            )

    # ── Formatting ────────────────────────────────────────────────────────
    ax_win.set_ylabel("Win Rate (rolling 20)", fontsize=12)
    ax_win.set_ylim(0, 1.05)
    ax_win.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_win.legend(fontsize=9, loc="upper left", ncol=2)
    ax_win.grid(True, linestyle="--", alpha=0.35)
    ax_win.set_title("Learning Curves — All PPO Runs", fontsize=14, fontweight="bold")

    ax_ill.set_xlabel("Game", fontsize=12)
    ax_ill.set_ylabel(f"Illegal Action Rate (smooth={SMOOTH_WINDOW})", fontsize=12)
    ax_ill.set_ylim(0, 1.05)
    ax_ill.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_ill.legend(fontsize=9, loc="upper right", ncol=2)
    ax_ill.grid(True, linestyle="--", alpha=0.35)
    ax_ill.set_title("Illegal Action Rate over Training", fontsize=13)

    ax_win.set_xlim(left=1)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig_learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    plot_learning_curves()
