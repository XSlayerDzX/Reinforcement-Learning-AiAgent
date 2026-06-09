"""Figure 1 — Learning curves.

Reads training_log.csv for all 5 PPO runs and plots win_rate_last_20
vs game_id on a single figure.  Saves to figures/fig_learning_curves.png.

Run:
    python -m analytics.plot_learning_curves
"""

import sys
from pathlib import Path

# Ensure project root is on the path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Ai.models.run_config import PPO_RUN_IDS, ppo_log_dir

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Visual style per run
RUN_STYLE = {
    "BCPPO_s1":        {"color": "#2196F3", "linestyle": "-",  "label": "BC→PPO (seed 1)"},
    "BCPPO_s2":        {"color": "#64B5F6", "linestyle": "--", "label": "BC→PPO (seed 2)"},
    "PPOScratch_s1":   {"color": "#F44336", "linestyle": "-",  "label": "PPO-Scratch (seed 1)"},
    "PPOScratch_s2":   {"color": "#EF9A9A", "linestyle": "--", "label": "PPO-Scratch (seed 2)"},
    "BCPPO_NoMask_s1": {"color": "#FF9800", "linestyle": "-.", "label": "BC→PPO No-Mask (seed 1)"},
}


def load_training_log(run_id: str) -> pd.DataFrame | None:
    path = ppo_log_dir(run_id) / "training_log.csv"
    if not path.exists():
        print(f"[WARN] training_log.csv not found for {run_id}: {path}")
        return None
    df = pd.read_csv(path)
    if "game_id" not in df.columns or "win_rate_last_20" not in df.columns:
        print(f"[WARN] Required columns missing in {path}")
        return None
    return df


def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(10, 6))

    any_plotted = False
    for run_id in PPO_RUN_IDS:
        df = load_training_log(run_id)
        if df is None:
            continue
        style = RUN_STYLE.get(run_id, {})
        ax.plot(
            df["game_id"],
            df["win_rate_last_20"],
            color     = style.get("color",     "gray"),
            linestyle = style.get("linestyle", "-"),
            linewidth = 2,
            label     = style.get("label",     run_id),
            alpha     = 0.9,
        )
        any_plotted = True

    if not any_plotted:
        print("[ERROR] No training logs found. Run training first.")
        return

    ax.set_xlabel("Game", fontsize=13)
    ax.set_ylabel("Win Rate (rolling 20-game window)", fontsize=13)
    ax.set_title("Learning Curves — All PPO Runs", fontsize=15, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig_learning_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    plot_learning_curves()
