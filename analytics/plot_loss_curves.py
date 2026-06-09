"""Figure 2 — Policy loss and value loss curves.

Reads training_log.csv for all 5 PPO runs and plots:
  - policy_loss vs game_id  (top panel)
  - value_loss  vs game_id  (bottom panel)

Saves to figures/fig_loss_curves.png.

Run:
    python -m analytics.plot_loss_curves
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Ai.models.run_config import PPO_RUN_IDS, ppo_log_dir

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RUN_STYLE = {
    "BCPPO_s1":        {"color": "#2196F3", "linestyle": "-",  "label": "BC→PPO (seed 1)"},
    "BCPPO_s2":        {"color": "#64B5F6", "linestyle": "--", "label": "BC→PPO (seed 2)"},
    "PPOScratch_s1":   {"color": "#F44336", "linestyle": "-",  "label": "PPO-Scratch (seed 1)"},
    "PPOScratch_s2":   {"color": "#EF9A9A", "linestyle": "--", "label": "PPO-Scratch (seed 2)"},
    "BCPPO_NoMask_s1": {"color": "#FF9800", "linestyle": "-.", "label": "BC→PPO No-Mask (seed 1)"},
}


def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply a rolling mean for visual clarity."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_loss_curves():
    fig, (ax_pol, ax_val) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    any_plotted = False
    for run_id in PPO_RUN_IDS:
        path = ppo_log_dir(run_id) / "training_log.csv"
        if not path.exists():
            print(f"[WARN] training_log.csv not found for {run_id}")
            continue
        df = pd.read_csv(path)
        if "policy_loss" not in df.columns or "value_loss" not in df.columns:
            print(f"[WARN] Loss columns missing for {run_id}")
            continue

        style = RUN_STYLE.get(run_id, {})
        color = style.get("color", "gray")
        ls    = style.get("linestyle", "-")
        label = style.get("label", run_id)

        ax_pol.plot(
            df["game_id"], _smooth(df["policy_loss"]),
            color=color, linestyle=ls, linewidth=2, label=label, alpha=0.9,
        )
        ax_val.plot(
            df["game_id"], _smooth(df["value_loss"]),
            color=color, linestyle=ls, linewidth=2, label=label, alpha=0.9,
        )
        any_plotted = True

    if not any_plotted:
        print("[ERROR] No training logs found. Run training first.")
        return

    ax_pol.set_ylabel("Policy Loss", fontsize=13)
    ax_pol.set_title("Loss Curves — All PPO Runs", fontsize=15, fontweight="bold")
    ax_pol.legend(fontsize=10, loc="upper right")
    ax_pol.grid(True, linestyle="--", alpha=0.4)

    ax_val.set_xlabel("Game", fontsize=13)
    ax_val.set_ylabel("Value Loss", fontsize=13)
    ax_val.legend(fontsize=10, loc="upper right")
    ax_val.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_loss_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    plot_loss_curves()
