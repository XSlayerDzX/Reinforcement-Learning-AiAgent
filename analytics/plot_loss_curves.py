"""Figure 2 — PPO loss curves.

Reads updates.json for all 5 PPO runs and plots:
  - policy_loss  (left y-axis)
  - value_loss   (right y-axis)
  - clip_fraction (separate panel)
  - action_entropy (separate panel)

Saves to figures/fig_loss_curves.png.

Run:
    python -m analytics.plot_loss_curves
"""

import json
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

from Ai.models.run_config import PPO_RUN_IDS, ppo_log_dir

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SMOOTH_WINDOW = 10

RUN_STYLE = {
    "BCPPO_s1":        {"color": "#2196F3", "linestyle": "-",  "label": "BC→PPO (s1)"},
    "BCPPO_s2":        {"color": "#64B5F6", "linestyle": "--", "label": "BC→PPO (s2)"},
    "PPOScratch_s1":   {"color": "#F44336", "linestyle": "-",  "label": "PPO-Scratch (s1)"},
    "PPOScratch_s2":   {"color": "#EF9A9A", "linestyle": "--", "label": "PPO-Scratch (s2)"},
    "BCPPO_NoMask_s1": {"color": "#FF9800", "linestyle": "-.", "label": "BC→PPO No-Mask (s1)"},
}


def _load_updates(run_id: str) -> pd.DataFrame | None:
    path = ppo_log_dir(run_id) / "updates.json"
    if not path.exists():
        print(f"[WARN] updates.json not found for {run_id}")
        return None
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None
    df = pd.DataFrame(data)
    # Keep only rows that are real update entries (have policy_loss)
    df = df[df["policy_loss"].notna()] if "policy_loss" in df.columns else df
    return df if not df.empty else None


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_loss_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_pol, ax_val, ax_clip, ax_ent = axes.flatten()

    any_plotted = False

    for run_id in PPO_RUN_IDS:
        df = _load_updates(run_id)
        if df is None:
            continue
        style = RUN_STYLE.get(run_id, {"color": "gray", "linestyle": "-", "label": run_id})
        kw = dict(color=style["color"], linestyle=style["linestyle"],
                  linewidth=1.8, alpha=0.85, label=style["label"])

        x = df["update_id"] if "update_id" in df.columns else range(len(df))

        if "policy_loss" in df.columns:
            ax_pol.plot(x, _smooth(df["policy_loss"],   SMOOTH_WINDOW), **kw)
        if "value_loss" in df.columns:
            ax_val.plot(x, _smooth(df["value_loss"],    SMOOTH_WINDOW), **kw)
        if "clip_fraction" in df.columns:
            ax_clip.plot(x, _smooth(df["clip_fraction"], SMOOTH_WINDOW), **kw)
        if "action_entropy" in df.columns:
            ax_ent.plot(x, _smooth(df["action_entropy"], SMOOTH_WINDOW), **kw)

        any_plotted = True

    if not any_plotted:
        print("[ERROR] No updates.json found. Run training first.")
        return

    ax_pol.set_title("Policy Loss",    fontsize=12, fontweight="bold")
    ax_val.set_title("Value Loss",     fontsize=12, fontweight="bold")
    ax_clip.set_title("Clip Fraction", fontsize=12, fontweight="bold")
    ax_ent.set_title("Action Entropy", fontsize=12, fontweight="bold")

    for ax in axes.flatten():
        ax.set_xlabel("Game", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle("PPO Training Diagnostics", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig_loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    plot_loss_curves()
