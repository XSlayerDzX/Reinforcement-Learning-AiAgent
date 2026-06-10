"""Figure 3 — Action distribution: baselines vs PPO.

Two sub-figures:
  A) Stacked bar: aggregate action distribution for each model
     (all games averaged), sorted by policy type.
  B) Evolution bars: action distribution at game snapshots
     [1, 25, 50, 100] for the best PPO run (BCPPO_s1).

Reads:
  - eval_{policy}_games.csv  for baselines (action_dist column, JSON string)
  - training_log.csv         for PPO runs  (action_dist column, JSON string)

Saves to figures/fig_action_dist.png.

Run:
    python -m analytics.plot_action_dist
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Ai.models.run_config import (
    PPO_RUN_IDS,
    BASELINE_POLICY_NAMES,
    baseline_eval_dir,
    ppo_log_dir,
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SNAPSHOT_GAMES = [1, 25, 50, 100]
EVOL_RUN       = "BCPPO_s1"    # run used for evolution plot

# Card action IDs -> readable labels (matches AVAIL_FEATURE_TO_ACTION_ID)
ACTION_LABELS = {
    "0":  "wait",
    "1":  "mini pekka",
    "2":  "knight",
    "4":  "goblins",
    "5":  "giant",
    "6":  "spear goblins",
    "7":  "archers",
    "9":  "minions",
    "10": "musketeer",
    "11": "goblin cage",
}

CARD_COLORS = [
    "#607D8B",  # wait  — grey
    "#F44336", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#8BC34A", "#FF5722",
    "#E91E63", "#FFC107",
]


def _parse_dist_col(series: pd.Series) -> dict:
    """Aggregate action_dist JSON strings from a DataFrame column into one dict."""
    total = defaultdict(int)
    for val in series.dropna():
        try:
            d = json.loads(val) if isinstance(val, str) else val
            for k, v in d.items():
                total[str(k)] += int(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return dict(total)


def _load_baseline_dist(policy_id: str) -> dict:
    path = baseline_eval_dir(policy_id) / f"eval_{policy_id}_games.csv"
    if not path.exists():
        print(f"[WARN] Games CSV not found: {path}")
        return {}
    df = pd.read_csv(path)
    if "action_dist" not in df.columns:
        print(f"[WARN] action_dist column missing in {path}")
        return {}
    return _parse_dist_col(df["action_dist"])


def _load_ppo_dist(run_id: str) -> dict:
    path = ppo_log_dir(run_id) / "training_log.csv"
    if not path.exists():
        print(f"[WARN] training_log.csv not found for {run_id}")
        return {}
    df = pd.read_csv(path)
    if "action_dist" not in df.columns:
        print(f"[WARN] action_dist column missing in {path}")
        return {}
    return _parse_dist_col(df["action_dist"])


def _load_ppo_dist_at_game(run_id: str, game_id: int) -> dict:
    """Load action_dist for a specific game from training_log.csv."""
    path = ppo_log_dir(run_id) / "training_log.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "action_dist" not in df.columns or "game_id" not in df.columns:
        return {}
    row = df[df["game_id"] == game_id]
    if row.empty:
        # Fallback: closest game_id
        closest_idx = (df["game_id"] - game_id).abs().idxmin()
        row = df.loc[[closest_idx]]
    val = row["action_dist"].iloc[0]
    try:
        return json.loads(val) if isinstance(val, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _normalise(dist: dict) -> tuple[list, list]:
    """Return (labels, fractions) sorted by action id."""
    if not dist:
        return [], []
    total = max(sum(dist.values()), 1)
    sorted_items = sorted(dist.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99)
    labels = [ACTION_LABELS.get(k, f"act_{k}") for k, _ in sorted_items]
    fracs  = [v / total for _, v in sorted_items]
    return labels, fracs


def plot_action_dist():
    all_policies = list(BASELINE_POLICY_NAMES) + list(PPO_RUN_IDS)
    display = {
        "random":          "Random",
        "heuristic":       "Heuristic",
        "bc_only":         "BC-only",
        "PPOScratch_s1":   "PPO-Scratch (s1)",
        "PPOScratch_s2":   "PPO-Scratch (s2)",
        "BCPPO_s1":        "BC→PPO (s1)",
        "BCPPO_s2":        "BC→PPO (s2)",
        "BCPPO_NoMask_s1": "BC→PPO No-Mask",
    }

    # ── Figure A: aggregate dist per model ──────────────────────────────────
    fig_a, axes_a = plt.subplots(
        1, len(all_policies),
        figsize=(2.8 * len(all_policies), 5),
        sharey=True,
    )
    if len(all_policies) == 1:
        axes_a = [axes_a]

    all_action_ids = set()
    dists = {}
    for pid in all_policies:
        d = _load_baseline_dist(pid) if pid in BASELINE_POLICY_NAMES else _load_ppo_dist(pid)
        dists[pid] = d
        all_action_ids.update(d.keys())

    sorted_ids = sorted(all_action_ids, key=lambda x: int(x) if x.isdigit() else 99)
    color_map  = {aid: CARD_COLORS[i % len(CARD_COLORS)] for i, aid in enumerate(sorted_ids)}

    any_plotted_a = False
    for ax, pid in zip(axes_a, all_policies):
        d = dists[pid]
        if not d:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(display.get(pid, pid), fontsize=8)
            continue
        labels, fracs = _normalise(d)
        action_ids_sorted = sorted(d.keys(), key=lambda x: int(x) if x.isdigit() else 99)
        colors = [color_map[aid] for aid in action_ids_sorted]
        ax.bar(range(len(labels)), fracs, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(display.get(pid, pid), fontsize=9, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        any_plotted_a = True

    axes_a[0].set_ylabel("Fraction of actions", fontsize=11)
    fig_a.suptitle("Aggregate Action Distribution per Model", fontsize=13, fontweight="bold")
    fig_a.tight_layout()
    out_a = FIGURES_DIR / "fig_action_dist_aggregate.png"
    fig_a.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    if any_plotted_a:
        print(f"[OK] Saved -> {out_a}")

    # ── Figure B: evolution snapshots for BCPPO_s1 ────────────────────────
    fig_b, axes_b = plt.subplots(1, len(SNAPSHOT_GAMES), figsize=(4.5 * len(SNAPSHOT_GAMES), 5))
    if len(SNAPSHOT_GAMES) == 1:
        axes_b = [axes_b]

    any_plotted_b = False
    for ax, g in zip(axes_b, SNAPSHOT_GAMES):
        d = _load_ppo_dist_at_game(EVOL_RUN, g)
        if not d:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Game {g}", fontsize=11)
            continue
        labels, fracs = _normalise(d)
        action_ids_sorted = sorted(d.keys(), key=lambda x: int(x) if x.isdigit() else 99)
        colors = [color_map.get(aid, "#BDBDBD") for aid in action_ids_sorted]
        ax.bar(range(len(labels)), fracs, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Game {g}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        any_plotted_b = True

    axes_b[0].set_ylabel("Fraction of actions", fontsize=11)
    fig_b.suptitle(f"Action Distribution Evolution — {EVOL_RUN}", fontsize=13, fontweight="bold")
    fig_b.tight_layout()
    out_b = FIGURES_DIR / "fig_action_dist_evolution.png"
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    if any_plotted_b:
        print(f"[OK] Saved -> {out_b}")


if __name__ == "__main__":
    plot_action_dist()
