"""Figure 3 — Action distribution evolution.

For each PPO run reads updates.json and plots stacked bar charts
showing the action distribution at games 1, 25, 50, and 100.
Shows how the policy specialises over training.

Saves to figures/fig_action_dist.png.

Run:
    python -m analytics.plot_action_dist
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Ai.models.run_config import PPO_RUN_IDS, ppo_log_dir

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SNAPSHOT_GAMES = [1, 25, 50, 100]

# Consistent colour per card across all subplots
CARD_COLORS = [
    "#4CAF50", "#2196F3", "#F44336", "#FF9800",
    "#9C27B0", "#00BCD4", "#8BC34A", "#FF5722",
    "#607D8B", "#E91E63", "#FFC107", "#3F51B5", "#795548",
]


def _load_updates(run_id: str):
    path = ppo_log_dir(run_id) / "updates.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _get_dist_at_game(updates: list, game_id: int) -> dict:
    """
    Return the action_dist dict for the entry closest to game_id.
    Falls back to the last available entry if game_id exceeds log length.
    """
    # updates are indexed by update_id which equals game_id
    for entry in updates:
        if entry.get("update_id") == game_id:
            return entry.get("action_dist", {})
    # fallback: closest
    if not updates:
        return {}
    closest = min(updates, key=lambda e: abs(e.get("update_id", 0) - game_id))
    return closest.get("action_dist", {})


def plot_action_dist():
    n_runs = len(PPO_RUN_IDS)
    n_snap = len(SNAPSHOT_GAMES)

    fig, axes = plt.subplots(
        n_runs, n_snap,
        figsize=(4 * n_snap, 3.5 * n_runs),
        sharey="row",
    )
    # Ensure axes is always 2-D
    if n_runs == 1:
        axes = [axes]

    any_plotted = False
    all_cards   = set()

    # First pass: collect all card names for consistent ordering
    for run_id in PPO_RUN_IDS:
        updates = _load_updates(run_id)
        for g in SNAPSHOT_GAMES:
            dist = _get_dist_at_game(updates, g)
            all_cards.update(dist.keys())

    all_cards = sorted(all_cards)
    color_map = {card: CARD_COLORS[i % len(CARD_COLORS)] for i, card in enumerate(all_cards)}

    for row_idx, run_id in enumerate(PPO_RUN_IDS):
        updates = _load_updates(run_id)
        if not updates:
            print(f"[WARN] No updates.json for {run_id}")
            continue

        for col_idx, game_id in enumerate(SNAPSHOT_GAMES):
            ax   = axes[row_idx][col_idx]
            dist = _get_dist_at_game(updates, game_id)

            if not dist:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            cards  = list(dist.keys())
            counts = [dist[c] for c in cards]
            total  = max(sum(counts), 1)
            fracs  = [c / total for c in counts]
            colors = [color_map.get(c, "#BDBDBD") for c in cards]

            ax.bar(range(len(cards)), fracs, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(cards)))
            ax.set_xticklabels(cards, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1)

            if col_idx == 0:
                ax.set_ylabel(run_id.replace("_", "\n"), fontsize=8)
            if row_idx == 0:
                ax.set_title(f"Game {game_id}", fontsize=11, fontweight="bold")

            any_plotted = True

    if not any_plotted:
        print("[ERROR] No updates.json found. Run training first.")
        return

    fig.suptitle("Action Distribution at Training Snapshots", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_action_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    plot_action_dist()
