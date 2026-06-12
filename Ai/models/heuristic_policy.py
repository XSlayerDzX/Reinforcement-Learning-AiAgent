"""Heuristic baseline policy.

Rule-based policy that uses card availability and elixir cost to make
the cheapest legal play at a fixed default lane position.  No learning.

Decision logic (in order):
  1. If currently in post-play cooldown -> wait.
  2. If elixir < ELIXIR_THRESHOLD -> wait (save up before playing).
  3. Among cards currently in-hand (slot_1 .. slot_4) that are both
     available (*_avab > 0) AND affordable (elixir >= cost),
     pick the one with the LOWEST elixir cost.
  4. If no affordable card is available -> wait.
  5. Place the chosen card at the default attack position (centre lane,
     just across the river on the enemy side), then enter cooldown.

To run directly (e.g. from PyCharm): just edit SEED below, then
run this file.  No command-line arguments needed.

To run from terminal:
    python -m Ai.models.heuristic_policy --seed 42
"""

import argparse

from Ai.models.eval_runner import run_evaluation
from Ai.models.run_config import (
    DEFAULT_WINDOW_TITLE,
    EVAL_GAMES,
    baseline_eval_dir,
    baseline_log_dir,
)
from Ai.Behavior_Cloning.action_masking_config import AVAIL_FEATURE_TO_ACTION_ID
from Ai.ClashRoyalData import ElixirCost
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Data_Cleaning import final_clean

# ── Edit these to launch directly from PyCharm / file-run ────────────────────
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

# Default placement: grid column 4 (centre), row 10 (just across river)
_DEFAULT_GX = 4
_DEFAULT_GY = 10

# Minimum elixir required before the heuristic will attempt to play any card.
ELIXIR_THRESHOLD = 4.0

# Number of steps to wait after playing a card before playing again.
# Each env step is ~1.5 s, so 3 steps ≈ 4.5 s cooldown.
POST_PLAY_COOLDOWN = 3

# Reverse map: action_id -> card_name (for elixir lookup)
_ACTION_ID_TO_CARD = {
    aid: feat.replace("_avab", "")
    for feat, aid in AVAIL_FEATURE_TO_ACTION_ID.items()
    if aid is not None
}


class HeuristicPolicy:
    """
    Deterministic rule-based policy.  Plays the cheapest affordable
    available card at a fixed default position, or waits.
    Includes a post-play cooldown and elixir threshold to avoid spamming.
    """

    def __init__(self, window_title: str = DEFAULT_WINDOW_TITLE):
        self.window_title   = window_title
        self._cooldown_left = 0

    def reset(self):
        self._cooldown_left = 0

    def select_action(self, obs, slots) -> dict:
        wait_result = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

        if obs is None or obs.empty:
            return wait_result

        # Post-play cooldown
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            print(f"[HeuristicPolicy] Cooldown: {self._cooldown_left + 1} steps remaining — waiting")
            return wait_result

        # Run final_clean so that _avab columns exist
        try:
            cleaned = final_clean(obs)
            if cleaned is None or cleaned.empty:
                return wait_result
            last_row = cleaned.iloc[0]
        except Exception as e:
            print(f"[HeuristicPolicy] final_clean failed: {e} — defaulting to wait")
            return wait_result

        # Elixir threshold check
        try:
            current_elixir = float(last_row.get("Elixir", 0))
        except (TypeError, ValueError):
            current_elixir = 0.0

        if current_elixir < ELIXIR_THRESHOLD:
            print(f"[HeuristicPolicy] Elixir={current_elixir:.1f} < threshold={ELIXIR_THRESHOLD} — waiting")
            return wait_result

        # Find cheapest affordable available card
        best_action_id = None
        best_cost      = float("inf")

        for feat, action_id in AVAIL_FEATURE_TO_ACTION_ID.items():
            if action_id is None:
                continue
            avail = 0.0
            if feat in last_row.index:
                try:
                    avail = float(last_row[feat])
                except (TypeError, ValueError):
                    avail = 0.0
            if avail <= 0:
                continue
            card_name = feat.replace("_avab", "")
            cost = ElixirCost.get(card_name, float("inf"))
            if current_elixir < cost:
                continue
            if cost < best_cost:
                best_cost      = cost
                best_action_id = action_id

        if best_action_id is None:
            return wait_result

        try:
            bs_x, bs_y = grid_to_pixel(_DEFAULT_GX, _DEFAULT_GY)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[HeuristicPolicy] Coordinate conversion failed: {e} — using (-1, -1)")
            pos_x, pos_y = -1.0, -1.0

        self._cooldown_left = POST_PLAY_COOLDOWN

        print(f"[HeuristicPolicy] PLAY action_id={best_action_id} "
              f"({_ACTION_ID_TO_CARD.get(best_action_id, '?')}) "
              f"cost={best_cost} elixir={current_elixir:.1f} "
              f"pos=({pos_x:.0f},{pos_y:.0f}) "
              f"cooldown={POST_PLAY_COOLDOWN}")

        return {"action_id": best_action_id, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def _run(seed: int, n_games: int, window: str):
    policy = HeuristicPolicy(window_title=window)
    print(f"\n{'='*60}")
    print(f"Heuristic Policy Evaluation  |  seed={seed}  |  games={n_games}")
    print(f"{'='*60}\n")
    run_evaluation(
        policy       = policy,
        policy_name  = "heuristic",
        seed         = seed,
        log_dir      = baseline_log_dir("heuristic"),
        eval_dir     = baseline_eval_dir("heuristic"),
        n_games      = n_games,
        window_title = window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heuristic baseline evaluation")
    parser.add_argument("--seed",    type=int, default=SEED,                help="Seed (logged only, policy is deterministic)")
    parser.add_argument("--n_games", type=int, default=EVAL_GAMES,          help="Number of evaluation games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE,help="BlueStacks window title")
    args = parser.parse_args()
    _run(args.seed, args.n_games, args.window)
