"""Heuristic baseline policy.

Rule-based policy that uses card availability and elixir cost to make
the cheapest legal play at a fixed default lane position.  No learning.

Decision logic (in order):
  1. If elixir < min affordable card cost -> wait.
  2. Among cards currently in-hand (slot_1 .. slot_4) that are both
     available (*_avab > 0) AND affordable (elixir >= cost),
     pick the one with the LOWEST elixir cost.
  3. If no affordable card is available -> wait.
  4. Place the chosen card at the default attack position (centre lane,
     just across the river on the enemy side).

Run:
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

# Default placement: grid column 4 (centre), row 10 (just across river)
_DEFAULT_GX = 4
_DEFAULT_GY = 10

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
    """

    def __init__(self, window_title: str = DEFAULT_WINDOW_TITLE):
        self.window_title = window_title

    def reset(self):
        """Called by eval_runner at the start of every game."""
        pass  # stateless — nothing to reset

    def select_action(self, obs, slots) -> dict:
        """
        Args:
            obs   : pd.DataFrame  (current environment observation)
            slots : dict          (current card slots, ignored — we use obs directly)

        Returns:
            {"action_id": int, "pos_x": float, "pos_y": float}
        """
        wait_result = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

        if obs is None or obs.empty:
            return wait_result

        # ── CRITICAL: run final_clean so that _avab columns exist ────────────
        # The raw obs from the env does NOT contain _avab columns — they are
        # computed by card_avable() inside final_clean().  Without this call
        # every avail lookup returns 0 and the policy always waits.
        try:
            cleaned = final_clean(obs)
            if cleaned is None or cleaned.empty:
                return wait_result
            last_row = cleaned.iloc[0]
        except Exception as e:
            print(f"[HeuristicPolicy] final_clean failed: {e} — defaulting to wait")
            return wait_result

        # Read current elixir
        try:
            current_elixir = float(last_row.get("Elixir", 0))
        except (TypeError, ValueError):
            current_elixir = 0.0

        # Find cheapest affordable available card
        best_action_id = None
        best_cost      = float("inf")

        for feat, action_id in AVAIL_FEATURE_TO_ACTION_ID.items():
            if action_id is None:
                continue

            # Check availability (column now exists after final_clean)
            avail = 0.0
            if feat in last_row.index:
                try:
                    avail = float(last_row[feat])
                except (TypeError, ValueError):
                    avail = 0.0
            if avail <= 0:
                continue

            # Check affordability
            card_name = feat.replace("_avab", "")
            cost = ElixirCost.get(card_name, float("inf"))
            if current_elixir < cost:
                continue

            # Pick lowest cost
            if cost < best_cost:
                best_cost      = cost
                best_action_id = action_id

        if best_action_id is None:
            return wait_result

        # Convert default grid position to screen coordinates
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

        print(f"[HeuristicPolicy] action_id={best_action_id} "
              f"({_ACTION_ID_TO_CARD.get(best_action_id, '?')}) "
              f"cost={best_cost} elixir={current_elixir:.1f} "
              f"pos=({pos_x:.0f},{pos_y:.0f})")

        return {"action_id": best_action_id, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def main():
    parser = argparse.ArgumentParser(description="Heuristic baseline evaluation")
    parser.add_argument("--seed",    type=int, default=42,         help="Seed (logged only, policy is deterministic)")
    parser.add_argument("--n_games", type=int, default=EVAL_GAMES, help="Number of evaluation games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE, help="BlueStacks window title")
    args = parser.parse_args()

    policy = HeuristicPolicy(window_title=args.window)

    print(f"\n{'='*60}")
    print(f"Heuristic Policy Evaluation  |  seed={args.seed}  |  games={args.n_games}")
    print(f"{'='*60}\n")

    run_evaluation(
        policy       = policy,
        policy_name  = "heuristic",
        seed         = args.seed,
        log_dir      = baseline_log_dir("heuristic"),
        eval_dir     = baseline_eval_dir("heuristic"),
        n_games      = args.n_games,
        window_title = args.window,
    )


if __name__ == "__main__":
    main()
