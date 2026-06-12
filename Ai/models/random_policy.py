"""Random baseline policy evaluation.

To run directly (e.g. from PyCharm): just edit SEED below, then
run this file.  No command-line arguments needed.

To run from terminal:
    python -m Ai.models.random_policy --seed 42
"""
import argparse
import random

from Ai.models.eval_runner import run_evaluation
from Ai.models.run_config import (
    DEFAULT_WINDOW_TITLE,
    EVAL_GAMES,
    baseline_eval_dir,
    baseline_log_dir,
)
from Ai.Behavior_Cloning.action_masking_config import AVAIL_FEATURE_TO_ACTION_ID, WAIT_ID
from Ai.ClashRoyalData import ElixirCost
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Data_Cleaning import final_clean

# ── Edit these to launch directly from PyCharm / file-run ────────────────────
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_GX = 4
_DEFAULT_GY = 10

_ACTION_IDS = [aid for aid in AVAIL_FEATURE_TO_ACTION_ID.values() if aid is not None]


class RandomPolicy:
    """Uniformly random legal action, or wait if nothing is affordable."""

    def __init__(self, seed: int = 42, window_title: str = DEFAULT_WINDOW_TITLE):
        self.rng          = random.Random(seed)
        self.window_title = window_title

    def reset(self):
        pass

    def select_action(self, obs, slots) -> dict:
        wait_result = {"action_id": WAIT_ID, "pos_x": -1.0, "pos_y": -1.0}
        if obs is None or obs.empty:
            return wait_result

        try:
            cleaned  = final_clean(obs)
            if cleaned is None or cleaned.empty:
                return wait_result
            last_row = cleaned.iloc[0]
        except Exception:
            return wait_result

        try:
            current_elixir = float(last_row.get("Elixir", 0))
        except (TypeError, ValueError):
            current_elixir = 0.0

        legal = []
        for feat, action_id in AVAIL_FEATURE_TO_ACTION_ID.items():
            if action_id is None:
                continue
            avail = 0.0
            if feat in last_row.index:
                try:
                    avail = float(last_row[feat])
                except (TypeError, ValueError):
                    pass
            if avail <= 0:
                continue
            card_name = feat.replace("_avab", "")
            cost = ElixirCost.get(card_name, float("inf"))
            if current_elixir >= cost:
                legal.append(action_id)

        if not legal:
            return wait_result

        action_id = self.rng.choice(legal)

        try:
            bs_x, bs_y = grid_to_pixel(_DEFAULT_GX, _DEFAULT_GY)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[RandomPolicy] Coordinate error: {e}")
            pos_x, pos_y = -1.0, -1.0

        return {"action_id": action_id, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def _run(seed: int, n_games: int, window: str):
    policy = RandomPolicy(seed=seed, window_title=window)
    print(f"\n{'='*60}")
    print(f"Random Policy Evaluation  |  seed={seed}  |  games={n_games}")
    print(f"{'='*60}\n")
    run_evaluation(
        policy       = policy,
        policy_name  = "random",
        seed         = seed,
        log_dir      = baseline_log_dir("random"),
        eval_dir     = baseline_eval_dir("random"),
        n_games      = n_games,
        window_title = window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random baseline evaluation")
    parser.add_argument("--seed",    type=int, default=SEED,                help="Random seed")
    parser.add_argument("--n_games", type=int, default=EVAL_GAMES,          help="Evaluation games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE,help="BlueStacks window")
    args = parser.parse_args()
    _run(args.seed, args.n_games, args.window)
