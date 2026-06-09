"""Random baseline policy.

At every step picks a random action from the full unmasked action space
with a random valid grid position.  No elixir check, no card availability
check — this is intentional: it anchors the absolute floor in the paper.

Run:
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
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.models.run_config import NUM_ACTIONS

# Grid dimensions (must match coordinate_utils.py)
_GRID_W = 9
_GRID_H = 18


class RandomPolicy:
    """
    Selects a uniformly random action ID and a uniformly random grid cell
    as the placement position.  No masking of any kind.
    """

    def __init__(self, seed: int = 42, window_title: str = DEFAULT_WINDOW_TITLE):
        self.seed         = seed
        self.window_title = window_title
        self._rng         = random.Random(seed)

    def reset(self):
        """Called by eval_runner at the start of every game."""
        pass  # stateless — nothing to reset

    def select_action(self, obs, slots) -> dict:
        """
        Returns a random action and a random screen position.

        Args:
            obs   : pd.DataFrame  (current environment observation)
            slots : dict          (current card slots, ignored here)

        Returns:
            {"action_id": int, "pos_x": float, "pos_y": float}
        """
        action_id = self._rng.randint(0, NUM_ACTIONS - 1)

        if action_id == 0:  # wait — no position needed
            return {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

        # Random grid cell -> BlueStacks pixel -> global screen coords
        gx = self._rng.randint(0, _GRID_W - 1)
        gy = self._rng.randint(0, _GRID_H - 1)

        try:
            bs_x, bs_y = grid_to_pixel(gx, gy)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[RandomPolicy] Coordinate conversion failed: {e} — using (-1, -1)")
            pos_x, pos_y = -1.0, -1.0

        return {"action_id": action_id, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def main():
    parser = argparse.ArgumentParser(description="Random baseline evaluation")
    parser.add_argument("--seed",    type=int, default=42,      help="Random seed")
    parser.add_argument("--n_games", type=int, default=EVAL_GAMES, help="Number of evaluation games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE, help="BlueStacks window title")
    args = parser.parse_args()

    policy = RandomPolicy(seed=args.seed, window_title=args.window)

    print(f"\n{'='*60}")
    print(f"Random Policy Evaluation  |  seed={args.seed}  |  games={args.n_games}")
    print(f"{'='*60}\n")

    run_evaluation(
        policy       = policy,
        policy_name  = "random",
        seed         = args.seed,
        log_dir      = baseline_log_dir("random"),
        eval_dir     = baseline_eval_dir("random"),
        n_games      = args.n_games,
        window_title = args.window,
    )


if __name__ == "__main__":
    main()
