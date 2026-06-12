"""PPOScratch_s1 / PPOScratch_s2  —  random init + action masking ON.

To run directly (e.g. from PyCharm): just edit SEED and RUN_ID below, then
run this file.  No command-line arguments needed.

To run from terminal:
    python -m Ai.models.pposcratch_policy --seed 1 --run_id PPOScratch_s1
"""
import argparse

from Ai.RL.PPO_Main import main
from Ai.models.run_config import DEFAULT_WINDOW_TITLE, PPO_TRAINING_GAMES

# ── Edit these to launch directly from PyCharm / file-run ────────────────────
SEED   = 1
RUN_ID = "PPOScratch_s1"   # options: "PPOScratch_s1"  |  "PPOScratch_s2"
# ─────────────────────────────────────────────────────────────────────────────


def _run(seed: int, run_id: str, n_games: int, window: str):
    main(
        run_id         = run_id,
        seed           = seed,
        use_pretrained = False,
        use_masking    = True,
        n_games        = n_games,
        window_title   = window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO from scratch (random init + masking)")
    parser.add_argument("--seed",    type=int, default=SEED,                help="Random seed")
    parser.add_argument("--run_id",  type=str, default=RUN_ID,              help="Run identifier")
    parser.add_argument("--n_games", type=int, default=PPO_TRAINING_GAMES,  help="Training games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE,help="BlueStacks window")
    args = parser.parse_args()
    _run(args.seed, args.run_id, args.n_games, args.window)
