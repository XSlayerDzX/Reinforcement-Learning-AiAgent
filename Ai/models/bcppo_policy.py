"""BC -> PPO training script  (seeds 1 and 2).

This is the MAIN METHOD of the paper.
Initializes the PPO_LSTM_Model with BC warm-start weights (lstm.pth),
keeps action masking ON, and trains for 100 games.

Run seed 1:
    python -m Ai.models.bcppo_policy --seed 1

Run seed 2:
    python -m Ai.models.bcppo_policy --seed 2

What this produces:
    Ai/logs/ppo/BCPPO_s{seed}/training_log.csv
    Ai/logs/ppo/BCPPO_s{seed}/updates.json
    Ai/logs/ppo/BCPPO_s{seed}/rollouts.json
    Ai/logs/ppo/BCPPO_s{seed}/winrate.json
    Ai/logs/ppo/BCPPO_s{seed}/run_summary.json
    Ai/checkpoints/ppo/BCPPO_s{seed}/game_020.pth ... game_100.pth
    Ai/checkpoints/ppo/BCPPO_s{seed}/best.pth
"""

import argparse
from Ai.RL.PPO_Main import main
from Ai.models.run_config import PPO_TRAINING_GAMES, DEFAULT_WINDOW_TITLE


def run(seed: int, n_games: int = PPO_TRAINING_GAMES, window_title: str = DEFAULT_WINDOW_TITLE):
    """Entry point callable from other scripts or notebooks."""
    run_id = f"BCPPO_s{seed}"
    return main(
        run_id         = run_id,
        seed           = seed,
        use_pretrained = True,   # BC warm-start ON
        use_masking    = True,   # action masking ON
        n_games        = n_games,
        window_title   = window_title,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BC->PPO training (main method). Runs one seed."
    )
    parser.add_argument(
        "--seed", type=int, required=True, choices=[1, 2],
        help="Seed index: 1 -> BCPPO_s1, 2 -> BCPPO_s2"
    )
    parser.add_argument(
        "--n_games", type=int, default=PPO_TRAINING_GAMES,
        help=f"Training games (default: {PPO_TRAINING_GAMES})"
    )
    parser.add_argument(
        "--window", type=str, default=DEFAULT_WINDOW_TITLE,
        help="BlueStacks window title"
    )
    args = parser.parse_args()
    run(seed=args.seed, n_games=args.n_games, window_title=args.window)
