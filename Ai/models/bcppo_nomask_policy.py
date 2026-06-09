"""BC->PPO No-Mask training script  (seed 1 only).

Ablation: tests whether action masking is doing meaningful work.
BC warm-start is ON, but action masking is fully disabled —
build_action_mask_from_obs and the elixir guard are both bypassed
(every step gets an all-True mask).

One seed only because this is a secondary ablation.

Run:
    python -m Ai.models.bcppo_nomask_policy

What this produces:
    Ai/logs/ppo/BCPPO_NoMask_s1/training_log.csv
    Ai/logs/ppo/BCPPO_NoMask_s1/updates.json
    Ai/logs/ppo/BCPPO_NoMask_s1/rollouts.json
    Ai/logs/ppo/BCPPO_NoMask_s1/winrate.json
    Ai/logs/ppo/BCPPO_NoMask_s1/run_summary.json
    Ai/checkpoints/ppo/BCPPO_NoMask_s1/game_020.pth ... game_100.pth
    Ai/checkpoints/ppo/BCPPO_NoMask_s1/best.pth
"""

import argparse
from Ai.RL.PPO_Main import main
from Ai.models.run_config import PPO_TRAINING_GAMES, DEFAULT_WINDOW_TITLE

_RUN_ID = "BCPPO_NoMask_s1"
_SEED   = 1


def run(n_games: int = PPO_TRAINING_GAMES, window_title: str = DEFAULT_WINDOW_TITLE):
    """Entry point callable from other scripts or notebooks."""
    return main(
        run_id         = _RUN_ID,
        seed           = _SEED,
        use_pretrained = True,   # BC warm-start ON
        use_masking    = False,  # action masking OFF — ablation
        n_games        = n_games,
        window_title   = window_title,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BC->PPO No-Mask training (ablation: masking disabled)."
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
    run(n_games=args.n_games, window_title=args.window)
