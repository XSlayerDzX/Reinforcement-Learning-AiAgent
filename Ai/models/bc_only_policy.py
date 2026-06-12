"""BC-only baseline evaluation  —  frozen BC model, no PPO fine-tuning.

To run directly (e.g. from PyCharm): just edit SEED below, then
run this file.  No command-line arguments needed.

To run from terminal:
    python -m Ai.models.bc_only_policy --seed 42
"""
import argparse
import os
from pathlib import Path

import torch

from Ai.models.eval_runner import run_evaluation
from Ai.models.run_config import (
    DEFAULT_WINDOW_TITLE,
    EVAL_GAMES,
    BC_CHECKPOINT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    INPUT_SIZE,
    NUM_ACTIONS,
    WINDOW_SIZE,
    baseline_eval_dir,
    baseline_log_dir,
)
from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
from Ai.RL.PPO_Trainer import sequenece_buffering, build_action_mask_from_obs
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Behavior_Cloning.action_masking_config import WAIT_ID
from collections import deque

# ── Edit these to launch directly from PyCharm / file-run ────────────────────
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_GX = 4
_DEFAULT_GY = 10


class BCOnlyPolicy:
    """Runs the frozen BC (behaviour-cloning) LSTM model with action masking."""

    def __init__(self, window_title: str = DEFAULT_WINDOW_TITLE):
        self.window_title  = window_title
        self._seq_buf      = deque(maxlen=WINDOW_SIZE)
        self.model         = PPO_LSTM_Model(
            input_size            = INPUT_SIZE,
            hidden_size           = HIDDEN_SIZE,
            num_layers            = NUM_LAYERS,
            num_actions           = NUM_ACTIONS,
            pretrained_model_path = str(BC_CHECKPOINT),
        )
        self.model.eval()
        print(f"[BCOnly] Loaded BC checkpoint from {BC_CHECKPOINT}")

    def reset(self):
        self._seq_buf.clear()

    def select_action(self, obs, slots) -> dict:
        wait_result = {"action_id": WAIT_ID, "pos_x": -1.0, "pos_y": -1.0}
        if obs is None or obs.empty:
            return wait_result

        window = sequenece_buffering(obs, self._seq_buf, WINDOW_SIZE, INPUT_SIZE)
        window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_logits, pos_logits, _, _ = self.model(window_t)

        action_logits = action_logits.squeeze(0)
        pos_logits    = pos_logits.squeeze(0)

        mask = build_action_mask_from_obs(obs, num_actions=action_logits.shape[-1])
        masked_logits = action_logits.masked_fill(~mask, -1e9)

        dist   = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample().item()

        if action == WAIT_ID:
            return wait_result

        try:
            gx = int(round(pos_logits[0].item()))
            gy = int(round(pos_logits[1].item()))
            bs_x, bs_y = grid_to_pixel(gx, gy)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[BCOnly] Coordinate error: {e}")
            pos_x, pos_y = -1.0, -1.0

        return {"action_id": action, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def _run(seed: int, n_games: int, window: str):
    policy = BCOnlyPolicy(window_title=window)
    print(f"\n{'='*60}")
    print(f"BC-Only Evaluation  |  seed={seed}  |  games={n_games}")
    print(f"{'='*60}\n")
    run_evaluation(
        policy       = policy,
        policy_name  = "bc_only",
        seed         = seed,
        log_dir      = baseline_log_dir("bc_only"),
        eval_dir     = baseline_eval_dir("bc_only"),
        n_games      = n_games,
        window_title = window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC-only baseline evaluation")
    parser.add_argument("--seed",    type=int, default=SEED,                help="Random seed")
    parser.add_argument("--n_games", type=int, default=EVAL_GAMES,          help="Evaluation games")
    parser.add_argument("--window",  type=str, default=DEFAULT_WINDOW_TITLE,help="BlueStacks window")
    args = parser.parse_args()
    _run(args.seed, args.n_games, args.window)
