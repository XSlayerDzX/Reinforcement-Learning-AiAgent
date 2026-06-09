"""BC-only baseline policy.

Loads the trained LSTM checkpoint (lstm.pth) and runs it in pure
inference mode — no PPO fine-tuning, no gradient updates, no online
learning.  Action selection is deterministic: argmax over masked logits.

This is the most important baseline for the paper — it is the direct
comparison target for BC→PPO.  If BC→PPO does not beat this, the
central claim of the paper does not hold.

Run:
    python -m Ai.models.bc_only_policy --seed 42
"""

import argparse

import torch

from Ai.models.eval_runner import run_evaluation
from Ai.models.run_config import (
    BC_CHECKPOINT,
    DEFAULT_WINDOW_TITLE,
    EVAL_GAMES,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_ACTIONS,
    NUM_LAYERS,
    WINDOW_SIZE,
    baseline_eval_dir,
    baseline_log_dir,
)
from Ai.Behavior_Cloning.lstm_inference_pipeline import LSTM_Inference_Pipeline
from Ai.Behavior_Cloning.action_masking_config import get_masking_kwargs
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords


class BCOnlyPolicy:
    """
    Wraps LSTM_Inference_Pipeline for use with eval_runner.
    Deterministic: always picks argmax action after legal-action masking.
    """

    def __init__(
        self,
        checkpoint_path=BC_CHECKPOINT,
        window_title: str = DEFAULT_WINDOW_TITLE,
    ):
        self.window_title = window_title
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[BCOnlyPolicy] Loading checkpoint: {checkpoint_path}")
        self._pipeline = LSTM_Inference_Pipeline(
            model_path  = str(checkpoint_path),
            device      = device,
            window_size = WINDOW_SIZE,
            input_size  = INPUT_SIZE,
            output_size = NUM_ACTIONS,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            **get_masking_kwargs(),
        )
        print("[BCOnlyPolicy] Checkpoint loaded successfully.")

    def reset(self):
        """Clear the LSTM sequence buffer at the start of every game."""
        self._pipeline.reset_sequence_buffer()

    def select_action(self, obs, slots) -> dict:
        """
        Run the LSTM inference pipeline on the current observation.

        Args:
            obs   : pd.DataFrame  (current environment observation)
            slots : dict          (ignored — masking is handled inside pipeline)

        Returns:
            {"action_id": int, "pos_x": float, "pos_y": float}
        """
        wait_result = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

        if obs is None or obs.empty:
            return wait_result

        try:
            prediction = self._pipeline.predict(obs)
        except Exception as e:
            print(f"[BCOnlyPolicy] Inference failed: {e} — defaulting to wait")
            return wait_result

        action_id = prediction["action_id"]
        gx, gy    = prediction["pos_pred"]

        if action_id == 0:
            return wait_result

        # Convert grid position to screen coordinates
        try:
            bs_x, bs_y = grid_to_pixel(gx, gy)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[BCOnlyPolicy] Coordinate conversion failed: {e} — using (-1, -1)")
            pos_x, pos_y = -1.0, -1.0

        return {"action_id": action_id, "pos_x": float(pos_x), "pos_y": float(pos_y)}


def main():
    parser = argparse.ArgumentParser(description="BC-only baseline evaluation")
    parser.add_argument("--seed",       type=int,  default=42,            help="Seed (logged only)")
    parser.add_argument("--n_games",    type=int,  default=EVAL_GAMES,    help="Number of evaluation games")
    parser.add_argument("--checkpoint", type=str,  default=str(BC_CHECKPOINT), help="Path to lstm.pth")
    parser.add_argument("--window",     type=str,  default=DEFAULT_WINDOW_TITLE, help="BlueStacks window title")
    args = parser.parse_args()

    policy = BCOnlyPolicy(
        checkpoint_path = args.checkpoint,
        window_title    = args.window,
    )

    print(f"\n{'='*60}")
    print(f"BC-Only Policy Evaluation  |  seed={args.seed}  |  games={args.n_games}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    run_evaluation(
        policy       = policy,
        policy_name  = "bc_only",
        seed         = args.seed,
        log_dir      = baseline_log_dir("bc_only"),
        eval_dir     = baseline_eval_dir("bc_only"),
        n_games      = args.n_games,
        window_title = args.window,
    )


if __name__ == "__main__":
    main()
