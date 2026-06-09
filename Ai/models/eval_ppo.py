"""Post-training PPO evaluation script.

Loads a trained PPO checkpoint (by run_id + game number, or best.pth)
and runs it in PURE INFERENCE MODE for N games using the shared
eval_runner.  No gradient updates, no learning — evaluation only.

This produces the final numbers that go into Table 1 of the paper.
Run AFTER training is complete for each run_id.

Examples:
    # Evaluate best checkpoint of BCPPO_s1
    python -m Ai.models.eval_ppo --run_id BCPPO_s1 --checkpoint best

    # Evaluate game-100 checkpoint of PPOScratch_s1
    python -m Ai.models.eval_ppo --run_id PPOScratch_s1 --checkpoint 100

    # Evaluate all 5 runs using their best checkpoint
    python -m Ai.models.eval_ppo --run_id BCPPO_s1       --checkpoint best
    python -m Ai.models.eval_ppo --run_id BCPPO_s2       --checkpoint best
    python -m Ai.models.eval_ppo --run_id PPOScratch_s1  --checkpoint best
    python -m Ai.models.eval_ppo --run_id PPOScratch_s2  --checkpoint best
    python -m Ai.models.eval_ppo --run_id BCPPO_NoMask_s1 --checkpoint best

What this produces (per run):
    Ai/evaluations/ppo/{run_id}/eval_{run_id}_games.csv
    Ai/evaluations/ppo/{run_id}/eval_{run_id}_aggregate.csv
    Ai/logs/ppo/{run_id}_eval/training_log.csv
    Ai/logs/ppo/{run_id}_eval/run_summary.json
"""

import argparse
from pathlib import Path

import torch
import pandas as pd

from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
from Ai.RL.PPO_Trainer import build_action_mask_from_obs, sequenece_buffering
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Behavior_Cloning.action_masking_config import WAIT_ID
from Ai.ClashRoyalData import ElixirCost
from Ai.Behavior_Cloning.action_masking_config import AVAIL_FEATURE_TO_ACTION_ID

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
    PPO_RUN_IDS,
    ppo_checkpoint_dir,
    ppo_checkpoint_path,
    ppo_best_checkpoint_path,
    ppo_eval_dir,
    ppo_log_dir,
)

from collections import deque


# ---------------------------------------------------------------------------
# PPO inference policy  (wraps trained model for use with eval_runner)
# ---------------------------------------------------------------------------

class PPOInferencePolicy:
    """
    Loads a trained PPO_LSTM_Model checkpoint and exposes the
    eval_runner interface: .reset() and .select_action(obs, slots).

    Action selection mirrors the rollout collection logic exactly:
      - masking ON by default (controlled by use_masking flag)
      - deterministic: argmax over masked logits
      - elixir safety guard applied when masking is ON
    """

    def __init__(
        self,
        checkpoint_path: Path,
        use_masking: bool     = True,
        window_title: str     = DEFAULT_WINDOW_TITLE,
    ):
        self.use_masking  = use_masking
        self.window_title = window_title
        self._window_buf  = deque(maxlen=WINDOW_SIZE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        print(f"[PPOInferencePolicy] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Detect whether the run used BC warm-start from checkpoint metadata
        run_id = ckpt.get("run_id", "")
        use_pretrained = "Scratch" not in run_id

        self._model = PPO_LSTM_Model(
            input_size            = INPUT_SIZE,
            hidden_size           = HIDDEN_SIZE,
            num_layers            = NUM_LAYERS,
            num_actions           = NUM_ACTIONS,
            pretrained_model_path = None,       # weights loaded from ckpt below
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        print(
            f"[PPOInferencePolicy] Loaded run_id={run_id!r}  "
            f"game={ckpt.get('game_id','?')}  "
            f"wr={ckpt.get('win_rate_last_20','?')}"
        )

    def reset(self):
        """Clear LSTM sequence buffer between games."""
        self._window_buf.clear()

    def select_action(self, obs: pd.DataFrame, slots: dict) -> dict:
        """
        Run one forward pass and return the greedy (argmax) action.

        Args:
            obs   : pd.DataFrame  (current environment observation)
            slots : dict          (ignored — masking uses obs directly)

        Returns:
            {"action_id": int, "pos_x": float, "pos_y": float}
        """
        wait_result = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

        if obs is None or obs.empty:
            return wait_result

        # Build sequence window
        try:
            current_window = sequenece_buffering(
                obs, self._window_buf, WINDOW_SIZE, INPUT_SIZE
            )
        except Exception as e:
            print(f"[PPOInferencePolicy] sequenece_buffering failed: {e}")
            return wait_result

        window_tensor = torch.tensor(
            current_window, dtype=torch.float32, device=self._device
        ).unsqueeze(0)  # [1, window, features]

        with torch.no_grad():
            action_logits, pos_logits, _, _ = self._model(window_tensor)

        action_logits = action_logits.squeeze(0)  # [num_actions]
        pos_logits    = pos_logits.squeeze(0)      # [2]

        # --- action mask ---
        if self.use_masking:
            action_mask = build_action_mask_from_obs(
                obs, num_actions=action_logits.shape[-1]
            )
            # Elixir guard
            try:
                current_elixir = float(obs["Elixir"].iloc[0])
            except (KeyError, ValueError, TypeError):
                current_elixir = 10.0

            for card_name, cost in ElixirCost.items():
                avab_key = card_name + "_avab"
                aid = AVAIL_FEATURE_TO_ACTION_ID.get(avab_key)
                if aid is not None and current_elixir < cost:
                    action_mask[aid] = False
        else:
            action_mask    = torch.ones(action_logits.shape[-1], dtype=torch.bool)
            current_elixir = 10.0

        masked_logits = action_logits.masked_fill(~action_mask, -1e9)

        # Deterministic: argmax
        action_val = int(torch.argmax(masked_logits).item())

        # Final elixir safety check (masking ON only)
        if self.use_masking and action_val != WAIT_ID:
            ACTION_ID_TO_CARD = {
                v: k.replace("_avab", "")
                for k, v in AVAIL_FEATURE_TO_ACTION_ID.items() if v is not None
            }
            if action_val in ACTION_ID_TO_CARD:
                card_name = ACTION_ID_TO_CARD[action_val]
                card_cost = ElixirCost.get(card_name, 0)
                try:
                    live_elixir = float(obs["Elixir"].iloc[0]) - 1
                except (KeyError, ValueError, TypeError):
                    live_elixir = 0.0
                if live_elixir < card_cost:
                    action_val = WAIT_ID

        if action_val == WAIT_ID:
            return wait_result

        # Convert position to screen coordinates
        gx = int(round(pos_logits[0].item()))
        gy = int(round(pos_logits[1].item()))
        try:
            bs_x, bs_y = grid_to_pixel(gx, gy)
            pos_x, pos_y = bluestacks_to_global_coords(
                bs_x, bs_y,
                bluestacks_resolution=(540, 960),
                window_title=self.window_title,
            )
        except Exception as e:
            print(f"[PPOInferencePolicy] Coordinate conversion failed: {e}")
            pos_x, pos_y = -1.0, -1.0

        return {"action_id": action_val, "pos_x": float(pos_x), "pos_y": float(pos_y)}


# ---------------------------------------------------------------------------
# checkpoint path resolver
# ---------------------------------------------------------------------------

def resolve_checkpoint(run_id: str, checkpoint: str) -> Path:
    """
    Resolve the checkpoint path from run_id + checkpoint specifier.

    Args:
        run_id     : e.g. "BCPPO_s1"
        checkpoint : "best" | integer string e.g. "100" | full path string

    Returns:
        Path to the .pth file.

    Raises:
        FileNotFoundError if the resolved path does not exist.
    """
    if checkpoint == "best":
        path = ppo_best_checkpoint_path(run_id)
    elif checkpoint.isdigit():
        path = ppo_checkpoint_path(run_id, int(checkpoint))
    else:
        path = Path(checkpoint)  # treat as literal path

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Available checkpoints in {ppo_checkpoint_dir(run_id)}:\n"
            + "\n".join(
                str(p) for p in sorted(ppo_checkpoint_dir(run_id).glob("*.pth"))
            )
        )
    return path


# ---------------------------------------------------------------------------
# main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    run_id: str,
    checkpoint: str       = "best",
    seed: int             = 42,
    n_games: int          = EVAL_GAMES,
    window_title: str     = DEFAULT_WINDOW_TITLE,
):
    """
    Load a trained PPO checkpoint and evaluate it for n_games.

    Args:
        run_id     : one of PPO_RUN_IDS
        checkpoint : "best", an integer game number (e.g. "100"), or a full path
        seed       : logged in the output CSVs
        n_games    : number of evaluation games
        window_title: BlueStacks window title

    Returns:
        (summary dict, list of per-game record dicts)
    """
    assert run_id in PPO_RUN_IDS, (
        f"Unknown run_id {run_id!r}. Valid: {PPO_RUN_IDS}"
    )

    ckpt_path  = resolve_checkpoint(run_id, checkpoint)
    use_masking = "NoMask" not in run_id

    policy = PPOInferencePolicy(
        checkpoint_path = ckpt_path,
        use_masking     = use_masking,
        window_title    = window_title,
    )

    # Eval logs go to a separate subfolder so they don't mix with training logs
    eval_log_dir = ppo_log_dir(run_id).parent / f"{run_id}_eval"

    print(f"\n{'='*60}")
    print(f"PPO Evaluation  |  run_id={run_id}  |  checkpoint={checkpoint}")
    print(f"use_masking={use_masking}  |  seed={seed}  |  games={n_games}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    return run_evaluation(
        policy       = policy,
        policy_name  = run_id,
        seed         = seed,
        log_dir      = eval_log_dir,
        eval_dir     = ppo_eval_dir(run_id),
        n_games      = n_games,
        window_title = window_title,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-training PPO evaluation. Run after training is complete."
    )
    parser.add_argument(
        "--run_id", type=str, required=True,
        choices=PPO_RUN_IDS,
        help="Which PPO run to evaluate"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best",
        help="'best', an integer game number e.g. '100', or a full path to a .pth file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed value logged in output CSVs (default: 42)"
    )
    parser.add_argument(
        "--n_games", type=int, default=EVAL_GAMES,
        help=f"Number of evaluation games (default: {EVAL_GAMES})"
    )
    parser.add_argument(
        "--window", type=str, default=DEFAULT_WINDOW_TITLE,
        help="BlueStacks window title"
    )
    args = parser.parse_args()

    evaluate(
        run_id      = args.run_id,
        checkpoint  = args.checkpoint,
        seed        = args.seed,
        n_games     = args.n_games,
        window_title= args.window,
    )
