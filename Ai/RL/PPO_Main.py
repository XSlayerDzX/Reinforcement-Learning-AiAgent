from collections import deque
from time import sleep

from Ai.RL.PPO_Trainer import sequenece_buffering, build_action_mask_from_obs , compute_returns_and_advantages,actor_critic_update
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Behavior_Cloning.action_masking_config import WAIT_ID
from Ai.ClashRoyalData import ElixirCost
from Ai.Behavior_Cloning.action_masking_config import AVAIL_FEATURE_TO_ACTION_ID

import torch
import pandas as pd
import traceback
import sys

from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model



## here we will implement the main training loop for PPO, including interaction with the environment, collecting transitions,
# and updating the policy and value networks using the data in the PPOBuffer. We will also include logging and model saving functionality.

def _ensure_dataframe(obs):
    """
    Normalize an observation to a pandas DataFrame expected by the cleaning functions.
    - If obs is already a DataFrame, return a copy.
    - If obs is a dict (single row), wrap into DataFrame([obs]).
    - If obs is None or other unexpected type, return None.
    """
    if obs is None:
        return None
    if isinstance(obs, pd.DataFrame):
        return obs.copy()
    if isinstance(obs, dict):
        try:
            return pd.DataFrame([obs])
        except Exception:
            return None
    return None

def collect_rollout(env, model, rollouts_to_collect=1):
    rollouts = []
    window_buffer = deque(maxlen=10)

    for rollout in range(rollouts_to_collect):
        model.eval()
        window_buffer.clear()
        windows = []
        actions = []
        x = []
        y = []
        log_probs = []
        rewards = []
        values = []
        masks = []
        done = False
        slots = {}
        current_slots = {}

        state_raw, current_slots = env.reset()
        state = _ensure_dataframe(state_raw)
        if state is None or "slot_1" not in state.columns:
            print("Failed to get valid initial obs from env.reset(), aborting rollout")
            return rollouts

        with torch.no_grad():
            while not done:
                # Build 10x205 window from the current raw obs (pd.DataFrame)
                current_window = sequenece_buffering(state, window_buffer, 10, 205)
                window_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)

                # Forward pass
                action_logits, pos_logits, value_estimate, (h_s, c_s) = model(window_tensor)

                # Remove batch dimension
                action_logits = action_logits.squeeze(0)
                pos_logits = pos_logits.squeeze(0)

                # === 1) Build legal‑action mask from current obs ===
                action_mask = build_action_mask_from_obs(state, num_actions=action_logits.shape[-1])

                # Secondary elixir guard using live current_slots (more reliable than parsed obs)
                current_elixir = state["Elixir"].iloc[0] if "Elixir" in state.columns else 10
                try:
                    current_elixir = float(current_elixir)
                except (ValueError, TypeError):
                    current_elixir = 10.0

                for card_name, cost in ElixirCost.items():
                    avab_key = card_name + "_avab"
                    aid = AVAIL_FEATURE_TO_ACTION_ID.get(avab_key)
                    if aid is not None and current_elixir < cost:
                        action_mask[aid] = False

                # Mask illegal actions like in BC: illegal -> very negative logit
                masked_action_logits = action_logits.masked_fill(~action_mask, -1e9)

                # === 2) Sample action and position ===
                dist_action = torch.distributions.Categorical(logits=masked_action_logits)
                dist_pos = torch.distributions.Normal(loc=pos_logits, scale=1.0)

                action = dist_action.sample()
                pos = dist_pos.sample()

                action_val = action.item()

                # === 3) Final elixir safety check — force wait if card is unaffordable ===
                # Map action_id back to card name
                ACTION_ID_TO_CARD = {v: k.replace("_avab", "") for k, v in AVAIL_FEATURE_TO_ACTION_ID.items() if
                                     v is not None}
                if action_val in ACTION_ID_TO_CARD:
                    card_name = ACTION_ID_TO_CARD[action_val]
                    card_cost = ElixirCost.get(card_name, 0)
                    try:
                        live_elixir = float(state["Elixir"].iloc[0]) - 1  # -1 safety buffer
                    except (KeyError, ValueError, TypeError):
                        live_elixir = 0.0
                    if live_elixir < card_cost:
                        print(f"[SAFETY] Forced wait: {card_name} costs {card_cost}, elixir={live_elixir:.1f}")
                        action_val = WAIT_ID
                        # Recompute log_prob for wait under the masked distribution
                        action = torch.tensor(WAIT_ID, dtype=torch.long)

                # === 4) Interpret position as grid coords from the BC head ===
                if action_val == WAIT_ID:
                    gx, gy = -1, -1
                    pos_x, pos_y = -1.0, -1.0
                else:
                    gx = int(round(pos[0].item()))
                    gy = int(round(pos[1].item()))
                    bs_x, bs_y = grid_to_pixel(gx, gy)
                    pos_x, pos_y = bluestacks_to_global_coords(
                        bs_x,
                        bs_y,
                        bluestacks_resolution=(540, 960),
                        window_title="BlueStacks App Player 4",
                    )

                # === 4) Log probabilities from the masked distributions ===
                log_prob_action = dist_action.log_prob(action)
                log_prob_pos = dist_pos.log_prob(pos).sum(dim=-1)
                log_prob = (log_prob_action + log_prob_pos).item()

                # Step env with final discrete action + global screen coords
                next_state_raw, reward, done, slots = env.step(action_val, pos_x, pos_y, state_raw, current_slots)

                # If step returned None, retry using the last valid raw state (state_raw)
                retry_attempts = 0
                while next_state_raw is None and retry_attempts < 5:
                    print("[WARN] Step returned None, retrying after short delay...")
                    sleep(0.5)
                    next_state_raw, reward, done, slots = env.step(action_val, pos_x, pos_y, state_raw, current_slots)
                    retry_attempts += 1

                # If env.step returns a terminal string status
                if isinstance(next_state_raw, str):
                    print(f"[INFO] Received terminal status '{next_state_raw}' from env step, treating as done")
                    done = True
                    reward = env.reward_win if next_state_raw.lower() == "win" else env.reward_lose if next_state_raw.lower() == "loss" else 0.0
                    next_state_raw = None

                # Normalize next_state to DataFrame for downstream processing
                next_state = _ensure_dataframe(next_state_raw)

                # If normalization failed, keep previous valid state to avoid crashes
                if next_state is None:
                    print("[WARN] Received unexpected obs type from env; keeping previous state for next step")
                    next_state = state.copy()
                    # keep current_slots unchanged or update from slots if available
                else:
                    state_raw = next_state_raw
                    state = next_state
                    current_slots = slots

                # Store transition
                windows.append(current_window)
                masks.append(action_mask)
                actions.append(action_val)
                values.append(value_estimate.item())
                x.append(pos_x)
                y.append(pos_y)
                log_probs.append(log_prob)
                rewards.append(reward)

        r = {
            "windows": windows,
            "actions": actions,
            "x": x,
            "y": y,
            "log_probs": log_probs,
            "rewards": rewards,
            "values": values,
            "masks": masks,
        }
        r_a = compute_returns_and_advantages(r)
        rollouts.append(r_a)

    return rollouts





def main():
    print("[DEBUG] Starting main function")
    try:
        print("[DEBUG] Attempting to instantiate ClashRoyalEnv...")
        env = ClashRoyalEnv()
        print("[DEBUG] ClashRoyalEnv instantiated successfully")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate ClashRoyalEnv: {e}")
        traceback.print_exc()
        return

    try:
        print("[DEBUG] Attempting to instantiate PPO_LSTM_Model...")

        model = PPO_LSTM_Model(
            input_size=205,
            hidden_size=128,
            num_layers=2,
            num_actions=13,
            pretrained_model_path=r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\Behavior_Cloning\lstm.pth"
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        # ── Resume from PPO checkpoint if it exists ──────────────────────────
        import os
        ppo_save_path = r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\RL\ppo_model.pth"

        if os.path.exists(ppo_save_path):
            print("[DEBUG] PPO checkpoint found, resuming from saved weights...")
            checkpoint = torch.load(ppo_save_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[DEBUG] Model and optimizer state restored successfully")
        else:
            print("[DEBUG] No PPO checkpoint found, starting from BC warm-start weights")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate PPO_LSTM_Model: {e}")
        traceback.print_exc()
        return

    try:
        print("[DEBUG] Starting rollout collection...")
        rollouts = collect_rollout(env, model, rollouts_to_collect=1)
        print(f"[DEBUG] Rollouts collected: {len(rollouts)} rollouts")
        if len(rollouts) > 0:
            first = rollouts[0]
            print("actions :", len(first.get("actions", [])))
            print("values :", len(first.get("values", [])))
            print("log_probs :", len(first.get("log_probs", [])))
            print("rewards :", len(first.get("rewards", [])))
            print("windows :", len(first.get("windows", [])))
        else:
            print("[DEBUG] No rollouts returned")
    except Exception as e:
        print(f"[ERROR] Failed to collect rollouts: {e}")
        traceback.print_exc()
        return

    try:
        print("[DEBUG] Starting PPO update...")
        for rollout in rollouts:
            policy_loss, value_loss = actor_critic_update(
                actor_critic_network=model,
                optimizer=opt,
                rollout=rollout,
            )
            print(f"[DEBUG] Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")

        # Save model and optimizer state after update
        save_path = r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\RL\ppo_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }, save_path)
        print(f"[DEBUG] Model saved to {save_path}")

    except Exception as e:
        print(f"[ERROR] Failed during PPO update: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    print("[DEBUG] PPO_Main.py started")
    main()
    print("[DEBUG] PPO_Main.py completed")
