from collections import deque
from time import sleep

from Ai.RL.PPO_Trainer import sequenece_buffering, build_action_mask_from_obs , compute_returns_and_advantages
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Behavior_Cloning.action_masking_config import WAIT_ID

import torch
import traceback
import sys

from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model



## here we will implement the main training loop for PPO, including interaction with the environment, collecting transitions,
# and updating the policy and value networks using the data in the PPOBuffer. We will also include logging and model saving functionality.

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
        done = False
        slots = {}
        current_slots = {}

        state, current_slots = env.reset()
        if state is None:
            print("faild to get the env first obs")
            return
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

                # Mask illegal actions like in BC: illegal -> very negative logit
                masked_action_logits = action_logits.masked_fill(~action_mask, -1e9)

                # === 2) Sample action and position ===
                dist_action = torch.distributions.Categorical(logits=masked_action_logits)
                dist_pos = torch.distributions.Normal(loc=pos_logits, scale=1.0)

                action = dist_action.sample()
                pos = dist_pos.sample()

                action_val = action.item()

                # === 3) Interpret position as grid coords from the BC head ===
                if action_val == WAIT_ID:
                    # Consistent with BC inference: wait => no placement
                    gx, gy = -1, -1
                    pos_x, pos_y = -1.0, -1.0
                else:
                    # Treat model outputs as grid indices and clamp
                    gx = int(round(pos[0].item()))
                    gy = int(round(pos[1].item()))

                    # grid_to_pixel itself clamps to [0, GRID_W-1] / [0, GRID_H-1]
                    bs_x, bs_y = grid_to_pixel(gx, gy)

                    # Map Bluestacks virtual coords -> global screen coords
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
                next_state, reward, done, slots = env.step(action_val, pos_x, pos_y, state,current_slots)
                while next_state is None:
                    print("[WARN] Step returned None, retrying after short delay...")
                    next_state, reward, done, slots = env.step(action_val, pos_x, pos_y, next_state,current_slots)
                    sleep(0.5)
                if isinstance(next_state, str):
                    print(f"[INFO] Received terminal status '{next_state}' from env step, treating as done")
                    done = True
                    reward = env.reward_win if next_state.lower() == "win" else env.reward_lose if next_state.lower() == "loss" else 0.0
                state = next_state
                current_slots = slots

                # Store transition
                windows.append(current_window)
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
        print("[DEBUG] PPO_LSTM_Model instantiated successfully")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate PPO_LSTM_Model: {e}")
        traceback.print_exc()
        return

    try:
        print("[DEBUG] Starting rollout collection...")
        rollouts = collect_rollout(env, model, rollouts_to_collect=1)
        print(f"[DEBUG] Rollouts collected: {len(rollouts)} rollouts")
        print(rollouts)
    except Exception as e:
        print(f"[ERROR] Failed to collect rollouts: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    print("[DEBUG] PPO_Main.py started")
    main()
    print("[DEBUG] PPO_Main.py completed")
