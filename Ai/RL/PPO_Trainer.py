
import torch.nn.functional as F
import torch
import torch.nn as nn
from pyclipper import log_action
from Ai.Behavior_Cloning.action_masking_config import (
    WAIT_ID,
    ALWAYS_ALLOW_WAIT,
    AVAIL_FEATURE_TO_ACTION_ID,
)


import pandas as pd


from Ai.Data_Cleaning import final_clean


def clean_obs(obs):
    cleaned_frame = final_clean(obs)
    cleaned_frame = cleaned_frame.drop(columns=["match_id", "id"], errors="ignore")
    cleaned_frame = cleaned_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    last_cleaned_columns = cleaned_frame.columns.tolist()

    if cleaned_frame.shape[1] != 205:
        raise ValueError(
            f"Expected {205} features, got {cleaned_frame.shape[1]} after cleaning."
        )

    # Return a flat numeric vector for sequence buffering.
    return cleaned_frame.iloc[0].astype(float).tolist()

def build_action_mask_from_obs(obs, num_actions=13):
    """
    Build a [num_actions] bool mask from a raw env observation (pd.DataFrame),
    using the same availability logic as behavior cloning.

    True  => action is legal
    False => action is masked out (logit -> -1e9)
    """
    try:
        # Reuse your existing cleaning pipeline so *_avab columns are created.
        cleaned_df = final_clean(obs)
        if not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
            return torch.ones(num_actions, dtype=torch.bool)

        last_row = cleaned_df.iloc[0]
    except Exception:
        # On any failure, fail open: allow all actions.
        return torch.ones(num_actions, dtype=torch.bool)

    mask = torch.ones(num_actions, dtype=torch.bool)

    # Start by marking all mapped ids illegal, then re‑enable when their *_avab > 0
    mapped_ids = {v for v in AVAIL_FEATURE_TO_ACTION_ID.values() if v is not None}
    for aid in mapped_ids:
        if 0 <= aid < num_actions:
            mask[aid] = False

    for feat, aid in AVAIL_FEATURE_TO_ACTION_ID.items():
        if aid is None or not (0 <= aid < num_actions):
            continue
        if feat in last_row.index and last_row[feat] > 0:
            mask[aid] = True

    # Wait is always allowed
    if ALWAYS_ALLOW_WAIT and 0 <= WAIT_ID < num_actions:
        mask[WAIT_ID] = True

    return mask

def sequenece_buffering(obs,sequence_buffer,window_size,input_size):
    cleaned_frame = clean_obs(obs)

    sequence_buffer.append(cleaned_frame)

    current_sequence = list(sequence_buffer)

    if len(current_sequence) < window_size:
        pad_rows = window_size - len(current_sequence)
        padding = [[0.0] * input_size for _ in range(pad_rows)]
        current_sequence = padding + current_sequence

    return current_sequence

def reset_sequence_buffer(sequence_buffer):
    sequence_buffer.clear()

## masking actions based on availability features in the obs can be done later after the prototype is working end to end,
## for now we will just let the model learn to not take illegal actions by giving it negative rewards when it does so.


###


###

###

def actor_critic_update(
        actor_critic_network,
        optimizer,
        rollout,          # dict from collect_rollout, must include "advantages" and "returns"
        vf=0.5,
        ent_coef=0.01,
        epsilon=0.2,
):
    device = next(actor_critic_network.parameters()).device

    # --- Unpack rollout dict ---
    windows       = rollout["windows"]       # list of T items, each [10, 205]
    actions       = rollout["actions"]       # list of T ints
    xs            = rollout["x"]             # list of T floats (global screen x)
    ys            = rollout["y"]             # list of T floats (global screen y)
    old_log_probs = rollout["log_probs"]     # list of T floats (action+pos combined)
    advantages    = rollout["advantages"]    # list of T floats (precomputed outside)
    returns       = rollout["returns"]       # list of T floats (precomputed outside)

    T = len(windows)

    # --- Convert to tensors ---
    states_t      = torch.tensor(windows,       dtype=torch.float32, device=device)  # [T, 10, 205]
    actions_t     = torch.tensor(actions,       dtype=torch.long,    device=device)  # [T]
    positions_t   = torch.stack([
                        torch.tensor(xs, dtype=torch.float32, device=device),
                        torch.tensor(ys, dtype=torch.float32, device=device)
                    ], dim=1)                                                         # [T, 2]
    old_lp_t      = torch.tensor(old_log_probs, dtype=torch.float32, device=device)  # [T]
    advantages_t  = torch.tensor(advantages,    dtype=torch.float32, device=device)  # [T]
    returns_t     = torch.tensor(returns,       dtype=torch.float32, device=device)  # [T]

    # Normalize advantages (stabilizes training)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # --- Forward pass: re-run model on all windows under the CURRENT (updated) policy ---
    action_logits, pos_logits, value_estimate, _ = actor_critic_network(states_t)
    # action_logits:  [T, 13]
    # pos_logits:     [T, 2]
    # value_estimate: [T, 1]

    # --- Build distributions from fresh logits ---
    dist_action = torch.distributions.Categorical(logits=action_logits)      # discrete over 13 actions
    dist_pos    = torch.distributions.Normal(loc=pos_logits, scale=1.0)      # continuous over (x, y)

    # --- Current log probs under the NEW policy ---
    action_log_probs  = dist_action.log_prob(actions_t)                      # [T]
    pos_log_probs     = dist_pos.log_prob(positions_t).sum(dim=-1)           # [T, 2] -> sum -> [T]
    current_log_probs = action_log_probs + pos_log_probs                     # [T]

    # --- PPO clipped policy loss ---
    ratios    = torch.exp(current_log_probs - old_lp_t)                      # [T]
    surr1     = ratios * advantages_t                                         # [T]
    surr2     = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages_t  # [T]
    policy_loss = -torch.min(surr1, surr2).mean()

    # --- Value loss ---
    values      = value_estimate.squeeze(-1)                                  # [T, 1] -> [T]
    value_loss  = F.mse_loss(values, returns_t)

    # --- Entropy bonus ---
    entropy     = dist_action.entropy() + dist_pos.entropy().sum(-1)         # [T]
    entropy_loss = -ent_coef * entropy.mean()

    # --- Total loss ---
    loss = policy_loss + vf * value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic_network.parameters(), max_norm=0.5)
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def compute_returns_and_advantages(rollout, gamma=0.99):
    """
    Takes a rollout dict (from collect_rollout) and adds:
      - rollout["returns"]    : discounted return-to-go G_t for each step
      - rollout["advantages"] : A_t = G_t - V_t for each step

    Returns the same dict with those two keys added.
    """
    rewards = rollout["rewards"]   # list of T floats
    values  = rollout["values"]    # list of T floats (critic estimates from rollout)
    T       = len(rewards)

    returns    = [0.0] * T
    advantages = [0.0] * T

    # --- Compute returns-to-go backwards ---
    # G_{T-1} = r_{T-1}
    # G_t     = r_t + gamma * G_{t+1}
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    # --- Compute advantages: A_t = G_t - V_t ---
    for t in range(T):
        advantages[t] = returns[t] - values[t]

    rollout["returns"]    = returns
    rollout["advantages"] = advantages

    return rollout


