import torch.nn.functional as F
import torch
import torch.nn as nn
from Ai.Behavior_Cloning.action_masking_config import (
    WAIT_ID,
    ALWAYS_ALLOW_WAIT,
    AVAIL_FEATURE_TO_ACTION_ID,
)

import pandas as pd

from Ai.Data_Cleaning import final_clean

EXPECTED_FEATURES = 205


def clean_obs(obs):
    """
    Clean a raw env observation into a flat list of EXPECTED_FEATURES floats.
    Pads with zeros (too few cols) or truncates (too many) to avoid crashes.
    """
    cleaned_frame = final_clean(obs)
    cleaned_frame = cleaned_frame.drop(columns=["match_id", "id"], errors="ignore")
    cleaned_frame = cleaned_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n_cols = cleaned_frame.shape[1]
    if n_cols != EXPECTED_FEATURES:
        print(
            f"[clean_obs] WARNING: expected {EXPECTED_FEATURES} features, "
            f"got {n_cols}. Padding/truncating to {EXPECTED_FEATURES}."
        )
        row = cleaned_frame.iloc[0].astype(float).tolist()
        if n_cols < EXPECTED_FEATURES:
            row = row + [0.0] * (EXPECTED_FEATURES - n_cols)
        else:
            row = row[:EXPECTED_FEATURES]
        return row

    return cleaned_frame.iloc[0].astype(float).tolist()


# Reverse map: action_id -> card_name (no _avab suffix)
_ACTION_ID_TO_CARD = {
    v: k.replace("_avab", "")
    for k, v in AVAIL_FEATURE_TO_ACTION_ID.items()
    if v is not None
}

# All action IDs that correspond to a known card
_KNOWN_CARD_ACTION_IDS = set(_ACTION_ID_TO_CARD.keys())


def build_action_mask_from_obs(obs, num_actions=13):
    """
    Build a [num_actions] bool mask from a raw env observation (pd.DataFrame).
    True  => action is legal
    False => action is masked out (logit -> -1e9)

    Masking logic:
      1. Start with all actions BLOCKED (False).
      2. Always allow WAIT_ID.
      3. For every card in AVAIL_FEATURE_TO_ACTION_ID, allow its action ID
         only if its _avab feature > 0 in the cleaned observation.
         (_avab already encodes: card is in hand + enough elixir.)
      4. Unmapped action IDs (no entry in AVAIL_FEATURE_TO_ACTION_ID) stay
         blocked permanently.

    Falls back to all-True mask if obs cannot be cleaned (safe degradation).
    """
    try:
        cleaned_df = final_clean(obs)
        if not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
            return torch.ones(num_actions, dtype=torch.bool)
        last_row = cleaned_df.iloc[0]
    except Exception:
        return torch.ones(num_actions, dtype=torch.bool)

    # Step 1 — block everything by default
    mask = torch.zeros(num_actions, dtype=torch.bool)

    # Step 2 — wait is always legal
    if ALWAYS_ALLOW_WAIT and 0 <= WAIT_ID < num_actions:
        mask[WAIT_ID] = True

    # Step 3 — allow card action if _avab > 0
    # The _avab feature is already 1 only when the card is in hand and
    # the player has enough elixir, so no separate slot check is needed.
    for feat, aid in AVAIL_FEATURE_TO_ACTION_ID.items():
        if aid is None or not (0 <= aid < num_actions):
            continue
        try:
            if float(last_row[feat]) > 0:
                mask[aid] = True
        except (KeyError, TypeError, ValueError):
            pass

    return mask


def sequenece_buffering(obs, sequence_buffer, window_size, input_size):
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


def actor_critic_update(
        actor_critic_network,
        optimizer,
        rollout,
        vf=0.5,
        ent_coef=0.01,
        epsilon=0.2,
):
    """
    PPO clipped update.

    Returns:
        policy_loss   (float)
        value_loss    (float)
        clip_fraction (float)  -- fraction of steps where ratio was clipped
        action_entropy(float)  -- mean action distribution entropy
    """
    device = next(actor_critic_network.parameters()).device

    windows       = rollout["windows"]
    actions       = rollout["actions"]
    xs            = rollout["x"]
    ys            = rollout["y"]
    old_log_probs = rollout["log_probs"]
    advantages    = rollout["advantages"]
    returns       = rollout["returns"]

    states_t      = torch.tensor(windows,       dtype=torch.float32, device=device)
    actions_t     = torch.tensor(actions,       dtype=torch.long,    device=device)
    positions_t   = torch.stack([
                        torch.tensor(xs, dtype=torch.float32, device=device),
                        torch.tensor(ys, dtype=torch.float32, device=device)
                    ], dim=1)
    old_lp_t      = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    advantages_t  = torch.tensor(advantages,    dtype=torch.float32, device=device)
    returns_t     = torch.tensor(returns,       dtype=torch.float32, device=device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    masks_list = rollout["masks"]
    masks_t    = torch.stack(masks_list, dim=0).to(device)

    action_logits, pos_logits, value_estimate, _ = actor_critic_network(states_t)

    masked_action_logits = action_logits.masked_fill(~masks_t, -1e9)

    dist_action = torch.distributions.Categorical(logits=masked_action_logits)
    dist_pos    = torch.distributions.Normal(loc=pos_logits, scale=1.0)

    action_log_probs  = dist_action.log_prob(actions_t)
    pos_log_probs     = dist_pos.log_prob(positions_t).sum(dim=-1)
    current_log_probs = action_log_probs + pos_log_probs

    ratios    = torch.exp(current_log_probs - old_lp_t)
    surr1     = ratios * advantages_t
    surr2     = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages_t
    policy_loss = -torch.min(surr1, surr2).mean()

    clipped       = ((ratios < 1.0 - epsilon) | (ratios > 1.0 + epsilon)).float()
    clip_fraction = float(clipped.mean().item())

    values      = value_estimate.squeeze(-1)
    value_loss  = F.mse_loss(values, returns_t)

    entropy      = dist_action.entropy() + dist_pos.entropy().sum(-1)
    entropy_loss = -ent_coef * entropy.mean()
    action_entropy = float(dist_action.entropy().mean().item())

    loss = policy_loss + vf * value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic_network.parameters(), max_norm=0.5)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), clip_fraction, action_entropy


def compute_returns_and_advantages(rollout, gamma=0.99):
    """
    Adds rollout["returns"] and rollout["advantages"] in-place and returns the dict.
    """
    rewards = rollout["rewards"]
    values  = rollout["values"]
    T       = len(rewards)

    returns    = [0.0] * T
    advantages = [0.0] * T

    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t]     = running_return

    for t in range(T):
        advantages[t] = returns[t] - values[t]

    rollout["returns"]    = returns
    rollout["advantages"] = advantages
    return rollout
