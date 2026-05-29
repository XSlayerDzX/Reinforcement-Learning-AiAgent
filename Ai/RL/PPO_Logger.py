import json
import os
from datetime import datetime

# Action ID -> human readable name for dashboard display
ACTION_ID_TO_NAME = {
    0:  "wait",
    1:  "mini pekka",
    2:  "knight",
    4:  "goblins",
    5:  "giant",
    6:  "spear goblins",
    7:  "archers",
    9:  "minions",
    10: "musketeer",
    11: "goblin cage",
}

LOG_PATH = r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\RL\logs"

def _decode_action_dist(action_counts: dict) -> dict:
    """Convert {action_id: count} -> {card_name: count} for dashboard readability."""
    return {
        ACTION_ID_TO_NAME.get(int(k), f"action_{k}"): v
        for k, v in action_counts.items()
    }

def _ensure_log_dir():
    os.makedirs(LOG_PATH, exist_ok=True)

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_update(update_id, rollouts, policy_loss, value_loss, outcome=None):
    """
    Called once per main() run after all rollouts are updated.
    outcome: "win" | "loss" | "draw" | None (unknown)
    """
    _ensure_log_dir()
    path = os.path.join(LOG_PATH, "updates.json")
    history = _load_json(path)

    # Aggregate stats across all rollouts in this run
    all_rewards   = [r for ro in rollouts for r in ro["rewards"]]
    all_values    = [v for ro in rollouts for v in ro["values"]]
    all_returns   = [g for ro in rollouts for g in ro["returns"]]
    all_actions   = [a for ro in rollouts for a in ro["actions"]]
    total_steps   = sum(len(ro["rewards"]) for ro in rollouts)

    # Explained variance
    import numpy as np
    returns_arr = np.array(all_returns)
    values_arr  = np.array(all_values)
    residuals   = returns_arr - values_arr
    explained_var = float(1 - np.var(residuals) / (np.var(returns_arr) + 1e-8))

    # Action distribution (count per action id)
    from collections import Counter
    action_counts = dict(Counter(all_actions))

    entry = {
        "update_id":        update_id,
        "timestamp":        datetime.now().isoformat(),
        "num_rollouts":     len(rollouts),
        "total_steps":      total_steps,
        "outcome":          outcome,                              # win/loss/draw
        "policy_loss":      round(policy_loss, 6),
        "value_loss":       round(value_loss, 6),
        "mean_reward":      round(float(np.mean(all_rewards)), 4),
        "total_reward":     round(float(np.sum(all_rewards)), 4),
        "mean_return":      round(float(np.mean(all_returns)), 4),
        "mean_value_est":   round(float(np.mean(all_values)), 4),
        "explained_var":    round(explained_var, 4),
        "mean_ep_length":   round(total_steps / max(len(rollouts), 1), 1),
        "action_dist": _decode_action_dist(action_counts),
    }

    history.append(entry)
    _save_json(path, history)
    print(f"[LOG] Update #{update_id} logged → {path}")
    return entry


def log_rollout(update_id, rollout_idx, rollout):
    """
    Logs raw rollout data per rollout (not per step — just summary).
    """
    _ensure_log_dir()
    path = os.path.join(LOG_PATH, "rollouts.json")
    history = _load_json(path)

    import numpy as np
    from collections import Counter

    rewards  = rollout["rewards"]
    values   = rollout["values"]
    returns  = rollout["returns"]
    actions  = rollout["actions"]

    forced_waits = actions.count(0)   # WAIT_ID = 0

    entry = {
        "update_id":      update_id,
        "rollout_idx":    rollout_idx,
        "timestamp":      datetime.now().isoformat(),
        "steps":          len(rewards),
        "total_reward":   round(float(np.sum(rewards)), 4),
        "mean_reward":    round(float(np.mean(rewards)), 4),
        "mean_return":    round(float(np.mean(returns)), 4),
        "mean_value":     round(float(np.mean(values)), 4),
        "forced_waits":   forced_waits,
        "action_dist":    _decode_action_dist(dict(Counter(actions))),
    }

    history.append(entry)
    _save_json(path, entry if not history else history)
    print(f"[LOG] Rollout {rollout_idx} of update #{update_id} logged")
    return entry


def log_winrate(outcome):
    """
    Call with "win", "loss", or "draw" at end of each run.
    Maintains a running win rate across all time.
    """
    _ensure_log_dir()
    path = os.path.join(LOG_PATH, "winrate.json")
    history = _load_json(path)

    history.append({
        "timestamp": datetime.now().isoformat(),
        "outcome":   outcome,
    })

    wins   = sum(1 for e in history if e["outcome"] == "win")
    losses = sum(1 for e in history if e["outcome"] == "loss")
    draws  = sum(1 for e in history if e["outcome"] == "draw")
    total  = wins + losses + draws
    winrate = round(wins / total, 4) if total > 0 else 0.0

    _save_json(path, history)
    print(f"[LOG] Win rate: {winrate*100:.1f}% ({wins}W / {losses}L / {draws}D over {total} games)")
    return winrate


def get_next_update_id():
    """Auto-increment update ID based on existing log."""
    _ensure_log_dir()
    path = os.path.join(LOG_PATH, "updates.json")
    history = _load_json(path)
    return len(history)