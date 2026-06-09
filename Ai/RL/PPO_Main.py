"""PPO training main loop.

This file is the single entry point for all 5 PPO runs:
    BCPPO_s1 / BCPPO_s2         -- BC warm-start + masking ON
    PPOScratch_s1 / s2           -- random init   + masking ON
    BCPPO_NoMask_s1              -- BC warm-start + masking OFF

Run:
    python -m Ai.RL.PPO_Main --run_id BCPPO_s1 --seed 1
    python -m Ai.RL.PPO_Main --run_id PPOScratch_s1 --seed 1
    python -m Ai.RL.PPO_Main --run_id BCPPO_NoMask_s1 --seed 1
"""
import argparse
import traceback
import sys
from collections import deque
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import torch

from Ai.RL.PPO_Trainer import (
    sequenece_buffering,
    build_action_mask_from_obs,
    compute_returns_and_advantages,
    actor_critic_update,
)
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
from Ai.Behavior_Cloning.action_masking_config import WAIT_ID
from Ai.ClashRoyalData import ElixirCost
from Ai.Behavior_Cloning.action_masking_config import AVAIL_FEATURE_TO_ACTION_ID
from Ai.Agent.start_end_game import auto_play, load_template
from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
from Ai.Stream_to_frame import Frame_Handler

from Ai.models.logger import RunLogger
from Ai.models.run_config import (
    BC_CHECKPOINT,
    CHECKPOINT_INTERVAL,
    DEFAULT_WINDOW_TITLE,
    GAMMA,
    EPSILON,
    VF_COEF,
    ENT_COEF,
    GRAD_CLIP,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    NUM_ACTIONS,
    NUM_LAYERS,
    PPO_TRAINING_GAMES,
    WINDOW_SIZE,
    ppo_checkpoint_dir,
    ppo_checkpoint_path,
    ppo_best_checkpoint_path,
    ppo_log_dir,
)

# ---------------------------------------------------------------------------
# Template paths (relative to the Agent folder)
# ---------------------------------------------------------------------------

_AGENT_DIR = Path(__file__).resolve().parent.parent / "Agent"

_TEMPLATE_PATHS = {
    "ok":            _AGENT_DIR / "ok_end.jpg",
    "menu":          _AGENT_DIR / "menu_button.png",
    "training_camp": _AGENT_DIR / "training_camp.png",
    "ok_training":   _AGENT_DIR / "ok_play.png",
}


def _load_templates() -> dict:
    """Load all button templates once and return as a dict."""
    return {key: load_template(path) for key, path in _TEMPLATE_PATHS.items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dataframe(obs):
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


def _save_checkpoint(model, optimizer, run_id, game_id, win_rate, path):
    """Save model + optimizer state with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "game_id":              game_id,
        "run_id":               run_id,
        "win_rate_last_20":     round(win_rate, 4),
    }, path)
    print(f"[{run_id}] Checkpoint saved -> {path}")


# ---------------------------------------------------------------------------
# rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env,
    model,
    run_id,
    templates: dict,
    use_masking=True,
    window_title=DEFAULT_WINDOW_TITLE,
    stop_flag=None,
):
    """
    Play one game and collect all transitions.
    Returns the rollout dict (with returns + advantages added) plus
    per-game diagnostic values needed for logging.

    Args:
        templates: preloaded template images dict (call _load_templates() once
                   in main() and pass down here — never reload inside the loop).
    """
    model.eval()
    window_buffer = deque(maxlen=WINDOW_SIZE)

    windows, actions, x_list, y_list = [], [], [], []
    log_probs, rewards, values, masks = [], [], [], []
    elixir_at_action = []   # elixir value at every non-wait step

    done    = False
    started = False

    # Wait for game to be ready (ok_training = the "OK" button on the training-camp screen)
    while not started:
        if stop_flag and stop_flag.is_set():
            return None, {}
        frame, zone = Frame_Handler(window_title=window_title)
        if auto_play(frame, zone, templates) == "ok_training":
            started = True
        sleep(2.5)

    state_raw, current_slots = env.reset()

    for attempt in range(5):
        if state_raw is not None and _ensure_dataframe(state_raw) is not None:
            break
        print(f"[WARN] env.reset() invalid, retry {attempt+1}/5")
        sleep(2.0)
        state_raw, current_slots = env.reset()

    state = _ensure_dataframe(state_raw)
    if state is None or "slot_1" not in state.columns:
        print("[ERROR] Could not get valid initial obs — skipping game.")
        return None, {}

    with torch.no_grad():
        while not done:
            if stop_flag and stop_flag.is_set():
                break

            current_window = sequenece_buffering(state, window_buffer, WINDOW_SIZE, INPUT_SIZE)
            window_tensor  = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)

            action_logits, pos_logits, value_estimate, _ = model(window_tensor)
            action_logits = action_logits.squeeze(0)
            pos_logits    = pos_logits.squeeze(0)

            # --- action mask ---
            if use_masking:
                action_mask = build_action_mask_from_obs(state, num_actions=action_logits.shape[-1])

                # Elixir guard
                try:
                    current_elixir = float(state["Elixir"].iloc[0])
                except (KeyError, ValueError, TypeError):
                    current_elixir = 10.0

                ACTION_ID_TO_CARD = {
                    v: k.replace("_avab", "")
                    for k, v in AVAIL_FEATURE_TO_ACTION_ID.items() if v is not None
                }
                for card_name, cost in ElixirCost.items():
                    avab_key = card_name + "_avab"
                    aid = AVAIL_FEATURE_TO_ACTION_ID.get(avab_key)
                    if aid is not None and current_elixir < cost:
                        action_mask[aid] = False
            else:
                # masking OFF: allow everything
                action_mask    = torch.ones(action_logits.shape[-1], dtype=torch.bool)
                current_elixir = 10.0

            masked_logits = action_logits.masked_fill(~action_mask, -1e9)

            dist_action = torch.distributions.Categorical(logits=masked_logits)
            dist_pos    = torch.distributions.Normal(loc=pos_logits, scale=1.0)

            action  = dist_action.sample()
            pos     = dist_pos.sample()
            action_val = action.item()

            # Final elixir safety check (masking ON only)
            if use_masking:
                ACTION_ID_TO_CARD = {
                    v: k.replace("_avab", "")
                    for k, v in AVAIL_FEATURE_TO_ACTION_ID.items() if v is not None
                }
                if action_val in ACTION_ID_TO_CARD:
                    card_name = ACTION_ID_TO_CARD[action_val]
                    card_cost = ElixirCost.get(card_name, 0)
                    try:
                        live_elixir = float(state["Elixir"].iloc[0]) - 1
                    except (KeyError, ValueError, TypeError):
                        live_elixir = 0.0
                    if live_elixir < card_cost:
                        print(f"[SAFETY] Forced wait: {card_name} costs {card_cost}, elixir={live_elixir:.1f}")
                        action_val = WAIT_ID
                        action     = torch.tensor(WAIT_ID, dtype=torch.long)

            # Position
            if action_val == WAIT_ID:
                pos_x, pos_y = -1.0, -1.0
            else:
                gx  = int(round(pos[0].item()))
                gy  = int(round(pos[1].item()))
                bs_x, bs_y = grid_to_pixel(gx, gy)
                pos_x, pos_y = bluestacks_to_global_coords(
                    bs_x, bs_y,
                    bluestacks_resolution=(540, 960),
                    window_title=window_title,
                )
                # Track elixir at non-wait actions
                try:
                    elixir_at_action.append(float(state["Elixir"].iloc[0]))
                except Exception:
                    pass

            log_prob_action = dist_action.log_prob(action)
            log_prob_pos    = dist_pos.log_prob(pos).sum(dim=-1)
            log_prob        = (log_prob_action + log_prob_pos).item()

            next_state_raw, reward, done, slots, frame = env.step(
                action_val, pos_x, pos_y, state_raw, current_slots
            )

            for _ in range(2):
                if next_state_raw is not None:
                    break
                print("[WARN] Step returned None, retrying...")
                sleep(1.5)
                next_state_raw, reward, done, slots, frame = env.step(
                    action_val, pos_x, pos_y, state_raw, current_slots
                )

            if isinstance(next_state_raw, str):
                done   = True
                reward = (
                    env.reward_win  if next_state_raw.lower() == "win"  else
                    env.reward_lose if next_state_raw.lower() == "loss" else 0.0
                )
                next_state_raw = None

            next_state = _ensure_dataframe(next_state_raw)
            if next_state is None:
                next_state = state.copy()
            else:
                state_raw     = next_state_raw
                state         = next_state
                current_slots = slots

            windows.append(current_window)
            masks.append(action_mask)
            actions.append(action_val)
            values.append(value_estimate.item())
            x_list.append(pos_x)
            y_list.append(pos_y)
            log_probs.append(log_prob)
            rewards.append(reward)

            # Wait for end-of-game screen to clear
            while done:
                if stop_flag and stop_flag.is_set():
                    break
                c_f, z = Frame_Handler(window_title=window_title)
                if auto_play(c_f, z, templates) == "ok":
                    sleep(2)
                    break
                sleep(1)

    rollout = {
        "windows":   windows,
        "actions":   actions,
        "x":         x_list,
        "y":         y_list,
        "log_probs": log_probs,
        "rewards":   rewards,
        "values":    values,
        "masks":     masks,
    }
    rollout = compute_returns_and_advantages(rollout, gamma=GAMMA)

    # --- per-game diagnostics ---
    ep_len      = max(len(rewards), 1)
    wait_count  = actions.count(WAIT_ID)
    terminal_r  = rewards[-1] if rewards else 0.0
    outcome = (
        "win"  if terminal_r >= env.reward_win  else
        "loss" if terminal_r <= env.reward_lose else
        "draw"
    )

    diagnostics = {
        "outcome":             outcome,
        "episodic_return":     round(float(np.sum(rewards)), 4),
        "episode_length":      ep_len,
        "total_actions":       ep_len,
        "wait_actions":        wait_count,
        "wait_rate":           round(wait_count / ep_len, 4),
        "mean_advantage":      round(float(np.mean(rollout["advantages"])), 4),
        "std_advantage":       round(float(np.std(rollout["advantages"])),  4),
        "mean_elixir_at_action": round(float(np.mean(elixir_at_action)), 4) if elixir_at_action else 0.0,
        "elixir_overflow_proxy": round(
            sum(1 for r in rewards if r == 0.0) / ep_len, 4
        ),
    }

    return rollout, diagnostics


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def main(
    run_id: str,
    seed: int,
    use_pretrained: bool = True,
    use_masking: bool    = True,
    n_games: int         = PPO_TRAINING_GAMES,
    window_title: str    = DEFAULT_WINDOW_TITLE,
):
    print(f"\n{'='*60}")
    print(f"PPO Training  |  run_id={run_id}  |  seed={seed}")
    print(f"use_pretrained={use_pretrained}  |  use_masking={use_masking}")
    print(f"n_games={n_games}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)

    # --- load templates once for the entire run ---
    print(f"[{run_id}] Loading button templates...")
    templates = _load_templates()
    print(f"[{run_id}] Templates loaded: {list(templates.keys())}")

    # --- environment ---
    env = ClashRoyalEnv()

    # --- model ---
    pretrained_path = str(BC_CHECKPOINT) if use_pretrained else None
    model = PPO_LSTM_Model(
        input_size          = INPUT_SIZE,
        hidden_size         = HIDDEN_SIZE,
        num_layers          = NUM_LAYERS,
        num_actions         = NUM_ACTIONS,
        pretrained_model_path = pretrained_path,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- resume from latest checkpoint if one exists ---
    ckpt_dir   = ppo_checkpoint_dir(run_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_game = 1

    existing = sorted(ckpt_dir.glob("game_*.pth"))
    if existing:
        latest_ckpt = existing[-1]
        print(f"[{run_id}] Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_game = ckpt["game_id"] + 1
        print(f"[{run_id}] Resuming from game {start_game}")

    # --- logger ---
    logger = RunLogger(
        log_dir = ppo_log_dir(run_id),
        run_id  = run_id,
        mode    = "ppo",
    )

    best_winrate = logger.best_win_rate

    # --- training loop ---
    for game_id in range(start_game, n_games + 1):
        print(f"\n[{run_id}] —— Game {game_id}/{n_games} ——")

        try:
            rollout, diagnostics = collect_rollout(
                env          = env,
                model        = model,
                run_id       = run_id,
                templates    = templates,
                use_masking  = use_masking,
                window_title = window_title,
            )
        except Exception as e:
            print(f"[ERROR] collect_rollout failed on game {game_id}: {e}")
            traceback.print_exc()
            continue

        if rollout is None or not rollout["rewards"]:
            print(f"[WARN] Empty rollout on game {game_id}, skipping update.")
            continue

        # --- PPO update ---
        try:
            policy_loss, value_loss, clip_fraction, action_entropy = actor_critic_update(
                actor_critic_network = model,
                optimizer            = optimizer,
                rollout              = rollout,
                vf                  = VF_COEF,
                ent_coef            = ENT_COEF,
                epsilon             = EPSILON,
            )
        except Exception as e:
            print(f"[ERROR] actor_critic_update failed on game {game_id}: {e}")
            traceback.print_exc()
            continue

        # --- explained variance ---
        returns_arr = np.array(rollout["returns"])
        values_arr  = np.array(rollout["values"])
        residuals   = returns_arr - values_arr
        explained_var = float(
            1 - np.var(residuals) / (np.var(returns_arr) + 1e-8)
        )

        # --- log this game ---
        record = {
            "game_id":              game_id,
            "seed":                 seed,
            "outcome":              diagnostics["outcome"],
            "episodic_return":      diagnostics["episodic_return"],
            "episode_length":       diagnostics["episode_length"],
            "total_actions":        diagnostics["total_actions"],
            "wait_actions":         diagnostics["wait_actions"],
            "wait_rate":            diagnostics["wait_rate"],
            "mean_elixir_at_action":diagnostics["mean_elixir_at_action"],
            "elixir_overflow_proxy":diagnostics["elixir_overflow_proxy"],
            "policy_loss":          round(policy_loss,    6),
            "value_loss":           round(value_loss,     6),
            "explained_variance":   round(explained_var,  4),
            "mean_advantage":       diagnostics["mean_advantage"],
            "std_advantage":        diagnostics["std_advantage"],
            "clip_fraction":        round(clip_fraction,  4),
            "action_entropy":       round(action_entropy, 4),
            "checkpoint_saved":     False,
        }

        current_wr = logger.log_game(record)

        logger.log_ppo_update({
            "update_id":        game_id,
            "policy_loss":      round(policy_loss,   6),
            "value_loss":       round(value_loss,    6),
            "clip_fraction":    round(clip_fraction, 4),
            "action_entropy":   round(action_entropy,4),
            "explained_var":    round(explained_var, 4),
            "mean_advantage":   diagnostics["mean_advantage"],
            "std_advantage":    diagnostics["std_advantage"],
            "outcome":          diagnostics["outcome"],
            "total_reward":     diagnostics["episodic_return"],
            "total_steps":      diagnostics["episode_length"],
        })

        logger.log_ppo_rollout({
            "game_id":         game_id,
            "steps":           diagnostics["episode_length"],
            "total_reward":    diagnostics["episodic_return"],
            "forced_waits":    diagnostics["wait_actions"],
            "mean_advantage":  diagnostics["mean_advantage"],
        })

        # --- checkpoint rotation ---
        checkpoint_saved = False

        if game_id % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                model, optimizer, run_id, game_id, current_wr,
                path=ppo_checkpoint_path(run_id, game_id),
            )
            checkpoint_saved = True

        if current_wr > best_winrate:
            best_winrate = current_wr
            _save_checkpoint(
                model, optimizer, run_id, game_id, current_wr,
                path=ppo_best_checkpoint_path(run_id),
            )
            print(f"[{run_id}] New best win rate: {best_winrate:.0%} at game {game_id}")
            checkpoint_saved = True

        if checkpoint_saved:
            logger.log_ppo_update({"update_id": game_id, "checkpoint_saved": True})

    # --- end of run ---
    _save_checkpoint(
        model, optimizer, run_id, n_games, logger.current_win_rate,
        path=ppo_checkpoint_path(run_id, n_games),
    )

    summary = logger.finalize(run_config={
        "run_id":         run_id,
        "seed":           seed,
        "use_pretrained": use_pretrained,
        "use_masking":    use_masking,
        "n_games":        n_games,
        "learning_rate":  LEARNING_RATE,
        "gamma":          GAMMA,
        "epsilon":        EPSILON,
        "vf_coef":        VF_COEF,
        "ent_coef":       ENT_COEF,
    })

    print(f"\n[{run_id}] Training complete.")
    print(f"  Final win rate : {summary['final_win_rate']:.0%}")
    print(f"  Best win rate  : {summary['best_win_rate']:.0%} (game {summary['best_win_rate_game']})")
    print(f"  Mean return    : {summary['mean_return']}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training loop")
    parser.add_argument("--run_id",        type=str,  required=True,
                        choices=["BCPPO_s1","BCPPO_s2","PPOScratch_s1","PPOScratch_s2","BCPPO_NoMask_s1"],
                        help="Run identifier")
    parser.add_argument("--seed",          type=int,  default=1,
                        help="Random seed")
    parser.add_argument("--n_games",       type=int,  default=PPO_TRAINING_GAMES,
                        help="Total training games")
    parser.add_argument("--no_pretrain",   action="store_true",
                        help="Disable BC warm-start (PPO from scratch)")
    parser.add_argument("--no_mask",       action="store_true",
                        help="Disable action masking")
    parser.add_argument("--window",        type=str,  default=DEFAULT_WINDOW_TITLE,
                        help="BlueStacks window title")
    args = parser.parse_args()

    use_pretrained = not args.no_pretrain
    use_masking    = not args.no_mask

    if "Scratch" in args.run_id:
        use_pretrained = False
    if "NoMask" in args.run_id:
        use_masking = False

    main(
        run_id         = args.run_id,
        seed           = args.seed,
        use_pretrained = use_pretrained,
        use_masking    = use_masking,
        n_games        = args.n_games,
        window_title   = args.window,
    )
