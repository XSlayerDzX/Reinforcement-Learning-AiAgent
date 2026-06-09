"""Shared evaluation loop used by all baseline policies and final PPO evaluation.

Usage:
    from Ai.models.eval_runner import run_evaluation
    from Ai.models.run_config  import baseline_eval_dir, EVAL_GAMES

    run_evaluation(
        policy       = my_policy_instance,
        policy_name  = "random",          # used for log and eval filenames
        seed         = 42,
        n_games      = EVAL_GAMES,
        log_dir      = baseline_log_dir("random"),
        eval_dir     = baseline_eval_dir("random"),
        window_title = DEFAULT_WINDOW_TITLE,
    )

A policy object must implement:
    policy.select_action(obs: pd.DataFrame, slots: dict) -> dict
        Returns: {"action_id": int, "pos_x": float, "pos_y": float}
    policy.reset()   -> None   (called at the start of every game)
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.models.logger import RunLogger
from Ai.models.run_config import DEFAULT_WINDOW_TITLE, EVAL_GAMES


# ── aggregate CSV columns ─────────────────────────────────────────────────────

AGGREGATE_COLUMNS = [
    "policy", "n_games", "seed",
    "win_rate_mean", "win_rate_std",
    "return_mean", "return_std",
    "episode_length_mean", "episode_length_std",
    "wait_rate_mean", "wait_rate_std",
    "elixir_overflow_mean",
    "finalized_at",
]


def run_evaluation(
    policy,
    policy_name: str,
    seed: int,
    log_dir: Path,
    eval_dir: Path,
    n_games: int = EVAL_GAMES,
    window_title: str = DEFAULT_WINDOW_TITLE,
):
    """
    Run `n_games` evaluation episodes and write:
      - {log_dir}/training_log.csv     — per-game rows (via RunLogger)
      - {log_dir}/winrate.json         — per-game win rate (via RunLogger)
      - {log_dir}/run_summary.json     — aggregate summary (via RunLogger)
      - {eval_dir}/eval_{policy_name}_games.csv      — per-game CSV copy
      - {eval_dir}/eval_{policy_name}_aggregate.csv  — one-row aggregate

    Args:
        policy       : object with .select_action(obs, slots) -> dict
                       and .reset() -> None
        policy_name  : short name used in filenames and log output
        seed         : integer seed (logged, not used to seed env)
        log_dir      : where RunLogger writes its files
        eval_dir     : where final eval CSVs are written
        n_games      : number of games to play
        window_title : BlueStacks window title
    """
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    logger = RunLogger(log_dir=log_dir, run_id=policy_name, mode="baseline")
    env    = ClashRoyalEnv(window_title=window_title)

    game_records = []

    for game_id in range(1, n_games + 1):
        print(f"\n{'='*60}")
        print(f"[{policy_name}] Starting game {game_id}/{n_games}")
        print(f"{'='*60}")

        policy.reset()
        t_start = time.time()

        obs, slots = env.reset()
        if obs is None:
            print(f"[WARN] env.reset() returned None on game {game_id}, skipping.")
            continue

        # ── episode loop ──────────────────────────────────────────────────────
        done            = False
        episode_return  = 0.0
        episode_length  = 0
        wait_actions    = 0
        total_actions   = 0
        overflow_steps  = 0   # steps where elixir >= 10

        outcome = "draw"

        while not done:
            # Get elixir for overflow tracking
            try:
                elixir_val = float(obs["Elixir"].iloc[0]) if "Elixir" in obs.columns else 0.0
            except Exception:
                elixir_val = 0.0
            if elixir_val >= 10.0:
                overflow_steps += 1

            # Policy selects action
            try:
                decision = policy.select_action(obs, slots)
            except Exception as e:
                print(f"[WARN] policy.select_action raised: {e} — defaulting to wait")
                decision = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

            action_id = decision.get("action_id", 0)
            pos_x     = decision.get("pos_x", -1.0)
            pos_y     = decision.get("pos_y", -1.0)

            # Step environment
            next_obs, reward, done, next_slots, _ = env.step(
                action_id, pos_x, pos_y, obs, slots
            )

            episode_return += reward
            episode_length += 1
            total_actions  += 1
            if action_id == 0:
                wait_actions += 1

            # Terminal outcome
            if isinstance(next_obs, str):
                outcome = next_obs.lower()   # "win" / "loss"
                done    = True
            elif done:
                if reward >= 1.0:
                    outcome = "win"
                elif reward <= -1.0:
                    outcome = "loss"

            if not done:
                obs   = next_obs if next_obs is not None else obs
                slots = next_slots if next_slots is not None else slots

        duration = round(time.time() - t_start, 2)
        ep_len   = max(episode_length, 1)

        record = {
            "game_id":              game_id,
            "seed":                 seed,
            "outcome":              outcome,
            "episodic_return":      round(episode_return, 4),
            "episode_length":       episode_length,
            "total_actions":        total_actions,
            "wait_actions":         wait_actions,
            "wait_rate":            round(wait_actions / ep_len, 4),
            "elixir_overflow_proxy":round(overflow_steps / ep_len, 4),
            "duration_seconds":     duration,
        }

        logger.log_game(record)
        game_records.append(record)

    # ── finalize ──────────────────────────────────────────────────────────────
    summary = logger.finalize(run_config={"policy": policy_name, "seed": seed, "n_games": n_games})

    # Write per-game CSV to eval_dir
    games_csv_path = eval_dir / f"eval_{policy_name}_games.csv"
    if game_records:
        fieldnames = list(game_records[0].keys())
        with open(games_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_records)
        print(f"[{policy_name}] Per-game CSV written to {games_csv_path}")

    # Write aggregate CSV to eval_dir
    agg_csv_path = eval_dir / f"eval_{policy_name}_aggregate.csv"
    outcomes   = [r["outcome"] for r in game_records]
    returns_   = [r["episodic_return"] for r in game_records]
    lengths_   = [r["episode_length"] for r in game_records]
    wait_rates = [r["wait_rate"] for r in game_records]
    overflows  = [r["elixir_overflow_proxy"] for r in game_records]
    n          = max(len(game_records), 1)
    wins       = outcomes.count("win")

    agg_row = {
        "policy":               policy_name,
        "n_games":              n,
        "seed":                 seed,
        "win_rate_mean":        round(wins / n, 4),
        "win_rate_std":         0.0,
        "return_mean":          round(float(np.mean(returns_)), 4)   if returns_   else 0.0,
        "return_std":           round(float(np.std(returns_)),  4)   if returns_   else 0.0,
        "episode_length_mean":  round(float(np.mean(lengths_)), 2)   if lengths_   else 0.0,
        "episode_length_std":   round(float(np.std(lengths_)),  2)   if lengths_   else 0.0,
        "wait_rate_mean":       round(float(np.mean(wait_rates)), 4) if wait_rates else 0.0,
        "wait_rate_std":        round(float(np.std(wait_rates)),  4) if wait_rates else 0.0,
        "elixir_overflow_mean": round(float(np.mean(overflows)),  4) if overflows  else 0.0,
        "finalized_at":         datetime.now().isoformat(),
    }

    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerow(agg_row)
    print(f"[{policy_name}] Aggregate CSV written to {agg_csv_path}")

    return summary, game_records
