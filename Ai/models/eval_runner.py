"""Shared evaluation loop used by all baseline policies and final PPO evaluation.

Usage:
    from Ai.models.eval_runner import run_evaluation
    from Ai.models.run_config  import baseline_eval_dir, EVAL_GAMES

    run_evaluation(
        policy       = my_policy_instance,
        policy_name  = "random",
        seed         = 42,
        n_games      = EVAL_GAMES,
        log_dir      = baseline_log_dir("random"),
        eval_dir     = baseline_eval_dir("random"),
        window_title = DEFAULT_WINDOW_TITLE,
    )

A policy object must implement:
    policy.select_action(obs: pd.DataFrame, slots: dict) -> dict
        Returns: {"action_id": int, "pos_x": float, "pos_y": float}
    policy.reset()   -> None
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd

from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.models.logger import RunLogger
from Ai.models.run_config import DEFAULT_WINDOW_TITLE, EVAL_GAMES
from Ai.Agent.start_end_game import auto_play, load_template
from Ai.Stream_to_frame import Frame_Handler


# ── Template paths ────────────────────────────────────────────────────────────

_AGENT_DIR = Path(__file__).resolve().parent.parent / "Agent"

_TEMPLATE_PATHS = {
    "ok":            _AGENT_DIR / "ok_end.jpg",
    "menu":          _AGENT_DIR / "menu_button.png",
    "training_camp": _AGENT_DIR / "training_camp.png",
    "ok_training":   _AGENT_DIR / "ok_play.png",
}


def _load_templates() -> dict:
    return {key: load_template(path) for key, path in _TEMPLATE_PATHS.items()}


def _navigate_to_game(templates: dict, window_title: str, policy_name: str = "") -> bool:
    tag = f"[{policy_name}][NAV]" if policy_name else "[NAV]"
    print(f"{tag} Waiting for game to start...")
    tick = 0
    while True:
        result = Frame_Handler(window_title=window_title)
        if result is None:
            print(f"{tag} Frame_Handler returned None, retrying...")
            sleep(1.5)
            tick += 1
            continue

        frame_path, monitor = result
        zone = {"left": monitor.get("left", 0), "top": monitor.get("top", 0)}
        detected = auto_play(frame_path, zone, templates)
        tick += 1

        if detected is None:
            print(f"{tag} tick={tick}  nothing matched — waiting...")
            sleep(1.5)
        elif detected == "menu":
            print(f"{tag} tick={tick}  'menu' clicked — waiting for training-camp screen...")
            sleep(2.5)
        elif detected == "training_camp":
            print(f"{tag} tick={tick}  'training_camp' clicked — waiting for OK button...")
            sleep(3.0)
        elif detected == "ok_training":
            print(f"{tag} tick={tick}  'ok_training' clicked — game launching...")
            sleep(4.0)
            return True
        elif detected == "ok":
            print(f"{tag} tick={tick}  'ok' (end screen) — clicking and waiting for menu...")
            sleep(2.0)
        else:
            print(f"{tag} tick={tick}  unknown '{detected}' — waiting...")
            sleep(1.5)


def _dismiss_end_screen(templates: dict, window_title: str, policy_name: str = "", timeout_ticks: int = 20) -> None:
    """Poll for the 'ok' end-screen button, click it, then wait 4 s for the menu to load."""
    tag = f"[{policy_name}][END]" if policy_name else "[END]"
    print(f"{tag} Waiting for end-screen OK...")
    for tick in range(1, timeout_ticks + 1):
        result = Frame_Handler(window_title=window_title)
        if result is None:
            sleep(1)
            continue
        frame_path, monitor = result
        zone = {"left": monitor.get("left", 0), "top": monitor.get("top", 0)}
        detected = auto_play(frame_path, zone, templates)
        print(f"{tag} tick={tick}  detected={detected}")
        if detected == "ok":
            print(f"{tag} OK clicked — waiting 4 s for menu to load...")
            sleep(4.0)
            return
        sleep(1)
    print(f"{tag} Timed out waiting for end-screen OK — continuing anyway.")


# ── aggregate CSV columns ─────────────────────────────────────────────────────

AGGREGATE_COLUMNS = [
    "policy", "n_games", "seed",
    "win_rate_mean",
    "outcome_return_mean", "outcome_return_std",   # +1/-1/0 terminal only
    "episode_length_mean", "episode_length_std",
    "duration_seconds_mean", "duration_seconds_std", # wall-clock per game
    "wait_rate_mean", "wait_rate_std",
    "mean_elixir_at_action_mean",                   # avg elixir when playing a card
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
    Run n_games evaluation episodes.

    episodic_return is TERMINAL-ONLY for all policies run through this function:
        +1  win
        -1  loss
         0  draw

    mean_elixir_at_action: average elixir bar value at the moment a
    non-wait card action was taken. Tracks elixir efficiency.
    """
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"[{policy_name}] Loading button templates...")
    templates = _load_templates()
    print(f"[{policy_name}] Templates loaded: {list(templates.keys())}")

    logger = RunLogger(log_dir=log_dir, run_id=policy_name, mode="baseline")
    env    = ClashRoyalEnv(window_title=window_title)

    game_records = []

    for game_id in range(1, n_games + 1):
        print(f"\n{'='*60}")
        print(f"[{policy_name}] Starting game {game_id}/{n_games}")
        print(f"{'='*60}")

        _navigate_to_game(templates, window_title, policy_name=policy_name)

        policy.reset()
        t_start = time.time()

        obs, slots = env.reset()
        if obs is None:
            print(f"[WARN] env.reset() returned None on game {game_id}, skipping.")
            continue

        # ── episode loop ──────────────────────────────────────────────────────
        done              = False
        episode_length    = 0
        wait_actions      = 0
        total_actions     = 0
        overflow_steps    = 0
        elixir_at_action  = []   # elixir value each time a non-wait card is played
        outcome           = "draw"

        while not done:
            try:
                elixir_val = float(obs["Elixir"].iloc[0]) if "Elixir" in obs.columns else 0.0
            except Exception:
                elixir_val = 0.0
            if elixir_val >= 10.0:
                overflow_steps += 1

            try:
                decision = policy.select_action(obs, slots)
            except Exception as e:
                print(f"[WARN] policy.select_action raised: {e} — defaulting to wait")
                decision = {"action_id": 0, "pos_x": -1.0, "pos_y": -1.0}

            action_id = decision.get("action_id", 0)
            pos_x     = decision.get("pos_x", -1.0)
            pos_y     = decision.get("pos_y", -1.0)

            # Track elixir at non-wait actions
            if action_id != 0:
                elixir_at_action.append(elixir_val)

            # Step — _step_reward intentionally ignored for non-RL policies
            next_obs, _step_reward, done, next_slots, _ = env.step(
                action_id, pos_x, pos_y, obs, slots
            )

            episode_length += 1
            total_actions  += 1
            if action_id == 0:
                wait_actions += 1

            if isinstance(next_obs, str):
                outcome = next_obs.lower()
                done    = True
            elif done:
                outcome = "draw"

            if not done:
                obs   = next_obs if next_obs is not None else obs
                slots = next_slots if next_slots is not None else slots

        # Terminal-only return: +1 win / -1 loss / 0 draw
        if outcome == "win":
            episodic_return = 1.0
        elif outcome == "loss":
            episodic_return = -1.0
        else:
            episodic_return = 0.0

        _dismiss_end_screen(templates, window_title, policy_name=policy_name)

        duration = round(time.time() - t_start, 2)
        ep_len   = max(episode_length, 1)
        mean_elixir = round(float(np.mean(elixir_at_action)), 4) if elixir_at_action else 0.0

        record = {
            "game_id":               game_id,
            "seed":                  seed,
            "outcome":               outcome,
            "episodic_return":       episodic_return,
            "episode_length":        episode_length,
            "total_actions":         total_actions,
            "wait_actions":          wait_actions,
            "wait_rate":             round(wait_actions / ep_len, 4),
            "mean_elixir_at_action": mean_elixir,
            "elixir_overflow_proxy": round(overflow_steps / ep_len, 4),
            "duration_seconds":      duration,
        }

        logger.log_game(record)
        game_records.append(record)
        print(f"[{policy_name}] Game {game_id} done — outcome={outcome}  length={episode_length}  elixir_mean={mean_elixir}  duration={duration}s")

    # ── finalize ──────────────────────────────────────────────────────────────
    summary = logger.finalize(run_config={"policy": policy_name, "seed": seed, "n_games": n_games})

    # Per-game CSV
    games_csv_path = eval_dir / f"eval_{policy_name}_games.csv"
    if game_records:
        fieldnames = list(game_records[0].keys())
        with open(games_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_records)
        print(f"[{policy_name}] Per-game CSV written to {games_csv_path}")

    # Aggregate CSV
    agg_csv_path = eval_dir / f"eval_{policy_name}_aggregate.csv"
    outcomes   = [r["outcome"]               for r in game_records]
    returns_   = [r["episodic_return"]        for r in game_records]
    lengths_   = [r["episode_length"]         for r in game_records]
    durations_ = [r["duration_seconds"]       for r in game_records]
    wait_rates = [r["wait_rate"]              for r in game_records]
    elixirs_   = [r["mean_elixir_at_action"]  for r in game_records]
    overflows  = [r["elixir_overflow_proxy"]  for r in game_records]
    n          = max(len(game_records), 1)
    wins       = outcomes.count("win")

    agg_row = {
        "policy":                   policy_name,
        "n_games":                  n,
        "seed":                     seed,
        "win_rate_mean":            round(wins / n, 4),
        "outcome_return_mean":      round(float(np.mean(returns_)),   4) if returns_   else 0.0,
        "outcome_return_std":       round(float(np.std(returns_)),    4) if returns_   else 0.0,
        "episode_length_mean":      round(float(np.mean(lengths_)),   2) if lengths_   else 0.0,
        "episode_length_std":       round(float(np.std(lengths_)),    2) if lengths_   else 0.0,
        "duration_seconds_mean":    round(float(np.mean(durations_)), 2) if durations_ else 0.0,
        "duration_seconds_std":     round(float(np.std(durations_)),  2) if durations_ else 0.0,
        "wait_rate_mean":           round(float(np.mean(wait_rates)), 4) if wait_rates else 0.0,
        "wait_rate_std":            round(float(np.std(wait_rates)),  4) if wait_rates else 0.0,
        "mean_elixir_at_action_mean":round(float(np.mean(elixirs_)),  4) if elixirs_   else 0.0,
        "elixir_overflow_mean":     round(float(np.mean(overflows)),  4) if overflows  else 0.0,
        "finalized_at":             datetime.now().isoformat(),
    }

    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerow(agg_row)
    print(f"[{policy_name}] Aggregate CSV written to {agg_csv_path}")

    return summary, game_records
