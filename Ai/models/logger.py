"""Unified logger for all runs — baselines and PPO training.

Writes:
  - per-game CSV row  →  {log_dir}/training_log.csv
  - updates JSON      →  {log_dir}/updates.json        (PPO only)
  - rollouts JSON     →  {log_dir}/rollouts.json       (PPO only)
  - winrate JSON      →  {log_dir}/winrate.json
  - run summary JSON  →  {log_dir}/run_summary.json    (written at end)

Usage:
    logger = RunLogger(log_dir=ppo_log_dir("BCPPO_s1"), run_id="BCPPO_s1")
    logger.log_game(game_record)        # call after every game
    logger.log_ppo_update(update_dict)  # call after every PPO update (PPO only)
    logger.finalize()                   # call once at the end of all games
"""

import csv
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────

def _ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)

def _load_json(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── CSV column order ────────────────────────────────────────────────────────

BASELINE_CSV_COLUMNS = [
    "policy", "game_id", "seed", "outcome",
    "episodic_return",          # +1 win / -1 loss / 0 draw (terminal only)
    "episode_length",
    "total_actions", "wait_actions", "wait_rate",
    "mean_elixir_at_action",    # avg elixir when a non-wait card was played
    "elixir_overflow_proxy",
    "duration_seconds",         # wall-clock game duration
    "timestamp",
]

PPO_CSV_COLUMNS = [
    "run_id", "game_id", "seed", "outcome",
    "episodic_return",          # summed shaped step rewards (RL return)
    "episode_length",
    "total_actions", "wait_actions", "wait_rate",
    "mean_elixir_at_action",
    "elixir_overflow_proxy",
    "policy_loss", "value_loss", "explained_variance",
    "mean_advantage", "std_advantage", "clip_fraction",
    "action_entropy", "cumulative_wins", "win_rate_last_20",
    "duration_seconds",
    "timestamp", "checkpoint_saved",
]


# ── main class ────────────────────────────────────────────────────────────

class RunLogger:
    """
    Single logger instance per run. Works for both baselines and PPO runs.
    Pass mode="baseline" or mode="ppo" to select the correct CSV schema.
    """

    def __init__(self, log_dir: Path, run_id: str, mode: str = "ppo"):
        assert mode in ("ppo", "baseline"), f"mode must be 'ppo' or 'baseline', got {mode!r}"
        self.log_dir = Path(log_dir)
        self.run_id  = run_id
        self.mode    = mode
        _ensure_dir(self.log_dir)

        self.csv_columns   = PPO_CSV_COLUMNS if mode == "ppo" else BASELINE_CSV_COLUMNS
        self.csv_path      = self.log_dir / "training_log.csv"
        self.winrate_path  = self.log_dir / "winrate.json"
        self.updates_path  = self.log_dir / "updates.json"
        self.rollouts_path = self.log_dir / "rollouts.json"
        self.summary_path  = self.log_dir / "run_summary.json"

        self._game_count        = 0
        self._cumulative_wins   = 0
        self._outcome_window    = deque(maxlen=20)
        self._all_returns       = []
        self._all_lengths       = []
        self._all_durations     = []
        self._best_winrate      = 0.0
        self._best_winrate_game = 0

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    # ── per-game logging ────────────────────────────────────────────────

    def log_game(self, record: dict):
        """
        Append one row to training_log.csv and update winrate.json.

        Baseline record keys (required):
            game_id, seed, outcome, episodic_return, episode_length,
            total_actions, wait_actions, mean_elixir_at_action,
            elixir_overflow_proxy, duration_seconds

        PPO record additionally needs:
            policy_loss, value_loss, explained_variance,
            mean_advantage, std_advantage, clip_fraction,
            action_entropy, checkpoint_saved
        """
        self._game_count += 1
        outcome = record.get("outcome", "unknown")

        self._outcome_window.append(outcome)
        if outcome == "win":
            self._cumulative_wins += 1

        episodic_return = record.get("episodic_return", 0.0)
        ep_length       = record.get("episode_length", 0)
        duration        = record.get("duration_seconds", 0.0)
        self._all_returns.append(episodic_return)
        self._all_lengths.append(ep_length)
        self._all_durations.append(duration)

        wins_in_window  = sum(1 for o in self._outcome_window if o == "win")
        win_rate_last20 = round(wins_in_window / len(self._outcome_window), 4)

        if win_rate_last20 > self._best_winrate:
            self._best_winrate      = win_rate_last20
            self._best_winrate_game = record.get("game_id", self._game_count)

        row = dict(record)
        if self.mode == "ppo":
            row["run_id"]           = self.run_id
            row["cumulative_wins"]  = self._cumulative_wins
            row["win_rate_last_20"] = win_rate_last20
        else:
            row["policy"] = self.run_id

        row["timestamp"] = datetime.now().isoformat()

        if "wait_rate" not in row or row["wait_rate"] is None:
            ep_len = max(row.get("episode_length", 1), 1)
            row["wait_rate"] = round(row.get("wait_actions", 0) / ep_len, 4)

        # Default missing fields to 0.0 so CSV never has blank cells
        row.setdefault("mean_elixir_at_action", 0.0)
        row.setdefault("duration_seconds", 0.0)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns, extrasaction="ignore")
            writer.writerow(row)

        winrate_history = _load_json(self.winrate_path)
        winrate_history.append({
            "game_id":          record.get("game_id", self._game_count),
            "timestamp":        row["timestamp"],
            "outcome":          outcome,
            "win_rate_last_20": win_rate_last20,
            "cumulative_wins":  self._cumulative_wins,
        })
        _save_json(self.winrate_path, winrate_history)

        print(
            f"[{self.run_id}] Game {record.get('game_id', self._game_count):>3} | "
            f"{outcome:>4} | return={episodic_return:+.3f} | "
            f"WR(20)={win_rate_last20:.0%} | cumW={self._cumulative_wins} | "
            f"{duration:.0f}s"
        )
        return win_rate_last20

    # ── PPO-specific logging ────────────────────────────────────────────

    def log_ppo_update(self, update: dict):
        if self.mode != "ppo":
            return
        history = _load_json(self.updates_path)
        update.setdefault("timestamp", datetime.now().isoformat())
        update.setdefault("run_id", self.run_id)
        history.append(update)
        _save_json(self.updates_path, history)

    def log_ppo_rollout(self, rollout_summary: dict):
        if self.mode != "ppo":
            return
        history = _load_json(self.rollouts_path)
        rollout_summary.setdefault("timestamp", datetime.now().isoformat())
        rollout_summary.setdefault("run_id", self.run_id)
        history.append(rollout_summary)
        _save_json(self.rollouts_path, history)

    # ── end-of-run summary ────────────────────────────────────────────────

    def finalize(self, run_config: Optional[dict] = None):
        """Write run_summary.json. Call once after all games are logged."""
        import numpy as np

        total_games   = self._game_count
        winrate_hist  = _load_json(self.winrate_path)
        total_wins    = self._cumulative_wins
        final_winrate = round(total_wins / max(total_games, 1), 4)

        summary = {
            "run_id":               self.run_id,
            "mode":                 self.mode,
            "total_games":          total_games,
            "total_wins":           total_wins,
            "total_losses":         sum(1 for e in winrate_hist if e["outcome"] == "loss"),
            "total_draws":          sum(1 for e in winrate_hist if e["outcome"] == "draw"),
            "final_win_rate":       final_winrate,
            "best_win_rate":        round(self._best_winrate, 4),
            "best_win_rate_game":   self._best_winrate_game,
            "mean_return":          round(float(np.mean(self._all_returns))  if self._all_returns  else 0.0, 4),
            "std_return":           round(float(np.std(self._all_returns))   if self._all_returns  else 0.0, 4),
            "mean_episode_length":  round(float(np.mean(self._all_lengths))  if self._all_lengths  else 0.0, 1),
            "mean_duration_seconds":round(float(np.mean(self._all_durations))if self._all_durations else 0.0, 1),
            "finalized_at":         datetime.now().isoformat(),
        }
        if run_config:
            summary["run_config"] = run_config

        _save_json(self.summary_path, summary)
        print(f"[{self.run_id}] Run finalized. Summary written to {self.summary_path}")
        return summary

    # ── convenience properties ────────────────────────────────────────────

    @property
    def current_win_rate(self) -> float:
        if not self._outcome_window:
            return 0.0
        wins = sum(1 for o in self._outcome_window if o == "win")
        return round(wins / len(self._outcome_window), 4)

    @property
    def best_win_rate(self) -> float:
        return self._best_winrate
