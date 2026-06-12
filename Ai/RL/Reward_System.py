from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_set = PROJECT_ROOT / "datasets" / "linked_dataset.csv"

# ── Elixir overflow penalty ───────────────────────────────────────────────────
_ELIXIR_MAX                = 10.0
_ELIXIR_OVERFLOW_THRESHOLD = 9.5
_ELIXIR_OVERFLOW_PENALTY   = -0.15

# ── Illegal action penalty ───────────────────────────────────────────────────
ILLEGAL_ACTION_PENALTY = -0.5


def compute_reward(match_data_set):
    df = pd.read_csv(match_data_set)
    df["r"] = 0.0

    enemy_troops = [col for col in df.columns if col.endswith("_enemy")]
    ally_towers  = ["ally_prince_tower_left", "ally_prince_tower_right", "ally_king_tower"]
    enemy_towers = ["enemy_prince_tower_left", "enemy_prince_tower_right", "enemy_king_tower"]

    killed_troops = (df[enemy_troops].shift(1) == 1) & (df[enemy_troops] == 0)
    df.loc[killed_troops.any(axis=1), "r"] += 0.15

    unkilled_troops = (df[enemy_troops].shift(3) == 1) & (df[enemy_troops] == 1)
    df.loc[unkilled_troops.any(axis=1), "r"] -= 0.05

    destroyed_ally_towers = (df[ally_towers].shift(1) == 1) & (df[ally_towers] == 0)
    df.loc[destroyed_ally_towers.any(axis=1), "r"] -= 0.75

    destroyed_enemy_towers = (df[enemy_towers].shift(1) == 1) & (df[enemy_towers] == 0)
    df.loc[destroyed_enemy_towers.any(axis=1), "r"] += 0.75

    rtg = np.zeros(len(df))
    rtg[-1] = df['r'].iloc[-1]
    for t in range(len(df)-2, -1, -1):
        rtg[t] = df['r'].iloc[t] + rtg[t+1]
    df['rtg'] = rtg

    output_path = match_data_set.replace('.csv', '_rewarded.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("RTG_0:", df['rtg'].iloc[0])
    return df


def compute_step_reward(prev_obs, curr_obs):
    """Compute per-step reward from consecutive observations."""
    if prev_obs is None or curr_obs is None:
        return 0.0

    def to_series(o):
        if isinstance(o, pd.DataFrame):
            return o.iloc[0] if o.shape[0] >= 1 else pd.Series(dtype="float64")
        if isinstance(o, pd.Series):
            return o
        if isinstance(o, dict):
            return pd.Series(o)
        return pd.Series(dtype="float64")

    prev_s = to_series(prev_obs)
    curr_s = to_series(curr_obs)

    if prev_s.empty or curr_s.empty:
        return 0.0

    common = prev_s.index.intersection(curr_s.index)
    if common.empty:
        return 0.0

    prev_common = prev_s[common]
    curr_common = curr_s[common]

    numeric_cols = common[
        pd.to_numeric(prev_common, errors='coerce').notna() &
        pd.to_numeric(curr_common, errors='coerce').notna()
    ]

    if numeric_cols.empty:
        return 0.0

    prev_aligned = prev_s[numeric_cols].fillna(0).astype(float)
    curr_aligned = curr_s[numeric_cols].fillna(0).astype(float)

    diff = prev_aligned - curr_aligned
    reward = 0.0

    for col, d in diff.items():
        if d <= 0:
            continue
        if str(col).startswith("enemy_"):
            reward += float(d)
        elif str(col).startswith("ally_"):
            reward -= float(d)

    try:
        elixir_val = float(curr_s["Elixir"])
        if elixir_val >= _ELIXIR_OVERFLOW_THRESHOLD:
            reward += _ELIXIR_OVERFLOW_PENALTY
    except (KeyError, TypeError, ValueError):
        pass

    return float(reward)


def compute_tower_hp_reward(
    frame_paths: List[str],
    side_coef:      Optional[float] = None,
    king_coef:      Optional[float] = None,
    ally_side_coef: Optional[float] = None,
    ally_king_coef: Optional[float] = None,
    hp_norm:        Optional[float] = None,
) -> List[float]:
    """Compute per-step tower HP shaping rewards from a list of screenshot paths.

    Called ONCE after the game ends — never during live play.

    Separate coefficients for attack (enemy HP lost) and defence (ally HP lost)
    allow the agent to be penalised more harshly for letting its own towers take
    damage than it is rewarded for dealing the same damage to the enemy.

    Returns a list of length len(frame_paths) with float rewards.
    """
    from Ai.models.run_config import (
        TOWER_HP_SIDE_COEF,
        TOWER_HP_KING_COEF,
        TOWER_HP_ALLY_SIDE_COEF,
        TOWER_HP_ALLY_KING_COEF,
        HP_NORM,
    )
    from Ai.tower_hp_ocr import run_ocr

    _atk_side  = side_coef      if side_coef      is not None else TOWER_HP_SIDE_COEF
    _atk_king  = king_coef      if king_coef      is not None else TOWER_HP_KING_COEF
    _def_side  = ally_side_coef if ally_side_coef is not None else TOWER_HP_ALLY_SIDE_COEF
    _def_king  = ally_king_coef if ally_king_coef is not None else TOWER_HP_ALLY_KING_COEF
    _norm      = hp_norm        if hp_norm        is not None else HP_NORM

    n = len(frame_paths)
    rewards = [0.0] * n

    if n == 0:
        return rewards

    hp_readings = []
    for path in frame_paths:
        try:
            reading = run_ocr(path)
        except Exception as e:
            print(f"[tower_hp_reward] OCR exception for {path}: {e}")
            reading = None
        hp_readings.append(reading)

    for i in range(1, n):
        prev_hp = hp_readings[i - 1]
        curr_hp = hp_readings[i]

        if prev_hp is None or curr_hp is None:
            continue

        step_reward = 0.0

        # ── Enemy side towers (attack reward) ──────────────────────────────
        for prev_val, curr_val in [
            (prev_hp.enemy_left,  curr_hp.enemy_left),
            (prev_hp.enemy_right, curr_hp.enemy_right),
        ]:
            if prev_val is not None and curr_val is not None:
                delta = prev_val - curr_val
                if delta > 0:
                    step_reward += (delta / _norm) * _atk_side

        # ── Enemy king (attack reward) ───────────────────────────────────
        if prev_hp.enemy_king is not None and curr_hp.enemy_king is not None:
            delta = prev_hp.enemy_king - curr_hp.enemy_king
            if delta > 0:
                step_reward += (delta / _norm) * _atk_king

        # ── Ally side towers (defence penalty — stronger coefficient) ────────
        for prev_val, curr_val in [
            (prev_hp.ally_left,  curr_hp.ally_left),
            (prev_hp.ally_right, curr_hp.ally_right),
        ]:
            if prev_val is not None and curr_val is not None:
                delta = prev_val - curr_val
                if delta > 0:
                    step_reward -= (delta / _norm) * _def_side

        # ── Ally king (defence penalty — strongest coefficient) ─────────────
        if prev_hp.ally_king is not None and curr_hp.ally_king is not None:
            delta = prev_hp.ally_king - curr_hp.ally_king
            if delta > 0:
                step_reward -= (delta / _norm) * _def_king

        rewards[i] = round(step_reward, 5)

    return rewards
