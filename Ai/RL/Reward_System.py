import pandas as pd
import numpy as np


data_set = r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\datasets\linked_dataset.csv"


def compute_reward(match_data_set):
    df = pd.read_csv(match_data_set)
    df["r"] = 0.0

    enemy_troops = [col for col in df.columns if col.endswith("_enemy")]
    ally_towers = ["ally_prince_tower_left", "ally_prince_tower_right", "ally_king_tower"]
    enemy_towers = ["enemy_prince_tower_left", "enemy_prince_tower_right", "enemy_king_tower"]

    killed_troops = (df[enemy_troops].shift(1) == 1) & (df[enemy_troops] == 0)
    df.loc[killed_troops.any(axis=1), "r"] += 0.15

    unkilled_troops = (df[enemy_troops].shift(3) == 1) & (df[enemy_troops] == 1)
    df.loc[unkilled_troops.any(axis=1), "r"] -= 0.01

    destroyed_ally_towers = (df[ally_towers].shift(1) == 1) & (df[ally_towers] == 0)
    df.loc[destroyed_ally_towers.any(axis=1), "r"] -= 0.4

    destroyed_enemy_towers = (df[enemy_towers].shift(1) == 1) & (df[enemy_towers] == 0)
    df.loc[destroyed_enemy_towers.any(axis=1), "r"] += 0.75

    #checking for who won based on pixels
    #tayeb task


    rtg = np.zeros(len(df))
    rtg[-1] = df['r'].iloc[-1]
    for t in range(len(df)-2, -1, -1):
        rtg[t] = df['r'].iloc[t] + rtg[t+1]  # gamma=1.0
    df['rtg'] = rtg

    output_path = match_data_set.replace('.csv', '_rewarded.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("RTG_0:", df['rtg'].iloc[0])
    return df


def compute_step_reward(prev_obs, curr_obs):
    """
    Compute a scalar reward from previous and current observations.
    Works with dict, pd.Series, or single-row pd.DataFrame.
    Reward logic: count features that went from 1 -> 0.
      - if an `enemy_` feature dropped: +1 per drop
      - if an `ally_` feature dropped: -1 per drop
    Returns float.
    """
    if prev_obs is None or curr_obs is None:
        return 0.0

    def to_series(o):
        if isinstance(o, pd.DataFrame):
            if o.shape[0] >= 1:
                return o.iloc[0]
            return pd.Series(dtype="int64")
        if isinstance(o, pd.Series):
            return o
        if isinstance(o, dict):
            return pd.Series(o)
        return pd.Series(dtype="int64")

    prev_s = to_series(prev_obs)
    curr_s = to_series(curr_obs)

    if prev_s.empty or curr_s.empty:
        return 0.0

    common = prev_s.index.intersection(curr_s.index)
    if common.empty:
        return 0.0

    prev_aligned = prev_s[common].fillna(0).astype(int)
    curr_aligned = curr_s[common].fillna(0).astype(int)

    diff = prev_aligned - curr_aligned  # positive means value decreased
    reward = 0.0

    for col, d in diff.items():
        if d <= 0:
            continue
        # positive decrease happened
        if str(col).startswith("enemy_"):
            reward += float(d)
        elif str(col).startswith("ally_"):
            reward -= float(d)
        else:
            # neutral/unknown features: treat decreases as negative to agent (optional)
            reward += 0.0

    return float(reward)



