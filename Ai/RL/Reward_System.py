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


def compute_step_reward(current_frame, next_frame):
    reward = 0.0

    # Identify lists of entities based on the keys
    enemy_troops = [col for col in current_frame.keys() if str(col).endswith("_enemy")]
    ally_towers = ["ally_prince_tower_left", "ally_prince_tower_right", "ally_king_tower"]
    enemy_towers = ["enemy_prince_tower_left", "enemy_prince_tower_right", "enemy_king_tower"]

    # 1. Killed and Unkilled Enemy Troops
    for troop in enemy_troops:
        curr_val = current_frame.get(troop, 0)
        next_val = next_frame.get(troop, 0)

        if curr_val == 1 and next_val == 0:
            reward += 0.15
        elif curr_val == 1 and next_val == 1:
            reward -= 0.01

    # 2. Destroyed Ally Towers
    for tower in ally_towers:
        if current_frame.get(tower, 0) == 1 and next_frame.get(tower, 0) == 0:
            reward -= 0.4

    # 3. Destroyed Enemy Towers
    for tower in enemy_towers:
        if current_frame.get(tower, 0) == 1 and next_frame.get(tower, 0) == 0:
            reward += 0.75

    #4 slight negative reward for each step to encourage shorter games
    reward -= 0.01
    # tayeb task: checking for who won based on pixels would go here

    return reward



