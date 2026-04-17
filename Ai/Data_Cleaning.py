import pandas as pd
import numpy as np
from Ai.ClashRoyalData import *
from Ai.Create_DataSet import *





def link_frames(input_csv, output_csv,match_id):
    # Load the input and output datasets
    # df_input = pd.read_csv(input_csv)
    # df_output = pd.read_csv(output_csv)

    # Ensure both datasets have a common 'id' column for linking

    # Merge the datasets on the 'id' column
    linked_df = pd.merge(input_csv, output_csv, on='id', how='left')
    print(linked_df)
    print(linked_df.shape)

    # Save the linked dataset to a new CSV file
    return linked_df



def output_cleaning(match_csv):
    df = match_csv

    df["action"] = df["action"].fillna("wait")
    df["pos_x"] = df["pos_x"].fillna(-1)
    df["pos_y"] = df["pos_y"].fillna(-1)
    print("output cleaned")
    return df

def slot_cleaning(match_csv):
    df = match_csv

    for slot in ["slot_1", "slot_2", "slot_3", "slot_4"]:

        if pd.isnull(df[slot].iloc[-1]):
            mode = df[slot].mode()
            if not mode.empty:
                df.at[df.index[-1], slot] = mode[0]

        for index in range(len(df) - 2, -1, -1):
            if pd.isnull(df.at[index, slot]):
                df.at[index, slot] = df.at[index + 1, slot]

    return df

def general_cleaning(match_csv):
    df = match_csv

    df.fillna(-1, inplace=True)

    print("general cleaning done")
    return df

## whats needs to be added :

## grid system instead of pos_x and pos_y

arena_w  = 540
arena_h = 960

Grid_w = 9
Grid_h = 18

def pixel_to_grid_x(x_px):
    gx = int(x_px / arena_w  * Grid_w)
    gx = min(max(gx, 0), Grid_w - 1)
    return gx

def pixel_to_grid_y(y_px):
    gy = int(y_px / arena_h * Grid_h)
    gy = min(max(gy, 0), Grid_h - 1)
    return gy

# def grid_to_pixel(gx, gy):
#     x_px = (gx + 0.5) / Grid_w * arena_w
#     y_px = (gy + 0.5) / Grid_h * arena_h
#     return x_px, y_px

def clean_positions(match_csv):
    df = match_csv

    pos_col = [col for col in df.columns if col.endswith("_x") or col.endswith("_y")]

    for col in pos_col:
        cleaned_col = (df[col] != -1)
        df.loc[cleaned_col, col] = df.loc[cleaned_col, col].apply(lambda px: pixel_to_grid_x(px) if col.endswith("_x") else pixel_to_grid_y(px))
    print("cleaned positions done")
    return df

def card_avable(match_csv):
        df = match_csv

        cards_avab_col = ["archers","giant","minions","goblin cage","goblin gang","goblin hut","goblins","knight","mini pekka","musketeer","spear goblins"]

        # Create all availability columns at once to avoid fragmentation
        new_cols = {}
        for col in cards_avab_col:
            rows = (df["slot_1"] == col) | (df["slot_2"] == col) | (df["slot_3"] == col) | (df["slot_4"] == col)
            avail = (rows & (df["Elixir"] >= ElixirCost[col])).astype(int)
            new_cols[col + "_avab"] = avail

        # Concatenate all new columns at once
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        print("card availability done")
        return df




def clean_output_with_groundtruth(
    output_df: pd.DataFrame,
    validation_df: pd.DataFrame,) -> pd.DataFrame:

    out = output_df.copy()
    out = out.reset_index().rename(columns={"index": "orig_idx"})


    val_map = (
        validation_df
        .drop_duplicates(subset=["id"])       # just in case
        .set_index("id")["action"]
    )

    gt_actions = out["id"].map(val_map)

    out.loc[gt_actions.notna(), "action"] = gt_actions[gt_actions.notna()]

    out = out.sort_values("orig_idx")

    out = out.drop_duplicates(subset=["id"], keep="first")

    out = out.sort_values("orig_idx")

    consecutive_same_action = out["action"].eq(out["action"].shift(1))

    out = out[~consecutive_same_action]

    if "pos_x" in out.columns and "pos_y" in out.columns:
        zero_pos_mask = (out["pos_x"] == 0) & (out["pos_y"] == 0)
        out = out[~zero_pos_mask]

    out = (
        out
        .sort_values("orig_idx")
        .drop(columns=["orig_idx"])
        .reset_index(drop=True)
    )
    return out


def distance_columns_cleaning(match_csv):
    df = match_csv
    for col in distance_columns:
        df[col] = df[col].replace(10000000, -1)
    print("distance columns cleaned")

    return df

def drop_slot_columns(input):
    df = input
    df.drop(columns=["slot_1", "slot_2", "slot_3", "slot_4"], inplace=True)
    print("slot columns dropped")

    return df

def final_clean(input,match_id,val= None,output = None):
    input_df = pd.read_csv(input)
    output_uncleaned_df = pd.read_csv(output) if output else None
    val_df = pd.read_csv(val) if val else None

    output_df = clean_output_with_groundtruth(output_uncleaned_df, val_df) if val_df else output_uncleaned_df

    df = link_frames(input_df,output_df,match_id) if output_df else pd.Dataframe(input_df)
    df = output_cleaning(df) if output_df else df
    df = slot_cleaning(df)
    df = general_cleaning(df)
    df = clean_positions(df) if output_df else df
    df = card_avable(df)
    df = distance_columns_cleaning(df)
    df = drop_slot_columns(df)
    df = df.to_csv(f"C:/Users/SlayerDz/PycharmProjects/clash-royale-rl-agent/Ai/final_cleaned_dataset/match_{match_id}_final_cleaned_dataset.csv", index=False) if val_df else (
         df.to_csv(f"frame_{match_id}_agent.csv_")
)
    return df

# for i in range(2,19):
#     input = f"C:/Users/SlayerDz/PycharmProjects/clash-royale-rl-agent/Ai/uncleaned_match_data_sets/match_input_{i}.csv"
#     output = f"C:/Users/SlayerDz/PycharmProjects/clash-royale-rl-agent/Ai/uncleaned_match_data_sets/match_output_{i}.csv"
#     val = f"C:/Users/SlayerDz/PycharmProjects/clash-royale-rl-agent/Ai/uncleaned_match_data_sets/match_output_action_validation_{i}.csv"
#     final_clean(input,output,val,i)





