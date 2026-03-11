import pandas as pd
import numpy as np
from sympy import pretty_print

set = r"C:\Users\abdoa\PycharmProjects\Reinforcement-Learning-AiAgent\Ai\datasets\linked_dataset.csv"
df = pd.read_csv(set)


def link_frames(input_csv, output_csv,match_id):
    # Load the input and output datasets
    df_input = pd.read_csv(input_csv)
    df_output = pd.read_csv(output_csv)

    # Ensure both datasets have a common 'id' column for linking
    if 'id' not in df_input.columns or 'id' not in df_output.columns:
        raise ValueError("Both datasets must contain an 'id' column for linking.")

    # Merge the datasets on the 'id' column
    linked_df = pd.merge(df_input, df_output, on='id', how='left')
    print(linked_df)
    print(linked_df.shape)

    # Save the linked dataset to a new CSV file
    linked_df.to_csv(f"match_{match_id}_dataset.csv",index=False)



def output_cleaning(match_csv):
    df = pd.read_csv(match_csv)

    df["action"] = df["action"].fillna("wait")
    df["pos_x"] = df["pos_x"].fillna(-1)
    df["pos_y"] = df["pos_y"].fillna(-1)
    print("output cleaned")
    df.to_csv(match_csv, index=False)

def slot_cleaning(match_csv):
    df = pd.read_csv(match_csv)

    for slot in ["slot_1", "slot_2", "slot_3", "slot_4"]:

        if pd.isnull(df[slot].iloc[-1]):
            mode = df[slot].mode()
            if not mode.empty:
                df.at[df.index[-1], slot] = mode[0]

        for index in range(len(df) - 2, -1, -1):
            if pd.isnull(df.at[index, slot]):
                df.at[index, slot] = df.at[index + 1, slot]

    print("slots cleaned")
    df.to_csv(match_csv, index=False)

def general_cleaning(match_csv):
    df = pd.read_csv(match_csv)

    df.fillna(-1, inplace=True)

    print("general cleaning done")
    df.to_csv(match_csv, index=False)

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
    df = pd.read_csv(match_csv)

    pos_col = [col for col in df.columns if col.endswith("_x") or col.endswith("_y")]

    for col in pos_col:
        cleaned_col = (df[col] != -1)
        df.loc[cleaned_col, col] = df.loc[cleaned_col, col].apply(lambda px: pixel_to_grid_x(px) if col.endswith("_x") else pixel_to_grid_y(px))
    print("cleaned positions done")
    return df



new = clean_positions(match_csv=set)
pretty_print(new)












