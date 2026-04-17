#script to handle the main loop for the agent that will play
# this script is supposed to do the following:
# step 1 : use the streamtoframe function to take a frame from the window of the game
# step 2 : input that frame to our cv model to get the current state of the game
# step 3 : clean and transform the output of the cv model to be ready for the lstm model
# step 4 : have a buffer that stores frames and padd with zeros if the buffer is not full
# step 5 : input that buffer window into the lstm to get a prediction of action,pos x and y
# step 6 : here the agent is going to start doing its work : ------> {
    # if action predicted is "wait" then the agent wont do anything and skips
    # if action is anything else then "wait" then the agent will first get the slot of that action card
    # then the agent will translate the grid of x and y into pure pixel coordinates
    # then excutes pyautogui to press the slot number in the keyboard and move the mouse to the mouse coordinates
    # and click at that position ( masking for position grids is not done yet )
from time import sleep

import pandas as pd
# Note : each frame information is saved into a csv for logging to be used later for training and improving the model
# Note : we will implement the auto start match in this script as well

#importing
import os
import pyautogui as pya
from marshmallow.fields import Boolean
from pymsgbox import password

from Ai.Behavior_Cloning.lstm_inference_pipeline import LSTM_Inference_Pipeline
from Ai.Create_DataSet import Create_Dataset_Row
from Ai.Data_Cleaning import final_clean
from Ai.Stream_to_frame import Frame_Handler
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {
    "LSTM": LSTM_Inference_Pipeline(
        model_path=r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Behavior_Cloning\lstm.pth",
        device=DEVICE,
        window_size=10,
        input_size=205,
        output_size=13,
    )
}

Agent_State = {
    "current_frame" : "",
    "current_card" : "",
    "current_elixir" : "",
    "current_id" : "",
    "current_match_id" : "",
    "current_towers" : {},
    "current_slots": {},
}

import json
from pathlib import Path

STATE_FILE = Path(r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Agent\agent_global_state.json")

DEFAULT_AGENT_GLOBAL_STATE = {
    "current_match_id": 0,
    "match_id_state": {}
}

def load_agent_global_state():
    if not STATE_FILE.exists():
        return DEFAULT_AGENT_GLOBAL_STATE.copy()

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure expected keys exist
        if "current_match_id" not in data:
            data["current_match_id"] = 0
        if "match_id_state" not in data or not isinstance(data["match_id_state"], dict):
            data["match_id_state"] = {}
        return data
    except Exception as e:
        print(f"Failed to load state file {STATE_FILE}: {e}")
        return DEFAULT_AGENT_GLOBAL_STATE.copy()

def save_agent_global_state(state):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        temp_file = STATE_FILE.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Atomic replace on same filesystem
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        print(f"Failed to save state file {STATE_FILE}: {e}")


# Load persisted state once at module import time
Agent_global_state = load_agent_global_state()

match_csv = {}


ACTION_ID_TO_NAME = {
    0: "wait",
    1: "mini pekka",
    2: "knight",
    3: "arrows",
    4: "goblins",
    5: "giant",
    6: "spear goblins",
    7: "archers",
    8: "fireball",
    9: "minions",
    10: "musketeer",
    11: "goblin cage",
}

def get_slot_for_action(action_name, current_slots):
    for slot_key, card_name in current_slots.items():
        if card_name == action_name:
            return slot_key
    return None

def react_agent(action_id, pos_x, pos_y, current_slots):
    # 1) wait
    if action_id == 0:
        return

    #convert action id to card name
    action_name = ACTION_ID_TO_NAME.get(action_id)
    if action_name is None:
        print(f"Unknown action_id: {action_id}")
        return

    #find the slot that contains that card
    slot_key = get_slot_for_action(action_name, current_slots)
    if slot_key is None:
        print(f"Card '{action_name}' not found in current slots: {current_slots}")
        return

    slot_number = slot_key.split("_")[-1]

    pya.press(slot_number)

    pya.moveTo(pos_x, pos_y, duration=0.1)
    pya.click()

def Agent(model_name,state = True):
    #check if bluestack window is open
    windows = pya.getWindowsWithTitle("BlueStacks App Player 1")
    if not windows or windows[0].isActive == False:
        print("Please open the BlueStacks window and start the game.")
        return
    current_model = models_dict.get(model_name)
    if current_model is None:
        print(f"Unknown model key: {model_name}")
        return


    current_frame = None
    current_card = None
    current_elixir = None
    current_id = None
    current_match_id = Agent_global_state["current_match_id"]
    current_towers = None
    current_slots = None
    current_id = 0

    #start the match if the start pixels are found ( there should be a made function for this already )
    # else keep looping untill its found

    # if match is started we call the function that takes a frame from the game
    # here there should be a loop that keeps taking frames
    while state:

        current_frame = Frame_Handler()
        row_dict = Create_Dataset_Row(current_frame, current_id, current_match_id)

        if row_dict:
            current_slots = {
                "slot_1" : row_dict["slot_1"],
                "slot_2" : row_dict["slot_2"],
                "slot_3" : row_dict["slot_3"],
                "slot_4" : row_dict["slot_4"],
            }
            current_elixir = row_dict["Elixir"]
            Agent_State["current_slots"] = current_slots
            Agent_State["current_frame"] = current_frame
            Agent_State["current_id"] = current_id
            Agent_State["current_elixir"] = current_elixir

            row_csv = row_dict.to_csv()
            match_csv[current_id] = row_csv

            # Default to wait/no-op unless a model-specific prediction overrides it.
            action = 0
            pos_x = -1
            pos_y = -1

            if model_name == "LSTM":
                prediction = current_model.predict(row_csv)
                action = prediction["action_id"]
                pos_x = prediction["pos_pred"][0]
                pos_y = prediction["pos_pred"][-1]

            elif model_name == "Transformer":
                pass
            elif model_name == "PPO":
                pass
            #now we can call the react function to execute the action (wait do nothing else execute the action )
            react_agent(action,pos_x,pos_y,Agent_State["current_slots"])

            current_id += 1
            sleep(5)
        else:
            #call for the win/loss function to check if it was a win or loss
            #check = checker(current_frame)
            # if check == "win" or check == "loss":
            #     state = False
            #     print(f"Match {current_match_id} ended with a {check}.")
            #     Agent_global_state["match_id_state"][current_match_id] = check
            #     save_agent_global_state(Agent_global_state)
            #     break
            print(f"Failed to capture a valid frame {current_id}. Skipping.")
            if current_frame and os.path.exists(current_frame):
                os.remove(current_frame)
            sleep(1)
            continue

    # Persist frame log safely
    match_df = pd.DataFrame.from_dict(match_csv, orient="index", columns=["frame_csv"])
    match_df.to_csv(
        f"C:/Users/SlayerDz/PycharmProjects/clash-royale-rl-agent/Ai/lstm_matches.match_{current_match_id}.csv",
        index_label="id",
    )

    # Persist global state across sessions
    Agent_global_state["current_match_id"] = current_match_id + 1
    save_agent_global_state(Agent_global_state)


