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

import cv2
import pandas as pd
# Note : each frame information is saved into a csv for logging to be used later for training and improving the model
# Note : we will implement the auto start match in this script as well

#importing
import os
import pyautogui as pya

from Ai.Behavior_Cloning.lstm_inference_pipeline import LSTM_Inference_Pipeline
from Ai.Behavior_Cloning.action_masking_config import get_masking_kwargs
from Ai.Create_DataSet import Create_Dataset_Row
from Ai.Stream_to_frame import Frame_Handler
from Ai.Agent.coordinate_utils import grid_to_pixel, bluestacks_to_global_coords
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {
    "LSTM": LSTM_Inference_Pipeline(
        model_path=r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Behavior_Cloning\lstm.pth",
        device=DEVICE,
        window_size=10,
        input_size=205,
        output_size=13,
        hidden_size=128,
        num_layers=2,
        **get_masking_kwargs(),
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
LSTM_MATCHES_DIR = Path(r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\lstm_matches")
ACTION_LOGS_DIR = Path(r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\lstm_matches\action_logs")


DEFAULT_AGENT_GLOBAL_STATE = {
    "current_match_id": 0,
    "match_id_state": {},
    "matches": {},
}


def _to_json_safe_dict(row_dict):
    safe = {}
    for key, value in row_dict.items():
        # Convert numpy/pandas scalars to native Python types for JSON serialization.
        safe[key] = value.item() if hasattr(value, "item") else value
    return safe

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
        if "matches" not in data or not isinstance(data["matches"], dict):
            data["matches"] = {}
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
        "returning actions is wait"
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


def maybe_check_match_end(frame_path):
    checker_fn = globals().get("checker")
    if callable(checker_fn):
        return checker_fn(frame_path)
    return None

def Agent(model_name, state=True):
    current_match_id = Agent_global_state["current_match_id"]
    current_id = 0
    current_frame = None
    current_slots = None
    should_save_global_state = False
    match_csv.clear()
    match_actions_log = []

    try:
        windows = pya.getWindowsWithTitle("BlueStacks App Player 1")
        if not windows or windows[0].isActive == False:
            print("Please open the BlueStacks window and start the game.")
            return

        current_model = models_dict.get(model_name)
        if current_model is None:
            print(f"Unknown model key: {model_name}")
            return

        print("Model and bluestack ready")
        #cv2.namedWindow("Detections", cv2.WINDOW_NORMAL) # should be disabled when not recording
        while state:
            current_frame = Frame_Handler(current_id)
            row_dict = Create_Dataset_Row(current_frame, current_id, current_match_id)

            if row_dict:
                #print(f"valid frame_{current_id}")
                current_slots = {
                    "slot_1": row_dict["slot_1"],
                    "slot_2": row_dict["slot_2"],
                    "slot_3": row_dict["slot_3"],
                    "slot_4": row_dict["slot_4"],
                }

                current_elixir = row_dict["Elixir"]
                Agent_State["current_slots"] = current_slots
                Agent_State["current_frame"] = current_frame
                Agent_State["current_id"] = current_id
                Agent_State["current_elixir"] = current_elixir

                row_df = pd.DataFrame([row_dict])  # one-row DataFrame
                match_csv[str(current_id)] = _to_json_safe_dict(row_dict)

                action = 0
                pos_x = -1
                pos_y = -1

                if model_name == "LSTM":
                    #print("predicted using the lstm model")
                    prediction = current_model.predict(row_df)
                    action = prediction["action_id"]
                    gx = prediction["pos_pred"][0]
                    gy = prediction["pos_pred"][1]

                    bs_x, bs_y = grid_to_pixel(gx, gy)
                    pos_x, pos_y = bluestacks_to_global_coords(
                        bs_x,
                        bs_y,
                        bluestacks_resolution=(540, 960),
                        window_title="BlueStacks App Player 1", # this needs to be changed to the local name of ur bluestack
                    )

                    print(f"predicted action_id: {ACTION_ID_TO_NAME[action]}")
                    print("predicted pos_pred: {pos_pred}".format(pos_pred=prediction["pos_pred"]))

                elif model_name == "Transformer":
                    pass
                elif model_name == "PPO":
                    pass

                if action == 0:
                    pos_x = -1
                    pos_y = -1

                match_actions_log.append(
                    {
                        "id": current_id,
                        "action_id": int(action),
                        "action_name": ACTION_ID_TO_NAME.get(action, "unknown"),
                        "pos_x": int(pos_x),
                        "pos_y": int(pos_y),
                    }
                )

                react_agent(action, pos_x, pos_y, Agent_State["current_slots"])
                current_id += 1
                print("sleeping")
                sleep(2)
                #print("sleeping ended going for next frame")
            else:
                #call for the win/loss function to check if it was a win or loss
                # check = maybe_check_match_end(current_frame)
                # if check == "win" or check == "loss" or check is True:
                #     state = False
                #     status = check if isinstance(check, str) else "ended"
                #     print(f"Match {current_match_id} ended with a {status}.")
                #     Agent_global_state["match_id_state"][current_match_id] = status
                #     should_save_global_state = True
                #     break

                print(f"Failed to capture a valid frame {current_id}. Skipping.")
                if current_frame and os.path.exists(current_frame):
                    os.remove(current_frame)
                sleep(1)
                continue

    except KeyboardInterrupt:
        print("Agent stopped by keyboard interrupt. Saving state...")
        should_save_global_state = True

    except Exception as e:
        print(f"Agent crashed with error: {e}")

    finally:
        # Persist frame log safely
        try:
            if match_csv:
                match_df = pd.DataFrame.from_dict(match_csv, orient="index")
                match_df.index.name = "id"
                LSTM_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
                match_df.to_csv(LSTM_MATCHES_DIR / f"match_{current_match_id}.csv")
        except Exception as e:
            print(f"Failed to save match CSV: {e}")

        # Persist executed actions in a separate per-match file.
        try:
            if match_actions_log:
                ACTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
                actions_df = pd.DataFrame(match_actions_log)
                actions_df.to_csv(ACTION_LOGS_DIR / f"match_{current_match_id}_actions.csv", index=False)
        except Exception as e:
            print(f"Failed to save actions log CSV: {e}")

        # Persist collected match rows in global state.
        try:
            if match_csv:
                match_id_key = str(current_match_id)
                Agent_global_state.setdefault("match_id_state", {})
                Agent_global_state["match_id_state"][match_id_key] = "unknown"
                Agent_global_state["current_match_id"] = current_match_id + 1
                save_agent_global_state(Agent_global_state)
                print("match data saved in json file")
            else:
                print("Skipping global state save: no frame rows were captured.")
        except Exception as e:
            print(f"Failed to save agent global state: {e}")


Agent(model_name="LSTM",state=True)