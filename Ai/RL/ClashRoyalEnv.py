from time import sleep

import pyautogui as pya
import pandas as pd
from debugpy.common.timestamp import current

from Ai.Stream_to_frame import Frame_Handler
from Ai.Create_DataSet import Create_Dataset_Row
from Ai.Agent.Agent_main import react_agent,ACTION_ID_TO_NAME,get_slot_for_action
from Reward_System import compute_step_reward
from Ai.check_status import check_match_status






def Observation(id=0, match_id=0):
    current_frame = Frame_Handler()
    current_slots = {}
    if current_frame is None:
        return None
    # Process the frame to extract game state information
    row_dict = Create_Dataset_Row(current_frame, id, match_id)  # Example IDs
    if row_dict:
        current_slots = {
            "slot_1": row_dict["slot_1"],
            "slot_2": row_dict["slot_2"],
            "slot_3": row_dict["slot_3"],
            "slot_4": row_dict["slot_4"],
        }
        return row_dict , current_slots
    else:
        #check if the match is done
        state = check_match_status(current_frame)
        if state == "win" or state == "loss":
            return state, None
        else:
            return None, None


class ClashRoyalEnv:
    def __init__(self,step_delay= 0.5, max_steps= 300, reward_win= 100, reward_lose= -100, reward_draw= 0):
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_draw = reward_draw
        self.current_step = 0
        self.done = False
        self.prev_obs = None
        self.next_obs = None
        self.obs = None
        self.current_slots = {}

    def reset(self):

        windows = pya.getWindowsWithTitle("BlueStacks App Player 4") ## change this if the title changes
        if not windows or windows[0].isActive == False:
            print("Please open the BlueStacks window and start the game.")
            return

        self.current_step = 0
        self.done = False
        self.prev_obs = None
        self.next_obs = None
        self.current_slots = {}
        # Reset the game state here (e.g., start a new match)
        while self.obs is None: ## wait until the game state is ready
            self.obs, self.current_slots = Observation(self.current_step)
            sleep(0.5) ## wait for the game to update
        self.obs = pd.DataFrame([self.obs])
        return self.obs


    def step(self, action,pos_x=None, pos_y=None):
        if self.done:
            raise Exception("Episode is done. Please reset the environment.")

        # Execute the action (e.g., simulate a tap or card placement)
        react_agent(action,pos_x,pos_y,current_slots=self.current_slots)
        self.prev_obs = self.obs.copy() if self.obs is not None else None

        # Wait for the game state to update
        pya.sleep(self.step_delay)

        # Get the new observation
        self.current_step += 1
        self.obs, self.current_slots = Observation(self.current_step)  ## Get the new observation and current slots after executing the action, this function will return None if the game state is not ready yet, so we need to wait until it returns a valid observation
        while self.obs is None: ## wait until the game state is ready or until the check function for win or loss can be applied
            self.obs, self.current_slots = Observation(self.current_step) ## Get the new observation and current slots after executing the action
            sleep(0.5)
        if self.obs == "win":
            reward = self.reward_win
            self.done = True
            return self.obs, reward, self.done
        elif self.obs == "loss":
            reward = self.reward_lose
            self.done = True
            return self.obs, reward, self.done
        self.obs = pd.DataFrame([self.obs]) ## Convert to DataFrame for easier processing
        # Calculate reward based on the new observation and previous observation
        reward = compute_step_reward(self.prev_obs.iloc[0], self.obs.iloc[0])

        # Update previous observation
        self.next_obs = self.obs.copy()

        return self.next_obs, self.prev_obs,reward, self.done







