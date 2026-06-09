from time import sleep
import traceback
import pyautogui as pya
import pandas as pd
from debugpy.common.timestamp import current

from Ai.Stream_to_frame import Frame_Handler
from Ai.Create_DataSet import Create_Dataset_Row
from Ai.Agent.Agent_main import react_agent, ACTION_ID_TO_NAME, get_slot_for_action
from Ai.RL.Reward_System import compute_step_reward
from Ai.check_status import check_match_status





def Observation(id=0, match_id=0, window_title="BlueStacks App Player 1"):
    print(f"[DEBUG] Observation called with id={id}, match_id={match_id}, window_title={window_title}")
    try:
        current_frame, _ = Frame_Handler(id, window_title=window_title)
    except Exception as e:
        print(f"[ERROR] Failed to get current frame: {e}")
        traceback.print_exc()
        return None, None, None

    if current_frame is None:
        print("[DEBUG] current_frame is None, returning None")
        return None, None, None

    current_slots = {}
    try:
        row_dict = Create_Dataset_Row(current_frame, id, match_id)
        print(f"[DEBUG] Dataset row created successfully, keys: {list(row_dict.keys()) if row_dict else 'None'}")
    except Exception as e:
        print(f"[ERROR] Failed to create dataset row: {e}")
        traceback.print_exc()
        row_dict = None

    try:
        state = check_match_status(current_frame)
        print(f"[DEBUG] Match status checked: {state}")
        if state == "win" or state == "loss":
            return state, None, current_frame
    except Exception as e:
        print(f"[ERROR] Failed to check match status: {e}")
        traceback.print_exc()

    if row_dict:
        try:
            current_slots = {
                "slot_1": row_dict["slot_1"],
                "slot_2": row_dict["slot_2"],
                "slot_3": row_dict["slot_3"],
                "slot_4": row_dict["slot_4"],
            }
            return row_dict, current_slots, current_frame
        except Exception as e:
            print(f"[ERROR] Failed to extract slots: {e}")
            traceback.print_exc()

    return None, None, current_frame


class ClashRoyalEnv:
    """Environment wrapper for the Clash Royale agent."""

    def __init__(self, step_delay=1.5, max_steps=500, reward_win=1, reward_lose=-1, reward_draw=0, window_title="BlueStacks App Player 1"):
        print(f"[DEBUG] ClashRoyalEnv.__init__ called with step_delay={step_delay}, max_steps={max_steps}, window_title={window_title}")
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_draw = reward_draw
        self.window_title = window_title

        self.current_step = 0
        self.done = False
        self.prev_obs = None
        self.obs = None
        self.current_slots = {}
        # ensure these exist to avoid AttributeError when calling Observation(...)
        self.id = 0
        self.match_id = 0
        print("[DEBUG] ClashRoyalEnv initialized successfully")

    def reset(self, max_attempts=30, wait_between=1.0):
        """
        Reset environment and wait for initial observation.
        Returns: (observation, slots) where observation is a pandas.DataFrame or None,
        and slots is a dict or None on failure.
        Side-effect: updates self.current_slots, self.obs, and self.last_frame.
        """
        print("[DEBUG] reset() called")
        self.current_step = 0
        self.done = False
        self.prev_obs = None
        self.obs = None
        self.current_slots = {}
        self.last_frame = None

        attempt = 0
        while attempt < max_attempts:
            print(f"[DEBUG] Waiting for initial observation... (attempt {attempt + 1})")
            try:
                row, slots, frame = Observation(self.id, self.match_id, window_title=self.window_title)
            except Exception as e:
                print(f"[ERROR] Observation() raised: {e}")
                traceback.print_exc()
                row, slots, frame = None, None, None

            if isinstance(row, str) and row in ("win", "loss"):
                print(f"[DEBUG] Observation returned terminal status {row} during reset, retrying...")
            elif row:
                try:
                    df = pd.DataFrame([row])
                    self.obs = df
                    self.current_slots = slots or {}
                    self.last_frame = frame
                    print(
                        f"[DEBUG] Initial observation obtained, shape: {self.obs.shape}, "
                        f"slots: {self.current_slots}"
                    )
                    self.id += 1
                    return self.obs, self.current_slots
                except Exception as e:
                    print(f"[ERROR] converting observation to DataFrame: {e}")
                    traceback.print_exc()
                    return None, None
            else:
                sleep(wait_between)

            attempt += 1

        print("[ERROR] Failed to get initial observation after retries")
        return None, None

    def step(self, action, pos_x, pos_y, state, current_slots=None):
        """
        Execute action and return a 5-tuple:
        (next_obs, reward, done, slots, frame)

        next_obs: pd.DataFrame or None or status string
        reward: float
        done: bool
        slots: dict (may be {} or None)
        frame: screenshot path returned by Frame_Handler / Observation
        """
        if state is None:
            print("[ERROR] ClashRoyalEnv.step() called with state None")
            obs_raw, slots, frame = Observation(getattr(self, "id", self.id), getattr(self, "match_id", 0), window_title=self.window_title)
            self.current_slots = slots or getattr(self, "current_slots", {})

            if obs_raw is not None and isinstance(obs_raw, pd.DataFrame):
                reward = 0.0
                if self.prev_obs is not None:
                    try:
                        reward = float(compute_step_reward(self.prev_obs, obs_raw) or 0.0)
                    except Exception as e:
                        print(f"[WARN] compute_step_reward failed on initial state: {e}")
                        traceback.print_exc()
                        reward = 0.0

                self.prev_obs = obs_raw
                self.obs = obs_raw
                self.id += 1
                return obs_raw, reward, False, self.current_slots, frame

            return obs_raw, 0.0, False, self.current_slots, frame

        try:
            slots_to_use = current_slots if current_slots is not None else getattr(self, "current_slots", {})
            try:
                react_agent(action, pos_x, pos_y, slots_to_use)
            except TypeError:
                try:
                    react_agent(action, pos_x, pos_y)
                except Exception:
                    pass

            self.prev_obs = self.obs
            sleep(self.step_delay)

            obs_raw, slots, frame = Observation(getattr(self, "id", self.id), getattr(self, "match_id", 0), window_title=self.window_title)
            self.current_slots = slots or getattr(self, "current_slots", {})
            self.id += 1

            if isinstance(obs_raw, str):
                status = obs_raw.lower()
                if status == "win":
                    print("win detected in step observation, returning terminal state")
                    return status, float(self.reward_win), True, self.current_slots, frame
                if status == "loss":
                    print("loss detected in step observation, returning terminal state")
                    return status, float(self.reward_lose), True, self.current_slots, frame
                print("draw detected in step observation, returning terminal state")
                return status, 0.0, False, self.current_slots, frame

            if obs_raw is None:
                print("[WARN] Observation returned None, treating as no change")
                return None, 0.0, False, self.current_slots, frame

            try:
                obs_df = obs_raw if isinstance(obs_raw, pd.DataFrame) else pd.DataFrame([obs_raw])
                self.obs = obs_df
            except Exception:
                traceback.print_exc()
                return None, 0.0, False, self.current_slots, frame

            reward = 0.0
            done = False
            try:
                reward = float(compute_step_reward(self.prev_obs, self.obs) or 0.0)
            except Exception as e:
                print(f"[WARN] compute_step_reward failed: {e}")
                traceback.print_exc()
                reward = 0.0

            return self.obs, reward, done, self.current_slots, frame

        except Exception as e:
            print(f"[ERROR] step() failed: {e}")
            traceback.print_exc()
            return None, 0.0, True, None, None
