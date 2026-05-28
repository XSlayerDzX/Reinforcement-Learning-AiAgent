from collections import deque

import torch
import Ai.RL.ClashRoyalEnv
import Ai.RL.PPO_LSTM_Model
import Ai.RL.PPO_Trainer
from Ai.RL.PPO_Trainer import sequenece_buffering


## here we will implement the main training loop for PPO, including interaction with the environment, collecting transitions,
# and updating the policy and value networks using the data in the PPOBuffer. We will also include logging and model saving functionality.

def collect_rollout(env, model,rollouts_to_collect=1):
    rollouts = []
    window_buffer = deque(maxlen=10)


    for rollout in range(rollouts_to_collect):
        window_buffer.clear()
        states = []
        windows = []
        actions = []
        x = []
        y = []
        log_probs = []
        rewards = []
        values = []
        next_states = []
        h_s_list = []
        c_s_list = []
        done = False

        h_s, c_s = None, None  # Initialize LSTM states
        state = env.reset()  ## Reset the environment to get the initial state
        while not done:
           current_window =  sequenece_buffering(state, window_buffer,10,205)
           action_logits, pos_logits , value_estimate , (h_s, c_s) = model.forward(current_window)
           ##get the action and log probability
           action = torch.argmax(action_logits, dim=-1).item()  # Get the action
           log_prob = action_logits.gather(-1, action).squeeze(-1)
           pos_x = pos_logits[0, 0].item()  # Get the x position
           pos_y = pos_logits[0, 1].item()  # Get the y











