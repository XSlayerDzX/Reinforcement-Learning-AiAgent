import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


from Ai.RL.ClashRoyalEnv import ClashRoyalEnv
from Ai.RL.PPO_Buffer import PPOBuffer
from Ai.RL.PPO_LSTM_Model import PPO_LSTM_Model
from Ai.Data_Cleaning import final_clean


def clean_obs(obs):
    cleaned_frame = final_clean(obs)
    cleaned_frame = cleaned_frame.drop(columns=["match_id", "id"], errors="ignore")
    cleaned_frame = cleaned_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    last_cleaned_columns = cleaned_frame.columns.tolist()

    if cleaned_frame.shape[1] != 205:
        raise ValueError(
            f"Expected {205} features, got {cleaned_frame.shape[1]} after cleaning."
        )

    # Return a flat numeric vector for sequence buffering.
    return cleaned_frame.iloc[0].astype(float).tolist()

def sequenece_buffering(obs,sequence_buffer,window_size,input_size):
    cleaned_frame = clean_obs(obs)

    sequence_buffer.append(cleaned_frame)

    current_sequence = list(sequence_buffer)

    if len(current_sequence) < window_size:
        pad_rows = window_size - len(current_sequence)
        padding = [[0.0] * input_size for _ in range(pad_rows)]
        current_sequence = padding + current_sequence

    return current_sequence

def reset_sequence_buffer(sequence_buffer):
    sequence_buffer.clear()

## masking actions based on availability features in the obs can be done later after the prototype is working end to end,
## for now we will just let the model learn to not take illegal actions by giving it negative rewards when it does so.


###


###

###

def ppo_update(actor_network, optimizer, states, actions, old_log_probs, advantages, epsilon=0.2):
    # 1. Get the current policy's log probabilities for the actions taken
    # (Assuming actor_network outputs logits and we use a Categorical distribution)
    logits = actor_network(states)
    dist = torch.distributions.Categorical(logits=logits)
    current_log_probs = dist.log_prob(actions)

    # 2. Calculate the probability ratio: r_t(theta)
    # Using log probabilities makes this numerically stable (exp(log_a - log_b) = a/b)
    ratios = torch.exp(current_log_probs - old_log_probs)

    # 3. Calculate the two parts of the PPO min() function
    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # 4. The PPO Objective (Maximization)
    ppo_objective = torch.min(surrogate1, surrogate2)

    # 5. Convert to a Loss for Gradient Descent (Minimization) by taking the negative mean
    loss = -ppo_objective.mean()

    # 6. Perform the Gradient calculation and network update
    optimizer.zero_grad()
    loss.backward()  # PyTorch calculates the gradients here
    optimizer.step()  # PyTorch updates the network weights here

    return loss.item()

### PPO value update function
def critic_update(critic_network, optimizer, states, returns):
    """
    Updates the Critic (Value) network using Mean Squared Error.

    Args:
        critic_network: The PyTorch neural network predicting V(s).
        optimizer: The PyTorch optimizer (e.g., Adam) assigned to the critic.
        states: Tensor of states from the mini-batch, shape (batch_size, input_dim).
        returns: Tensor of actual calculated returns, shape (batch_size,).

    Returns:
        float: The calculated loss for logging purposes.
    """
    # 1. Get the Critic's current predictions for the batch of states
    values = critic_network(states)

    # PyTorch networks often output shape (batch_size, 1).
    # We need to squeeze it to (batch_size,) so it matches the 'returns' tensor perfectly.
    values = values.squeeze(-1)

    # 2. Calculate the Loss (Mean Squared Error)
    loss_fn = nn.MSELoss()
    critic_loss = loss_fn(values, returns)

    # 3. Perform Gradient Descent
    optimizer.zero_grad()  # Clear old gradients
    critic_loss.backward()  # Calculate new gradients

    # Optional but highly recommended: Gradient Clipping to prevent training crashes
    torch.nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=0.5)

    optimizer.step()  # Update the network weights

    return critic_loss.item()  # Return the loss value for logging

### PPO Trainer function that combines all the steps together
# def train_loop(env, model, buffer, epochs=10, batch_size=64):
#     # ==========================================
#     # PHASE 1: Rollout (Data Collection)
#     # ==========================================
#     model.eval()  # Set to evaluation mode for playing
#
#     state = env.reset()
#
#     h_s , c_s = model.get_initial_lstm_states()  # Get initial LSTM states (if using LSTM)
#     while not buffer.is_full():
#         with torch.no_grad():  # Don't track gradients during gameplay
#             action_logits, value_estimate = model(state)
#             action = torch.argmax(action_logits, dim=-1)  # Get the action with the highest logit
#             log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
#
#         # Step the environment and get the RAW reward
#         next_state, reward, done = env.step(action)
#
#         # Store in PPO_Buffer
#         buffer.add(state, action, log_prob, reward, value_estimate, h_s, c_s, done)
#         state = next_state
#         h_s = h_s
#         c_s = c_s
#
#         if done:
#             state = env.reset()
#             h_s , c_s = model.get_initial_lstm_states()  # Reset LSTM states for new episode
#
#     # ==========================================
#     # PHASE 2: Advantage Estimation (GAE)
#     # ==========================================
#     # Get the value of the final state to bootstrap GAE
#     with torch.no_grad():
#         _, next_value = model(next_state)
#
#     # Calculate Advantages and Returns inside the buffer
#     buffer.compute_gae(next_value)
#
#     # ==========================================
#     # PHASE 3: Network Updates
#
#     model.train()  # Switch back to training mode to update weights
#
#     # Extract the prepared data from the buffer
#     states, actions, old_log_probs, returns, advantages = buffer.get_training_data()
#
#     for epoch in range(epochs):
#         # Yield mini-batches of size 64
#         for batch in buffer.get_mini_batches(batch_size):
#             # 1. Update the Critic (Value Network)
#             critic_loss = critic_update(model.critic, batch.states, batch.returns)
#
#             # 2. Update the Actor (Policy Network)
#             actor_loss = ppo_update(model.actor, batch.states, batch.actions, batch.old_log_probs, batch.advantages)
#
#     # Clear the buffer for the next game
#     buffer.reset_buffer()


# if __name__ == "__main__":
#     # Example usage (assuming you have an environment, model, and buffer set up)
#     env = ClashRoyalEnv()  # Your ClashRoyalEnv here
#     input_size = 205
#     hidden_size = 128
#     model = PPO_LSTM_Model(input_size= input_size , hidden_size=hidden_size, num_layers=2, num_actions= 11 , pretrained_model_path="path_to_pretrained_model.pth")
#     buffer = PPOBuffer()
#
#     train_loop(env, model, buffer)



