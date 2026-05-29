
import torch.nn.functional as F
import torch
import torch.nn as nn
from pyclipper import log_action


import pandas as pd


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

def actor_critic_update(
        actor_Critic_network,
        optimizer,
        states,
        actions,
        returns,
        old_log_probs,
        advantages,
        h_s,  # Initial hidden state for the batch
        c_s,  # Initial cell state for the batch
        vf = 0.5,
        ent_coef=0.01,
        epsilon=0.2):

    # 1. Get the current policy's log probabilities for the actions taken
    # (Assuming actor_network outputs logits , and critic_network outputs value estimates, and both can take LSTM states as input)

    actions_logits ,pos_logits , value_estimate , _ = actor_Critic_network(states, h_s, c_s)  # Get action logits, position logits, and value estimate from the network

    # torch.distributions.Categorical applies softmax internally, so we can directly pass the logits without applying softmax ourselves.
    dist_action = torch.distributions.Categorical(logits=actions_logits)
    dist_pos = torch.distributions.Normal(loc=pos_logits, scale=1.0)

    # log_prob applies log to the probabilities, so we can directly pass the logits without applying softmax ourselves.
    action_log_probs = dist_action.log_prob(actions[0])
    pos_log_prob = dist_pos.log_prob(torch.stack([actions[1], actions[2]],dim=-1))  # Assuming actions[1] and actions[2] are the position coordinates
    current_log_probs = action_log_probs + pos_log_prob.sum(dim=-1)  # Total log prob for the combined action

    # 2. Calculate the probability ratio: r_t(theta)
    # Using log probabilities makes this numerically stable (exp(log_a - log_b) = a/b)
    ratios = torch.exp(current_log_probs - old_log_probs)

    # 3. Calculate the two parts of the PPO min() function
    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # 4. The PPO Objective (Maximization)
    policy_loss = - torch.min(surrogate1, surrogate2).mean()

    # PyTorch networks often output shape (batch_size, 1).
    # We need to squeeze it to (batch_size,) so it matches the 'returns' tensor perfectly.
    values = value_estimate.squeeze(-1)
    values_loss = F.mse_loss(values, returns)

    entropy = dist_action.entropy() + dist_pos.entropy().sum(-1) # Total entropy from both distributions
    entropy_loss =  - ent_coef * entropy.mean()  # Entropy bonus to encourage exploration (adjust coefficient as needed)

    loss = policy_loss + vf * values_loss + entropy_loss

    optimizer.zero_grad()  # Clear old gradients
    loss.backward()  # Calculate new gradients
    torch.nn.utils.clip_grad_norm_(actor_Critic_network.parameters(), max_norm=0.5)
    optimizer.step()  # Update the network weights

    return   policy_loss.item(), values_loss.item()

### PPO Trainer function that combines all the steps together
# def train_loop(env, model, buffer, epochs=10, batch_size=64 , lr=3e-4):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # You can adjust the learning rate as needed
#
#     # PHASE 1: Rollout (Data Collection)
#     model.eval()  # Set to evaluation mode for playing
#     state = env.reset()
#     h_s , c_s = model.get_initial_lstm_states()  # Get initial LSTM states (if using LSTM)
#
#     while not buffer.is_full():
#         with torch.no_grad():  # Don't track gradients during gameplay
#             # Add a sequence dimension for the single frame [Batch=1, Time=1, Features]
#             state_inpt = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
#             action_logits, pos_logits, value_estimate , (next_h_s, next_c_s) = model(state_inpt, h_s, c_s)
#
#             # Remove the batch/time dimensions for sampling
#             action_logits = action_logits.squeeze()
#             pos_logits = pos_logits.squeeze()
#             value_estimate = value_estimate.squeeze()
#
#             dist_action = torch.distributions.Categorical(logits=action_logits).sample()
#             dist_pos = torch.distributions.Normal(loc=pos_logits, scale=1.0).sample()
#
#             log_action = dist_action.log_prob(dist_action)
#             log_pos = dist_pos.log_prob(dist_pos).sum(dim=-1)  # Total log prob for the combined action
#             log_prob = log_action + log_pos
#
#             action = (dist_action.item(), dist_pos[0].item(), dist_pos[1].item())
#         # Step the environment and get the RAW reward
#         # verify env.step((dist_action.cpu().numpy(), dist_pos.cpu().numpy()))
#         next_state, reward, done = env.step(action)  # Assuming your env can take the action in this format
#
#         # Store in PPO_Buffer
#         buffer.add(state, action, log_prob, reward, value_estimate, h_s, c_s, done)
#         state = next_state
#         h_s , c_s = next_h_s, next_c_s  # Update LSTM states for the next step
#
#         if done:
#             state = env.reset()
#             h_s , c_s = model.get_initial_lstm_states()  # Reset LSTM states for new episode


    # PHASE 2: Advantage Estimation (GAE)
    # Get the value of the final state to bootstrap GAE
    # with torch.no_grad():
    #     next_state_input = torch.tensor(next_state).unsqueeze(0).unsqueeze(0)
    #     _,_,next_value,_ = model(next_state_input, h_s, c_s)  # Get value estimate for the final state
    #     next_value = next_value.squeeze()
    # buffer.compute_gae(next_value)
    #
    #
    # # PHASE 3: Network Updates
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    # model.train()  # Switch back to training mode to update weights
    #
    # for epoch in range(epochs):
    #     # Yield mini-batches of size 64
    #     for batch in buffer.generate_sequential_batches(sequence_length=batch_size, device=device):
    #         state_seq = batch.states.unsqueeze(0)  # Add batch dimension for LSTM input
    #         actor_loss , critic_loss = actor_critic_update(model=model,
    #                                                        optimizer=optimizer,
    #                                                        states=state_seq,
    #                                                        actions=batch.actions,
    #                                                        returns=batch.returns,
    #                                                        old_log_probs=batch.old_log_probs,
    #                                                        advantages=batch.advantages,
    #                                                        h_s=batch.h_s,
    #                                                        c_s=batch.c_s,
    #                                                        vf=0.5,
    #                                                        ent_coef=0.01,
    #                                                        epsilon=0.2)
    #         print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    #
    # # Clear the buffer for the next game
    # buffer.reset_buffer()


