import random
import torch


class Transition:
    def __init__(self, state, action, log_prob, reward, value, h_s, c_s, done):
        self.state = state
        self.action = action  # Expecting a tuple: (act_idx, pos_x, pos_y)
        self.log_prob = log_prob
        self.reward = reward
        self.value = value
        self.h_s = h_s
        self.c_s = c_s
        self.done = done


# Simple data class to pass batches to the trainer cleanly
class Batch:
    def __init__(self, states, actions, old_log_probs, returns, advantages, h_s, c_s):
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.returns = returns
        self.advantages = advantages
        self.h_s = h_s
        self.c_s = c_s


class PPOBuffer:
    def __init__(self):
        self.buffer = []
        self.returns_to_go = []
        self.advantages = []

    def reset_buffer(self):
        self.buffer = []
        self.advantages = []
        self.returns_to_go = []

    def add(self, state, action, log_prob, reward, value, h_s, c_s, done):
        self.buffer.append(Transition(state, action, log_prob, reward, value, h_s, c_s, done))

    def is_full(self, max_size=2048):
        return len(self.buffer) >= max_size

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """Calculates advantages and returns using GAE."""
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.buffer], dtype=torch.float32)


        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        self.advantages = advantages
        self.returns_to_go = advantages + values

    def generate_sequential_batches(self, sequence_length=64, device='cpu'):
        """
        Yields sequential chunks of data as tensors for LSTM training.
        """
        total_steps = len(self.buffer)
        indices = list(range(0, total_steps, sequence_length))

        # We can shuffle the chunks, but the frames INSIDE the chunk stay in temporal order
        random.shuffle(indices)

        # Normalize advantages across the whole buffer first (Good for stability)
        norm_adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for start_idx in indices:
            end_idx = min(start_idx + sequence_length, total_steps)
            chunk = self.buffer[start_idx:end_idx]

            # 1. Stack States
            states = torch.tensor([t.state for t in chunk], dtype=torch.float32).to(device)

            # 2. Unzip and Stack Actions (act_idx, pos_x, pos_y)
            act_idx = torch.tensor([t.action[0] for t in chunk], dtype=torch.float32).to(device)
            pos_x = torch.tensor([t.action[1] for t in chunk], dtype=torch.float32).to(device)
            pos_y = torch.tensor([t.action[2] for t in chunk], dtype=torch.float32).to(device)
            actions = (act_idx, pos_x, pos_y)

            # 3. Stack other metrics
            old_log_probs = torch.stack([t.log_prob for t in chunk]).to(device)
            returns = self.returns_to_go[start_idx:end_idx].to(device)
            advantages = norm_adv[start_idx:end_idx].to(device)

            # 4. Get the LSTM hidden states from the VERY FIRST frame of this sequence chunk
            h_s = chunk[0].h_s.to(device)
            c_s = chunk[0].c_s.to(device)

            yield Batch(states, actions, old_log_probs, returns, advantages, h_s, c_s)


