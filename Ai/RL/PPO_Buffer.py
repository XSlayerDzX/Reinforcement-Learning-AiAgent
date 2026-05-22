import random


class Transition:
    def __init__(self, state, action, reward, value, next_state, h_s, c_s, done): ## A simple class to store a transition in the environment, including LSTM states
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value
        self.next_state = next_state
        self.h_s = h_s
        self.c_s = c_s
        self.done = done




class PPOBuffer:
    def __init__(self,mini_batch_size=32): ## A buffer to store transitions and related data for PPO training, including LSTM states
        self.buffer = []
        self.mini_batch_size = mini_batch_size
        self.minibatch = []
        self.returns_to_go = []
        self.advantages = []
        self.lstm_states = []

    def reset_buffer(self): ## Clear the buffer and related data structures for a new episode or training iteration
        self.buffer = []
        self.minibatch = []
        self.advantages = []
        self.returns_to_go = []
        self.lstm_states = []


    def add(self, state, action, reward, value,next_state, h_s, c_s, done): ## Add a transition to the buffer
        transition = Transition(state, action, reward, value,next_state, h_s, c_s, done)
        self.buffer.append(transition)
        self.lstm_states.append((h_s, c_s))




    def compute_returns_to_go_and_advantages(self, gamma=0.99, lam=0.95): ## Compute returns-to-go and advantages for the transitions in the buffer, using Monte Carlo returns and a simple advantage calculation (can be improved with GAE)
        self.returns_to_go = []
        self.advantages = []
        G = 0
        A = 0
        for transition in reversed(self.buffer):
            G = transition.reward + gamma * G * (1 - transition.done) ## Compute the return-to-go using the reward and the discounted future return, accounting for episode termination
            # calculate advantage using monte carlo return and value function
            # should be changed to use GAE (Generalized Advantage Estimation) for better performance
            #
            #
            A = G - transition.value
            #
            #
            #
            self.returns_to_go.insert(0, G) ## insert at the beginning of the list to maintain the correct order
            self.advantages.insert(0, A) ## same here


    def sample(self): ## Sample a mini-batch of transitions from the buffer for training, ensuring that there are enough transitions to sample
        if len(self.buffer) < self.mini_batch_size:
            return None
        self.minibatch = random.sample(self.buffer, self.mini_batch_size)
        return self.minibatch


