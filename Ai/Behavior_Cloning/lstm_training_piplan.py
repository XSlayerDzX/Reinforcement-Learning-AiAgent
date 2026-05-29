import torch
import torch.nn as nn


class LSTM_BehaviorCloning(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions, dropout=0.1):
        super(LSTM_BehaviorCloning, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.pos_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)   # [B, T, H]
        last = lstm_out[:, -1, :]
        last_transformed = self.shared(last)

        action_logits = self.action_head(last_transformed)
        pos_logits = self.pos_head(last_transformed)

        return action_logits, pos_logits
