import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions, pretrained_model_path=None):
        super(PPO_LSTM_Model, self).__init__()

        # Shared feature extractor (will be copied from pretrained)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Shared transformation layer (will be copied from pretrained)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor heads (will be copied from pretrained)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.pos_head = nn.Linear(hidden_size, 2)

        # NEW Critic head (value network - initialized randomly, NOT from pretrained)
        self.critic_head = nn.Linear(hidden_size, 1)

        if pretrained_model_path:
            self.warm_start(pretrained_model_path)

    def forward(self, x):
        lstm_out, (h_n, c_n)= self.lstm(x)  # [B, T, H]
        last = lstm_out[:, -1, :]
        last_transformed = self.shared(last)

        # Actor outputs
        action_logits = self.action_head(last_transformed)
        pos_logits = self.pos_head(last_transformed)

        # Critic output (NEW)
        state_value = self.critic_head(last_transformed)

        return action_logits, pos_logits, state_value , (h_n, c_n)

    def warm_start(self, pretrained_model_path):
        """Load weights from the pretrained behavior cloning model."""
        pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Get the current model's state dict keys
        current_keys = set(self.state_dict().keys())

        # Load only matching keys (lstm, shared, action_head, pos_head)
        # The critic_head won't exist in pretrained_state_dict, so it stays randomly initialized
        matched_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key in current_keys:
                matched_state_dict[key] = value

        # Load the matched weights
        self.load_state_dict(matched_state_dict, strict=False)
        print(f"Warm-started from {pretrained_model_path}")
        print(f"Loaded {len(matched_state_dict)} layers from pretrained model")
        print(f"Uninitialized (randomly initialized): critic_head")