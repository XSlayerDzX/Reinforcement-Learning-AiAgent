import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

class PPO_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions, pretrained_model_path=None):
        print(f"[DEBUG] PPO_LSTM_Model.__init__ called with:")
        print(f"  - input_size: {input_size}")
        print(f"  - hidden_size: {hidden_size}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - num_actions: {num_actions}")
        print(f"  - pretrained_model_path: {pretrained_model_path}")

        super(PPO_LSTM_Model, self).__init__()

        try:
            # Shared feature extractor (will be copied from pretrained)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
            print("[DEBUG] LSTM layer created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create LSTM layer: {e}")
            traceback.print_exc()
            raise

        try:
            # Shared transformation layer (will be copied from pretrained)
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            print("[DEBUG] Shared transformation layer created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create shared layer: {e}")
            traceback.print_exc()
            raise

        try:
            # Actor heads (will be copied from pretrained)
            self.action_head = nn.Linear(hidden_size, num_actions)
            self.pos_head = nn.Linear(hidden_size, 2)
            print("[DEBUG] Actor heads created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create actor heads: {e}")
            traceback.print_exc()
            raise

        try:
            # NEW Critic head (value network - initialized randomly, NOT from pretrained)
            self.critic_head = nn.Linear(hidden_size, 1)
            print("[DEBUG] Critic head created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create critic head: {e}")
            traceback.print_exc()
            raise

        if pretrained_model_path:
            try:
                self.warm_start(pretrained_model_path)
            except Exception as e:
                print(f"[ERROR] Warm start failed: {e}")
                traceback.print_exc()
                raise

        print("[DEBUG] PPO_LSTM_Model initialization complete")

    def forward(self, x):
        print(f"[DEBUG] forward() called with input shape: {x.shape}")

        try:
            lstm_out, (h_n, c_n) = self.lstm(x)
            print(f"[DEBUG] LSTM output shape: {lstm_out.shape}")
            print(f"[DEBUG] Hidden state shape: {h_n.shape}, Cell state shape: {c_n.shape}")
        except Exception as e:
            print(f"[ERROR] LSTM forward pass failed: {e}")
            traceback.print_exc()
            raise

        try:
            last = lstm_out[:, -1, :]
            print(f"[DEBUG] Last timestep extracted, shape: {last.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to extract last timestep: {e}")
            traceback.print_exc()
            raise

        try:
            last_transformed = self.shared(last)
            print(f"[DEBUG] Shared transformation applied, shape: {last_transformed.shape}")
        except Exception as e:
            print(f"[ERROR] Shared transformation failed: {e}")
            traceback.print_exc()
            raise

        try:
            # Actor outputs
            action_logits = self.action_head(last_transformed)
            print(f"[DEBUG] Action logits computed, shape: {action_logits.shape}")
            print(f"[DEBUG] Action logits values: {action_logits}")
        except Exception as e:
            print(f"[ERROR] Action head failed: {e}")
            traceback.print_exc()
            raise

        try:
            pos_logits = self.pos_head(last_transformed)
            print(f"[DEBUG] Position logits computed, shape: {pos_logits.shape}")
            print(f"[DEBUG] Position logits values: {pos_logits}")
        except Exception as e:
            print(f"[ERROR] Position head failed: {e}")
            traceback.print_exc()
            raise

        try:
            # Critic output (NEW)
            state_value = self.critic_head(last_transformed)
            print(f"[DEBUG] State value computed, shape: {state_value.shape}, value: {state_value.item()}")
        except Exception as e:
            print(f"[ERROR] Critic head failed: {e}")
            traceback.print_exc()
            raise

        print("[DEBUG] forward() returning all outputs successfully")
        return action_logits, pos_logits, state_value, (h_n, c_n)

    def warm_start(self, pretrained_model_path):
        """Load weights from the pretrained behavior cloning model."""
        print(f"[DEBUG] warm_start() called with path: {pretrained_model_path}")

        try:
            pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')
            print(f"[DEBUG] Pretrained model loaded, keys: {list(pretrained_state_dict.keys())}")
        except Exception as e:
            print(f"[ERROR] Failed to load pretrained model: {e}")
            traceback.print_exc()
            raise

        try:
            # Get the current model's state dict keys
            current_keys = set(self.state_dict().keys())
            print(f"[DEBUG] Current model keys: {current_keys}")

            # Load only matching keys (lstm, shared, action_head, pos_head)
            # The critic_head won't exist in pretrained_state_dict, so it stays randomly initialized
            matched_state_dict = {}
            for key, value in pretrained_state_dict.items():
                if key in current_keys:
                    matched_state_dict[key] = value
                    print(f"[DEBUG] Matched key: {key}, shape: {value.shape}")
                else:
                    print(f"[DEBUG] Skipped key (not in current model): {key}")

            print(f"[DEBUG] Matched {len(matched_state_dict)} out of {len(pretrained_state_dict)} layers")
        except Exception as e:
            print(f"[ERROR] Failed to match state dicts: {e}")
            traceback.print_exc()
            raise

        try:
            # Load the matched weights
            self.load_state_dict(matched_state_dict, strict=False)
            print(f"[DEBUG] State dict loaded successfully (strict=False)")
        except Exception as e:
            print(f"[ERROR] Failed to load state dict: {e}")
            traceback.print_exc()
            raise

        print(f"[DEBUG] Warm-started from {pretrained_model_path}")
        print(f"[DEBUG] Loaded {len(matched_state_dict)} layers from pretrained model")
        print(f"[DEBUG] Uninitialized (randomly initialized): critic_head")
