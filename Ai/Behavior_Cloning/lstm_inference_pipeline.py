from collections import deque
import torch

# Directly import the specific classes and functions to avoid circular namespace initializations
from Ai.Behavior_Cloning.lstm_training_piplan import LSTM_BehaviorCloning
from Ai.Data_Cleaning import final_clean


class LSTM_Inference_Pipeline:
    def __init__(self, model_path, device, window_size, input_size, output_size, hidden_size=128, num_layers=2, **kwargs):
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_buffer = deque(maxlen=window_size)
        self.model = LSTM_BehaviorCloning(self.input_size, hidden_size, num_layers=num_layers, num_actions=self.output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def reset_sequence_buffer(self):
        self.sequence_buffer.clear()

    def clean_frame(self, frame_row):
        cleaned_frame = frame_row.drop(columns=["match_id", "id"])
        final_clean(cleaned_frame)
