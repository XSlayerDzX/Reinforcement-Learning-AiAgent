import torch
from collections import deque
import numpy as np
from Ai.Data_Cleaning import final_clean
from Ai.Behavior_Cloning.lstm_training_piplan import LSTM_BehaviorCloning


class LSTM_Inference_Pipeline:
    def __init__(self, model_path, device, window_size, input_size , output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_buffer = deque(maxlen=window_size)
        self.model = LSTM_BehaviorCloning(self.input_size, 64, num_layers=2, num_actions=self.output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)).to(self.device)
        self.model.eval()

    def reset_sequence_buffer(self):
        self.sequence_buffer.clear()

    def clean_frame(self, frame_row):
        cleaned_frame = frame_row.drop(columns=["match_id", "id"])
        #cleaned_frame = cleaned_frame.astype(float)
        final_clean(cleaned_frame)
        return cleaned_frame

    def buffer_handler(self, frame_row):
        cleaned_frame = self.clean_frame(frame_row)

        self.sequence_buffer.append(cleaned_frame)

        current_sequence = list(self.sequence_buffer)
        #don't use np.vstack as it will create a new array and consume more memory


        if len(current_sequence) < self.window_size:
            pad_rows = self.window_size - len(current_sequence)
            padding = [[0.0] * self.input_size] * pad_rows
            current_sequence = padding + current_sequence

        return current_sequence

    def predict(self, frame_row):

        with torch.no_grad():
            # torch.float32 provides better precision for LSTM
            input_sequence = torch.tensor(self.buffer_handler(frame_row), dtype=torch.float32).unsqueeze(0).to(self.device)

            # Unpack the multi-task outputs
            action_logits , pos_pred = self.model(input_sequence)
            # Process the Action (Classification)
            #  Apply softmax to get probabilities, then grab the index of the highest probability
            # dim = -1 means the last dimension
            action_probs = torch.softmax(action_logits, dim=-1)
            action_id = torch.argmax(action_probs, dim=-1).item()

            # Process the Position (Regression)
            # Squeeze out the batch dimension and convert to a numpy array
            pos_pred = pos_pred.squeeze(0).cpu().numpy()
            result = {"action_id": action_id, "pos_pred": pos_pred}
            return result