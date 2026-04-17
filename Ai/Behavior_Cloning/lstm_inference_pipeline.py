import torch
from collections import deque
import pandas as pd
from Ai.Data_Cleaning import final_clean
from Ai.Behavior_Cloning.lstm_training_piplan import LSTM_BehaviorCloning


class LSTM_Inference_Pipeline:
    def __init__(
        self,
        model_path,
        device,
        window_size,
        input_size,
        output_size,
        hidden_size=128,
        num_layers=2,
    ):
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_buffer = deque(maxlen=window_size)

        self.model = LSTM_BehaviorCloning(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            num_actions=self.output_size,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Make architecture mismatch errors explicit before load_state_dict raises generic shape errors.
        if "lstm.weight_ih_l0" in state_dict:
            ckpt_hidden_size = int(state_dict["lstm.weight_ih_l0"].shape[0] // 4)
            if ckpt_hidden_size != self.hidden_size:
                raise ValueError(
                    f"Checkpoint hidden_size={ckpt_hidden_size} does not match inference hidden_size={self.hidden_size}."
                )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def reset_sequence_buffer(self):
        self.sequence_buffer.clear()

    def clean_frame(self, frame_row):
        cleaned_frame = final_clean(frame_row)
        cleaned_frame = cleaned_frame.drop(columns=["match_id", "id"], errors="ignore")
        cleaned_frame = cleaned_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if cleaned_frame.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {cleaned_frame.shape[1]} after cleaning."
            )

        # Return a flat numeric vector for sequence buffering.
        return cleaned_frame.iloc[0].astype(float).tolist()

    def buffer_handler(self, frame_row):
        cleaned_frame = self.clean_frame(frame_row)

        self.sequence_buffer.append(cleaned_frame)

        current_sequence = list(self.sequence_buffer)

        if len(current_sequence) < self.window_size:
            pad_rows = self.window_size - len(current_sequence)
            padding = [[0.0] * self.input_size for _ in range(pad_rows)]
            current_sequence = padding + current_sequence

        return current_sequence

    def predict(self, frame_row):

        with torch.no_grad():
            # torch.float32 provides better precision for LSTM
            input_sequence = torch.tensor(
                self.buffer_handler(frame_row), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

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
            if action_id == 0:
                pos_pred[0] = -1
                pos_pred[1] = -1
            result = {"action_id": action_id, "pos_pred": pos_pred}
            return result