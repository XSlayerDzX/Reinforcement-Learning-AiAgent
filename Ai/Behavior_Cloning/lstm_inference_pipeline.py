from collections import deque
import torch
import pandas as pd
from typing import cast

# Directly import the specific classes and functions to avoid circular namespace initializations
from Ai.Behavior_Cloning.lstm_training_piplan import LSTM_BehaviorCloning
from Ai.Data_Cleaning import final_clean


class LSTM_Inference_Pipeline:
    def __init__(self, model_path, device, window_size, input_size, output_size, hidden_size=128, num_layers=2, **kwargs):
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.wait_id = kwargs.get("wait_id", 0)
        self.avail_feature_to_action_id = kwargs.get("avail_feature_to_action_id", {})
        self.always_allow_wait = kwargs.get("always_allow_wait", True)
        self.full_elixir_threshold = kwargs.get("full_elixir_threshold", 10.0)
        self.wait_bias_at_full_elixir = kwargs.get("wait_bias_at_full_elixir", 0.0)
        self.sequence_buffer = deque(maxlen=window_size)
        self.model = LSTM_BehaviorCloning(self.input_size, hidden_size, num_layers=num_layers, num_actions=self.output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def reset_sequence_buffer(self):
        self.sequence_buffer.clear()

    def clean_frame(self, frame_row):
        cleaned_frame = cast(pd.DataFrame, final_clean(frame_row))
        if isinstance(cleaned_frame, pd.DataFrame):
            cleaned_frame = cleaned_frame.drop(columns=["match_id", "id"], errors="ignore")
        return cleaned_frame

    def _build_action_mask(self, last_row):
        mask = torch.ones(self.output_size, dtype=torch.bool, device=self.device)
        mapped_ids = {int(aid) for aid in self.avail_feature_to_action_id.values() if aid is not None}
        for aid in mapped_ids:
            if 0 <= aid < self.output_size:
                mask[aid] = False

        for feature_name, action_id in self.avail_feature_to_action_id.items():
            if action_id is None:
                continue
            action_id = int(action_id)
            if not (0 <= action_id < self.output_size):
                continue
            if feature_name in last_row.index and float(last_row[feature_name]) > 0:
                mask[action_id] = True

        if self.always_allow_wait and 0 <= self.wait_id < self.output_size:
            mask[self.wait_id] = True

        return mask

    def predict(self, row_df):
        if row_df is None:
            raise ValueError("row_df cannot be None")

        cleaned = self.clean_frame(row_df)
        if cleaned is None or cleaned.empty:
            raise ValueError("Unable to clean input frame for LSTM inference")

        if len(cleaned) > 1:
            cleaned = cleaned.tail(1)

        last_row = cleaned.iloc[-1]
        sequence_row = cleaned.iloc[0].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).tolist()
        self.sequence_buffer.append(sequence_row)

        while len(self.sequence_buffer) < self.window_size:
            self.sequence_buffer.appendleft([0.0] * self.input_size)

        window = torch.tensor(list(self.sequence_buffer), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits, pos_pred = self.model(window)

        action_logits = action_logits.squeeze(0)
        pos_pred = pos_pred.squeeze(0)
        action_mask = self._build_action_mask(last_row)
        masked_logits = action_logits.masked_fill(~action_mask, -1e9)

        current_elixir = 0.0
        if "Elixir" in last_row.index:
            try:
                current_elixir = float(last_row["Elixir"])
            except (TypeError, ValueError):
                current_elixir = 0.0

        if current_elixir >= self.full_elixir_threshold and self.wait_bias_at_full_elixir > 0:
            if 0 <= self.wait_id < masked_logits.numel():
                masked_logits = masked_logits.clone()
                masked_logits[self.wait_id] -= float(self.wait_bias_at_full_elixir)

        action_id = int(torch.argmax(masked_logits).item())
        if action_id == self.wait_id:
            pos = [-1.0, -1.0]
        else:
            pos = [float(pos_pred[0].item()), float(pos_pred[1].item())]

        return {
            "action_id": action_id,
            "pos_pred": pos,
        }
