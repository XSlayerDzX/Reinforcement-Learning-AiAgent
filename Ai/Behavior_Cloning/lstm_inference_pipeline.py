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
        wait_id=0,
        avail_feature_to_action_id=None,
        always_allow_wait=True,
        full_elixir_threshold=10.0,
        wait_bias_at_full_elixir=0.0,
    ):
        self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.wait_id = wait_id
        self.always_allow_wait = always_allow_wait
        self.full_elixir_threshold = float(full_elixir_threshold)
        self.wait_bias_at_full_elixir = float(wait_bias_at_full_elixir)
        self.avail_feature_to_action_id = avail_feature_to_action_id or {}
        self.last_cleaned_columns = []
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
        self.last_cleaned_columns = cleaned_frame.columns.tolist()

        if cleaned_frame.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {cleaned_frame.shape[1]} after cleaning."
            )

        # Return a flat numeric vector for sequence buffering.
        return cleaned_frame.iloc[0].astype(float).tolist()

    def build_avail_mask_from_last_frame(self, last_frame):
        # Keep actions without availability features legal by default.
        # Only actions referenced in avail_feature_to_action_id are constrained.
        avail_mask = torch.ones(self.output_size, dtype=torch.bool, device=self.device)
        constrained_action_ids = set()

        for action_id in self.avail_feature_to_action_id.values():
            if action_id is None:
                continue
            action_id = int(action_id)
            if 0 <= action_id < self.output_size:
                constrained_action_ids.add(action_id)

        for action_id in constrained_action_ids:
            avail_mask[action_id] = False

        if self.always_allow_wait and 0 <= self.wait_id < self.output_size:
            avail_mask[self.wait_id] = True

        column_to_idx = {name: idx for idx, name in enumerate(self.last_cleaned_columns)}

        for feature_name, action_id in self.avail_feature_to_action_id.items():
            if action_id is None:
                continue
            action_id = int(action_id)
            if not (0 <= action_id < self.output_size):
                continue

            feature_idx = column_to_idx.get(feature_name)
            if feature_idx is None:
                continue

            if float(last_frame[feature_idx]) > 0:
                avail_mask[action_id] = True

        # Keep inference safe even if mapping is empty or feature extraction fails.
        if not bool(avail_mask.any()):
            if 0 <= self.wait_id < self.output_size:
                avail_mask[self.wait_id] = True
            else:
                avail_mask[0] = True

        return avail_mask

    @staticmethod
    def apply_action_mask(action_logits, avail_mask):
        return action_logits.masked_fill(~avail_mask.unsqueeze(0), -1e9)

    def _get_last_frame_elixir(self, last_frame):
        column_to_idx = {name: idx for idx, name in enumerate(self.last_cleaned_columns)}
        elixir_idx = column_to_idx.get("Elixir")
        if elixir_idx is None:
            return 0.0
        return float(last_frame[elixir_idx])

    def _apply_wait_bias_for_full_elixir(self, masked_action_logits, avail_mask, last_frame):
        if self.wait_bias_at_full_elixir <= 0:
            return masked_action_logits
        if not (0 <= self.wait_id < self.output_size):
            return masked_action_logits
        if not bool(avail_mask[self.wait_id]):
            return masked_action_logits

        non_wait_legal = avail_mask.clone()
        non_wait_legal[self.wait_id] = False
        if not bool(non_wait_legal.any()):
            return masked_action_logits

        elixir_value = self._get_last_frame_elixir(last_frame)
        if elixir_value < self.full_elixir_threshold:
            return masked_action_logits

        adjusted_logits = masked_action_logits.clone()
        adjusted_logits[:, self.wait_id] = adjusted_logits[:, self.wait_id] - self.wait_bias_at_full_elixir
        return adjusted_logits

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
            sequence = self.buffer_handler(frame_row)
            input_sequence = torch.tensor(
                sequence, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            # Unpack the multi-task outputs
            action_logits , pos_pred = self.model(input_sequence)

            last_frame = sequence[-1] if sequence else [0.0] * self.input_size
            avail_mask = self.build_avail_mask_from_last_frame(last_frame)
            masked_action_logits = self.apply_action_mask(action_logits, avail_mask)
            masked_action_logits = self._apply_wait_bias_for_full_elixir(
                masked_action_logits=masked_action_logits,
                avail_mask=avail_mask,
                last_frame=last_frame,
            )

            # Process the Action (Classification)
            #  Apply softmax to get probabilities, then grab the index of the highest probability
            # dim = -1 means the last dimension
            action_probs = torch.softmax(masked_action_logits, dim=-1)
            action_id = torch.argmax(action_probs, dim=-1).item()

            # Process the Position (Regression)
            # Squeeze out the batch dimension and convert to a numpy array
            pos_pred = pos_pred.squeeze(0).cpu().numpy()
            if action_id == self.wait_id:
                pos_pred[0] = -1
                pos_pred[1] = -1
            result = {"action_id": action_id, "pos_pred": pos_pred}
            return result