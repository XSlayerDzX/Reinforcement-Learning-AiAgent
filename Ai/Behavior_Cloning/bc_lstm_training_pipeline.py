import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Ai.Behavior_Cloning.lstm_training_piplan import LSTM_BehaviorCloning


class LSTM_DataSet(Dataset):
    def __init__(
        self,
        df,
        window_size,
        features,
        targets,
        num_actions,
        wait_id=0,
        avail_feature_to_action_id=None,
        always_allow_wait=True,
        elixir_feature_name="Elixir",
    ):
        self.df = df
        self.window_size = window_size
        self.features = features
        self.targets = targets
        self.num_actions = num_actions
        self.wait_id = wait_id
        self.always_allow_wait = always_allow_wait
        self.elixir_feature_name = elixir_feature_name
        self.avail_feature_to_action_id = avail_feature_to_action_id or {}
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.features)}
        self.elixir_idx = self.feature_to_idx.get(self.elixir_feature_name)

        self.X_per_match = []
        self.Y_per_match = []

        self.X_flat = []
        self.action_flat = []
        self.pos_flat = []
        self.pos_mask_flat = []
        self.avail_mask_flat = []
        self.elixir_last_flat = []

    def _build_avail_mask_from_last_frame(self, last_frame):
        # Keep non-mapped actions legal by default and constrain only mapped actions.
        mask = np.ones(self.num_actions, dtype=bool)
        constrained_action_ids = set()

        for action_id in self.avail_feature_to_action_id.values():
            if action_id is None:
                continue
            action_id = int(action_id)
            if 0 <= action_id < self.num_actions:
                constrained_action_ids.add(action_id)

        for action_id in constrained_action_ids:
            mask[action_id] = False

        if self.always_allow_wait and 0 <= self.wait_id < self.num_actions:
            mask[self.wait_id] = True

        for feature_name, action_id in self.avail_feature_to_action_id.items():
            if action_id is None:
                continue
            action_id = int(action_id)
            if not (0 <= action_id < self.num_actions):
                continue

            feature_idx = self.feature_to_idx.get(feature_name)
            if feature_idx is None:
                continue

            if float(last_frame[feature_idx]) > 0:
                mask[action_id] = True

        if not bool(mask.any()):
            if 0 <= self.wait_id < self.num_actions:
                mask[self.wait_id] = True
            else:
                mask[0] = True

        return mask

    def _extract_elixir_from_last_frame(self, last_frame):
        if self.elixir_idx is None:
            return 0.0
        return float(last_frame[self.elixir_idx])

    def get_match_ids(self):
        return self.df["match_id"].unique().tolist()

    def transform(self):
        self.X_per_match = []
        self.Y_per_match = []

        for match_id in self.get_match_ids():
            match_set = self.df[self.df["match_id"] == match_id]
            match_X = []
            match_Y = []

            for i in range(len(match_set)):
                start = max(0, i - self.window_size)

                if i == 0:
                    window_frames = []
                else:
                    window_frames = match_set.iloc[start:i][self.features].values.tolist()

                if len(window_frames) < self.window_size:
                    pad_rows = self.window_size - len(window_frames)
                    padding = [[0.0] * len(self.features)] * pad_rows
                    window_frames = padding + window_frames

                target_row = match_set.iloc[i][self.targets]
                target_list = target_row.values.tolist()

                match_X.append(window_frames)
                match_Y.append(target_list)

            self.X_per_match.append(match_X)
            self.Y_per_match.append(match_Y)

        self.flatten()

    def flatten(self):
        self.X_flat = []
        self.action_flat = []
        self.pos_flat = []
        self.pos_mask_flat = []
        self.avail_mask_flat = []
        self.elixir_last_flat = []

        for match_X, match_Y in zip(self.X_per_match, self.Y_per_match):
            for window_frames, target_list in zip(match_X, match_Y):
                action_id = int(target_list[0])
                pos_x = float(target_list[1])
                pos_y = float(target_list[2])

                last_frame = window_frames[-1] if window_frames else [0.0] * len(self.features)
                avail_mask = self._build_avail_mask_from_last_frame(last_frame)

                # Safety: force target action legal to avoid CE failures on noisy rows.
                if 0 <= action_id < self.num_actions:
                    avail_mask[action_id] = True

                self.X_flat.append(window_frames)
                self.action_flat.append(action_id)
                self.pos_flat.append([pos_x, pos_y])

                use_pos = (action_id != self.wait_id) and (pos_x != -1) and (pos_y != -1)
                self.pos_mask_flat.append(use_pos)
                self.avail_mask_flat.append(avail_mask)
                self.elixir_last_flat.append(self._extract_elixir_from_last_frame(last_frame))

    def __len__(self):
        return len(self.X_flat)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_flat[idx], dtype=torch.float32)
        action = torch.tensor(self.action_flat[idx], dtype=torch.long)
        pos = torch.tensor(self.pos_flat[idx], dtype=torch.float32)
        pos_mask = torch.tensor(self.pos_mask_flat[idx], dtype=torch.bool)
        avail_mask = torch.tensor(self.avail_mask_flat[idx], dtype=torch.bool)
        elixir_last = torch.tensor(self.elixir_last_flat[idx], dtype=torch.float32)
        return x, action, pos, pos_mask, avail_mask, elixir_last


def apply_action_mask(action_logits, avail_mask):
    return action_logits.masked_fill(~avail_mask, -1e9)


def compute_loss(
    action_logits,
    pos_pred,
    target_actions,
    target_pos,
    pos_mask,
    avail_mask,
    elixir_last,
    pos_weight=1.0,
    wait_id=0,
    wait_weight=0.07,
    leak_weight=0.2,
    full_elixir_threshold=10.0,
):
    num_actions = action_logits.size(1)
    class_weights = torch.ones(num_actions, device=action_logits.device)
    if 0 <= wait_id < num_actions:
        class_weights[wait_id] = wait_weight

    masked_action_logits = apply_action_mask(action_logits, avail_mask)
    action_loss = F.cross_entropy(masked_action_logits, target_actions, weight=class_weights)

    if pos_mask.any():
        pos_loss = F.mse_loss(pos_pred[pos_mask], target_pos[pos_mask], reduction="mean")
    else:
        pos_loss = torch.zeros((), device=action_logits.device, dtype=action_logits.dtype)

    action_probs = torch.softmax(masked_action_logits, dim=-1)
    p_wait = action_probs[:, wait_id]

    non_wait_legal = avail_mask.clone()
    if 0 <= wait_id < non_wait_legal.shape[1]:
        non_wait_legal[:, wait_id] = False
    has_playable_nonwait = non_wait_legal.any(dim=1)

    is_full_elixir = elixir_last >= full_elixir_threshold
    target_not_wait = target_actions != wait_id
    leak_gate = is_full_elixir & has_playable_nonwait & target_not_wait

    if leak_gate.any():
        leak_loss = p_wait[leak_gate].mean()
    else:
        leak_loss = torch.zeros((), device=action_logits.device, dtype=action_logits.dtype)

    total_loss = action_loss + (pos_weight * pos_loss) + (leak_weight * leak_loss)
    return total_loss, action_loss, pos_loss, leak_loss, masked_action_logits


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_one_epoch(
    model,
    data_loader,
    device,
    split,
    epoch_idx,
    optimizer=None,
    pos_weight=1.0,
    wait_id=0,
    wait_weight=0.07,
    leak_weight=0.2,
    full_elixir_threshold=10.0,
    log_file=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss_sum = 0.0
    action_loss_sum = 0.0
    pos_loss_sum = 0.0
    leak_loss_sum = 0.0

    total_samples = 0
    correct_actions = 0

    valid_pos_samples = 0
    valid_pos_coords = 0
    pos_abs_err_sum = 0.0

    leak_eval_samples = 0
    leak_pred_wait = 0

    t0 = time.time()

    for x, action, pos, pos_mask, avail_mask, elixir_last in data_loader:
        x = x.to(device)
        action = action.to(device)
        pos = pos.to(device)
        pos_mask = pos_mask.to(device)
        avail_mask = avail_mask.to(device)
        elixir_last = elixir_last.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            action_logits, pos_pred = model(x)
            total_loss, action_loss, pos_loss, leak_loss, masked_action_logits = compute_loss(
                action_logits=action_logits,
                pos_pred=pos_pred,
                target_actions=action,
                target_pos=pos,
                pos_mask=pos_mask,
                avail_mask=avail_mask,
                elixir_last=elixir_last,
                pos_weight=pos_weight,
                wait_id=wait_id,
                wait_weight=wait_weight,
                leak_weight=leak_weight,
                full_elixir_threshold=full_elixir_threshold,
            )

            if is_train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bs = x.size(0)
        total_samples += bs
        total_loss_sum += total_loss.item() * bs
        action_loss_sum += action_loss.item() * bs
        leak_loss_sum += leak_loss.item() * bs

        pred_actions = masked_action_logits.argmax(dim=1)
        correct_actions += (pred_actions == action).sum().item()

        non_wait_legal = avail_mask.clone()
        if 0 <= wait_id < non_wait_legal.shape[1]:
            non_wait_legal[:, wait_id] = False
        has_playable_nonwait = non_wait_legal.any(dim=1)
        is_full_elixir = elixir_last >= full_elixir_threshold
        leak_gate = is_full_elixir & has_playable_nonwait

        leak_eval_samples += int(leak_gate.sum().item())
        leak_pred_wait += int(((pred_actions == wait_id) & leak_gate).sum().item())

        if pos_mask.any():
            batch_valid_samples = int(pos_mask.sum().item())
            valid_pos_samples += batch_valid_samples
            pos_loss_sum += pos_loss.item() * batch_valid_samples

            abs_err = (pos_pred[pos_mask] - pos[pos_mask]).abs()
            pos_abs_err_sum += abs_err.sum().item()
            valid_pos_coords += abs_err.numel()

    epoch_record = {
        "event": "epoch",
        "epoch": epoch_idx,
        "split": split,
        "total_loss": total_loss_sum / max(total_samples, 1),
        "action_loss": action_loss_sum / max(total_samples, 1),
        "pos_loss": pos_loss_sum / max(valid_pos_samples, 1),
        "leak_loss": leak_loss_sum / max(total_samples, 1),
        "action_acc": correct_actions / max(total_samples, 1),
        "pos_mae": pos_abs_err_sum / max(valid_pos_coords, 1),
        "leak_rate_at_full_elixir": leak_pred_wait / max(leak_eval_samples, 1),
        "total_samples": total_samples,
        "valid_pos_samples": valid_pos_samples,
        "valid_pos_coords": valid_pos_coords,
        "correct_actions": correct_actions,
        "leak_eval_samples": leak_eval_samples,
        "leak_pred_wait": leak_pred_wait,
        "epoch_seconds": time.time() - t0,
        "timestamp": time.time(),
    }

    if log_file is not None:
        _append_jsonl(log_file, epoch_record)

    return epoch_record


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    num_epochs=20,
    pos_weight=1.0,
    wait_id=0,
    wait_weight=0.07,
    leak_weight=0.2,
    full_elixir_threshold=10.0,
    log_file="training_metrics.jsonl",
    reset_log_file=True,
):
    log_path = Path(log_file)
    if reset_log_file and log_path.exists():
        log_path.unlink()

    history = {
        "log_file": str(log_path),
        "train": [],
        "val": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_epoch = run_one_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            split="train",
            epoch_idx=epoch,
            optimizer=optimizer,
            pos_weight=pos_weight,
            wait_id=wait_id,
            wait_weight=wait_weight,
            leak_weight=leak_weight,
            full_elixir_threshold=full_elixir_threshold,
            log_file=log_path,
        )

        val_epoch = run_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            split="val",
            epoch_idx=epoch,
            optimizer=None,
            pos_weight=pos_weight,
            wait_id=wait_id,
            wait_weight=wait_weight,
            leak_weight=leak_weight,
            full_elixir_threshold=full_elixir_threshold,
            log_file=log_path,
        )

        history["train"].append(train_epoch)
        history["val"].append(val_epoch)

        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_epoch['total_loss']:.4f} train_acc={train_epoch['action_acc']:.4f} train_leak={train_epoch['leak_rate_at_full_elixir']:.4f} "
            f"val_loss={val_epoch['total_loss']:.4f} val_acc={val_epoch['action_acc']:.4f} val_leak={val_epoch['leak_rate_at_full_elixir']:.4f}"
        )

    return history


def build_dataloaders(
    train_csv,
    val_csv,
    features,
    targets,
    num_actions,
    window_size,
    batch_size,
    wait_id=0,
    avail_feature_to_action_id=None,
    always_allow_wait=True,
):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_set = LSTM_DataSet(
        train_df,
        window_size=window_size,
        features=features,
        targets=targets,
        num_actions=num_actions,
        wait_id=wait_id,
        avail_feature_to_action_id=avail_feature_to_action_id,
        always_allow_wait=always_allow_wait,
    )
    train_set.transform()

    val_set = LSTM_DataSet(
        val_df,
        window_size=window_size,
        features=features,
        targets=targets,
        num_actions=num_actions,
        wait_id=wait_id,
        avail_feature_to_action_id=avail_feature_to_action_id,
        always_allow_wait=always_allow_wait,
    )
    val_set.transform()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(input_size, hidden_size, num_layers, num_actions, dropout=0.1, device=None):
    model = LSTM_BehaviorCloning(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_actions=num_actions,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model

