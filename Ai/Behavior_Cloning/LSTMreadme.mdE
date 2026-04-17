# Behavior Cloning LSTM Guide

This README explains the full training and inference workflow for the Clash Royale LSTM behavior cloning system, including:

- legal-action masking from available cards (`*_avab` features)
- wait handling and class weighting
- position-loss masking
- full-elixir leak mitigation (avoid waiting too much at 10 elixir)
- notebook and Python-module usage

---

## 1) Files and roles

- `Ai/Behavior_Cloning/bc_lstm_training_pipeline.py`
  - Production training pipeline
  - Contains dataset, masking, losses, epoch loop, and trainer
- `Ai/Behavior_Cloning/BC_LSTM_Training.ipynb`
  - Notebook wrapper to train anytime and save `lstm.pth`
  - Uses the pipeline module above
- `Ai/Behavior_Cloning/lstm_inference_pipeline.py`
  - Inference sequence buffer + masked action prediction + optional anti-leak wait bias
- `Ai/Behavior_Cloning/action_masking_config.py`
  - Shared masking and fallback config used by inference (and importable for training)
- `Ai/Agent/Agent_main.py`
  - Runtime agent loop that calls `LSTM_Inference_Pipeline`

---

## 2) Action space and availability mapping

The model predicts over a fixed action space, but only legal actions should be selectable each frame.

Legal-action masking uses this feature-to-action mapping from `action_masking_config.py`:

```python
AVAIL_FEATURE_TO_ACTION_ID = {
    "spear goblins_avab": 6,
    "musketeer_avab": 10,
    "mini pekka_avab": 1,
    "knight_avab": 2,
    "goblins_avab": 4,
    "goblin hut_avab": None,
    "goblin gang_avab": None,
    "goblin cage_avab": 11,
    "minions_avab": 9,
    "giant_avab": 5,
    "archers_avab": 7,
}
```

Notes:

- `None` means the feature exists but currently has no action ID in the BC action space.
- `wait` is configured by `WAIT_ID` and is always legal if `ALWAYS_ALLOW_WAIT=True`.

---

## 3) Training pipeline details

## 3.1 Dataset output (`LSTM_DataSet`)

Each sample returns:

```python
x, action, pos, pos_mask, avail_mask, elixir_last
```

- `x`: `[T, F]` window of features
- `action`: target class ID
- `pos`: `[pos_x, pos_y]`
- `pos_mask`: bool, true only when position should be learned
- `avail_mask`: `[num_actions]` bool legal-action mask from the last frame
- `elixir_last`: scalar elixir value from the last frame in the window

### How `avail_mask` is built

`_build_avail_mask_from_last_frame(last_frame)`:

1. starts with `True` for all actions
2. marks mapped actions as temporarily `False`
3. enables mapped actions whose `*_avab` feature value is `> 0`
4. always enables `wait` when configured
5. final safety fallback: if somehow all false, force wait true

Training safety guard per sample:

```python
avail_mask[action_id] = True
```

This avoids cross-entropy errors from occasional feature/label mismatch.

## 3.2 Masked logits

Before CE loss, invalid actions are suppressed:

```python
masked_logits = action_logits.masked_fill(~avail_mask, -1e9)
```

This enforces legal action learning structurally.

## 3.3 Losses used

Total loss:

```python
L_total = L_action + pos_weight * L_pos + leak_weight * L_leak
```

### Action loss (`L_action`)

- CE over masked logits
- class-weighted to downweight `wait`

```python
class_weights = torch.ones(num_actions)
class_weights[wait_id] = wait_weight
L_action = F.cross_entropy(masked_logits, target_actions, weight=class_weights)
```

### Position loss (`L_pos`)

MSE only where `pos_mask=True`.

`pos_mask=True` iff:

- action is not wait
- `pos_x != -1`
- `pos_y != -1`

This ignores invalid/no-op position labels.

### Leak penalty (`L_leak`)

Goal: reduce waiting at full elixir when non-wait legal actions exist.

Compute:

```python
p_wait = softmax(masked_logits)[:, wait_id]
```

Gate samples where:

- `elixir_last >= full_elixir_threshold`
- at least one legal non-wait action exists
- target action is not wait

Then:

```python
L_leak = mean(p_wait over gated samples)
```

This keeps wait legal while discouraging wasteful wait behavior in spend-now states.

## 3.4 Metrics logged per epoch

`run_one_epoch` logs:

- `total_loss`, `action_loss`, `pos_loss`, `leak_loss`
- `action_acc` (computed using **masked logits**)
- `pos_mae`
- `leak_rate_at_full_elixir` (predicted wait rate on full-elixir + playable states)

Logs are appended to `training_metrics.jsonl`.

---

## 4) Notebook workflow (`BC_LSTM_Training.ipynb`)

The notebook is now aligned to the training pipeline and can be rerun anytime.

It does:

1. import pipeline + shared config
2. load dataset and compute `features/targets`
3. build train/val dataloaders with masking config
4. build model + optimizer
5. train with leak-aware and masking-aware loss
6. save model to `lstm.pth`
7. optionally export training history CSV and print best epochs

This keeps notebook logic small and avoids drift by centralizing core logic in `bc_lstm_training_pipeline.py`.

---

## 5) Inference pipeline details

`lstm_inference_pipeline.py` does:

1. clean incoming frame (`final_clean`)
2. maintain a fixed window buffer
3. build `avail_mask` from last frame using `last_cleaned_columns` + mapping
4. apply action mask on logits before softmax/argmax
5. optional full-elixir anti-wait logit bias
6. if predicted action is wait, force position to `[-1, -1]`

### Inference mask helper

```python
masked_logits = action_logits.masked_fill(~avail_mask.unsqueeze(0), -1e9)
```

### Optional anti-leak inference bias

When:

- `Elixir >= FULL_ELIXIR_THRESHOLD`
- wait legal
- at least one legal non-wait exists

then:

```python
masked_logits[:, wait_id] -= wait_bias_at_full_elixir
```

Default configured in `action_masking_config.py`.

---

## 6) How to train and save model

From notebook:

- run cells in order
- model is saved by:

```python
torch.save(model.state_dict(), 'lstm.pth')
```

Optional script-style usage from Python:

```python
import pandas as pd
import torch
from torch.optim import Adam

from Ai.Behavior_Cloning.bc_lstm_training_pipeline import build_dataloaders, build_model, train_model
from Ai.Behavior_Cloning.action_masking_config import (
    WAIT_ID, ALWAYS_ALLOW_WAIT, AVAIL_FEATURE_TO_ACTION_ID, FULL_ELIXIR_THRESHOLD
)

full_df = pd.read_csv('Ai/Behavior_Cloning/full_cleaned_dataset_lstm.csv')
features = [c for c in full_df.columns if c not in ['match_id', 'id', 'action', 'pos_x', 'pos_y']]
targets = ['action', 'pos_x', 'pos_y']

train_loader, val_loader = build_dataloaders(
    train_csv='Ai/Behavior_Cloning/train_cleaned_dataset.csv',
    val_csv='Ai/Behavior_Cloning/test_cleaned_dataset.csv',
    features=features,
    targets=targets,
    num_actions=13,
    window_size=10,
    batch_size=64,
    wait_id=WAIT_ID,
    avail_feature_to_action_id=AVAIL_FEATURE_TO_ACTION_ID,
    always_allow_wait=ALWAYS_ALLOW_WAIT,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(len(features), 128, 2, 13, dropout=0.1, device=device)
optimizer = Adam(model.parameters(), lr=1e-3)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    optimizer=optimizer,
    num_epochs=50,
    pos_weight=0.7,
    wait_id=WAIT_ID,
    wait_weight=0.1,
    leak_weight=0.2,
    full_elixir_threshold=FULL_ELIXIR_THRESHOLD,
    log_file='Ai/Behavior_Cloning/training_metrics.jsonl',
)

torch.save(model.state_dict(), 'Ai/Behavior_Cloning/lstm.pth')
```

---

## 7) Practical tuning tips

If the model still leaks elixir:

1. increase `leak_weight` gradually (e.g. `0.2 -> 0.35`)
2. increase inference `wait_bias_at_full_elixir` slightly
3. lower threshold to `9.5` if elixir detection is noisy
4. monitor `leak_rate_at_full_elixir` alongside global action accuracy

If model becomes too aggressive (bad overplaying):

1. reduce `leak_weight`
2. reduce `wait_bias_at_full_elixir`
3. check labels in full-elixir situations for strategic wait examples

---

## 8) What is implemented now (summary)

- legal-action masking in both training and inference
- wait always legal fallback
- wait class downweighting in CE
- position loss masked by valid-position + non-wait rule
- full-elixir leak-aware training penalty
- optional full-elixir anti-wait inference bias
- notebook synced to pipeline for repeatable training and model saving

---

## 9) Known assumptions

- `Elixir` column exists in cleaned features for leak gating.
- Action IDs in your label space match mapping IDs in `action_masking_config.py`.
- Unmapped actions remain legal by default in current design.

If you later want stricter behavior, you can switch to "deny-by-default" masking where only wait and explicitly mapped available actions are legal.

