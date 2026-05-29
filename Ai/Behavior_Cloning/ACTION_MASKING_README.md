# Action Masking and Masked-Logit Handling (LSTM Behavior Cloning)

This document explains how legal-action masking is implemented in the LSTM behavior-cloning pipeline, why masked logits are used in both training and inference, and how to configure the mapping.

## What the masking enforces

The model predicts over a fixed action space (`output_size=13`), but only some actions are legal at a given frame.

Legal-action masking enforces:

- card actions are legal only when their matching `*_avab` feature is active in the **last frame** of the input window
- `wait` remains a legal fallback
- position loss is computed only when position is meaningful (`action != wait` and position is valid)

Masking is applied by replacing illegal action logits with a large negative value before softmax/cross-entropy:

```python
masked_logits = action_logits.masked_fill(~avail_mask, -1e9)
```

## Files involved

- Shared masking config (single source of truth): `Ai/Behavior_Cloning/action_masking_config.py`
- Training notebook script: `Ai/Behavior_Cloning/BC_LSTM_Training.ipynb`
- Inference pipeline: `Ai/Behavior_Cloning/lstm_inference_pipeline.py`
- Live agent entrypoint: `Ai/Agent/Agent_main.py`
- One-shot inference runner: `Ai/Behavior_Cloning/run_lstm_inference_test.py`

## Availability columns currently used

You provided these availability columns, and they are now reflected in configuration:

- `spear goblins_avab`
- `musketeer_avab`
- `mini pekka_avab`
- `knight_avab`
- `goblins_avab`
- `goblin hut_avab`
- `goblin gang_avab`
- `goblin cage_avab`
- `minions_avab`
- `giant_avab`
- `archers_avab`

## Configuration: `avail_feature_to_action_id`

This dictionary maps cleaned availability feature names to action IDs.

Current mapping (defined centrally in `action_masking_config.py`):

```python
AVAIL_FEATURE_TO_ACTION_ID = {
    "mini pekka_avab": 1,
    "knight_avab": 2,
    "goblins_avab": 4,
    "goblin hut_avab": None,   # TODO: set if added to BC action labels
    "goblin gang_avab": None,  # TODO: set if added to BC action labels
    "giant_avab": 5,
    "spear goblins_avab": 6,
    "archers_avab": 7,
    "minions_avab": 9,
    "musketeer_avab": 10,
    "goblin cage_avab": 11,
}

ALWAYS_ALLOW_WAIT = True


def get_masking_kwargs():
    return {
        "wait_id": WAIT_ID,
        "avail_feature_to_action_id": AVAIL_FEATURE_TO_ACTION_ID,
        "always_allow_wait": ALWAYS_ALLOW_WAIT,
    }
```

### Why some entries are `None`

`goblin hut` and `goblin gang` have availability features in cleaned data, but they are not currently mapped to known BC action IDs in the live action map used by agent/inference.

`None` means:

- the feature is declared in config for completeness
- no action index is constrained from this feature yet
- no crash occurs (logic explicitly skips `None` values)

## Core logic design

### 1) Dataset returns `avail_mask` per sample

In `LSTM_DataSet`, each sample now returns:

- `x`: `[T, F]`
- `action`: scalar target action ID
- `pos`: `[2]`
- `pos_mask`: bool
- `avail_mask`: `[num_actions]` bool

The mask is built from the last frame in the window.

### 2) Constrain only mapped action IDs

To avoid accidentally blocking actions with no availability feature (for example spell actions), mask construction does this:

1. Start with all actions legal (`True`)
2. Find action IDs referenced in `avail_feature_to_action_id` (except `None`)
3. Mark those IDs illegal (`False`) by default
4. Re-enable mapped IDs when the corresponding `*_avab` feature is `> 0`
5. Keep `wait` legal when configured (`always_allow_wait=True`)

This means:

- mapped card actions are availability-constrained
- unmapped actions remain legal (not constrained by availability)

### 3) Safety for training labels

When flattening training samples:

```python
avail_mask[action_id] = True
```

This prevents cross-entropy failure if feature-label mismatch appears in data.

### 4) Masked logits are used for both training and metrics

Training loss:

- apply action mask first
- compute CE on masked logits
- downweight wait class (`wait_weight`, default `0.07`)

Accuracy:

- uses `argmax(masked_action_logits)` (not raw logits)

### 5) Position loss masking

Position loss is only computed when:

- action is not wait
- `pos_x != -1`
- `pos_y != -1`

This is encoded by `pos_mask` and used in MSE selection.

### 6) Inference uses identical mask behavior

In `LSTM_Inference_Pipeline.predict()`:

1. build sequence
2. run model -> `action_logits`, `pos_pred`
3. build `avail_mask` from last cleaned frame
4. mask logits with `-1e9`
5. softmax/argmax on masked logits
6. if predicted action is wait, force `pos_pred=[-1,-1]`

## Example: masked logits

Assume 5 actions and mask:

```python
action_logits = tensor([[2.3, 1.0, 5.2, -0.4, 0.7]])
avail_mask    = tensor([ True, False, True, False, True])
```

After masking:

```python
masked_logits = tensor([[ 2.3, -1e9, 5.2, -1e9, 0.7]])
```

Softmax cannot pick actions `1` or `3`.

## Example: constrained-vs-unconstrained behavior

If mapping includes only action IDs `{1,2,4}` and wait is `0`:

- Actions `{1,2,4}` are constrained by their `*_avab` feature
- Actions `{3,5,6,...}` stay legal by default unless you explicitly add them to mapping

This is intentional to avoid blocking classes that do not have availability features.

## Example: training call with wait weighting

```python
from Ai.Behavior_Cloning.action_masking_config import (
    WAIT_ID,
    ALWAYS_ALLOW_WAIT,
    AVAIL_FEATURE_TO_ACTION_ID,
)

history = train_model(
    model=model,
    train_loader=train_loaded,
    val_loader=test_loaded,
    device=device,
    optimizer=optimizer,
    num_epochs=30,
    pos_weight=0.7,
    wait_id=0,
    wait_weight=0.07,
    log_file="training_metrics.jsonl",
)
```

Dataset construction should use the same shared config:

```python
train_transformed = LSTM_DataSet(
    train_df,
    window_size=10,
    features=features,
    targets=targets,
    num_actions=13,
    wait_id=WAIT_ID,
    avail_feature_to_action_id=AVAIL_FEATURE_TO_ACTION_ID,
    always_allow_wait=ALWAYS_ALLOW_WAIT,
)
```

Inference callers can pass shared kwargs directly:

```python
from Ai.Behavior_Cloning.action_masking_config import get_masking_kwargs

pipeline = LSTM_Inference_Pipeline(
    model_path=model_path,
    device=device,
    window_size=10,
    input_size=205,
    output_size=13,
    hidden_size=128,
    num_layers=2,
    **get_masking_kwargs(),
)
```

## What to edit when action space changes

If you add new card actions/classes:

1. Add action to your action-id mapping (agent + evaluation)
2. Ensure cleaning creates a matching `*_avab` column
3. Add `"<card>_avab": <action_id>` to `AVAIL_FEATURE_TO_ACTION_ID`
4. Retrain so masked accuracy and logits align with new constraints

## Quick sanity checks

For an automated check, run:

```powershell
python Ai/Behavior_Cloning/validate_action_masking_config.py
```

The validator checks:

- every feature key in `AVAIL_FEATURE_TO_ACTION_ID` exists in both train/test cleaned CSVs
- every non-`None` mapped action id is inside `[0, output_size-1]`

It prints a JSON report and exits with code `0` on pass, `1` on config/data mismatch.

You can also pass custom paths and action-space size:

```powershell
python Ai/Behavior_Cloning/validate_action_masking_config.py --train-csv Ai/Behavior_Cloning/train_cleaned_dataset.csv --test-csv Ai/Behavior_Cloning/test_cleaned_dataset.csv --output-size 13
```

Use these checks after mapping edits:

```python
# 1) Every mapped feature should exist in cleaned dataframe columns
missing = [k for k in AVAIL_FEATURE_TO_ACTION_ID if k not in cleaned_df.columns]

# 2) Observe target labels not represented in mapping+wait
mapped_ids = {v for v in AVAIL_FEATURE_TO_ACTION_ID.values() if v is not None}
allowed_ids = mapped_ids | {WAIT_ID}
uncovered = sorted(set(cleaned_df["action"].unique()) - allowed_ids)
```

If `uncovered` is non-empty, those labels are not currently constrained by availability features.

## Why this is robust

- No illegal mapped card can win argmax after masking
- Training and inference use the same masking strategy
- Wait remains safe fallback
- Position regression is trained only where position is meaningful
- Partial mappings are supported without breaking unknown classes




