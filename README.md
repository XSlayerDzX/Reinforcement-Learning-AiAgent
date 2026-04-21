# Clash Royale LSTM Agent

This project builds an AI agent that plays Clash Royale from BlueStacks screenshots using:

- computer vision for state extraction
- behavior cloning with an LSTM policy
- a runtime agent loop that executes predicted actions

The project previously explored Decision Transformer, but that path has been removed from scope.

---

## Project Focus

Current focus is one clear pipeline:

1. Capture game state from frames.
2. Build and clean match datasets.
3. Train and run an LSTM behavior cloning policy.
4. Execute actions in BlueStacks through the agent loop.

High-level flow:

```text
[BlueStacks Frame]
      |
      v
[State Extraction + Features]
      |
      v
[Cleaned Dataset]
      |
      v
[LSTM Behavior Cloning]
      |
      v
[Live Agent Inference + Click/Key Actions]
```

---

## What Is Implemented

### Perception and state extraction

- `Ai/StatePredictor.py`:
  - Detects troops, towers, and elixir from frames.
- `Ai/CardPredictor.py`:
  - Detects hand cards for slots 1-4.
- `Ai/Stream_to_frame.py`:
  - Captures BlueStacks window frames.

### Data collection

- `Ai/Event_listners.py`:
  - Logs keyboard slot picks and mouse placements.
- `Ai/Create_DataSet.py`:
  - Builds one feature row per frame.
- `Ai/DataSet_Handler.py`:
  - Runs collection loop and saves per-match input/output CSV files.

### Data cleaning and rewards

- `Ai/Data_Cleaning.py`:
  - Merges logs, fills missing actions/positions, maps coordinates to grid, adds `*_avab` availability features.
- `Ai/Reward_System.py`:
  - Adds frame reward `r` and returns-to-go `rtg` for reward-aware experiments.

### LSTM training and inference

- `Ai/Behavior_Cloning/bc_lstm_training_pipeline.py`:
  - Training pipeline with action masking, wait handling, and position masking.
- `Ai/Behavior_Cloning/lstm_inference_pipeline.py`:
  - Runtime inference with sequence buffer and legal-action masking.
- `Ai/Agent/Agent_main.py`:
  - Live gameplay loop: predicts action/position and executes inputs.

---

## Roadmap (Updated)

| Phase | Label | Description |
|---|---|---|
| 0 | Perception | Detect troops, towers, elixir, and cards from BlueStacks frames. |
| 1 | Data Collection | Record state-action trajectories from gameplay. |
| 2 | Cleaning and Features | Merge/clean datasets and build train-ready features. |
| 3 | BC-LSTM | Train and validate LSTM imitation policy. |
| 4 | LSTM Agent Runtime | Run the model live with action masking and coordinate conversion. |
| 5 | PPO Warm Start (Future) | Initialize PPO from LSTM policy and fine-tune online. |
| 6 | Evaluation | Compare offline imitation and online fine-tuning performance. |

---

## Simple Setup Guide (v1: Run Agent with LSTM)

This is the fastest path to run the first working version.

### 1) Requirements

- Windows
- Python 3.8+
- BlueStacks (window title used by default: `BlueStacks App Player 1`)
- Roboflow inference endpoint (local or hosted)
- Trained model file at `Ai/Behavior_Cloning/lstm.pth`

### 2) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install dependencies

```powershell
pip install inference-sdk pynput pywin32 pygetwindow
pip install mss numpy pandas opencv-python pyautogui
pip install torch onnxruntime
```

If you plan to work on RL later:

```powershell
pip install gymnasium stable-baselines3
```

### 4) Configure Roboflow settings

Set API values in:

- `Ai/StatePredictor.py`
- `Ai/CardPredictor.py`

Expected variables:

```python
API_URL = "http://localhost:9001"
API_KEY = "your_api_key_here"
WORKSPACE = "your_workspace"
WORKFLOW = "your_workflow_id"
```

### 5) Verify local paths/window title

In `Ai/Agent/Agent_main.py`, check:

- `model_path` inside `models_dict` points to your `lstm.pth`
- BlueStacks window title matches your machine (`BlueStacks App Player 1` by default)

### 6) Run the agent

```powershell
python Ai/Agent/Agent_main.py
```

The script will:

- capture frames
- build feature rows
- run LSTM inference
- map predicted grid positions to screen coordinates
- press slot keys and click positions when action is not `wait`

### 7) Output files

During runs, logs are written under:

- `Ai/lstm_matches/`
- `Ai/lstm_matches/action_logs/`
- `Ai/Agent/agent_global_state.json`

---

## Current Limitations

- **CV inference latency (major bottleneck):** due to limited GPU resources, vision inference is slow and introduces about **4-5 seconds** delay. This means the agent can understand state changes late and cannot always react immediately to events in the match.
- **Limited behavior cloning data:** the current `LSTM` model was trained on only **70 matches**, which is small for imitation learning. Adding new matches is necessary for better generalization and stronger in-game behavior.
- **Limited card/arena coverage:** the current CV pipeline is trained for only a subset of classes and arena conditions, so the agent is not yet robust across all card types and environments.
- **Team capacity constraint:** AI-side development has mainly been handled by **2 contributors**, so expansion of data and CV coverage depends heavily on community help.

---

## Notes

- For deeper LSTM training/inference details, see `Ai/Behavior_Cloning/LSTMreadme.md`.
- If your model predicts mostly `wait`, check masking config in `Ai/Behavior_Cloning/action_masking_config.py` and training class-balance settings.
- A dedicated contribution guide (including how to add new match datasets and improve the CV models) will be added in a future README update.
