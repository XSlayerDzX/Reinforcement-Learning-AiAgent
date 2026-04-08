# 🤖 Clash Royale RL/IL Agent

An AI agent that uses Computer Vision, Behavior Cloning, Offline RL, and Online RL to play Clash Royale from BlueStacks screenshots.

---

## 🎯 Project Idea

Use one unified pipeline and dataset to compare three policy-learning approaches:

1. Behavior Cloning with LSTM – imitate a human player from recorded trajectories.
2. Decision Transformer (Offline RL) – learn a return-conditioned policy from the same offline data.
3. PPO (Online RL) – start from the best offline policy and fine-tune directly in the environment.

High-level diagram:

[Screen + Human Input]
        |
        v
[State & Action Dataset] ----> [Reward & RTG]
        |
        |----> BC-LSTM (Behavior Cloning)
        |----> Decision Transformer (Offline RL)
        v
[Best Offline Policy Weights]
        |
        v
        PPO (Online RL Fine-Tuning)

---

## 🏗️ What the Codebase Already Does

### Perception / State Extraction

- StatePredictor.py – Roboflow workflow for:
  - Troops (class, side, position, elixir cost)
  - Towers (which side)
  - Elixir amount
- CardPredictor.py – Roboflow workflow to detect cards in hand for slots 1–4.

### Data Collection

- Stream_to_frame.py – captures BlueStacks frames with mss and window geometry.
- Event_listners.py – keyboard/mouse listeners:
  - Keys 1–4 = choose card slot
  - Mouse click = placement; validated against elixir and current card
- State_Tracker.py – holds current image, id, card, elixir, towers, troops, slots, and intermediate dictionaries.
- Create_DataSet.py – builds a feature row per frame:
  - Slots, elixir, tower flags
  - Ally/enemy card presence and (x, y)
  - Dense ally-enemy distance features
- DataSet_Handler.py – main loop:
  - Captures frames, extracts features
  - Logs actions + positions on validated clicks
  - Saves per-match CSVs:
    - match_input_{id}.csv
    - match_output_{id}.csv
    - match_output_action_validation_{id}.csv

### Data Cleaning & Reward

- Data_Cleaning.py:
  - Merges input/output/validation on id
  - Fills missing actions as "wait", missing positions as -1
  - Fixes slot histories (back-filling last valid card)
  - Converts all position columns to a 9×18 grid, keeping -1 as “no unit/no position”
  - Adds per-card *_avab flags based on slots and elixir (archers_avab, giant_avab, etc.)
  - Writes match_{id}_final_cleaned_dataset.csv per match
- Reward_System.py:
  - Adds a reward column r:
    - +0.15 when enemy troops disappear
    - -0.01 when enemy troops stay alive across frames
    - -0.4 when ally towers die
    - +0.75 when enemy towers die
  - Computes returns-to-go rtg per frame (gamma=1) and saves {...}_rewarded.csv

The rewarded, cleaned match CSVs are the base for both BC-LSTM and Decision Transformer.

---

## 🔬 Planned Experiments

### 1️⃣ Behavior Cloning with LSTM

- Inputs per timestep:
  - Game state features from the cleaned CSVs (elixir, tower flags, per-card ally/enemy presence and grid positions, distance features, *_avab flags, etc.)
- Outputs:
  - Action head: multi-class over {wait} ∪ {all deck cards}
  - Position head: either grid-based (gx, gy) or continuous (x, y) for card placement
- Handling:
  - Class imbalance: weighted cross-entropy so "wait" doesn’t dominate
  - Position loss masked when action is "wait" (pos = -1)
  - At inference, action logits are masked to only allow cards with *_avab == 1 and sufficient elixir

### 2️⃣ Decision Transformer (Offline RL)

- Uses sequences of (state, action, reward, rtg) from the rewarded CSVs.
- Learns a return-conditioned policy:
  - Input: (rtg_t, state_t, action_{t-1}, ...)
  - Output: next action
- Objective: reach higher win-rate / returns than pure BC on the same offline data.

### 3️⃣ PPO with Warm Start

- Custom environment built around the same state representation.
- PPO policy network initialized from:
  - The BC-LSTM policy or
  - The Decision Transformer policy head (whichever performs best offline)
- Trains by interacting with the game using a similar reward signal to Reward_System.py.

---

## 📦 Project Structure (Simplified)

Reinforcement-Learning-AiAgent/
|
|-- Core CV & State
|   |-- StatePredictor.py
|   |-- CardPredictor.py
|   |-- ClashRoyalData.py
|   |-- State_Tracker.py
|
|-- Data Collection
|   |-- Stream_to_frame.py
|   |-- Event_listners.py
|   |-- Create_DataSet.py
|   |-- DataSet_Handler.py
|
|-- Data Cleaning & Rewards
|   |-- Data_Cleaning.py
|   |-- Reward_System.py
|
|-- BC-LSTM (notebooks / WIP)
|   |-- BC_LSTM_Data.ipynb
|   |-- BC_LSTM_Model.ipynb
|   |-- BC_LSTM_Training.ipynb
|   |-- BC_LSTM_Inference.ipynb
|
|-- Future RL Components (planned)
    |-- DecisionTransformer_*.py
    |-- PPO_Agent_*.py
    |-- Env_Wrapper.py
    |-- Evaluation_*.py

---

## 🗺️ Roadmap

| Phase | Label | Description |
|-------|--------|-------------|
| 0 | 🧱 Perception & State Extraction | Roboflow-based detectors for troops, towers, elixir, and cards in hand; mapping to structured features. |
| 1 | 🎮 Data Collection | Capture frames, listen to keyboard/mouse, log (state, action) per frame into per-match CSVs. |
| 2 | 🧹 Data Cleaning & Engineering | Merge input/output/validation, grid positions, fill missing values, add card availability flags. |
| 3 | 💰 Rewards & RTG | Design reward function and compute returns-to-go for each frame (offline RL ready). |
| 4 | 🧠 BC-LSTM | Train a multi-output LSTM to imitate human actions (card + placement). |
| 5 | 🧾 Decision Transformer | Train an offline RL Decision Transformer on the same dataset with RTG conditioning. |
| 6 | 🌀 PPO Warm Start | Wrap the game as an environment and fine-tune PPO using weights from the best offline policy. |
| 7 | 📊 Evaluation | Compare BC-LSTM, Decision Transformer, and PPO (win rate, rewards, sample efficiency). |

---

## ⚙️ Setup & Requirements (Short)

- Python 3.8+
- Windows + BlueStacks 4
- Roboflow account & local/remote inference server

Main Python packages:

pip install inference-sdk pynput pywin32 pygetwindow
pip install mss numpy pandas
pip install torch onnxruntime
pip install gymnasium stable-baselines3

Set your Roboflow config in StatePredictor.py and CardPredictor.py:

API_URL = "http://localhost:9001"
API_KEY = "your_api_key_here"
WORKSPACE = "your_workspace"
WORKFLOW = "your_workflow_id"
