# 🤖 Clash Royale AI Agent

An AI agent that uses **Computer Vision**, **Behavior Cloning (Imitation Learning)**, and **Reinforcement Learning** to play Clash Royale autonomously.

> **Project Status:** 🚧 Under Development - Second Year Computer Science Project at the Higher School of Computer Science

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Why Two Phases?](#-why-two-phases)
- [What's Implemented ✅](#-whats-implemented-)
- [What's Left 📝](#-whats-left-)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Setup & Requirements](#-setup--requirements)

---

## 🏗️ Architecture Overview

The project follows a **two-phase learning approach** where Reinforcement Learning benefits from the pre-trained Imitation Learning model:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: IMITATION LEARNING (Warm Start)                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                    │
│  │  Human Plays │───▶│  Record State│───▶│  Supervised  │                    │
│  │  (BlueStacks)│    │   + Actions  │    │    Training  │                    │
│  └──────────────┘    └──────────────┘    └──────────────┘                    │
│                                                      │                       │
│                                                      ▼                       │
│                                            ┌─────────────────┐              │
│                                            │  Pretrained     │              │
│                                            │  Policy Network │──────────────┼──────┐
│                                            │  (Knowledge)    │              │      │
│                                            └─────────────────┘              │      │
└─────────────────────────────────────────────────────────────────────────────┘      │
                                                                                     │
                                          │                                          │
                                          │  WEIGHTS INITIALIZATION                  │
                                          │  (Not random weights!)                   │
                                          ▼                                          │
┌─────────────────────────────────────────────────────────────────────────────┐      │
│                  PHASE 2: REINFORCEMENT LEARNING (Fine-tuning)               │      │
│                                                                              │      │
│   ┌─────────────────┐      ┌──────────────┐      ┌──────────────┐           │      │
│   │  Pretrained     │─────▶│   AI Agent   │─────▶│   Observe    │           │      │
│   │  Weights from   │      │    Plays     │      │   Rewards    │           │      │
│   │  Imitation      │      └──────────────┘      └──────┬───────┘           │      │
│   └─────────────────┘              ▲                   │                    │      │
│                                    │                   │                    │      │
│                                    └───────────────────┘                    │      │
│                                            (Policy Update Loop)              │      │
│                                                                              │      │
│   Benefits:                                                                  │      │
│   ✓ Starts with meaningful policy (not random)                               │      │
│   ✓ Faster convergence                                                       │      │
│   ✓ Better exploration in early episodes                                     │      │
│   ✓ Avoids catastrophic early failures                                       │      │
└─────────────────────────────────────────────────────────────────────────────┘◄─────┘
```

---

## 🎯 Why Two Phases?

### The Problem with Pure RL
Training an RL agent from **random weights** in a complex game like Clash Royale is extremely difficult:
- ❌ Agent makes completely random moves initially
- ❌ Takes forever to learn basic game mechanics
- ❌ May never discover good strategies through random exploration
- ❌ Catastrophic performance in early training phases

### The Solution: Imitation Learning → RL

| Phase | Purpose | Benefit for Next Phase |
|-------|---------|----------------------|
| **Phase 1: Imitation Learning** | Learn from human demonstrations | Provides a **pretrained policy** with basic game knowledge |
| **Phase 2: RL Fine-tuning** | Improve beyond human performance | Starts from a **good baseline** instead of random weights |

### Key Advantage
> 🧠 The RL agent **inherits the pretrained weights** from the imitation learning model, giving it a **"warm start"**. Instead of starting with random actions, it starts with human-level gameplay and learns to optimize from there.

---

## ✅ What's Implemented

### 1️⃣ Computer Vision Pipeline
| Component | Status | Description |
|-----------|--------|-------------|
| **StatePrediction** | ✅ Done | Detects troops, towers, and elixir level from game screenshots using Roboflow Inference API |
| **CardPrediction** | ✅ Done | Identifies cards in the player's hand (4 slots) using Roboflow workflows |
| **Object Detection** | ✅ Done | YOLO-based ONNX inference pipeline for local object detection |
| **Data Mapping** | ✅ Done | Dictionaries for elixir costs, troop sides (ally/enemy), and tower classification |

**Key Features:**
- Real-time detection of 13+ card types (Archers, Giant, Minions, Goblin Cage, etc.)
- Elixir level recognition (0-10)
- Troop position tracking (x, y coordinates)
- Ally/Enemy classification for all entities

### 2️⃣ Data Collection (Behavior Cloning Setup)
| Component | Status | Description |
|-----------|--------|-------------|
| **Mouse Listener** | ✅ Done | Captures mouse clicks and converts screen coords to BlueStacks coordinates |
| **Keyboard Listener** | ✅ Done | Detects F1-F4 key presses for card selection |
| **DPI Awareness** | ✅ Done | Handles high-DPI display scaling for accurate coordinate mapping |
| **Coordinate Conversion** | ✅ Done | Converts global screen coordinates to BlueStacks virtual resolution (540x960) |

### 3️⃣ Data Processing Utilities
| Component | Status | Description |
|-----------|--------|-------------|
| **Video to Frames** | ✅ Done | Converts gameplay videos to frame-by-frame images for dataset creation |
| **Frame Extraction** | ✅ Done | Configurable step size for frame sampling |

### 4️⃣ Model Infrastructure
| Component | Status | Description |
|-----------|--------|-------------|
| **ONNX Runtime** | ✅ Done | ObjectFinder class for running ONNX models with NMS |
| **Roboflow API** | ✅ Done | HTTP client for workflow-based inference |

### ⚠️ Experimental (Not Part of Main Project)
| Component | Status | Description |
|-----------|--------|-------------|
| **AdbControle.py** | 🧪 Experimental | ADB integration was explored but **not part of the main pipeline** |
| **test.py** | 🧪 Experimental | ADB debugging script |

---

## 📝 What's Left

### 🔴 Phase 1: Behavior Cloning (Imitation Learning)
| Task | Priority | Description |
|------|----------|-------------|
| **Dataset Builder** | 🔴 High | Combine recorded (State, Action) pairs into training dataset |
| **Action Encoder** | 🔴 High | Define action space: which card → where to play (x, y) |
| **Neural Network Model** | 🔴 High | Build CNN + MLP model that takes game state → predicts action |
| **Training Script** | 🔴 High | Supervised learning loop for behavior cloning |
| **Model Evaluation** | 🟡 Medium | Validation metrics for imitation learning performance |
| **Save Pretrained Weights** | 🔴 High | Export model weights for RL phase initialization |

### 🔴 Phase 2: Reinforcement Learning (Using Pretrained Weights!)
| Task | Priority | Description |
|------|----------|-------------|
| **Environment Wrapper** | 🔴 High | OpenAI Gym/Gymnasium interface for Clash Royale |
| **Reward Function** | 🔴 High | Design reward signals (tower damage, elixir advantage, win/loss, etc.) |
| **Load Pretrained Weights** | 🔴 High | Initialize RL policy network with imitation learning weights |
| **RL Algorithm** | 🔴 High | Implement PPO, DQN, or similar RL algorithm |
| **Action Execution** | 🔴 High | Send commands to game (mouse/keyboard automation) |
| **State Representation** | 🟡 Medium | Vectorized state space for RL agent |
| **Self-Play Loop** | 🟡 Medium | Automated gameplay with learning from interactions |

### 🟡 Infrastructure & Polish
| Task | Priority | Description |
|------|----------|-------------|
| **Real-time Screen Capture** | 🟡 Medium | Continuous screenshot pipeline (e.g., using mss or dxcam) |
| **Configuration System** | 🟢 Low | YAML/JSON config for paths, API keys, hyperparameters |
| **Logging & Monitoring** | 🟢 Low | TensorBoard/WandB integration for training visualization |
| **Requirements.txt** | 🟢 Low | Document all Python dependencies |
| **Documentation** | 🟢 Low | Code comments, docstrings, usage examples |

---

## 📁 Project Structure

```
Reinforcement-Learning-AiAgent/
│
├── 📄 README.md                 # This file
│
├── 🤖 Core Modules
│   ├── StatePredictor.py        # Game state detection (troops, towers, elixir)
│   ├── CardPredictor.py         # Hand card detection from screenshots
│   ├── ClashRoyalData.py        # Game data: elixir costs, troop sides
│   └── Train.py                 # ONNX object detection utilities
│
├── 🎮 Data Collection
│   └── Event_listners.py        # Mouse/keyboard listeners for recording (State, Action) pairs
│
├── 🛠️ Utilities
│   ├── Clip_To_Frames.py        # Video to frames converter
│   ├── AdbControle.py           # ⚠️ EXPERIMENTAL: ADB (not part of main project)
│   └── test.py                  # ⚠️ EXPERIMENTAL: ADB testing
│
└── 🧠 Future Modules (To Be Created)
    ├── dataset_builder.py       # (TODO) Build behavior cloning dataset
    ├── behavior_cloning.py      # (TODO) Imitation learning training
    ├── rl_environment.py        # (TODO) Gym environment wrapper
    ├── rl_agent.py              # (TODO) RL agent with pretrained weight loading
    └── action_executor.py       # (TODO) Execute actions (mouse automation)
```

---

## 🔧 How It Works

### Step 1: State Extraction
```python
# StatePredictor.py extracts:
Slots  = {"slot_1": "archers", "slot_2": "giant", ...}  # Cards in hand
Troops = {"knight": ((x, y), "ally", 3), ...}           # Troop positions + side
Towers = {"king_tower": "ally", ...}                    # Tower ownership
Elixir = 7                                               # Current elixir (0-10)
```

### Step 2: Action Recording (Behavior Cloning)
```python
# Event_listners.py captures:
Keyboard: F1-F4 selects card slot
Mouse:    Click position (x, y) → converted to BlueStacks coords
Result:   (State, Action) pair stored for training
```

### Step 3: Imitation Learning Training
```python
# behavior_cloning.py (TODO):
Input:   Game State (screenshot features)
Output:  Action (which_card, x, y)
Loss:    MSE or Cross-Entropy vs human actions
Result:  Pretrained policy network with saved weights
```

### Step 4: RL Fine-Tuning (The Key!)
```python
# rl_agent.py (TODO):
# 1. Load pretrained weights from imitation learning
policy_network.load_weights("imitation_model.pth")

# 2. Fine-tune with RL
while training:
    state = env.observe()
    action = policy_network.act(state)
    reward = env.execute(action)
    policy_network.update(state, action, reward)  # PPO/DQN/etc
    
# Result: Agent that plays better than humans!
```

---

## ⚙️ Setup & Requirements

### Prerequisites
- Python 3.8+
- BlueStacks emulator (Android)
- Roboflow account + API key

### Dependencies (inferred from code)
```bash
pip install inference-sdk          # Roboflow inference
pip install pynput                 # Input listeners
pip install pywin32 pygetwindow    # Windows GUI
pip install opencv-python          # CV utilities
pip install onnxruntime torch      # Model inference
pip install matplotlib numpy       # Utilities
pip install mss                    # (Suggested) Screen capture
pip install gymnasium              # (Suggested) RL environment
pip install stable-baselines3      # (Suggested) RL algorithms
```

### Configuration
Update these paths in the respective files:
```python
# CardPredictor.py / StatePredictor.py
API_URL = "http://localhost:9001"
API_KEY = "your_api_key_here"
```

---

## 🎯 Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Computer Vision pipeline for state extraction |
| 2 | ✅ | Data collection setup (mouse/keyboard listeners) |
| 3 | 🚧 | **Behavior Cloning** - Train model on human demonstrations |
| 4 | ⏳ | **Export pretrained weights** for RL warm start |
| 5 | ⏳ | **RL Fine-tuning** - Initialize with imitation weights and improve |
| 6 | ⏳ | Evaluation and optimization |

---

## 📌 Notes

- 🧠 **Key Concept**: The RL agent starts with **pretrained weights** from imitation learning, not random weights!
- The project uses **Roboflow Workflows** for inference - requires a local Roboflow server or API access
- `Train.py` contains ONNX utilities that may need refactoring for Clash Royale
- Coordinate conversion is calibrated for BlueStacks resolution **540x960**
- ⚠️ `AdbControle.py` and `test.py` are **experimental** and not part of the main project pipeline

---

*Made with ❤️ for a Second Year Computer Science Project*
