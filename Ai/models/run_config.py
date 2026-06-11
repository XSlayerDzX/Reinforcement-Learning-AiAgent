"""Central configuration for all runs — baselines, PPO training, and evaluation.
Every script imports constants from here. Nothing is hardcoded elsewhere.
"""
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
AI_ROOT          = PROJECT_ROOT / "Ai"

CHECKPOINTS_ROOT = AI_ROOT / "checkpoints"
LOGS_ROOT        = AI_ROOT / "logs"
EVALUATIONS_ROOT = AI_ROOT / "evaluations"

# ── Checkpoint paths ─────────────────────────────────────────────────────────
BC_CHECKPOINT    = CHECKPOINTS_ROOT / "bc" / "lstm.pth"

def ppo_checkpoint_dir(run_id: str) -> Path:
    """Return the checkpoint directory for a given PPO run_id."""
    return CHECKPOINTS_ROOT / "ppo" / run_id

def ppo_checkpoint_path(run_id: str, game_id: int) -> Path:
    """Return the path for a periodic checkpoint at game_id."""
    return ppo_checkpoint_dir(run_id) / f"game_{game_id:03d}.pth"

def ppo_best_checkpoint_path(run_id: str) -> Path:
    """Return the path for the best checkpoint of a run."""
    return ppo_checkpoint_dir(run_id) / "best.pth"

# ── Log paths ─────────────────────────────────────────────────────────────────
def baseline_log_dir(policy_name: str) -> Path:
    """Return the log directory for a baseline policy."""
    return LOGS_ROOT / "baselines" / policy_name

def ppo_log_dir(run_id: str) -> Path:
    """Return the log directory for a PPO run."""
    return LOGS_ROOT / "ppo" / run_id

# ── Evaluation paths ──────────────────────────────────────────────────────────
def baseline_eval_dir(policy_name: str) -> Path:
    """Return the evaluation output directory for a baseline policy."""
    return EVALUATIONS_ROOT / "baselines" / policy_name

def ppo_eval_dir(run_id: str) -> Path:
    """Return the evaluation output directory for a PPO run."""
    return EVALUATIONS_ROOT / "ppo" / run_id

# ── Experiment constants ───────────────────────────────────────────────────────
EVAL_GAMES          = 15      # games per baseline / final PPO eval
PPO_TRAINING_GAMES  = 100     # total games per PPO training run
CHECKPOINT_INTERVAL = 20      # save a periodic checkpoint every N games
WINRATE_WINDOW      = 20      # rolling window size for win-rate smoothing

# ── PPO model architecture (must match BC training) ───────────────────────────
INPUT_SIZE   = 205
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
NUM_ACTIONS  = 13
WINDOW_SIZE  = 10

# ── PPO hyperparameters ───────────────────────────────────────────────────────
LEARNING_RATE = 1e-4
GAMMA         = 0.99
EPSILON       = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.01
GRAD_CLIP     = 0.5

# ── Tower HP reward shaping ───────────────────────────────────────────────────
# Rewards are proportional to normalised HP change: delta_hp / HP_NORM
# Side towers: max HP ~1400  -> full destruction yields ~0.3 total shaping
# King tower:  max HP ~3000  -> full destruction handled by terminal reward
#              shaping only fires on progressive damage
TOWER_HP_SIDE_COEF = 0.3   # reward coefficient for side tower HP changes
TOWER_HP_KING_COEF = 0.5   # reward coefficient for king tower HP changes
HP_NORM            = 1000.0 # divisor to keep shaping in [-1, +1] range

# ── BlueStacks window ─────────────────────────────────────────────────────────
DEFAULT_WINDOW_TITLE = "BlueStacks App Player 4"

# ── Run registry — all valid run IDs ─────────────────────────────────────────
PPO_RUN_IDS = [
    "BCPPO_s1",
    "BCPPO_s2",
    "PPOScratch_s1",
    "PPOScratch_s2",
    "BCPPO_NoMask_s1",
]

BASELINE_POLICY_NAMES = ["random", "heuristic", "bc_only"]
