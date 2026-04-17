"""Shared configuration for legal-action masking across training and inference."""

WAIT_ID = 0
ALWAYS_ALLOW_WAIT = True
FULL_ELIXIR_THRESHOLD = 10.0
WAIT_BIAS_AT_FULL_ELIXIR = 0.4

# Availability feature -> action_id
# Use None for features that currently have no action id in the BC action space.
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


def get_masking_kwargs():
    """Return keyword args used by LSTM_Inference_Pipeline."""
    return {
        "wait_id": WAIT_ID,
        "avail_feature_to_action_id": AVAIL_FEATURE_TO_ACTION_ID,
        "always_allow_wait": ALWAYS_ALLOW_WAIT,
        "full_elixir_threshold": FULL_ELIXIR_THRESHOLD,
        "wait_bias_at_full_elixir": WAIT_BIAS_AT_FULL_ELIXIR,
    }

