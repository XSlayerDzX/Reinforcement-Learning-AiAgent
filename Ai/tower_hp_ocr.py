"""Tower HP OCR — wraps the Roboflow inference workflow.

Used post-game to extract tower HP values from captured screenshots.
Never called during live gameplay to avoid latency.

Workflow output format (list of one dict):
    [{
        'ally_right':  '1226', 'ally_left':  '0', 'ally_king':  '745',
        'enemy_right': '0',    'enemy_left': '1512', 'enemy_king': '1780',
        'qwen_error_1': '', ..., 'qwen_error_6': ''
    }]

Tower semantics:
  - Side tower HP  = 0  -> tower is destroyed
  - King tower HP  = 0  -> king took no damage yet (do NOT treat as destroyed)
  - Any field = '' or OCR error -> return None for that tower (skip silently)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

_CLIENT = None
_CLIENT_ERROR: Optional[str] = None

WORKSPACE   = "clashroyalbot-z9idj"
WORKFLOW_ID = "custom-workflow-3"
SERVER_URL  = "http://localhost:9001"

# Hard timeout (seconds) for every OCR HTTP call.
# If the Roboflow server is down or slow, the call will be aborted after
# this many seconds instead of hanging the training loop indefinitely.
_OCR_TIMEOUT = 10


def _get_client():
    """Lazily create the InferenceHTTPClient, caching it after first call."""
    global _CLIENT, _CLIENT_ERROR
    if _CLIENT is not None:
        return _CLIENT
    if _CLIENT_ERROR is not None:
        return None
    try:
        from inference_sdk import InferenceHTTPClient  # type: ignore
        import os
        api_key = os.environ.get("ROBOFLOW_API_KEY", "obQog4mAaBRuPZZBIoti")
        _CLIENT = InferenceHTTPClient(api_url=SERVER_URL, api_key=api_key)
        # Inject a default timeout so every subsequent run_workflow call
        # respects it — the SDK passes kwargs straight through to requests.
        _CLIENT.client_timeout = _OCR_TIMEOUT
    except Exception as e:
        _CLIENT_ERROR = str(e)
        print(f"[tower_hp_ocr] Could not create inference client: {e}")
    return _CLIENT


@dataclass
class TowerHP:
    ally_left:    Optional[int]
    ally_right:   Optional[int]
    ally_king:    Optional[int]
    enemy_left:   Optional[int]
    enemy_right:  Optional[int]
    enemy_king:   Optional[int]


def _parse_hp(raw: str, is_king: bool = False) -> Optional[int]:
    """Convert raw OCR string to int HP, or None on failure.

    - Empty string or whitespace -> None (OCR failed, skip frame)
    - '0' on a side tower        -> 0   (tower destroyed)
    - '0' on king tower          -> None (no damage yet, ignore)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    if v == 0 and is_king:
        return None
    return v


def run_ocr(image_path: str) -> Optional[TowerHP]:
    """Run the Roboflow OCR workflow on a single screenshot.

    Returns a TowerHP dataclass on success, or None if the call fails
    entirely (server down, timeout, network error, empty result, etc.).
    Individual tower fields may be None if that tower could not be read.
    """
    import socket
    import urllib.request

    # Fast pre-check: is the server port open at all?
    # Avoids waiting for the full SDK timeout on every frame when the
    # server is simply not running.
    try:
        with socket.create_connection(("localhost", 9001), timeout=2):
            pass
    except OSError:
        # Port not reachable — skip silently
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": image_path},
            use_cache=True,
        )
    except Exception as e:
        print(f"[tower_hp_ocr] OCR call failed for {image_path}: {e}")
        return None

    # Unwrap list wrapper
    if isinstance(result, list):
        if not result:
            return None
        result = result[0]

    if not isinstance(result, dict):
        return None

    return TowerHP(
        ally_left   = _parse_hp(result.get("ally_left",   ""), is_king=False),
        ally_right  = _parse_hp(result.get("ally_right",  ""), is_king=False),
        ally_king   = _parse_hp(result.get("ally_king",   ""), is_king=True),
        enemy_left  = _parse_hp(result.get("enemy_left",  ""), is_king=False),
        enemy_right = _parse_hp(result.get("enemy_right", ""), is_king=False),
        enemy_king  = _parse_hp(result.get("enemy_king",  ""), is_king=True),
    )
