"""
arena_client.py — ArenaBrain Web Integration Client
=====================================================
Drop-in HTTP client that mirrors what PPO_Logger.py writes to disk,
but ALSO pushes the same data to the ArenaBrain web dashboard.

Usage in PPO_GUI.py:
    from arena_web_integration.arena_client import ArenaBrainClient
    client = ArenaBrainClient(api_key="your_key_here", base_url="http://localhost:5000")
    client.validate()               # call at startup
    client.push_run(entry)          # call after log_update()
    client.push_rollouts(update_id, [entry, ...])  # call after log_rollout()
    client.push_winrate(outcome)    # call after log_winrate()
"""

import json
import threading
from datetime import datetime
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class ArenaBrainClient:
    """
    Thin wrapper around the ArenaBrain REST API.
    All network calls are fire-and-forget on a background thread
    so they never block the training loop.
    """

    def __init__(self, api_key: str = "", base_url: str = "http://localhost:5000"):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._enabled = False  # set to True after successful validate()
        self._session = None
        if _REQUESTS_AVAILABLE and self.api_key:
            self._session = requests.Session()
            self._session.headers.update({
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            })

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """
        Call once at training start. Pings /api/keys/validate.
        Returns True if the key is valid and active.
        Prints a clear message to the GUI log either way.
        """
        if not self._ready():
            return False

        try:
            resp = self._session.post(
                f"{self.base_url}/api/keys/validate",
                json={"apiKey": self.api_key},
                timeout=8,
            )
            if resp.status_code == 200 and resp.json().get("valid"):
                self._enabled = True
                print("[ArenaBrain] ✅ API key validated — telemetry will be pushed to dashboard.")
                return True
            else:
                print(f"[ArenaBrain] ❌ API key invalid or inactive (status {resp.status_code}). Running in offline mode.")
                return False
        except Exception as e:
            print(f"[ArenaBrain] ⚠️  Could not reach server ({e}). Running in offline mode.")
            return False

    def push_run(self, log_entry: dict):
        """
        Push a PPO update entry (as returned by PPO_Logger.log_update) to /api/logs/run.
        The log_entry dict keys match what PPO_Logger produces.
        """
        if not self._enabled:
            return
        payload = {
            "updateId":     log_entry.get("update_id", 0),
            "timestamp":    log_entry.get("timestamp", datetime.now().isoformat()),
            "numRollouts":  log_entry.get("num_rollouts", 0),
            "totalSteps":   log_entry.get("total_steps", 0),
            "outcome":      log_entry.get("outcome", "unknown"),
            "policyLoss":   log_entry.get("policy_loss", 0.0),
            "valueLoss":    log_entry.get("value_loss", 0.0),
            "meanReward":   log_entry.get("mean_reward", 0.0),
            "totalReward":  log_entry.get("total_reward", 0.0),
            "meanReturn":   log_entry.get("mean_return", 0.0),
            "meanValueEst": log_entry.get("mean_value_est", 0.0),
            "explainedVar": log_entry.get("explained_var", 0.0),
            "meanEpLength": log_entry.get("mean_ep_length", 0.0),
            "actionDist":   log_entry.get("action_dist", {}),
        }
        self._fire(f"{self.base_url}/api/logs/run", payload)

    def push_rollouts(self, update_id: int, rollout_entries: list):
        """
        Push a list of rollout log entries (as returned by PPO_Logger.log_rollout).
        """
        if not self._enabled:
            return
        rollouts_payload = [
            {
                "rolloutIdx":  entry.get("rollout_idx", i),
                "timestamp":   entry.get("timestamp", datetime.now().isoformat()),
                "steps":       entry.get("steps", 0),
                "totalReward": entry.get("total_reward", 0.0),
                "meanReward":  entry.get("mean_reward", 0.0),
                "meanReturn":  entry.get("mean_return", 0.0),
                "meanValue":   entry.get("mean_value", 0.0),
                "forcedWaits": entry.get("forced_waits", 0),
                "actionDist":  entry.get("action_dist", {}),
            }
            for i, entry in enumerate(rollout_entries)
        ]
        payload = {"updateId": update_id, "rollouts": rollouts_payload}
        self._fire(f"{self.base_url}/api/logs/rollouts", payload)

    def push_winrate(self, outcome: str):
        """
        Push a win/loss/draw outcome to /api/logs/winrate.
        """
        if not self._enabled:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
        }
        self._fire(f"{self.base_url}/api/logs/winrate", payload)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ready(self) -> bool:
        if not _REQUESTS_AVAILABLE:
            print("[ArenaBrain] ⚠️  'requests' not installed. Run: pip install requests")
            return False
        if not self.api_key:
            print("[ArenaBrain] ⚠️  No API key set. Skipping web push.")
            return False
        return True

    def _fire(self, url: str, payload: dict):
        """Non-blocking POST on a daemon thread."""
        def _send():
            try:
                resp = self._session.post(url, json=payload, timeout=10)
                if resp.status_code not in (200, 201):
                    print(f"[ArenaBrain] ⚠️  Push failed ({url.split('/')[-1]}): {resp.status_code} {resp.text[:100]}")
            except Exception as e:
                print(f"[ArenaBrain] ⚠️  Network error pushing to {url.split('/')[-1]}: {e}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()
