"""Table 1 — Main results table.

Reads all evaluation aggregate CSVs (baselines + PPO eval) and builds:
  - tables/table1_main_results.csv   (machine-readable)
  - tables/table1_main_results.tex   (LaTeX-formatted, paste into paper)

Row order (matches paper presentation):
  Random | Heuristic | BC-only | PPO-Scratch (s1) | PPO-Scratch (s2)
  | BC→PPO (s1) | BC→PPO (s2) | BC→PPO No-Mask (s1)

Run AFTER all evaluation scripts have been executed:
    python -m analytics.build_results_table
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from Ai.models.run_config import (
    PPO_RUN_IDS,
    BASELINE_POLICY_NAMES,
    baseline_eval_dir,
    ppo_eval_dir,
)

TABLES_DIR = PROJECT_ROOT / "tables"
TABLES_DIR.mkdir(exist_ok=True)

# Display names for the paper
DISPLAY_NAMES = {
    "random":          "Random",
    "heuristic":       "Heuristic",
    "bc_only":         "BC-only",
    "PPOScratch_s1":   "PPO-Scratch (s1)",
    "PPOScratch_s2":   "PPO-Scratch (s2)",
    "BCPPO_s1":        r"BC$\to$PPO (s1)",
    "BCPPO_s2":        r"BC$\to$PPO (s2)",
    "BCPPO_NoMask_s1": r"BC$\to$PPO No-Mask (s1)",
}

ROW_ORDER = [
    "random", "heuristic", "bc_only",
    "PPOScratch_s1", "PPOScratch_s2",
    "BCPPO_s1", "BCPPO_s2",
    "BCPPO_NoMask_s1",
]


def _load_aggregate(policy_id: str, eval_dir: Path) -> dict | None:
    path = eval_dir / f"eval_{policy_id}_aggregate.csv"
    if not path.exists():
        print(f"[WARN] Aggregate CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _fmt(val, fmt=".3f", pct=False):
    """Safe formatter that returns '-' when value is missing."""
    try:
        v = float(val)
        if pct:
            return f"{v:.1%}"
        return format(v, fmt)
    except (TypeError, ValueError):
        return "-"


def build_table():
    rows = []

    for policy_id in ROW_ORDER:
        if policy_id in BASELINE_POLICY_NAMES:
            eval_dir = baseline_eval_dir(policy_id)
        else:
            eval_dir = ppo_eval_dir(policy_id)

        agg = _load_aggregate(policy_id, eval_dir)
        if agg is None:
            print(f"[SKIP] No data for {policy_id}")
            continue

        # ---- column keys match the current AGGREGATE_COLUMNS in eval_runner.py ----
        win_rate      = agg.get("win_rate_mean", 0.0)
        ret_mean      = agg.get("outcome_return_mean", 0.0)   # fixed: was return_mean
        ret_std       = agg.get("outcome_return_std",  0.0)
        ep_len        = agg.get("episode_length_mean", 0.0)
        wait_rate     = agg.get("wait_rate_mean", 0.0)
        illegal_mean  = agg.get("illegal_action_rate_mean", 0.0)  # new
        illegal_std   = agg.get("illegal_action_rate_std",  0.0)  # new
        elixir_eff    = agg.get("mean_elixir_at_action_mean", 0.0)  # new
        overflow      = agg.get("elixir_overflow_mean", 0.0)
        n_games       = int(agg.get("n_games", 0))

        rows.append({
            "Method":           DISPLAY_NAMES.get(policy_id, policy_id),
            "Win Rate":         _fmt(win_rate, pct=True),
            "Return":           f"{_fmt(ret_mean, '+.3f')} ± {_fmt(ret_std, '.3f')}",
            "Ep. Length":       _fmt(ep_len, ".1f"),
            "Wait Rate":        _fmt(wait_rate, pct=True),
            "Illegal Rate":     f"{_fmt(illegal_mean, pct=True)} ± {_fmt(illegal_std, pct=True)}",  # new
            "Elixir @ Action":  _fmt(elixir_eff, ".2f"),   # new
            "Overflow":         _fmt(overflow, pct=True),
            "N":                n_games,
        })

    if not rows:
        print("[ERROR] No evaluation data found. Run all eval scripts first.")
        return

    df = pd.DataFrame(rows)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = TABLES_DIR / "table1_main_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved  -> {csv_path}")

    # ── LaTeX ───────────────────────────────────────────────────────────────
    # Exclude N column from LaTeX (it's metadata, not a paper column)
    tex_cols  = [c for c in df.columns if c != "N"]
    col_fmt   = "l" + "c" * (len(tex_cols) - 1)
    n_display = rows[0]["N"] if rows else 15

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Main evaluation results. Return is terminal-only ($+1$ win, $-1$ loss)."
        r" Illegal Rate = fraction of non-wait actions that violated the availability mask"
        r" or elixir constraint. Elixir @ Action = mean elixir bar value when a card was played."
        r" All models evaluated over " + str(n_display) + r" games.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
    ]

    latex_lines.append(" & ".join(tex_cols) + r" \\ \midrule")

    bc_only_idx     = next((i for i, r in enumerate(rows) if "BC-only"        in r["Method"]), None)
    scratch_s2_idx  = next((i for i, r in enumerate(rows) if "Scratch (s2)"   in r["Method"]), None)

    for i, row in enumerate(rows):
        vals = [str(row[c]) for c in tex_cols]
        line = " & ".join(vals) + r" \\"
        latex_lines.append(line)
        if bc_only_idx    is not None and i == bc_only_idx:   latex_lines.append(r"\midrule")
        if scratch_s2_idx is not None and i == scratch_s2_idx: latex_lines.append(r"\midrule")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex_str = "\n".join(latex_lines)
    tex_path  = TABLES_DIR / "table1_main_results.tex"
    tex_path.write_text(latex_str)
    print(f"[OK] LaTeX saved -> {tex_path}")
    print("\n" + latex_str)


if __name__ == "__main__":
    build_table()
