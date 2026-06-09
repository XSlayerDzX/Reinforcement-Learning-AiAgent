"""Table 1 — Main results table.

Reads all evaluation aggregate CSVs (baselines + PPO) and builds:
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


def build_table():
    rows = []

    for policy_id in ROW_ORDER:
        # Determine eval directory
        if policy_id in BASELINE_POLICY_NAMES:
            eval_dir = baseline_eval_dir(policy_id)
        else:
            eval_dir = ppo_eval_dir(policy_id)

        agg = _load_aggregate(policy_id, eval_dir)
        if agg is None:
            print(f"[SKIP] No data for {policy_id}")
            continue

        rows.append({
            "Method":          DISPLAY_NAMES.get(policy_id, policy_id),
            "Win Rate":        f"{float(agg.get('win_rate_mean', 0)):.1%}",
            "Mean Return":     f"{float(agg.get('return_mean', 0)):+.3f}",
            "Ep. Length":      f"{float(agg.get('episode_length_mean', 0)):.1f}",
            "Wait Rate":       f"{float(agg.get('wait_rate_mean', 0)):.1%}",
            "Elixir Overflow": f"{float(agg.get('elixir_overflow_mean', 0)):.1%}",
            "N Games":         int(agg.get('n_games', 0)),
        })

    if not rows:
        print("[ERROR] No evaluation data found. Run all eval scripts first.")
        return

    df = pd.DataFrame(rows)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = TABLES_DIR / "table1_main_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved  -> {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    col_fmt = "l" + "c" * (len(df.columns) - 1)
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results across all policies evaluated over "
        + str(rows[0].get('N Games', 15) if rows else 15)
        + r" games.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
    ]

    # Header
    header_cols = [c for c in df.columns if c != "N Games"]
    latex_lines.append(" & ".join(header_cols) + r" \\ \midrule")

    # Separator index: after bc_only row (3rd row)
    bc_only_idx = next(
        (i for i, r in enumerate(rows) if "BC-only" in r["Method"]), None
    )

    for i, row in enumerate(rows):
        vals = [str(row[c]) for c in header_cols]
        line = " & ".join(vals) + r" \\"
        latex_lines.append(line)
        # Add \midrule after baselines block and after scratch block
        if bc_only_idx is not None and i == bc_only_idx:
            latex_lines.append(r"\midrule")
        scratch_s2_idx = next(
            (j for j, r in enumerate(rows) if "Scratch (s2)" in r["Method"]), None
        )
        if scratch_s2_idx is not None and i == scratch_s2_idx:
            latex_lines.append(r"\midrule")

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
