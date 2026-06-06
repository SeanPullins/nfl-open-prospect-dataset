#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

PRED_FILE = Path("predictions/model_v7_1_recalibrated_summary.csv")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/look_up_player_v7_1.py 'Player Name'")

query = " ".join(sys.argv[1:]).lower()
df = pd.read_csv(PRED_FILE, low_memory=False)

name_cols = ["player_name_clean", "player_name", "pfr_player_name", "name", "display_name"]
name_col = next((c for c in name_cols if c in df.columns), None)

matches = df[df[name_col].astype(str).str.lower().str.contains(query, na=False)].copy()

if matches.empty:
    print(f"No matches for: {query}")
    raise SystemExit(0)

cols = [
    name_col,
    "season",
    "position",
    "v7_position_group",
    "v7_model_group",
    "college",
    "round",
    "pick",
    "v7_projected_value",
    "v7_actual_value",
    "v7_actual_position_percentile",
    "v7_predicted_position_percentile",
    "v7_position_score",
    "v7_1_final_score",
    "v7_1_ceiling_score",
    "v7_1_floor_score",
    "v7_starter_probability",
    "v7_elite_probability",
    "v7_bust_probability",
    "v7_1_tier",
    "v7_1_outlier_type",
    "v7_1_explanation",
]

cols = [c for c in cols if c in matches.columns]

print(matches[cols].sort_values("v7_1_final_score", ascending=False).to_string(index=False))
