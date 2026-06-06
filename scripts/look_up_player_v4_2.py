#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path

PRED_FILE = Path("predictions/prospect_projection_walkforward_v4_2_dbfallback.csv")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/look_up_player_v4_2.py 'Player Name'")

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
    "v4_position_group",
    "v4_model_group",
    "college",
    "round",
    "pick",
    "v4_projected_outcome_value",
    "v4_global_score_0_100",
    "v4_position_score_0_100",
    "v4_position_tier",
    "v4_actual_outcome_value",
    "v4_residual_actual_minus_projected",
    "v4_outlier_type",
    "v4_explanation",
]

cols = [c for c in cols if c in matches.columns]

print(matches[cols].sort_values("v4_position_score_0_100", ascending=False).to_string(index=False))
