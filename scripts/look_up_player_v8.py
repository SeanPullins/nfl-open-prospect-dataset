#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

DATA = Path("site_data/player_cards_v8.csv")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/look_up_player_v8.py 'Player Name'")

query = " ".join(sys.argv[1:]).lower()
df = pd.read_csv(DATA, low_memory=False)

matches = df[df["player"].astype(str).str.lower().str.contains(query, na=False)].copy()

if matches.empty:
    print(f"No matches for: {query}")
    raise SystemExit(0)

cols = [
    "player", "draft_year", "position", "college", "round", "pick",
    "outcome_grade_pff_powered", "outcome_tier",
    "overall_rank_in_class", "position_rank_in_class",
    "draft_value_vs_grade", "confidence_label", "hit_miss_label",
    "outlier_type", "historical_position_comps",
    "player_card_summary",
]

cols = [c for c in cols if c in matches.columns]

print(matches[cols].sort_values("outcome_grade_pff_powered", ascending=False).to_string(index=False))
