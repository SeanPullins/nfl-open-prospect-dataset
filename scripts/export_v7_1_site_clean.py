#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

IN_FILE = Path("site_data/prospect_scores_v7_1.csv")
OUT_FILE = Path("site_data/player_grades_v7_1.csv")

df = pd.read_csv(IN_FILE, low_memory=False)

rename = {
    "final_score": "outcome_grade_pff_powered",
    "position_score": "position_model_score",
    "ceiling_score": "ceiling_grade",
    "floor_score": "floor_grade",
    "tier": "outcome_tier",
    "model_explanation": "grade_explanation",
}

df = df.rename(columns=rename)

preferred_cols = [
    "player",
    "draft_year",
    "position",
    "position_group",
    "model_group",
    "college",
    "round",
    "pick",
    "outcome_grade_pff_powered",
    "position_model_score",
    "ceiling_grade",
    "floor_grade",
    "starter_probability",
    "elite_probability",
    "bust_probability",
    "outcome_tier",
    "outlier_type",
    "grade_explanation",
    "projected_value",
    "actual_value",
    "actual_position_percentile",
    "predicted_position_percentile",
]

cols = [c for c in preferred_cols if c in df.columns]
df = df[cols].copy()

df.to_csv(OUT_FILE, index=False)

print(f"WROTE: {OUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print(df.head(10).to_string(index=False))
