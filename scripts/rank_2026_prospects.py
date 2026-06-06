#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

DATA = Path("site_data/prospect_projections_2026_v1.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--position")
parser.add_argument("--top", type=int, default=50)
args = parser.parse_args()

if not DATA.exists():
    raise SystemExit(f"Missing file: {DATA}. Run: python scripts/project_2026_class_v1.py")

df = pd.read_csv(DATA, low_memory=False)

if args.position:
    q = args.position.upper()
    df = df[
        (df["position"].astype(str).str.upper() == q)
        | (df["position_group"].astype(str).str.upper() == q)
        | (df["model_group"].astype(str).str.upper() == q)
    ]

cols = [
    "overall_rank_2026",
    "position_rank_2026",
    "player",
    "position",
    "position_group",
    "college",
    "school",
    "projected_pick",
    "projected_round",
    "projection_score",
    "ceiling_score",
    "floor_score",
    "starter_probability",
    "elite_probability",
    "bust_probability",
    "projection_tier",
    "projection_explanation",
]

cols = [c for c in cols if c in df.columns]

print(
    df[cols]
    .sort_values("projection_score", ascending=False)
    .head(args.top)
    .to_string(index=False)
)
