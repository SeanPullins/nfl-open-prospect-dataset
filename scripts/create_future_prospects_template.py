#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

OUT = Path("site_data/future_prospects.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

cols = [
    "player",
    "draft_year",
    "position",
    "position_group",
    "college",
    "class_status",
    "watchlist_rank",
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
    "college_stats_status",
    "pff_status",
    "all22_status",
    "notes",
]

rows = [
    {
        "player": "Sample 2027 Player",
        "draft_year": 2027,
        "position": "QB",
        "position_group": "QB",
        "college": "TBD",
        "class_status": "future_watchlist",
        "watchlist_rank": 1,
        "projected_pick": "",
        "projected_round": "",
        "projection_score": "",
        "ceiling_score": "",
        "floor_score": "",
        "starter_probability": "",
        "elite_probability": "",
        "bust_probability": "",
        "projection_tier": "Unscored future watchlist",
        "projection_explanation": "Future class placeholder. Add college stats, PFF, and All-22 data as available.",
        "college_stats_status": "pending",
        "pff_status": "pending",
        "all22_status": "pending",
        "notes": "",
    }
]

df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT, index=False)

print(f"WROTE: {OUT}")
print(df.to_string(index=False))
