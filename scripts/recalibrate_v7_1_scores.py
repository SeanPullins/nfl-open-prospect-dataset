#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

IN_FILE = Path("predictions/model_v7_position_percentile_pff_summary.csv")
OUT_FILE = Path("predictions/model_v7_1_recalibrated_summary.csv")
SITE_FILE = Path("site_data/prospect_scores_v7_1.csv")

SITE_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE, low_memory=False)

required = [
    "v7_position_score",
    "v7_predicted_position_percentile",
    "v7_starter_probability",
    "v7_elite_probability",
    "v7_bust_probability",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

# New user-facing grade:
# Position score is the backbone. Probabilities refine it.
df["v7_1_final_score"] = (
    0.70 * df["v7_position_score"]
    + 0.15 * df["v7_predicted_position_percentile"]
    + 0.10 * (100 * df["v7_starter_probability"])
    + 0.05 * (100 * df["v7_elite_probability"])
    - 0.05 * (100 * df["v7_bust_probability"])
).clip(0, 100).round(1)

df["v7_1_ceiling_score"] = np.maximum(
    df["v7_1_final_score"],
    (
        0.75 * df["v7_position_score"]
        + 0.25 * (100 * df["v7_elite_probability"])
    )
).clip(0, 100).round(1)

df["v7_1_floor_score"] = (100 * (1 - df["v7_bust_probability"])).clip(0, 100).round(1)

def tier(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 95:
        return "Elite / rare profile"
    if score >= 88:
        return "High-end starter / premium prospect profile"
    if score >= 75:
        return "Starter-caliber profile"
    if score >= 58:
        return "Rotational / contributor profile"
    if score >= 40:
        return "Depth / replacement-level profile"
    return "Low-probability profile"

df["v7_1_tier"] = df["v7_1_final_score"].map(tier)

def outlier(row):
    actual_pct = row.get("v7_actual_position_percentile")
    score = row.get("v7_1_final_score")
    if pd.isna(actual_pct) or pd.isna(score):
        return "unknown"

    if actual_pct >= 85 and score < 60:
        return "major_positive_outlier"
    if actual_pct <= 25 and score >= 80:
        return "major_negative_miss"
    if actual_pct - score >= 30:
        return "positive_outlier_exceeded_projection"
    if actual_pct - score <= -30:
        return "negative_outlier_missed_projection"
    return "near_expected"

df["v7_1_outlier_type"] = df.apply(outlier, axis=1)

def explanation(row):
    bits = []

    old = str(row.get("v7_explanation", "")).strip()
    if old and old.lower() != "nan":
        bits.append(old)

    if row.get("v7_1_final_score", 0) >= 90:
        bits.append("elite model grade")
    elif row.get("v7_1_final_score", 0) >= 80:
        bits.append("high-end model grade")

    if row.get("v7_position_score", 0) >= 95:
        bits.append("top-tier position profile")
    elif row.get("v7_position_score", 0) >= 85:
        bits.append("strong position profile")

    if row.get("v7_bust_probability", 0) >= 0.55:
        bits.append("meaningful downside risk")

    # Deduplicate while preserving order.
    final = []
    for b in bits:
        if b and b not in final:
            final.append(b)

    return "; ".join(final) if final else "limited usable model signal"

df["v7_1_explanation"] = df.apply(explanation, axis=1)

df.to_csv(OUT_FILE, index=False)

name_col = next(
    c for c in ["player_name_clean", "player_name", "pfr_player_name", "name", "display_name"]
    if c in df.columns
)

site_cols = [
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

site_cols = [c for c in site_cols if c in df.columns]

site = df[site_cols].copy()

site = site.rename(columns={
    name_col: "player",
    "season": "draft_year",
    "v7_position_group": "position_group",
    "v7_model_group": "model_group",
    "v7_projected_value": "projected_value",
    "v7_actual_value": "actual_value",
    "v7_actual_position_percentile": "actual_position_percentile",
    "v7_predicted_position_percentile": "predicted_position_percentile",
    "v7_position_score": "position_score",
    "v7_1_final_score": "final_score",
    "v7_1_ceiling_score": "ceiling_score",
    "v7_1_floor_score": "floor_score",
    "v7_starter_probability": "starter_probability",
    "v7_elite_probability": "elite_probability",
    "v7_bust_probability": "bust_probability",
    "v7_1_tier": "tier",
    "v7_1_outlier_type": "outlier_type",
    "v7_1_explanation": "model_explanation",
})

for c in site.columns:
    if c.endswith("_probability"):
        site[c] = pd.to_numeric(site[c], errors="coerce").round(3)
    elif c in [
        "projected_value",
        "actual_value",
        "actual_position_percentile",
        "predicted_position_percentile",
        "position_score",
        "final_score",
        "ceiling_score",
        "floor_score",
    ]:
        site[c] = pd.to_numeric(site[c], errors="coerce").round(1)

site.to_csv(SITE_FILE, index=False)

print(f"WROTE: {OUT_FILE}")
print(f"WROTE: {SITE_FILE}")
print(f"Rows: {len(site):,}")

print("")
print("Top 20 by final_score:")
print(site.sort_values("final_score", ascending=False).head(20).to_string(index=False))
