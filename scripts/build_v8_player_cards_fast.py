#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

IN_FILE = Path("site_data/player_grades_v7_1.csv")
OUT_FILE = Path("site_data/player_cards_v8.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(IN_FILE, low_memory=False)

num_cols = [
    "draft_year", "round", "pick", "outcome_grade_pff_powered",
    "position_model_score", "ceiling_grade", "floor_grade",
    "starter_probability", "elite_probability", "bust_probability",
    "projected_value", "actual_value", "actual_position_percentile",
    "predicted_position_percentile",
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["overall_rank_in_class"] = (
    df.groupby("draft_year")["outcome_grade_pff_powered"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

df["position_rank_in_class"] = (
    df.groupby(["draft_year", "position_group"])["outcome_grade_pff_powered"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

df["overall_class_percentile"] = (
    df.groupby("draft_year")["outcome_grade_pff_powered"]
    .rank(pct=True, method="average") * 100
).round(1)

df["position_class_percentile"] = (
    df.groupby(["draft_year", "position_group"])["outcome_grade_pff_powered"]
    .rank(pct=True, method="average") * 100
).round(1)

def draft_capital_bucket(row):
    pick = row.get("pick")
    if pd.isna(pick):
        return "Unknown draft capital"
    pick = float(pick)
    if pick <= 5:
        return "Top-5 pick"
    if pick <= 15:
        return "Top-15 pick"
    if pick <= 32:
        return "Round 1 pick"
    if pick <= 64:
        return "Round 2 pick"
    if pick <= 100:
        return "Day 2 pick"
    if pick <= 150:
        return "Early Day 3 pick"
    return "Late Day 3 pick"

df["draft_capital_bucket"] = df.apply(draft_capital_bucket, axis=1)

def draft_value_label(row):
    pick = row.get("pick")
    grade = row.get("outcome_grade_pff_powered")

    if pd.isna(pick) or pd.isna(grade):
        return "Unknown"

    pick = float(pick)
    grade = float(grade)

    if pick <= 15:
        if grade >= 90:
            return "Premium pick justified"
        if grade >= 75:
            return "Solid premium pick"
        if grade >= 58:
            return "Mixed return on premium pick"
        return "Major miss for draft slot"

    if pick <= 64:
        if grade >= 88:
            return "Major value"
        if grade >= 75:
            return "Strong value"
        if grade >= 58:
            return "Fair value"
        return "Miss for draft slot"

    if pick <= 100:
        if grade >= 85:
            return "Major Day 2 value"
        if grade >= 70:
            return "Strong Day 2 value"
        if grade >= 55:
            return "Fair Day 2 value"
        return "Day 2 miss"

    if grade >= 88:
        return "Massive late-round value"
    if grade >= 75:
        return "Strong late-round value"
    if grade >= 58:
        return "Useful late-round value"
    return "Low-impact outcome"

df["draft_value_vs_grade"] = df.apply(draft_value_label, axis=1)

def confidence_score(row):
    ceiling = row.get("ceiling_grade")
    floor = row.get("floor_grade")
    starter = row.get("starter_probability")
    bust = row.get("bust_probability")

    score = 70.0

    if not pd.isna(ceiling) and not pd.isna(floor):
        spread = float(ceiling) - float(floor)
        if spread <= 10:
            score += 15
        elif spread <= 20:
            score += 8
        elif spread >= 35:
            score -= 15

    if not pd.isna(starter):
        if starter >= 0.80:
            score += 8
        elif starter <= 0.35:
            score -= 8

    if not pd.isna(bust):
        if bust <= 0.10:
            score += 7
        elif bust >= 0.45:
            score -= 12

    return round(max(0, min(100, score)), 1)

df["confidence_score"] = df.apply(confidence_score, axis=1)

def confidence_label(score):
    if pd.isna(score):
        return "Unknown confidence"
    if score >= 85:
        return "High confidence"
    if score >= 70:
        return "Medium confidence"
    if score >= 55:
        return "Low-medium confidence"
    return "Low confidence"

df["confidence_label"] = df["confidence_score"].map(confidence_label)

def hit_miss_label(row):
    grade = row.get("outcome_grade_pff_powered")
    outlier = str(row.get("outlier_type", ""))

    if pd.isna(grade):
        return "Unknown"

    grade = float(grade)

    if "major_positive" in outlier:
        return "Massive positive outlier"
    if "major_negative" in outlier:
        return "Major negative miss"
    if grade >= 95:
        return "Elite hit"
    if grade >= 88:
        return "High-end hit"
    if grade >= 75:
        return "Starter-level hit"
    if grade >= 58:
        return "Useful contributor"
    if grade >= 40:
        return "Replacement/depth outcome"
    return "Miss"

df["hit_miss_label"] = df.apply(hit_miss_label, axis=1)

# Fast comps: same position, close score, capped at 250 candidates.
def make_comp_string(row, pool, max_comps=5):
    if pool.empty:
        return ""

    work = pool.copy()
    work["_dist"] = (
        (work["outcome_grade_pff_powered"] - row["outcome_grade_pff_powered"]).abs()
        + 0.15 * (work["pick"].fillna(999) - row["pick"]).abs()
        + 5.0 * (work["round"].fillna(9) - row["round"]).abs()
    )

    work = work.sort_values("_dist").head(max_comps)

    comps = []
    for _, r in work.iterrows():
        comps.append(
            f"{r['player']} ({int(r['draft_year'])}, {r['position']}, score {r['outcome_grade_pff_powered']:.1f})"
        )
    return " | ".join(comps)

print("Building fast comps...")

position_comps = []
overall_comps = []

for i, row in df.iterrows():
    prior = df[(df.index != i) & (df["draft_year"] < row["draft_year"])].copy()

    same_pos = prior[prior["position_group"] == row["position_group"]].copy()

    if len(same_pos) > 250:
        same_pos["_score_dist"] = (same_pos["outcome_grade_pff_powered"] - row["outcome_grade_pff_powered"]).abs()
        same_pos = same_pos.sort_values("_score_dist").head(250)

    if len(prior) > 250:
        prior["_score_dist"] = (prior["outcome_grade_pff_powered"] - row["outcome_grade_pff_powered"]).abs()
        prior = prior.sort_values("_score_dist").head(250)

    position_comps.append(make_comp_string(row, same_pos, 5))
    overall_comps.append(make_comp_string(row, prior, 5))

df["historical_position_comps"] = position_comps
df["historical_overall_comps"] = overall_comps

def build_card_summary(row):
    player = row.get("player", "Player")
    year = int(row["draft_year"]) if not pd.isna(row.get("draft_year")) else "Unknown year"
    pos = row.get("position", "")
    college = row.get("college", "")
    score = row.get("outcome_grade_pff_powered")
    tier = row.get("outcome_tier", "")
    rank = row.get("position_rank_in_class")
    bucket = row.get("draft_capital_bucket", "")
    value = row.get("draft_value_vs_grade", "")
    conf = row.get("confidence_label", "")
    explanation = row.get("grade_explanation", "")

    parts = []
    if not pd.isna(score):
        parts.append(f"{player} ({year}, {pos}, {college}) graded {score:.1f}.")
    else:
        parts.append(f"{player} ({year}, {pos}, {college}).")

    if tier:
        parts.append(f"Tier: {tier}.")
    if not pd.isna(rank):
        parts.append(f"Position rank in class: #{int(rank)}.")
    if bucket:
        parts.append(f"Draft capital: {bucket}.")
    if value:
        parts.append(f"Draft value read: {value}.")
    if conf:
        parts.append(f"Model confidence: {conf}.")
    if explanation:
        parts.append(f"Why: {explanation}.")

    return " ".join(parts)

df["player_card_summary"] = df.apply(build_card_summary, axis=1)

preferred_cols = [
    "player", "draft_year", "position", "position_group", "model_group",
    "college", "round", "pick", "draft_capital_bucket",
    "overall_rank_in_class", "position_rank_in_class",
    "overall_class_percentile", "position_class_percentile",
    "outcome_grade_pff_powered", "position_model_score",
    "ceiling_grade", "floor_grade", "starter_probability",
    "elite_probability", "bust_probability", "outcome_tier",
    "hit_miss_label", "draft_value_vs_grade",
    "confidence_score", "confidence_label", "outlier_type",
    "grade_explanation", "historical_position_comps",
    "historical_overall_comps", "player_card_summary",
    "projected_value", "actual_value",
    "actual_position_percentile", "predicted_position_percentile",
]

final_cols = [c for c in preferred_cols if c in df.columns]
out = df[final_cols].copy()

out.to_csv(OUT_FILE, index=False)

out.sort_values(["draft_year", "overall_rank_in_class"]).to_csv(
    REPORT_DIR / "v8_class_rankings.csv", index=False
)

out.sort_values("outcome_grade_pff_powered", ascending=False).head(500).to_csv(
    REPORT_DIR / "v8_top_500_overall.csv", index=False
)

print(f"WROTE: {OUT_FILE}")
print(f"Rows: {len(out):,}")
print(f"Columns: {len(out.columns):,}")

show_cols = [
    "player", "draft_year", "position", "college",
    "outcome_grade_pff_powered", "overall_rank_in_class",
    "position_rank_in_class", "outcome_tier",
    "draft_value_vs_grade", "confidence_label",
]
show_cols = [c for c in show_cols if c in out.columns]

print("")
print("Top 15 V8 cards:")
print(out.sort_values("outcome_grade_pff_powered", ascending=False)[show_cols].head(15).to_string(index=False))
