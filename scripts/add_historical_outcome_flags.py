#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

FILES = [
    Path("site_data/player_cards_v8.csv"),
    Path("docs/data/player_cards_v8.csv"),
]

def outcome_level(grade):
    if pd.isna(grade):
        return "Unknown"

    grade = float(grade)

    if grade >= 95:
        return "Elite Hit"
    if grade >= 88:
        return "High-End Hit"
    if grade >= 75:
        return "Starter Hit"
    if grade >= 58:
        return "Useful Contributor"
    if grade >= 40:
        return "Depth / Replacement"
    return "Miss"

def actual_outcome_flag(row):
    grade = row.get("outcome_grade_pff_powered")
    pick = row.get("pick")

    if pd.isna(grade):
        return "Unknown Outcome"

    grade = float(grade)

    if pd.isna(pick):
        if grade >= 88:
            return "Massive Undrafted Value"
        if grade >= 75:
            return "Major Undrafted Value"
        if grade >= 58:
            return "Useful Undrafted Outcome"
        if grade >= 40:
            return "Depth Outcome"
        return "Low-Impact Outcome"

    pick = float(pick)

    # Premium picks
    if pick <= 15:
        if grade >= 90:
            return "Premium Pick Hit"
        if grade >= 75:
            return "Solid Premium Pick"
        if grade >= 58:
            return "Mixed Premium Pick"
        return "Major Draft-Slot Miss"

    # Round 1/2
    if pick <= 64:
        if grade >= 88:
            return "Major Value"
        if grade >= 75:
            return "Strong Starter Value"
        if grade >= 58:
            return "Fair Return"
        return "Draft-Slot Miss"

    # Day 2
    if pick <= 100:
        if grade >= 85:
            return "Major Day 2 Value"
        if grade >= 70:
            return "Strong Day 2 Value"
        if grade >= 55:
            return "Fair Day 2 Return"
        return "Day 2 Miss"

    # Day 3 / late
    if grade >= 88:
        return "Massive Late-Round Value"
    if grade >= 75:
        return "Strong Late-Round Value"
    if grade >= 58:
        return "Useful Late-Round Value"
    if grade >= 40:
        return "Depth Late-Round Outcome"
    return "Low-Impact Outcome"

def miss_flag(row):
    grade = row.get("outcome_grade_pff_powered")
    pick = row.get("pick")
    outlier = str(row.get("outlier_type", "")).lower()

    if pd.isna(grade):
        return "unknown"

    grade = float(grade)

    if "major_negative" in outlier or "negative_outlier" in outlier:
        return "bad_miss"

    if pd.isna(pick):
        return "neutral" if grade >= 40 else "bad_miss"

    pick = float(pick)

    if pick <= 15 and grade < 58:
        return "bad_miss"
    if pick <= 64 and grade < 52:
        return "bad_miss"
    if pick <= 100 and grade < 45:
        return "bad_miss"
    if grade < 35:
        return "bad_miss"

    return "neutral"

def value_flag(row):
    grade = row.get("outcome_grade_pff_powered")
    pick = row.get("pick")
    outlier = str(row.get("outlier_type", "")).lower()

    if pd.isna(grade):
        return "unknown"

    grade = float(grade)

    if "major_positive" in outlier:
        return "good_miss"

    if pd.isna(pick):
        if grade >= 75:
            return "good_miss"
        return "neutral"

    pick = float(pick)

    if pick > 100 and grade >= 88:
        return "good_miss"
    if pick > 64 and grade >= 75:
        return "good_miss"
    if pick > 32 and grade >= 88:
        return "good_miss"

    return "neutral"

def card_class(row):
    if row.get("value_flag") == "good_miss":
        return "card-good-miss"
    if row.get("miss_flag") == "bad_miss":
        return "card-bad-miss"
    return "card-neutral"

def main():
    for path in FILES:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)

        df["outcome_grade_pff_powered"] = pd.to_numeric(
            df["outcome_grade_pff_powered"], errors="coerce"
        )

        if "pick" in df.columns:
            df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
        else:
            df["pick"] = np.nan

        df["actual_outcome_level"] = df["outcome_grade_pff_powered"].map(outcome_level)
        df["actual_outcome_flag"] = df.apply(actual_outcome_flag, axis=1)
        df["miss_flag"] = df.apply(miss_flag, axis=1)
        df["value_flag"] = df.apply(value_flag, axis=1)
        df["outcome_card_class"] = df.apply(card_class, axis=1)

        df["is_bad_miss"] = df["miss_flag"].eq("bad_miss")
        df["is_good_miss"] = df["value_flag"].eq("good_miss")

        df.to_csv(path, index=False)

        print(f"\nWROTE: {path}")
        print(df[[
            "player",
            "draft_year",
            "position",
            "pick",
            "outcome_grade_pff_powered",
            "actual_outcome_flag",
            "miss_flag",
            "value_flag",
            "outcome_card_class",
        ]].head(25).to_string(index=False))

if __name__ == "__main__":
    main()
