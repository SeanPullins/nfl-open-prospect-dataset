#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

IN = Path("site_data/player_cards_v8.csv")
OUT_SITE = Path("site_data/player_cards_v9.csv")
OUT_DOCS = Path("docs/data/player_cards_v9.csv")

PREMIUM_POSITIONS = {"QB", "OT", "T", "OL", "EDGE", "DE", "WR", "CB"}

def n(series):
    return pd.to_numeric(series, errors="coerce")

def score_mode(year):
    if pd.isna(year):
        return "unknown"
    year = int(float(year))
    if year <= 2021:
        return "actual_value"
    if year <= 2023:
        return "current_value"
    return "prospect_projection"

def score_label(mode):
    return {
        "actual_value": "Actual Value Score",
        "current_value": "Developing Value Score",
        "prospect_projection": "Prospect Profile Score",
    }.get(mode, "Score")

def pos_mult(pos):
    p = str(pos).upper().strip()
    if p == "QB": return 1.35
    if p in {"EDGE", "DE"}: return 1.15
    if p in {"OT", "T", "OL"}: return 1.12
    if p == "WR": return 1.10
    if p == "CB": return 1.08
    if p in {"DT", "IDL"}: return 1.02
    if p in {"S", "SAF"}: return 0.98
    if p in {"TE", "LB"}: return 0.95
    if p in {"IOL", "G", "C"}: return 0.92
    if p == "RB": return 0.88
    if p in {"ST", "K", "P", "LS"}: return 0.65
    return 1.00

def draft_score(row):
    pick = row.get("pick")
    pos = str(row.get("position_group") or row.get("position")).upper().strip()

    if pd.isna(pick):
        base = 42
    else:
        pick = float(pick)
        if pick <= 5: base = 92
        elif pick <= 15: base = 88
        elif pick <= 32: base = 82
        elif pick <= 64: base = 72
        elif pick <= 100: base = 63
        elif pick <= 150: base = 53
        elif pick <= 220: base = 45
        else: base = 38

    if pos == "QB":
        base += 4
    elif pos in PREMIUM_POSITIONS:
        base += 2
    elif pos in {"RB", "ST", "K", "P", "LS"}:
        base -= 2

    return max(0, min(99, base))

def av_to_score(av):
    if pd.isna(av):
        return np.nan
    av = float(av)
    if av >= 120: return 99
    if av >= 90: return 96
    if av >= 70: return 92
    if av >= 50: return 85
    if av >= 35: return 75
    if av >= 20: return 62
    if av >= 10: return 48
    if av >= 3: return 35
    return 20

def position_bonus(pos):
    p = str(pos).upper().strip()
    if p == "QB":
        return 3
    if p in {"EDGE", "DE", "OT", "T", "OL", "WR", "CB"}:
        return 2
    if p in {"DT", "IDL", "S", "SAF", "TE", "LB"}:
        return 0
    if p in {"RB", "IOL", "G", "C"}:
        return -1
    if p in {"ST", "K", "P", "LS"}:
        return -8
    return 0


def actual_value_to_team(row):
    """
    Mature classes: actual value to team.

    Do not multiply by positional value here. Multiplication was flattening elite QBs
    into 99s and unfairly dragging strong RB outcomes down. Use actual outcome quality
    plus a small positional-context bonus instead.
    """
    pct = row.get("actual_position_percentile")
    av = row.get("actual_value")
    grade = row.get("outcome_grade_pff_powered")
    pos = row.get("position_group") or row.get("position")

    av_score = av_to_score(av)

    if not pd.isna(pct) and not pd.isna(av_score):
        base = 0.70 * float(pct) + 0.30 * float(av_score)
    elif not pd.isna(pct):
        base = float(pct)
    elif not pd.isna(av_score):
        base = float(av_score)
    elif not pd.isna(grade):
        base = float(grade)
    else:
        return np.nan

    base += position_bonus(pos)

    return round(max(0, min(99, base)), 1)

def current_value_to_team(row):
    grade = row.get("outcome_grade_pff_powered")
    av = row.get("actual_value")
    pos = str(row.get("position_group") or row.get("position")).upper().strip()
    ds = draft_score(row)

    profile = ds if pd.isna(grade) else float(grade)

    av_score = av_to_score(av)
    if pd.isna(av_score):
        av_score = 50

    score = 0.60 * profile + 0.25 * av_score + 0.15 * ds

    if pos == "QB":
        score += 3
    elif pos in PREMIUM_POSITIONS:
        score += 2
    elif pos in {"RB", "ST", "K", "P", "LS"}:
        score -= 2

    if profile < 85:
        score = min(score, 84)

    if pos == "QB" and profile < 85 and not pd.isna(av) and float(av) < 15:
        score = min(score, 78)

    return round(max(0, min(99, score)), 1)

def prospect_profile_score(row):
    grade = row.get("outcome_grade_pff_powered")
    pick = row.get("pick")
    pos = str(row.get("position_group") or row.get("position")).upper().strip()
    ds = draft_score(row)

    if pd.isna(grade):
        score = ds
    else:
        score = 0.85 * ds + 0.15 * float(grade)

    if not pd.isna(pick):
        pk = float(pick)
        if pos in PREMIUM_POSITIONS:
            if pk <= 5: score = max(score, 88)
            elif pk <= 10: score = max(score, 86)
            elif pk <= 15: score = max(score, 84)
            elif pk <= 32: score = max(score, 78)
        else:
            if pk <= 5: score = max(score, 84)
            elif pk <= 15: score = max(score, 80)
            elif pk <= 32: score = max(score, 74)

    return round(max(0, min(99, score)), 1)

def bucket(score):
    if pd.isna(score): return "Unknown"
    score = float(score)
    if score >= 97: return "Top-5 caliber"
    if score >= 93: return "Top-15 caliber"
    if score >= 88: return "Round 1 caliber"
    if score >= 80: return "Round 2 caliber"
    if score >= 70: return "Day 2 caliber"
    if score >= 58: return "Early Day 3 caliber"
    if score >= 40: return "Late Day 3 / priority depth"
    return "Undrafted / replacement outcome"

def bucket_pick(label):
    return {
        "Top-5 caliber": 5,
        "Top-15 caliber": 15,
        "Round 1 caliber": 32,
        "Round 2 caliber": 64,
        "Day 2 caliber": 100,
        "Early Day 3 caliber": 150,
        "Late Day 3 / priority depth": 220,
        "Undrafted / replacement outcome": 999,
    }.get(label, 999)

def regrade(row):
    mode = row.get("score_mode")
    pick = row.get("pick")
    should = bucket_pick(row.get("should_have_been_drafted"))

    if mode == "prospect_projection":
        return "Too Early"

    if pd.isna(pick):
        if should <= 100: return "Massive undrafted steal"
        if should <= 220: return "Useful undrafted value"
        return "Undrafted-level outcome"

    pick = float(pick)

    if should <= 15 and pick > 64: return "Massive steal"
    if should <= 32 and pick > 100: return "Massive steal"
    if should <= 64 and pick > 150: return "Major value"
    if should <= 100 and pick > 150: return "Good value"
    if should + 40 < pick: return "Drafted too late"

    if pick <= 15 and should > 100: return "Major overdraft"
    if pick <= 32 and should > 150: return "Major overdraft"
    if pick <= 64 and should > 220: return "Overdraft"
    if pick + 50 < should: return "Drafted too early"

    return "Drafted about right"


def class_rank_regrade(row):
    """
    Draft-slot regrade based on actual pick vs. class-aware re-draft rank.
    This avoids saying Class #9 players were drafted too early just because
    their global bucket says Round 2 caliber.
    """
    mode = row.get("score_mode")
    pick = row.get("pick")
    rank = row.get("redraft_class_rank")

    if mode == "prospect_projection":
        return "Too Early"

    if pd.isna(rank):
        return row.get("draft_slot_regrade", "Unknown")

    rank = float(rank)

    if pd.isna(pick):
        if rank <= 32:
            return "Massive undrafted steal"
        if rank <= 100:
            return "Useful undrafted value"
        return "Undrafted-level outcome"

    pick = float(pick)

    # Steals: player should have gone much earlier in his own class.
    if rank <= 10 and pick > 25:
        return "Major value"
    if rank <= 15 and pick > 64:
        return "Massive steal"
    if rank <= 32 and pick > 100:
        return "Massive steal"
    if rank <= 64 and pick > 150:
        return "Major value"
    if rank <= 100 and pick > 150:
        return "Good value"
    if rank + 40 < pick:
        return "Drafted too late"

    # Overdrafts: player went much earlier than his class re-rank.
    if pick <= 5 and rank > 32:
        return "Major overdraft"
    if pick <= 15 and rank > 64:
        return "Major overdraft"
    if pick <= 32 and rank > 100:
        return "Overdraft"
    if pick + 35 < rank:
        return "Drafted too early"

    return "Drafted about right"


def good_bad_type(row):
    if row.get("score_mode") == "prospect_projection":
        return "neutral"

    rg = row.get("draft_slot_regrade")
    if rg in {"Massive steal", "Major value", "Good value", "Massive undrafted steal", "Useful undrafted value"}:
        return "good_miss"
    if rg in {"Major overdraft", "Overdraft"}:
        return "bad_miss"
    return "neutral"

def card_class(t):
    if t == "good_miss": return "card-good-miss"
    if t == "bad_miss": return "card-bad-miss"
    return "card-neutral"

def outcome_flag(row):
    mode = row.get("score_mode")
    rg = row.get("draft_slot_regrade")

    if mode == "prospect_projection":
        return "Too Early to Regrade"

    if mode == "current_value":
        return f"Developing: {rg}"

    # Mature classes should display the V9 regrade label directly.
    # This avoids stale old labels like "Drafted too early" when the new
    # class-rank regrade says "Drafted about right."
    return rg

def main():
    df = pd.read_csv(IN, low_memory=False)

    for c in ["draft_year", "pick", "actual_value", "actual_position_percentile", "outcome_grade_pff_powered"]:
        if c in df.columns:
            df[c] = n(df[c])

    df["score_mode"] = df["draft_year"].map(score_mode)

    df["actual_value_to_team"] = df.apply(actual_value_to_team, axis=1)
    df["current_value_to_team"] = df.apply(current_value_to_team, axis=1)
    df["prospect_similarity_score"] = df.apply(prospect_profile_score, axis=1)

    df["site_display_score"] = np.select(
        [
            df["score_mode"].eq("actual_value"),
            df["score_mode"].eq("current_value"),
            df["score_mode"].eq("prospect_projection"),
        ],
        [
            df["actual_value_to_team"],
            df["current_value_to_team"],
            df["prospect_similarity_score"],
        ],
        default=df["outcome_grade_pff_powered"],
    )

    df["site_display_score_label"] = df["score_mode"].map(score_label)
    df["should_have_been_drafted"] = df["site_display_score"].map(bucket)
    df["should_have_pick_bucket"] = df["should_have_been_drafted"].map(bucket_pick)
    df["draft_slot_regrade"] = df.apply(regrade, axis=1)

    df["good_bad_miss_type"] = df.apply(good_bad_type, axis=1)
    df["miss_flag"] = np.where(df["good_bad_miss_type"].eq("bad_miss"), "bad_miss", "neutral")
    df["value_flag"] = np.where(df["good_bad_miss_type"].eq("good_miss"), "good_miss", "neutral")
    df["outcome_card_class"] = df["good_bad_miss_type"].map(card_class)
    df["actual_outcome_flag"] = df.apply(outcome_flag, axis=1)

    # Site compatibility
    df["display_grade"] = df["site_display_score"]
    df["display_grade_label"] = df["site_display_score_label"]

    df["_year"] = n(df["draft_year"])

    # Rank using a precision score so capped display scores do not create bad class ranks.
    # For mature players, actual_value_to_team + actual_value gives better ordering.
    df["_rank_score"] = n(df["site_display_score"])
    if "actual_value" in df.columns:
        df["_rank_score"] = df["_rank_score"] + (n(df["actual_value"]).fillna(0) * 0.03)
    if "outcome_grade_pff_powered" in df.columns:
        df["_rank_score"] = df["_rank_score"] + (n(df["outcome_grade_pff_powered"]).fillna(0) * 0.01)

    df["redraft_class_rank"] = (
        df.groupby("_year")["_rank_score"]
          .rank(method="first", ascending=False)
          .round()
          .astype("Int64")
    )

    df["redraft_position_rank"] = (
        df.groupby(["_year", "position_group"])["_rank_score"]
          .rank(method="first", ascending=False)
          .round()
          .astype("Int64")
    )

    def rank_label(row):
        parts = []
        if not pd.isna(row.get("redraft_class_rank")):
            parts.append(f"Class #{int(row.get('redraft_class_rank'))}")
        if not pd.isna(row.get("redraft_position_rank")):
            parts.append(f"{row.get('position_group')} #{int(row.get('redraft_position_rank'))}")
        return " / ".join(parts) if parts else row.get("should_have_been_drafted", "—")

    df["should_have_been_drafted_display"] = df.apply(rank_label, axis=1)

    # Recompute draft-slot regrade after class-aware ranks exist.
    df["draft_slot_regrade"] = df.apply(class_rank_regrade, axis=1)
    df["good_bad_miss_type"] = df.apply(good_bad_type, axis=1)
    df["miss_flag"] = np.where(df["good_bad_miss_type"].eq("bad_miss"), "bad_miss", "neutral")
    df["value_flag"] = np.where(df["good_bad_miss_type"].eq("good_miss"), "good_miss", "neutral")
    df["outcome_card_class"] = df["good_bad_miss_type"].map(card_class)
    df["actual_outcome_flag"] = df.apply(outcome_flag, axis=1)

    df = df.drop(columns=["_year", "_rank_score"], errors="ignore")

    df.to_csv(OUT_SITE, index=False)
    df.to_csv(OUT_DOCS, index=False)

    print(f"WROTE: {OUT_SITE}")
    print(f"WROTE: {OUT_DOCS}")
    print(f"Rows: {len(df):,}")
    print("\nScore modes:")
    print(df["score_mode"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
