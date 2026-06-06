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

    # Only treat negative outliers as bad misses if they had real draft capital.
    # Late-round failures are usually expected, not true misses.
    if pd.isna(pick):
        return "neutral"

    pick = float(pick)

    if "major_negative" in outlier or "negative_outlier" in outlier:
        if pick <= 100:
            return "bad_miss"

    # Premium draft capital miss.
    if pick <= 15 and grade < 58:
        return "bad_miss"

    # Round 1/2 miss.
    if pick <= 64 and grade < 52:
        return "bad_miss"

    # Day 2 miss.
    if pick <= 100 and grade < 45:
        return "bad_miss"

    # Day 3 players should not be red unless the model says the outcome was truly awful
    # and they still had relatively meaningful early Day 3 capital.
    if pick <= 150 and grade < 25:
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
    # Bad miss takes priority over good miss if both flags somehow fire.
    if row.get("miss_flag") == "bad_miss":
        return "card-bad-miss"
    if row.get("value_flag") == "good_miss":
        return "card-good-miss"
    return "card-neutral"


def positional_value_multiplier(pos):
    p = str(pos).upper().strip()

    # Premium-position hierarchy.
    # QB gets the biggest boost, then EDGE/OT/WR/CB.
    if p == "QB":
        return 1.35
    if p in {"EDGE", "DE"}:
        return 1.15
    if p in {"OT", "T"}:
        return 1.12
    if p == "WR":
        return 1.10
    if p == "CB":
        return 1.08
    if p in {"IDL", "DT"}:
        return 1.02
    if p in {"S", "SAF"}:
        return 0.98
    if p in {"TE", "LB"}:
        return 0.95
    if p in {"IOL", "G", "C"}:
        return 0.92
    if p == "RB":
        return 0.88
    if p in {"ST", "K", "P", "LS"}:
        return 0.65

    return 1.00


def should_have_been_drafted(row):
    """
    Re-draft label.

    Mature classes use actual NFL outcome.
    Developing classes use a developing profile score so 2022/2023 players are not
    unfairly dragged down by incomplete career samples.
    """
    pos = row.get("position_group") or row.get("position")
    maturity = row.get("outcome_maturity")

    actual_pct = row.get("actual_position_percentile")
    actual_value = row.get("actual_value")
    grade = row.get("outcome_grade_pff_powered")

    base = np.nan

    # 2022/2023: use developing profile, not final career outcome.
    # This blends model/profile grade, draft capital, and early NFL value.
    # Draft capital matters, but should not let shaky young QBs auto-grade as Top-5.
    if maturity == "developing":
        draft_score = draft_capital_profile_score(row)

        if pd.isna(grade):
            profile_grade = draft_score
        else:
            profile_grade = float(grade)

        av_score = np.nan
        if not pd.isna(actual_value):
            av = float(actual_value)
            if av >= 50:
                av_score = 90
            elif av >= 40:
                av_score = 84
            elif av >= 30:
                av_score = 78
            elif av >= 20:
                av_score = 68
            elif av >= 10:
                av_score = 56
            elif av >= 5:
                av_score = 45
            else:
                av_score = 30

        if pd.isna(av_score):
            base = 0.75 * profile_grade + 0.25 * draft_score
        else:
            base = 0.65 * profile_grade + 0.20 * draft_score + 0.15 * av_score

        p = str(pos).upper().strip()
        if p == "QB":
            base += 3
        elif p in {"OT", "T", "OL", "EDGE", "DE", "WR", "CB"}:
            base += 2
        elif p in {"RB", "ST", "K", "P", "LS"}:
            base -= 2

        # Floors for high-end developing profiles.
        # If the profile grade is already excellent, avoid under-labeling young hits.
        if profile_grade >= 94:
            if p in {"RB", "TE", "LB", "S", "SAF"}:
                base = max(base, 88)   # Round 1 caliber
            else:
                base = max(base, 93)   # Top-15 caliber
        elif profile_grade >= 92:
            base = max(base, 88)       # Round 1 caliber
        elif profile_grade >= 88:
            base = max(base, 80)       # Round 2 caliber

        # Premium top-5 developing picks with strong enough profile should not be
        # called Round 2 unless early profile is truly weak.
        if not pd.isna(row.get("pick")):
            pk = float(row.get("pick"))
            if pk <= 5 and p in {"QB", "EDGE", "DE", "LB", "OT", "T", "OL", "WR", "CB"} and profile_grade >= 88:
                base = max(base, 88)   # Round 1 caliber

        # Caps for shaky developing players.
        # They can still improve later, but we should not label them Top-5 yet.
        if profile_grade < 85:
            base = min(base, 84)   # max Round 2 caliber
        if p == "QB" and profile_grade < 85 and not pd.isna(actual_value) and float(actual_value) < 15:
            base = min(base, 78)   # max Day 2 caliber for now

        adjusted = max(0, min(100, base))

    # 2024/2025: still too early; label gets overwritten later, but keep sane fallback.
    elif maturity == "too_early":
        adjusted = display_grade_for_site(row)

    else:
        # Best signal for mature classes: actual outcome percentile within position.
        if not pd.isna(actual_pct):
            base = float(actual_pct)

        # Second-best signal: actual value, scaled roughly into a percentile-ish score.
        elif not pd.isna(actual_value):
            av = float(actual_value)
            if av >= 120:
                base = 99
            elif av >= 90:
                base = 96
            elif av >= 70:
                base = 92
            elif av >= 50:
                base = 85
            elif av >= 35:
                base = 75
            elif av >= 20:
                base = 62
            elif av >= 10:
                base = 48
            elif av >= 3:
                base = 35
            else:
                base = 20

        # Fallback only.
        elif not pd.isna(grade):
            base = float(grade)

        if pd.isna(base):
            return "Unknown"

        adjusted = base * positional_value_multiplier(pos)
        adjusted = max(0, min(100, adjusted))

    if adjusted >= 97:
        return "Top-5 caliber"
    if adjusted >= 93:
        return "Top-15 caliber"
    if adjusted >= 88:
        return "Round 1 caliber"
    if adjusted >= 80:
        return "Round 2 caliber"
    if adjusted >= 70:
        return "Day 2 caliber"
    if adjusted >= 58:
        return "Early Day 3 caliber"
    if adjusted >= 40:
        return "Late Day 3 / priority depth"
    return "Undrafted / replacement outcome"


def should_pick_bucket(label):
    mapping = {
        "Top-5 caliber": 5,
        "Top-15 caliber": 15,
        "Round 1 caliber": 32,
        "Round 2 caliber": 64,
        "Day 2 caliber": 100,
        "Early Day 3 caliber": 150,
        "Late Day 3 / priority depth": 220,
        "Undrafted / replacement outcome": 999,
    }
    return mapping.get(label, 999)


def draft_slot_regrade(row):
    pick = row.get("pick")
    label = row.get("should_have_been_drafted")
    should_pick = should_pick_bucket(label)

    if pd.isna(pick):
        if should_pick <= 100:
            return "Massive undrafted steal"
        if should_pick <= 220:
            return "Useful undrafted value"
        return "Undrafted-level outcome"

    pick = float(pick)

    # Positive value: player should have gone much earlier.
    if should_pick <= 15 and pick > 64:
        return "Massive steal"
    if should_pick <= 32 and pick > 100:
        return "Massive steal"
    if should_pick <= 64 and pick > 150:
        return "Major value"
    if should_pick <= 100 and pick > 150:
        return "Good value"
    if should_pick + 40 < pick:
        return "Drafted too late"

    # Negative value: player should have gone much later.
    if pick <= 15 and should_pick > 100:
        return "Major overdraft"
    if pick <= 32 and should_pick > 150:
        return "Major overdraft"
    if pick <= 64 and should_pick > 220:
        return "Overdraft"
    if pick + 50 < should_pick:
        return "Drafted too early"

    return "Drafted about right"



def outcome_maturity(row):
    year = row.get("draft_year")
    if pd.isna(year):
        return "unknown"

    year = int(float(year))

    if year >= 2024:
        return "too_early"
    if year >= 2022:
        return "developing"
    return "mature"



def draft_capital_profile_score(row):
    pick = row.get("pick")
    pos = row.get("position_group") or row.get("position")

    if pd.isna(pick):
        base = 45
    else:
        pick = float(pick)

        if pick <= 5:
            base = 92
        elif pick <= 15:
            base = 88
        elif pick <= 32:
            base = 82
        elif pick <= 64:
            base = 72
        elif pick <= 100:
            base = 63
        elif pick <= 150:
            base = 53
        elif pick <= 220:
            base = 45
        else:
            base = 38

    # Small premium-position bump.
    p = str(pos).upper().strip()
    if p == "QB":
        base += 4
    elif p in {"EDGE", "DE", "OT", "T", "WR", "CB"}:
        base += 2
    elif p in {"RB", "ST", "K", "P", "LS"}:
        base -= 2

    return max(0, min(99, base))


def display_grade_for_site(row):
    maturity = row.get("outcome_maturity")
    existing = row.get("outcome_grade_pff_powered")
    pick = row.get("pick")
    pos = row.get("position_group") or row.get("position")

    # Mature/developing classes can use the historical outcome model.
    if maturity != "too_early":
        return existing

    # 2024/2025: show a prospect / early-career profile score.
    # Do not let tiny NFL production samples drag down recent premium prospects.
    draft_score = draft_capital_profile_score(row)

    p = str(pos).upper().strip()
    premium = p in {"QB", "OT", "T", "OL", "EDGE", "DE", "WR", "CB"}

    # Use probability model only as a light modifier for recent classes.
    starter = row.get("starter_probability")
    elite = row.get("elite_probability")
    bust = row.get("bust_probability")

    prob_score = None
    if not pd.isna(starter) or not pd.isna(elite) or not pd.isna(bust):
        s = 50 if pd.isna(starter) else float(starter) * 100
        e = 20 if pd.isna(elite) else float(elite) * 100
        b = 35 if pd.isna(bust) else float(bust) * 100
        prob_score = (0.55 * s) + (0.25 * e) + (0.20 * (100 - b))

    if prob_score is not None:
        profile = 0.85 * draft_score + 0.15 * prob_score
    else:
        profile = draft_score

    # Stronger floors for recent premium prospects.
    # These are display floors, not final career judgments.
    if not pd.isna(pick):
        pk = float(pick)

        if premium:
            if pk <= 5:
                profile = max(profile, 88)
            elif pk <= 10:
                profile = max(profile, 86)
            elif pk <= 15:
                profile = max(profile, 84)
            elif pk <= 32:
                profile = max(profile, 78)
        else:
            if pk <= 5:
                profile = max(profile, 84)
            elif pk <= 15:
                profile = max(profile, 80)
            elif pk <= 32:
                profile = max(profile, 74)

    return round(max(0, min(99, profile)), 1)


def display_grade_label(row):
    if row.get("outcome_maturity") == "too_early":
        return "Early Profile Score"
    if row.get("outcome_maturity") == "developing":
        return "Developing Outcome Score"
    return "Outcome Score"


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

        # A player should never be both a bad miss and a good miss.
        # For premium/Day 2 failures, bad miss wins.
        df.loc[df["miss_flag"].eq("bad_miss"), "value_flag"] = "neutral"

        df["outcome_maturity"] = df.apply(outcome_maturity, axis=1)

        df["should_have_been_drafted"] = df.apply(should_have_been_drafted, axis=1)
        df["should_have_pick_bucket"] = df["should_have_been_drafted"].map(should_pick_bucket)
        df["draft_slot_regrade"] = df.apply(draft_slot_regrade, axis=1)

        # Make the public-facing outcome label match the actual re-draft read.
        df.loc[df["miss_flag"].eq("bad_miss"), "actual_outcome_flag"] = df.loc[
            df["miss_flag"].eq("bad_miss"), "draft_slot_regrade"
        ]

        df.loc[df["value_flag"].eq("good_miss"), "actual_outcome_flag"] = df.loc[
            df["value_flag"].eq("good_miss"), "draft_slot_regrade"
        ]

        # 2024/2025 are too early to honestly call hits, misses, steals, or overdrafts.
        recent_mask = df["outcome_maturity"].eq("too_early")
        df.loc[recent_mask, "actual_outcome_flag"] = "Too Early to Regrade"
        df.loc[recent_mask, "should_have_been_drafted"] = "Projection / early-career profile"
        df.loc[recent_mask, "draft_slot_regrade"] = "Too Early"
        df.loc[recent_mask, "miss_flag"] = "neutral"
        df.loc[recent_mask, "value_flag"] = "neutral"

        # 2022/2023 can be shown, but should be marked developing.
        developing_mask = df["outcome_maturity"].eq("developing")
        df.loc[developing_mask, "actual_outcome_flag"] = (
            "Developing: " + df.loc[developing_mask, "actual_outcome_flag"].astype(str)
        )

        # Separate display score from actual outcome score.
        # This prevents 2024/2025 players from showing weird immature outcome grades.
        df["display_grade"] = df.apply(display_grade_for_site, axis=1)
        df["display_grade_label"] = df.apply(display_grade_label, axis=1)

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
