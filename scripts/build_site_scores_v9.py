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



PRIVATE_MASTER = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.csv")

CAREER_ENRICH_COLS = [
    "allpro",
    "probowls",
    "seasons_started",
    "w_av",
    "car_av",
    "dr_av",
    "games",
    "pass_completions",
    "pass_attempts",
    "pass_yards",
    "pass_tds",
    "pass_ints",
    "rush_atts",
    "rush_yards",
    "rush_tds",
    "receptions",
    "rec_yards",
    "rec_tds",
    "def_solo_tackles",
    "def_ints",
    "def_sacks",
]

def clean_name_key(s):
    return (
        s.astype(str)
         .str.lower()
         .str.replace(r"[^a-z0-9]+", " ", regex=True)
         .str.strip()
    )

def merge_career_enrichment(df):
    """
    Merge public-safe career stat columns from the private master into V9.
    Does not expose PFF columns. Only career outcomes/stats/awards.
    """
    if not PRIVATE_MASTER.exists():
        print(f"WARNING: private master missing: {PRIVATE_MASTER}")
        return df

    src = pd.read_csv(PRIVATE_MASTER, low_memory=False)

    name_col = "player_name_clean" if "player_name_clean" in src.columns else "player"
    year_col = "draft_year" if "draft_year" in src.columns else "season"

    if name_col not in src.columns or year_col not in src.columns:
        print("WARNING: could not find merge keys in private master")
        return df

    keep = [name_col, year_col]
    if "position" in src.columns:
        keep.append("position")

    keep += [c for c in CAREER_ENRICH_COLS if c in src.columns]
    src = src[keep].copy()

    src["_merge_name"] = clean_name_key(src[name_col])
    src["_merge_year"] = pd.to_numeric(src[year_col], errors="coerce")

    df = df.copy()
    df["_merge_name"] = clean_name_key(df["player"])
    df["_merge_year"] = pd.to_numeric(df["draft_year"], errors="coerce")

    src = src.drop_duplicates(["_merge_name", "_merge_year"], keep="first")

    merge_cols = ["_merge_name", "_merge_year"] + [c for c in CAREER_ENRICH_COLS if c in src.columns]
    enriched = df.merge(src[merge_cols], on=["_merge_name", "_merge_year"], how="left", suffixes=("", "_career"))

    # Fill/add career columns.
    for c in CAREER_ENRICH_COLS:
        career_c = f"{c}_career"
        if career_c in enriched.columns:
            if c in enriched.columns:
                enriched[c] = enriched[c].where(enriched[c].notna(), enriched[career_c])
            else:
                enriched[c] = enriched[career_c]
            enriched = enriched.drop(columns=[career_c], errors="ignore")

    enriched = enriched.drop(columns=["_merge_name", "_merge_year"], errors="ignore")

    return enriched

def as_num(row, col, default=0):
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def career_stat_raw(row):
    """
    Position-specific production raw score.
    This will later be converted to percentile within position group.
    """
    pos = str(row.get("position_group") or row.get("position")).upper().strip()

    pass_yards = as_num(row, "pass_yards")
    pass_tds = as_num(row, "pass_tds")
    pass_ints = as_num(row, "pass_ints")

    rush_yards = as_num(row, "rush_yards")
    rush_tds = as_num(row, "rush_tds")

    receptions = as_num(row, "receptions")
    rec_yards = as_num(row, "rec_yards")
    rec_tds = as_num(row, "rec_tds")

    sacks = as_num(row, "def_sacks")
    tackles = as_num(row, "def_solo_tackles")
    def_ints = as_num(row, "def_ints")

    if pos == "QB":
        return (
            np.log1p(pass_yards) * 8
            + np.sqrt(pass_tds) * 5
            - np.sqrt(max(pass_ints, 0)) * 1.5
            + np.log1p(rush_yards) * 2
            + np.sqrt(rush_tds) * 2
        )

    if pos == "RB":
        return (
            np.log1p(rush_yards) * 8
            + np.sqrt(rush_tds) * 5
            + np.log1p(rec_yards) * 3
            + np.sqrt(rec_tds) * 2
        )

    if pos in {"WR", "TE"}:
        return (
            np.log1p(rec_yards) * 9
            + np.sqrt(rec_tds) * 5
            + np.log1p(receptions) * 3
        )

    if pos in {"EDGE", "DE", "IDL", "DT", "LB"}:
        return (
            np.sqrt(sacks) * 9
            + np.log1p(tackles) * 4
            + np.sqrt(def_ints) * 3
        )

    if pos in {"CB", "S", "SAF", "DB"}:
        return (
            np.sqrt(def_ints) * 9
            + np.log1p(tackles) * 4
            + np.sqrt(sacks) * 2
        )

    return 0

def career_honors_raw(row):
    games = as_num(row, "games")
    starts = as_num(row, "seasons_started")
    probowls = as_num(row, "probowls")
    allpro = as_num(row, "allpro")

    return (
        np.sqrt(games) * 2.0
        + np.sqrt(starts) * 4.0
        + probowls * 8.0
        + allpro * 18.0
    )



def build_continuous_position_success(df):
    """
    Build a non-compressed historical success score.

    Percentile ranks make many successful players look like 99s.
    This uses a continuous raw impact score and scales it within position group.
    """
    work = df.copy()

    for c in [
        "actual_value", "games", "seasons_started", "probowls", "allpro",
        "career_stat_raw", "career_honors_raw", "draft_year"
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        else:
            work[c] = 0

    av = work["actual_value"].fillna(0).clip(lower=0)
    games = work["games"].fillna(0).clip(lower=0)
    starts = work["seasons_started"].fillna(0).clip(lower=0)
    probowls = work["probowls"].fillna(0).clip(lower=0)
    allpro = work["allpro"].fillna(0).clip(lower=0)
    stat_raw = work["career_stat_raw"].fillna(0).clip(lower=0)
    honors_raw = work["career_honors_raw"].fillna(0).clip(lower=0)

    # Raw career impact. This is intentionally continuous.
    # Log/sqrt transforms reduce extreme dominance but preserve separation.
    work["career_impact_raw"] = (
        np.log1p(av) * 28
        + np.sqrt(games) * 2.0
        + np.sqrt(starts) * 3.5
        + probowls * 7.5
        + allpro * 17.5
        + stat_raw * 0.35
        + honors_raw * 0.25
    )

    mature_mask = pd.to_numeric(work["draft_year"], errors="coerce").le(2022)

    def scale_group(s):
        s = pd.to_numeric(s, errors="coerce")
        valid = s.dropna()

        if len(valid) < 10:
            return pd.Series(np.nan, index=s.index)

        lo = valid.quantile(0.05)
        hi = valid.quantile(0.995)

        if hi <= lo:
            return pd.Series(np.nan, index=s.index)

        x = ((s - lo) / (hi - lo)).clip(lower=0, upper=1)

        # Football-readable curve:
        # lifts strong careers into realistic starter/star ranges
        # without flattening all elite players to 99.
        scaled = 35 + 64 * (x ** 0.72)

        return scaled.clip(lower=0, upper=99)

    work["position_success_score"] = (
        work.where(mature_mask)
            .groupby("position_group")["career_impact_raw"]
            .transform(scale_group)
            .round(1)
    )

    # Global version for cross-position class ranking/tiebreaks.
    valid = work.loc[mature_mask, "career_impact_raw"].dropna()
    if len(valid) > 10:
        lo = valid.quantile(0.05)
        hi = valid.quantile(0.995)
        if hi > lo:
            x = ((work["career_impact_raw"] - lo) / (hi - lo)).clip(0, 1)
            work["global_success_score"] = (35 + 64 * (x ** 0.72)).clip(0, 99).round(1)
        else:
            work["global_success_score"] = np.nan
    else:
        work["global_success_score"] = np.nan

    work.loc[~mature_mask, ["position_success_score", "global_success_score", "career_impact_raw"]] = np.nan

    return work[["career_impact_raw", "position_success_score", "global_success_score"]]


def actual_value_to_team(row):
    """
    Mature classes: actual value to team.

    Primary input is continuous position-relative career impact.
    This prevents every successful QB/star from being flattened to 99.
    """
    pos_success = row.get("position_success_score")
    grade = row.get("outcome_grade_pff_powered")

    if not pd.isna(pos_success):
        return round(max(0, min(99, float(pos_success))), 1)

    # Fallback if continuous success score is unavailable.
    pct = row.get("actual_position_percentile")
    av = row.get("actual_value")
    pos = row.get("position_group") or row.get("position")

    av_score = av_to_score(av)

    parts = []
    weights = []

    if not pd.isna(pct):
        parts.append(float(pct))
        weights.append(0.50)

    if not pd.isna(av_score):
        parts.append(float(av_score))
        weights.append(0.35)

    if not pd.isna(grade):
        parts.append(float(grade))
        weights.append(0.15)

    if not parts:
        return np.nan

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    base = float(np.dot(parts, weights))

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


def should_have_bucket_from_rank(row):
    """
    Convert redraft class rank into a should-have-gone bucket.

    Uses class rank as the main source of truth, then applies small guardrails
    for premium positions so top positional outcomes do not get under-labeled.
    """
    rank = row.get("redraft_class_rank")
    pos_rank = row.get("redraft_position_rank")
    pos = str(row.get("position_group") or row.get("position")).upper().strip()
    score = row.get("site_display_score")

    if pd.isna(rank):
        return "Unknown"

    rank = float(rank)
    pr = np.nan if pd.isna(pos_rank) else float(pos_rank)
    sc = np.nan if pd.isna(score) else float(score)

    # Base class-rank bucket.
    if rank <= 5:
        bucket = "Top-5 caliber"
    elif rank <= 15:
        bucket = "Top-15 caliber"
    elif rank <= 32:
        bucket = "Round 1 caliber"
    elif rank <= 64:
        bucket = "Round 2 caliber"
    elif rank <= 100:
        bucket = "Day 2 caliber"
    elif rank <= 150:
        bucket = "Early Day 3 caliber"
    elif rank <= 220:
        bucket = "Late Day 3 / priority depth"
    else:
        bucket = "Undrafted / replacement outcome"

    # Premium-position guardrails.
    # QB value in particular means a high-end QB outcome belongs in Round 1
    # even if raw score is not elite on an all-position scale.
    if pos == "QB":
        if not pd.isna(pr):
            if pr <= 1 and rank <= 20:
                bucket = "Top-5 caliber"
            elif pr <= 2 and rank <= 24:
                bucket = "Top-15 caliber"
            elif pr <= 3 and rank <= 40:
                bucket = "Round 1 caliber"

        # If QB score is solid and class rank is Round 1-ish, never call him Round 2.
        if not pd.isna(sc) and sc >= 75 and rank <= 40:
            if bucket in {"Round 2 caliber", "Day 2 caliber", "Early Day 3 caliber"}:
                bucket = "Round 1 caliber"

    elif pos in {"EDGE", "DE", "OT", "T", "OL", "WR", "CB"}:
        if not pd.isna(pr):
            if pr <= 1 and rank <= 20:
                bucket = "Top-15 caliber"
            elif pr <= 3 and rank <= 40:
                bucket = "Round 1 caliber"

    return bucket


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
    if pick <= 5 and rank > 64:
        return "Major overdraft"
    if pick <= 15 and rank > 100:
        return "Major overdraft"
    if pick <= 32 and rank > 140:
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

    # Bring in public-safe career stats/awards from private master.
    df = merge_career_enrichment(df)

    for c in [
        "draft_year", "pick", "actual_value", "actual_position_percentile",
        "outcome_grade_pff_powered", "games", "seasons_started", "probowls",
        "allpro", "w_av", "car_av", "dr_av", "pass_yards", "pass_tds",
        "pass_ints", "rush_yards", "rush_tds", "receptions", "rec_yards",
        "rec_tds", "def_solo_tackles", "def_ints", "def_sacks"
    ]:
        if c in df.columns:
            df[c] = n(df[c])

    # Position-relative stat and honors/longevity percentiles for mature comparisons.
    df["career_stat_raw"] = df.apply(career_stat_raw, axis=1)
    df["career_honors_raw"] = df.apply(career_honors_raw, axis=1)

    df["career_stat_score"] = (
        df.groupby("position_group")["career_stat_raw"]
          .rank(pct=True, method="average")
          .mul(100)
          .clip(upper=99)
          .round(1)
    )

    df["career_honors_score"] = (
        df.groupby("position_group")["career_honors_raw"]
          .rank(pct=True, method="average")
          .mul(100)
          .clip(upper=99)
          .round(1)
    )

    # Non-compressed mature historical success scores.
    continuous_success = build_continuous_position_success(df)
    df["career_impact_raw"] = continuous_success["career_impact_raw"]
    df["position_success_score"] = continuous_success["position_success_score"]
    df["global_success_score"] = continuous_success["global_success_score"]

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

    # Mature historical display score:
    # blend position-relative success with global impact.
    # Do NOT use old prospect/model grade here; mature players should be judged
    # by what actually happened, not what they looked like before the draft.
    mature_mask = df["score_mode"].eq("actual_value")

    pos_score = pd.to_numeric(df.get("position_success_score"), errors="coerce")
    global_score = pd.to_numeric(df.get("global_success_score"), errors="coerce")

    blended_actual = (
        pos_score.fillna(df["site_display_score"]) * 0.65
        + global_score.fillna(pos_score).fillna(df["site_display_score"]) * 0.35
    )

    df.loc[mature_mask, "site_display_score"] = blended_actual.loc[mature_mask].clip(0, 99).round(1)

    # Bust compression: low-career-value players should not float into the 50s/60s
    # just because position percentiles or old projection signals were generous.
    av = pd.to_numeric(df.get("actual_value"), errors="coerce")
    games = pd.to_numeric(df.get("games"), errors="coerce")
    probowls = pd.to_numeric(df.get("probowls"), errors="coerce").fillna(0)
    allpro = pd.to_numeric(df.get("allpro"), errors="coerce").fillna(0)

    no_honors = probowls.eq(0) & allpro.eq(0)

    df.loc[mature_mask & no_honors & av.le(3), "site_display_score"] = df.loc[
        mature_mask & no_honors & av.le(3), "site_display_score"
    ].clip(upper=32)

    df.loc[mature_mask & no_honors & av.le(8) & games.le(40), "site_display_score"] = df.loc[
        mature_mask & no_honors & av.le(8) & games.le(40), "site_display_score"
    ].clip(upper=42)

    df.loc[mature_mask & no_honors & av.le(15) & games.le(70), "site_display_score"] = df.loc[
        mature_mask & no_honors & av.le(15) & games.le(70), "site_display_score"
    ].clip(upper=52)

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

    # Recompute should-have-gone bucket from class-aware rank plus positional context.
    # This fixes cases like Josh Allen / Baker being under-labeled as Round 2.
    df["should_have_been_drafted"] = df.apply(should_have_bucket_from_rank, axis=1)
    df["should_have_pick_bucket"] = df["should_have_been_drafted"].map(bucket_pick)

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
