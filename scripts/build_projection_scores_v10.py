#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MASTER = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.csv")
V9 = Path("site_data/player_cards_v9.csv")

OUT_SITE = Path("site_data/player_cards_v10.csv")
OUT_DOCS = Path("docs/data/player_cards_v10.csv")
REPORT = Path("reports/v10_projection_feature_report.json")

TRAIN_END_YEAR = 2022


def clean_name(s):
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def choose_year_col(df):
    if "draft_year" in df.columns:
        return "draft_year"
    if "season" in df.columns:
        return "season"
    raise ValueError("No draft_year/season column found")


def choose_name_col(df):
    for c in ["player_name_clean", "player", "player_name", "name", "pfr_player_name"]:
        if c in df.columns:
            return c
    raise ValueError("No player name column found")


def candidate_feature_columns(df):
    """
    Projection-safe features only.

    Avoid final NFL outcome leakage:
    - no w_av / dr_av / car_av
    - no final games / seasons_started / probowls / allpro
    - no raw career NFL pass/rush/rec/sack outcome columns

    Allow:
    - draft capital
    - combine / size / athletic testing
    - college/PFF aggregates
    - limited early-NFL windows for 2023/2024
    """
    include_prefixes = [
        "pff_avg_passing_",
        "pff_avg_rushing_",
        "pff_avg_receiving_",
        "pff_avg_sacks_suffered",
        "pff_avg_pacr",
        "pff_avg_adot",
        "pff_avg_air_yards",
        "pff_avg_target_share",
        "pff_avg_combine_",
        "pff_avg_consensus_",
        "pff_avg_mock_",
        "combine_",
    ]

    include_exact = [
        "round",
        "pick",
        "season",
        "draft_year",
        "pff_avg_draft_round",
        "pff_avg_draft_pick",
        "player_draft_round",
        "player_draft_pick",

        # limited early NFL evidence
        "pff_avg_games_y1",
        "pff_avg_games_y2",
        "pff_avg_games_y3",
        "pff_avg_offensive_snaps_y1",
        "pff_avg_defensive_snaps_y1",
        "pff_avg_st_snaps_y1",
        "pff_avg_offensive_snaps_y2",
        "pff_avg_defensive_snaps_y2",
        "pff_avg_st_snaps_y2",
        "pff_avg_max_off_snap_pct_y1",
        "pff_avg_max_def_snap_pct_y1",
        "pff_avg_max_off_snap_pct_y2",
        "pff_avg_max_def_snap_pct_y2",
        "pff_avg_snap_share_starter_by_y3",
        "pff_avg_played_y1",
        "pff_avg_played_y2",
        "pff_avg_played_y3",
    ]

    exclude_contains = [
        # final NFL career outcomes / leakage
        "w_av",
        "car_av",
        "dr_av",
        "career_av",
        "weighted_av",
        "av_with_drafting_team",
        "probowls",
        "pro_bowls",
        "allpro",
        "all_pro",
        "hof",
        "career_games",
        "career_seasons",
        "seasons_started",
        "starter_any_season_career",
        "any_pro_bowl_career",
        "any_all_pro_career",
        "actual_value",
        "site_display_score",
        "redraft",
        "bust_relative",
        "steal_relative",
        "surplus_av",
        "expected_av",

        # raw NFL career stat outcomes
        "pass_completions",
        "pass_attempts",
        "pass_yards",
        "pass_tds",
        "pass_ints",
        "rush_atts",
        "rush_yards",
        "rush_tds",
        "receptions_x",
        "rec_yards",
        "rec_tds",
        "def_solo_tackles",
        "def_ints",
        "def_sacks",
        "games_recorded",
    ]

    features = []

    for c in df.columns:
        lc = c.lower()

        if any(x in lc for x in exclude_contains):
            continue

        if c in include_exact or any(lc.startswith(prefix) for prefix in include_prefixes):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() >= 0.03 and s.nunique(dropna=True) > 3:
                features.append(c)

    return sorted(set(features))


def position_group_col(df):
    if "position_group" in df.columns:
        return "position_group"
    if "position_group_model" in df.columns:
        return "position_group_model"
    if "pff_avg_position_group_model" in df.columns:
        return "pff_avg_position_group_model"
    if "position" in df.columns:
        return "position"
    raise ValueError("No position column")


def blend_by_year(year, projected, current):
    """
    Projection-first blend.

    2023 has real NFL evidence, but it is still too early to let current success
    dominate the long-term projection.
    2024 has even less evidence.
    2025+ should be almost entirely historical trait / prospect similarity.
    """
    if pd.isna(year):
        return projected

    year = int(float(year))

    if pd.isna(current):
        return projected

    if year == 2023:
        return 0.70 * projected + 0.30 * current
    if year == 2024:
        return 0.80 * projected + 0.20 * current
    if year >= 2025:
        return 0.95 * projected + 0.05 * current

    return projected


def tier(score):
    if pd.isna(score):
        return "Unknown"
    score = float(score)
    if score >= 93:
        return "Elite / rare profile"
    if score >= 86:
        return "High-end starter profile"
    if score >= 78:
        return "Starter-caliber profile"
    if score >= 68:
        return "Rotational / contributor profile"
    if score >= 55:
        return "Depth / replacement profile"
    return "Low-probability profile"



def calibrate_projection_score(row):
    """
    Final V10 projection calibration.

    Purpose:
    - 2023/2024 can use early NFL evidence.
    - 2025+ should use prospect similarity plus draft capital floors.
    - Avoid burying premium top picks with strong historical profiles.
    """
    score = row.get("v10_projection_score")
    trait = row.get("v10_trait_projection_score")
    current = row.get("current_value_to_team")
    year = row.get("draft_year")
    pick = row.get("pick")
    pos = str(row.get("position_group_v10") or row.get("position") or "").upper().strip()

    if pd.isna(score):
        return score

    score = float(score)
    trait = score if pd.isna(trait) else float(trait)

    yr = np.nan if pd.isna(year) else int(float(year))
    pk = np.nan if pd.isna(pick) else float(pick)

    premium = pos in {"QB", "OT", "OL", "T", "EDGE", "DE", "WR", "CB", "TE"}

    # 2023/2024 early NFL evidence booster.
    # Keep this modest: early production can confirm a projection, but should
    # not dominate the historical trait/similarity model.
    if yr in {2023, 2024} and not pd.isna(current):
        cur = float(current)
        if cur >= 84:
            score = max(score, 0.80 * score + 0.20 * cur)
        elif cur >= 80:
            score = max(score, 0.88 * score + 0.12 * cur)

    # Draft capital / premium-profile floors.
    # These are not final career judgments; they are projection floors.
    if not pd.isna(pk):
        if premium:
            if pk <= 5:
                score = max(score, 80)
            elif pk <= 10:
                score = max(score, 78)
            elif pk <= 15:
                score = max(score, 76)
            elif pk <= 32:
                score = max(score, 72)

            # QB-specific premium floor.
            # Require trait support so raw draft slot does not over-protect risky QBs.
            if pos == "QB":
                if pk <= 3:
                    if trait >= 78:
                        score = max(score, 80)
                    elif trait >= 74:
                        score = max(score, 77)
                elif pk <= 10:
                    if trait >= 78:
                        score = max(score, 77)
                    elif trait >= 74:
                        score = max(score, 75)
        else:
            if pk <= 5:
                score = max(score, 77)
            elif pk <= 15:
                score = max(score, 74)
            elif pk <= 32:
                score = max(score, 70)

    # If the trait model itself is strong, preserve that.
    if trait >= 82:
        score = max(score, trait - 1)
    elif trait >= 78:
        score = max(score, trait - 2)

    # Projection-first QB risk cap:
    # if the historical trait model is below starter level, do not let draft slot
    # alone push a QB into a clean starter-caliber score.
    if pos == "QB" and trait < 76:
        score = min(score, 77)

    # Special case: late-round 2023+ players with real current hit evidence.
    # Still projection-first: give some credit, but do not fully convert early
    # production into a mature-career grade.
    if yr in {2023, 2024} and not pd.isna(current):
        cur = float(current)
        if cur >= 86 and score < 78:
            score = max(score, 78)
        elif cur >= 82 and score < 74:
            score = max(score, 74)

    return round(max(0, min(99, score)), 1)




def bucket_from_rank(rank):
    if pd.isna(rank):
        return "Unknown"
    rank = float(rank)
    if rank <= 5:
        return "Top-5 caliber"
    if rank <= 15:
        return "Top-15 caliber"
    if rank <= 32:
        return "Round 1 caliber"
    if rank <= 64:
        return "Round 2 caliber"
    if rank <= 100:
        return "Day 2 caliber"
    if rank <= 150:
        return "Early Day 3 caliber"
    if rank <= 220:
        return "Late Day 3 / priority depth"
    return "Undrafted / replacement outcome"


def projection_rank_label(row):
    parts = []
    cr = row.get("redraft_class_rank")
    pr = row.get("redraft_position_rank")
    pg = row.get("position_group")

    if not pd.isna(cr):
        parts.append(f"Class #{int(cr)}")
    if not pd.isna(pr) and str(pg) not in {"", "nan", "None"}:
        parts.append(f"{pg} #{int(pr)}")

    return " / ".join(parts) if parts else row.get("should_have_been_drafted", "—")


def projection_regrade(row):
    """
    For 2023+ projection mode, do not call anything a final miss.
    Compare actual pick to V10 projected class rank.
    """
    pick = row.get("pick")
    rank = row.get("redraft_class_rank")

    if pd.isna(rank):
        return "Too Early"

    if pd.isna(pick):
        return "Projected undrafted / unknown"

    pick = float(pick)
    rank = float(rank)

    if rank <= 15 and pick > 64:
        return "Projected major value"
    if rank <= 32 and pick > 100:
        return "Projected major value"
    if rank <= 64 and pick > 150:
        return "Projected value"
    if rank + 40 < pick:
        return "Projected drafted too late"

    if pick <= 5 and rank > 50:
        return "Projected major overdraft risk"
    if pick <= 15 and rank > 100:
        return "Projected overdraft risk"
    if pick + 40 < rank:
        return "Projected drafted too early"

    return "Projected about right"



def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"Missing private master: {MASTER}")
    if not V9.exists():
        raise FileNotFoundError(f"Missing V9 file: {V9}")

    master = pd.read_csv(MASTER, low_memory=False)
    v9 = pd.read_csv(V9, low_memory=False)

    name_col = choose_name_col(master)
    year_col = choose_year_col(master)

    master = master.copy()
    master["_merge_name"] = clean_name(master[name_col])
    master["_merge_year"] = pd.to_numeric(master[year_col], errors="coerce")

    v9 = v9.copy()
    v9["_merge_name"] = clean_name(v9["player"])
    v9["_merge_year"] = pd.to_numeric(v9["draft_year"], errors="coerce")

    keep_v9 = [
        "_merge_name",
        "_merge_year",
        "player",
        "site_display_score",
        "score_mode",
        "current_value_to_team",
        "prospect_similarity_score",
        "redraft_class_rank",
        "redraft_position_rank",
        "draft_slot_regrade",
        "actual_outcome_flag",
    ]
    keep_v9 = [c for c in keep_v9 if c in v9.columns]

    base = master.merge(
        v9[keep_v9],
        on=["_merge_name", "_merge_year"],
        how="left",
        suffixes=("", "_v9"),
    )

    if "player_v9" in base.columns:
        base["player"] = base["player_v9"].fillna(base.get("player", base[name_col]))

    if "player" not in base.columns:
        base["player"] = base[name_col]

    base["draft_year"] = base["_merge_year"]
    pg_col = position_group_col(base)
    base["position_group_v10"] = base[pg_col].astype(str)

    feature_cols = candidate_feature_columns(base)

    target = pd.to_numeric(base["site_display_score"], errors="coerce")
    years = pd.to_numeric(base["draft_year"], errors="coerce")

    train_mask = years.le(TRAIN_END_YEAR) & target.notna()
    project_mask = years.ge(2023)

    X = base[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = target

    if train_mask.sum() < 500:
        raise ValueError(f"Too few training rows: {train_mask.sum()}")
    if len(feature_cols) < 5:
        raise ValueError(f"Too few features: {len(feature_cols)}")

    gbr = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.045,
            max_leaf_nodes=20,
            l2_regularization=0.15,
            random_state=42,
        ),
    )

    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    )

    gbr.fit(X.loc[train_mask], y.loc[train_mask])
    rf.fit(X.loc[train_mask], y.loc[train_mask])

    pred_gbr = gbr.predict(X)
    pred_rf = rf.predict(X)

    base["v10_trait_projection_score"] = (0.65 * pred_gbr + 0.35 * pred_rf).clip(0, 99).round(1)

    # Similarity comps
    scaler = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_scaled = scaler.fit_transform(X)

    successful_hist = base.loc[train_mask].copy()
    successful_hist["_hist_success"] = pd.to_numeric(successful_hist["site_display_score"], errors="coerce")

    comp_strings = []
    risk_strings = []

    for idx in base.index:
        if not bool(project_mask.loc[idx]):
            comp_strings.append("")
            risk_strings.append("")
            continue

        pos = str(base.loc[idx, "position_group_v10"])
        same_pos = successful_hist["position_group_v10"].astype(str).eq(pos)

        hist_subset = successful_hist.loc[same_pos].copy()
        if len(hist_subset) < 20:
            hist_subset = successful_hist.copy()

        hist_subset_idx = hist_subset.index.to_numpy()

        sims = cosine_similarity(X_scaled[idx].reshape(1, -1), X_scaled[hist_subset_idx])[0]
        hist_subset["_sim"] = sims

        good = hist_subset[hist_subset["_hist_success"] >= 78].sort_values("_sim", ascending=False).head(5)
        bad = hist_subset[hist_subset["_hist_success"] < 55].sort_values("_sim", ascending=False).head(5)

        def fmt(rows):
            vals = []
            for _, r in rows.iterrows():
                nm = r.get("player") or r.get(name_col)
                yr = int(r.get("draft_year")) if not pd.isna(r.get("draft_year")) else "?"
                sc = r.get("_hist_success")
                vals.append(f"{nm} ({yr}, score {sc:.1f})")
            return " | ".join(vals)

        comp_strings.append(fmt(good))
        risk_strings.append(fmt(bad))

    base["v10_successful_comps"] = comp_strings
    base["v10_risk_comps"] = risk_strings

    current = pd.to_numeric(base.get("current_value_to_team"), errors="coerce")
    base["v10_projection_score"] = [
        blend_by_year(y, p, c)
        for y, p, c in zip(base["draft_year"], base["v10_trait_projection_score"], current)
    ]
    base["v10_projection_score"] = pd.to_numeric(base["v10_projection_score"], errors="coerce").clip(0, 99).round(1)

    # Final projection calibration for 2023+.
    base.loc[project_mask, "v10_projection_score"] = base.loc[project_mask].apply(
        calibrate_projection_score,
        axis=1
    )

    base["site_display_score_v10"] = pd.to_numeric(base["site_display_score"], errors="coerce")
    base.loc[project_mask, "site_display_score_v10"] = base.loc[project_mask, "v10_projection_score"]

    base["site_display_score_label_v10"] = np.where(
        project_mask,
        np.where(base["draft_year"].le(2024), "Projection + Early NFL Score", "Prospect Similarity Score"),
        "Actual Value Score",
    )

    base["v10_projection_tier"] = base["site_display_score_v10"].map(tier)

    # Merge back onto V9 site shape.
    out = v9.drop(columns=["_merge_name", "_merge_year"], errors="ignore").copy()

    add_cols = [
        "player",
        "draft_year",
        "v10_trait_projection_score",
        "v10_projection_score",
        "v10_successful_comps",
        "v10_risk_comps",
        "site_display_score_v10",
        "site_display_score_label_v10",
        "v10_projection_tier",
    ]

    bridge = base[["_merge_name", "_merge_year"] + add_cols].copy()
    bridge = bridge.drop_duplicates(["_merge_name", "_merge_year"], keep="first")

    out["_merge_name"] = clean_name(out["player"])
    out["_merge_year"] = pd.to_numeric(out["draft_year"], errors="coerce")

    out = out.merge(bridge, on=["_merge_name", "_merge_year"], how="left", suffixes=("", "_new"))

    for c in add_cols:
        newc = c + "_new"
        if newc in out.columns:
            if c in out.columns:
                out[c] = out[newc].combine_first(out[c])
            else:
                out[c] = out[newc]
            out = out.drop(columns=[newc], errors="ignore")

    future_mask = pd.to_numeric(out["draft_year"], errors="coerce").ge(2023)

    out.loc[future_mask & out["site_display_score_v10"].notna(), "site_display_score"] = out.loc[
        future_mask & out["site_display_score_v10"].notna(), "site_display_score_v10"
    ]

    out.loc[future_mask & out["site_display_score_label_v10"].notna(), "site_display_score_label"] = out.loc[
        future_mask & out["site_display_score_label_v10"].notna(), "site_display_score_label_v10"
    ]

    out["display_grade"] = out["site_display_score"]
    out["display_grade_label"] = out["site_display_score_label"]

    # Recompute projected class/position ranks for 2023+ using V10 scores.
    future_mask = pd.to_numeric(out["draft_year"], errors="coerce").ge(2023)
    out["_rank_year"] = pd.to_numeric(out["draft_year"], errors="coerce")
    out["_rank_score_v10"] = pd.to_numeric(out["site_display_score"], errors="coerce")

    out.loc[future_mask, "redraft_class_rank"] = (
        out.loc[future_mask]
           .groupby("_rank_year")["_rank_score_v10"]
           .rank(method="first", ascending=False)
           .round()
    )

    out.loc[future_mask, "redraft_position_rank"] = (
        out.loc[future_mask]
           .groupby(["_rank_year", "position_group"])["_rank_score_v10"]
           .rank(method="first", ascending=False)
           .round()
    )

    out.loc[future_mask, "should_have_been_drafted"] = out.loc[
        future_mask, "redraft_class_rank"
    ].map(bucket_from_rank)

    out.loc[future_mask, "should_have_pick_bucket"] = out.loc[
        future_mask, "should_have_been_drafted"
    ].map({
        "Top-5 caliber": 5,
        "Top-15 caliber": 15,
        "Round 1 caliber": 32,
        "Round 2 caliber": 64,
        "Day 2 caliber": 100,
        "Early Day 3 caliber": 150,
        "Late Day 3 / priority depth": 220,
        "Undrafted / replacement outcome": 999,
        "Unknown": 999,
    })

    out.loc[future_mask, "should_have_been_drafted_display"] = out.loc[
        future_mask
    ].apply(projection_rank_label, axis=1)

    out.loc[future_mask, "draft_slot_regrade"] = out.loc[
        future_mask
    ].apply(projection_regrade, axis=1)

    out.loc[future_mask, "actual_outcome_flag"] = out.loc[
        future_mask, "draft_slot_regrade"
    ]

    out = out.drop(columns=["_rank_year", "_rank_score_v10"], errors="ignore")

    out["v10_projection_explanation"] = ""
    out.loc[future_mask, "v10_projection_explanation"] = (
        "Projected from historical 1980-2022 players with similar college/PFF/combine/draft profiles "
        "plus limited early NFL evidence where available."
    )

    out = out.drop(columns=["_merge_name", "_merge_year"], errors="ignore")

    # Clean display formatting: avoid years/ranks showing as 2025.0 / 9.0 on site.
    int_like_cols = [
        "draft_year",
        "round",
        "pick",
        "redraft_class_rank",
        "redraft_position_rank",
        "should_have_pick_bucket",
    ]

    for c in int_like_cols:
        if c in out.columns:
            vals = pd.to_numeric(out[c], errors="coerce")
            out[c] = vals.map(lambda x: "" if pd.isna(x) else str(int(round(x))))

    OUT_SITE.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_SITE, index=False)
    out.to_csv(OUT_DOCS, index=False)

    report = {
        "train_rows": int(train_mask.sum()),
        "project_rows_2023_plus": int(project_mask.sum()),
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }
    REPORT.write_text(json.dumps(report, indent=2))

    print(f"WROTE: {OUT_SITE}")
    print(f"WROTE: {OUT_DOCS}")
    print(f"WROTE: {REPORT}")
    print(f"Train rows: {train_mask.sum():,}")
    print(f"Project rows 2023+: {project_mask.sum():,}")
    print(f"Feature count: {len(feature_cols):,}")


if __name__ == "__main__":
    main()
