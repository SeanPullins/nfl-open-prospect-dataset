#!/usr/bin/env python3

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PRIVATE_FILE = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.csv")

OUT_DIR = Path("site_data")
REPORTS = Path("reports")
PREDS = Path("predictions")

for p in [OUT_DIR, REPORTS, PREDS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET_CANDIDATES = ["w_av", "weighted_av", "car_av", "career_av"]
YEAR_CANDIDATES = ["season", "draft_year", "year"]
NAME_CANDIDATES = ["player_name_clean", "player_name", "pfr_player_name", "name", "display_name"]
POS_CANDIDATES = ["position", "pos", "player_position", "combine_pos"]

BASE_NUMERIC_FEATURES = [
    "round",
    "pick",
    "overall_pick",
    "draft_ovr",
    "age",
    "player_height",
    "player_weight",
    "combine_ht",
    "combine_wt",
    "combine_forty",
    "combine_bench",
    "combine_vertical",
    "combine_broad_jump",
    "combine_cone",
    "combine_shuttle",
]

RANK_HINTS = [
    "consensus",
    "rank",
    "big_board",
    "board",
    "overall",
    "jacklich",
]

OUTCOME_PATTERNS = [
    r"^w_av$",
    r"weighted_av",
    r"^car_av$",
    r"career_av",
    r"draft_team_av",
    r"^dr_av$",
    r"allpro",
    r"all_pro",
    r"probowls",
    r"pro_bowls",
    r"hof",
    r"hall_of_fame",
    r"actual_outcome",
    r"residual",
    r"outlier",
    r"projection",
    r"score_0_100",
    r"v4_",
    r"v5_",
    r"v6_",
    r"v7_",
    r"v8_",
]


def log(x):
    print(x, flush=True)


def first_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None


def normalize_pos(x):
    if pd.isna(x):
        return "UNK"

    p = str(x).upper().strip().replace(" ", "")

    if p == "QB":
        return "QB"
    if p in {"RB", "HB", "FB"}:
        return "RB"
    if p in {"WR", "FL", "SE"}:
        return "WR"
    if p == "TE":
        return "TE"
    if p in {"OT", "T", "LT", "RT"}:
        return "OT"
    if p in {"OG", "G", "LG", "RG", "C"}:
        return "IOL"
    if p == "OL":
        return "OL"
    if p in {"DE", "EDGE", "OLB/DE", "DE/OLB"}:
        return "EDGE"
    if p in {"DT", "NT", "DL"}:
        return "IDL"
    if p in {"LB", "ILB", "MLB", "OLB"}:
        return "LB"
    if p in {"CB", "DB"}:
        return "CB"
    if p in {"S", "SS", "FS", "SAF"}:
        return "S"
    if p in {"K", "PK", "P", "LS"}:
        return "ST"

    return p


def height_or_float(x):
    if pd.isna(x):
        return np.nan

    txt = str(x).strip()
    if not txt or txt.lower() in {"nan", "none", "null"}:
        return np.nan

    m = re.match(r"^(\d+)\s*-\s*(\d{1,2})$", txt)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))

    m = re.match(r"^(\d+)\s*'\s*(\d{1,2})", txt)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))

    txt = txt.replace(",", "")
    txt = re.sub(r"[^0-9.\-]+", "", txt)

    try:
        return float(txt)
    except Exception:
        return np.nan


def is_outcome_col(c):
    c = str(c).lower()
    return any(re.search(p, c) for p in OUTCOME_PATTERNS)


def choose_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 500:
                return c
    raise ValueError("No usable target found.")


def select_features(df, target, year_col):
    features = []

    for c in BASE_NUMERIC_FEATURES:
        if c in df.columns and c != target and c != year_col:
            features.append(c)

    for c in df.columns:
        c_low = c.lower()

        if c == target or c == year_col:
            continue
        if is_outcome_col(c):
            continue

        # PFF-powered features.
        if "pff" in c_low:
            vals = df[c].map(height_or_float)
            if vals.notna().sum() >= 100:
                features.append(c)

    return list(dict.fromkeys(features))


def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(height_or_float)
    return out


def make_regressor(feature_cols, leaf=6):
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), feature_cols),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        n_estimators=900,
        min_samples_leaf=leaf,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("prep", prep), ("model", model)])


def make_classifier(feature_cols):
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), feature_cols),
        ],
        remainder="drop",
    )

    model = ExtraTreesClassifier(
        n_estimators=900,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline([("prep", prep), ("model", model)])


def fit_classifier_if_possible(X, y, feature_cols):
    y = pd.Series(y).astype(int)
    if y.nunique() < 2:
        return None
    pipe = make_classifier(feature_cols)
    pipe.fit(X[feature_cols], y)
    return pipe


def class_prob_or_base(pipe, X, base_rate):
    if pipe is None:
        return np.repeat(base_rate, len(X))
    probs = pipe.predict_proba(X)
    if probs.shape[1] == 1:
        return np.repeat(base_rate, len(X))
    return probs[:, 1]


def percentile(value, ref):
    vals = np.sort(np.asarray(pd.Series(ref).dropna()))
    if len(vals) == 0 or pd.isna(value):
        return np.nan
    return round(100 * (np.searchsorted(vals, value, side="right") / len(vals)), 1)


def tier(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 95:
        return "Elite / rare projection"
    if score >= 88:
        return "High-end starter / premium prospect projection"
    if score >= 75:
        return "Starter-caliber projection"
    if score >= 58:
        return "Rotational / contributor projection"
    if score >= 40:
        return "Depth / developmental projection"
    return "Low-probability projection"


def find_rank_proxy(df):
    candidates = []

    for c in df.columns:
        c_low = c.lower()
        if any(h in c_low for h in RANK_HINTS):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                candidates.append((c, vals.notna().sum(), vals.median()))

    # Prefer columns that look like rankings and have values.
    candidates = sorted(candidates, key=lambda x: (-x[1], x[2] if pd.notna(x[2]) else 999999))
    return candidates[0][0] if candidates else None


def pick_to_round(pick):
    if pd.isna(pick):
        return np.nan
    pick = float(pick)
    if pick <= 32:
        return 1
    if pick <= 64:
        return 2
    if pick <= 100:
        return 3
    if pick <= 140:
        return 4
    if pick <= 180:
        return 5
    if pick <= 220:
        return 6
    return 7


def explanation(row):
    bits = []

    rank = row.get("projection_input_pick")
    if not pd.isna(rank):
        rank = float(rank)
        if rank <= 15:
            bits.append("premium projected draft slot")
        elif rank <= 50:
            bits.append("top-50 projected draft slot")
        elif rank <= 100:
            bits.append("Day 2 projected draft slot")
        else:
            bits.append("later projected draft slot")

    if row.get("starter_probability", 0) >= 0.70:
        bits.append("strong starter probability")
    elif row.get("starter_probability", 0) <= 0.30:
        bits.append("low starter probability")

    if row.get("elite_probability", 0) >= 0.35:
        bits.append("meaningful elite ceiling")

    if row.get("bust_probability", 0) >= 0.55:
        bits.append("elevated bust risk")

    bits.append("PFF-powered model projection")

    return "; ".join(bits)


def main():
    if not PRIVATE_FILE.exists():
        raise FileNotFoundError(f"Missing private file: {PRIVATE_FILE}")

    df = pd.read_csv(PRIVATE_FILE, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    target = choose_target(df)
    year_col = first_col(df, YEAR_CANDIDATES)
    name_col = first_col(df, NAME_CANDIDATES)
    pos_col = first_col(df, POS_CANDIDATES)

    if not year_col or not name_col or not pos_col:
        raise ValueError("Missing required year/name/position column.")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    df["projection_position_group"] = df[pos_col].map(normalize_pos)
    df["projection_model_group"] = df["projection_position_group"]
    df.loc[df["projection_position_group"].isin(["CB", "S"]), "projection_model_group"] = "DB"

    # Identify 2026 class.
    class_2026 = df[df[year_col].eq(2026)].copy()

    if class_2026.empty:
        # Fallback: any unlabeled future-ish rows.
        class_2026 = df[df[target].isna() & df[year_col].ge(2026)].copy()

    if class_2026.empty:
        raise SystemExit("No 2026 rows found. Need to add/import 2026 prospect rows first.")

    labeled = df[df[target].notna() & df[year_col].notna()].copy()
    labeled = labeled[labeled["projection_position_group"].ne("UNK")].copy()

    # Historical outcome percentile within model group.
    labeled["actual_position_percentile"] = np.nan
    for group, idx in labeled.groupby("projection_model_group").groups.items():
        vals = labeled.loc[idx, target]
        labeled.loc[idx, "actual_position_percentile"] = vals.rank(pct=True, method="average") * 100

    labeled["target_starter"] = (labeled["actual_position_percentile"] >= 55).astype(int)
    labeled["target_elite"] = (labeled["actual_position_percentile"] >= 85).astype(int)
    labeled["target_bust"] = (labeled["actual_position_percentile"] <= 30).astype(int)

    # Proxy draft slot for 2026 if actual pick is empty.
    rank_col = find_rank_proxy(class_2026)

    class_2026["projection_rank_source"] = rank_col if rank_col else ""

    if "pick" not in df.columns:
        df["pick"] = np.nan
        labeled["pick"] = np.nan
        class_2026["pick"] = np.nan

    if "round" not in df.columns:
        df["round"] = np.nan
        labeled["round"] = np.nan
        class_2026["round"] = np.nan

    class_2026["projection_input_pick"] = pd.to_numeric(class_2026.get("pick"), errors="coerce")

    if rank_col:
        rank_vals = pd.to_numeric(class_2026[rank_col], errors="coerce")
        class_2026["projection_input_pick"] = class_2026["projection_input_pick"].fillna(rank_vals)

    class_2026["projection_input_round"] = pd.to_numeric(class_2026.get("round"), errors="coerce")
    class_2026["projection_input_round"] = class_2026["projection_input_round"].fillna(
        class_2026["projection_input_pick"].map(pick_to_round)
    )

    # Feed proxy into model's pick/round features.
    class_2026["pick"] = class_2026["pick"].fillna(class_2026["projection_input_pick"])
    class_2026["round"] = class_2026["round"].fillna(class_2026["projection_input_round"])

    feature_cols = select_features(labeled, target, year_col)
    pff_cols = [c for c in feature_cols if "pff" in c.lower()]

    print(f"Loaded rows: {len(df):,}")
    print(f"Historical labeled rows: {len(labeled):,}")
    print(f"2026 rows to project: {len(class_2026):,}")
    print(f"Target: {target}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"PFF feature count: {len(pff_cols)}")
    print(f"Rank proxy column: {rank_col}")

    preds = []

    for group, train_g in labeled.groupby("projection_model_group"):
        test_g = class_2026[class_2026["projection_model_group"].eq(group)].copy()

        if test_g.empty:
            continue

        if len(train_g) < 60:
            print(f"Skipping {group}: train rows too small")
            continue

        train_g = coerce_numeric(train_g, feature_cols)
        test_g = coerce_numeric(test_g, feature_cols)

        X_train = train_g[feature_cols]
        X_test = test_g[feature_cols]

        pct_pipe = make_regressor(feature_cols, leaf=6)
        pct_pipe.fit(X_train, train_g["actual_position_percentile"])

        value_pipe = make_regressor(feature_cols, leaf=8)
        value_pipe.fit(X_train, np.log1p(train_g[target].clip(lower=0)))

        starter_pipe = fit_classifier_if_possible(X_train, train_g["target_starter"], feature_cols)
        elite_pipe = fit_classifier_if_possible(X_train, train_g["target_elite"], feature_cols)
        bust_pipe = fit_classifier_if_possible(X_train, train_g["target_bust"], feature_cols)

        pred_pct = np.clip(pct_pipe.predict(X_test), 0, 100)
        train_pred_pct = np.clip(pct_pipe.predict(X_train), 0, 100)
        pred_value = np.expm1(value_pipe.predict(X_test)).clip(min=0)

        starter_prob = class_prob_or_base(starter_pipe, X_test, train_g["target_starter"].mean())
        elite_prob = class_prob_or_base(elite_pipe, X_test, train_g["target_elite"].mean())
        bust_prob = class_prob_or_base(bust_pipe, X_test, train_g["target_bust"].mean())

        out = test_g.copy()
        out["projected_value"] = pred_value
        out["predicted_position_percentile"] = pred_pct
        out["position_score"] = [percentile(x, train_pred_pct) for x in pred_pct]
        out["starter_probability"] = starter_prob
        out["elite_probability"] = elite_prob
        out["bust_probability"] = bust_prob

        out["projection_score"] = (
            0.70 * out["position_score"]
            + 0.15 * out["predicted_position_percentile"]
            + 0.10 * (100 * out["starter_probability"])
            + 0.05 * (100 * out["elite_probability"])
            - 0.05 * (100 * out["bust_probability"])
        ).clip(0, 100).round(1)

        out["ceiling_score"] = np.maximum(
            out["projection_score"],
            0.75 * out["position_score"] + 0.25 * (100 * out["elite_probability"])
        ).clip(0, 100).round(1)

        out["floor_score"] = (100 * (1 - out["bust_probability"])).clip(0, 100).round(1)
        out["projection_tier"] = out["projection_score"].map(tier)
        out["projection_explanation"] = out.apply(explanation, axis=1)

        preds.append(out)

        print(f"Projected {group}: {len(out):,} prospects")

    if not preds:
        raise RuntimeError("No 2026 predictions created.")

    pred = pd.concat(preds, ignore_index=True, sort=False)

    # Rankings.
    pred["overall_rank_2026"] = pred["projection_score"].rank(method="min", ascending=False).astype("Int64")
    pred["position_rank_2026"] = (
        pred.groupby("projection_position_group")["projection_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    keep = []
    for c in [
        name_col,
        year_col,
        pos_col,
        "projection_position_group",
        "projection_model_group",
        "college",
        "school",
        "projection_input_pick",
        "projection_input_round",
        "projection_rank_source",
        "overall_rank_2026",
        "position_rank_2026",
        "projection_score",
        "ceiling_score",
        "floor_score",
        "starter_probability",
        "elite_probability",
        "bust_probability",
        "projected_value",
        "predicted_position_percentile",
        "position_score",
        "projection_tier",
        "projection_explanation",
    ]:
        if c and c in pred.columns and c not in keep:
            keep.append(c)

    final = pred[keep].copy()

    final = final.rename(columns={
        name_col: "player",
        year_col: "draft_year",
        pos_col: "position",
        "projection_position_group": "position_group",
        "projection_model_group": "model_group",
        "projection_input_pick": "projected_pick",
        "projection_input_round": "projected_round",
    })

    for c in [
        "projection_score",
        "ceiling_score",
        "floor_score",
        "starter_probability",
        "elite_probability",
        "bust_probability",
        "projected_value",
        "predicted_position_percentile",
        "position_score",
    ]:
        if c in final.columns:
            final[c] = pd.to_numeric(final[c], errors="coerce").round(3)

    final = final.sort_values("overall_rank_2026")

    out_file = OUT_DIR / "prospect_projections_2026_v1.csv"
    pred_file = PREDS / "prospect_projections_2026_v1_FULL.csv"
    report_file = REPORTS / "prospect_projections_2026_v1_report.json"

    final.to_csv(out_file, index=False)
    pred.to_csv(pred_file, index=False)

    report = {
        "model": "2026_projection_v1",
        "source_file": str(PRIVATE_FILE),
        "rows_projected": int(len(final)),
        "target_used_for_training": target,
        "feature_count": len(feature_cols),
        "pff_feature_count": len(pff_cols),
        "rank_proxy_column": rank_col,
        "note": "Projected 2026 class using V7.1-style PFF-powered model. Actual outcomes are unknown.",
    }

    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print("DONE")
    print(f"2026 projections: {out_file}")
    print(f"Full predictions: {pred_file}")
    print(f"Report: {report_file}")
    print("")
    print("Top 30 2026 projections:")
    print(final.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
