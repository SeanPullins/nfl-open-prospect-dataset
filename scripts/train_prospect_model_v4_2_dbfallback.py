#!/usr/bin/env python3

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PRIVATE_FILE = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.csv")
PUBLIC_FILE = Path("data/nflverse_draft_player_master_WITH_COMBINE.csv")

REPORTS = Path("reports")
PREDS = Path("predictions")
MODELS = Path("models")

for p in [REPORTS, PREDS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET_CANDIDATES = ["w_av", "weighted_av", "car_av", "career_av"]
YEAR_CANDIDATES = ["season", "draft_year", "year"]
NAME_CANDIDATES = ["player_name_clean", "player_name", "pfr_player_name", "name", "display_name"]
POS_CANDIDATES = ["position", "pos", "player_position", "combine_pos"]

NUMERIC_FEATURE_CANDIDATES = [
    "round",
    "pick",
    "overall_pick",
    "draft_ovr",
    "age",

    "height",
    "weight",
    "ht",
    "wt",

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
    "combine_draft_round",
    "combine_draft_ovr",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "position",
    "pos",
    "player_position",
    "combine_pos",
    "college",
    "school",
    "combine_school",
]


def log(x):
    print(x, flush=True)


def first_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None


def height_to_inches_or_float(x):
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


def normalize_pos(x):
    if pd.isna(x):
        return "UNK"

    p = str(x).upper().strip()
    p = p.replace(" ", "")

    # QB
    if p in {"QB"}:
        return "QB"

    # Skill
    if p in {"RB", "HB", "FB"}:
        return "RB"
    if p in {"WR", "FL", "SE"}:
        return "WR"
    if p in {"TE"}:
        return "TE"

    # OL
    if p in {"OT", "T", "LT", "RT"}:
        return "OT"
    if p in {"OG", "G", "LG", "RG"}:
        return "IOL"
    if p in {"C"}:
        return "IOL"
    if p in {"OL"}:
        return "OL"

    # Defensive front
    if p in {"DE", "EDGE", "OLB/DE", "DE/OLB"}:
        return "EDGE"
    if p in {"DT", "NT", "DL"}:
        return "IDL"

    # LB
    if p in {"LB", "ILB", "MLB", "OLB"}:
        return "LB"

    # DB
    if p in {"CB", "DB"}:
        return "CB"
    if p in {"S", "SS", "FS", "SAF"}:
        return "S"

    # ST
    if p in {"K", "PK", "P", "LS"}:
        return "ST"

    return p


def load_data():
    if PRIVATE_FILE.exists():
        path = PRIVATE_FILE
        source = "private_file_loaded_but_only_strict_public_predraft_features_used"
    elif PUBLIC_FILE.exists():
        path = PUBLIC_FILE
        source = "public"
    else:
        raise FileNotFoundError("No dataset found.")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    log(f"Loaded: {path}")
    log(f"Rows: {len(df):,}")
    log(f"Cols: {len(df.columns):,}")
    return df, source, path


def choose_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 500:
                return c
    raise ValueError("No usable target found.")


def choose_year(df):
    c = first_col(df, YEAR_CANDIDATES)
    if not c:
        raise ValueError("No year column found.")
    return c


def choose_name(df):
    return first_col(df, NAME_CANDIDATES)


def choose_pos(df):
    c = first_col(df, POS_CANDIDATES)
    if not c:
        raise ValueError("No position column found.")
    return c


def select_features(df):
    numeric = [c for c in NUMERIC_FEATURE_CANDIDATES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURE_CANDIDATES if c in df.columns]

    # Avoid duplicate draft fields if base and combine both present.
    if "pick" in numeric and "combine_draft_ovr" in numeric:
        numeric.remove("combine_draft_ovr")
    if "round" in numeric and "combine_draft_round" in numeric:
        numeric.remove("combine_draft_round")

    categorical = [
        c for c in categorical
        if 1 < df[c].nunique(dropna=True) <= 300
    ]

    return numeric, categorical


def coerce_numeric(df, numeric_cols):
    out = df.copy()
    for c in numeric_cols:
        out[c] = out[c].map(height_to_inches_or_float)
    return out


def make_pipe(numeric_cols, categorical_cols):
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
            ]), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("prep", prep),
        ("model", model),
    ])


def percentile_score(value, reference_values):
    vals = np.sort(np.asarray(pd.Series(reference_values).dropna()))
    if len(vals) == 0 or pd.isna(value):
        return np.nan
    return round(100 * (np.searchsorted(vals, value, side="right") / len(vals)), 1)


def tier(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 95:
        return "Elite / rare position profile"
    if score >= 85:
        return "High-end starter position profile"
    if score >= 70:
        return "Starter-caliber position profile"
    if score >= 55:
        return "Rotational / contributor profile"
    if score >= 35:
        return "Depth / replacement-level profile"
    return "Low-probability hit profile"


def explain(row):
    reasons = []

    pick = None
    for c in ["pick", "overall_pick", "draft_ovr", "combine_draft_ovr"]:
        if c in row and pd.notna(row[c]):
            try:
                pick = float(row[c])
                break
            except Exception:
                pass

    if pick is not None:
        if pick <= 15:
            reasons.append("premium draft capital")
        elif pick <= 50:
            reasons.append("top-50 draft capital")
        elif pick <= 100:
            reasons.append("Day 2 draft capital")
        else:
            reasons.append("lower draft-capital profile")

    forty = None
    for c in ["combine_forty"]:
        if c in row and pd.notna(row[c]):
            try:
                forty = float(row[c])
                break
            except Exception:
                pass

    pos = row.get("v4_position_group", "")

    if forty is not None:
        if pos in {"WR", "CB", "RB", "S"} and forty <= 4.45:
            reasons.append("plus speed for position")
        elif pos in {"EDGE", "LB", "TE"} and forty <= 4.65:
            reasons.append("plus speed for position")
        elif forty >= 4.85 and pos not in {"OT", "IOL", "IDL", "ST"}:
            reasons.append("below-average timed speed")

    wt = None
    for c in ["combine_wt", "wt", "weight", "player_weight"]:
        if c in row and pd.notna(row[c]):
            try:
                wt = height_to_inches_or_float(row[c])
                break
            except Exception:
                pass

    if wt is not None and not pd.isna(wt):
        if pos in {"EDGE", "IDL", "OT", "IOL"} and wt >= 250:
            reasons.append("NFL-sized frame for front/line position")
        elif pos in {"WR", "CB", "S", "RB"} and wt <= 190:
            reasons.append("lighter frame")

    if not reasons:
        reasons.append("limited strict pre-draft signal available")

    return "; ".join(reasons)


def run_one_position_group(group_df, group, target, year_col, name_col, numeric_cols, categorical_cols):
    preds = []
    metrics_rows = []

    years = sorted(group_df[year_col].dropna().unique())

    for yr in years:
        train = group_df[group_df[year_col] < yr].copy()
        test = group_df[group_df[year_col] == yr].copy()

        # Position-specific models need enough history.
        # If too little data, skip that year/group.
        if len(train) < 60 or len(test) == 0:
            continue

        train = coerce_numeric(train, numeric_cols)
        test = coerce_numeric(test, numeric_cols)

        pipe = make_pipe(numeric_cols, categorical_cols)
        pipe.fit(train[numeric_cols + categorical_cols], train[target])

        pred = pipe.predict(test[numeric_cols + categorical_cols])

        # V4.1 calibration:
        # Score prospects against prior same-position MODEL PROJECTIONS,
        # not against actual final career outcomes. This prevents impossible
        # historic outliers from depressing elite prospect scores.
        train_pred_for_score = pipe.predict(train[numeric_cols + categorical_cols])

        tmp = test.copy()
        tmp["v4_model_group"] = group
        # v4_position_group remains the true display/report position from the original row
        tmp["v4_projected_outcome_value"] = pred
        tmp["v4_actual_outcome_value"] = tmp[target]
        tmp["v4_residual_actual_minus_projected"] = (
            tmp["v4_actual_outcome_value"] - tmp["v4_projected_outcome_value"]
        )
        tmp["v4_train_rows"] = len(train)
        tmp["v4_train_max_year"] = yr - 1

        # Position score: percentile of this player's projection vs prior same-position projections.
        tmp["v4_position_score_0_100"] = [
            percentile_score(x, train_pred_for_score) for x in pred
        ]

        # Also compute a global-like score against all prior target values in that group history.
        tmp["v4_position_tier"] = tmp["v4_position_score_0_100"].map(tier)

        preds.append(tmp)

        metrics_rows.append({
            "position_group": group,
            "year": int(yr),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        })

    if not preds:
        return pd.DataFrame(), pd.DataFrame(metrics_rows)

    out = pd.concat(preds, ignore_index=True, sort=False)
    return out, pd.DataFrame(metrics_rows)


def main():
    df, source, data_path = load_data()

    target = choose_target(df)
    year_col = choose_year(df)
    name_col = choose_name(df)
    pos_col = choose_pos(df)

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["v4_position_group"] = df[pos_col].map(normalize_pos)

    # V4.2 model grouping:
    # Keep display/report position as CB or S, but train both with a shared DB pool.
    # This prevents thin safety history from dropping players like Jamal Adams.
    df["v4_model_group"] = df["v4_position_group"]
    df.loc[df["v4_position_group"].isin(["CB", "S"]), "v4_model_group"] = "DB"

    labeled = df[df[target].notna() & df[year_col].notna()].copy()
    labeled = labeled[labeled["v4_position_group"].ne("UNK")].copy()

    numeric_cols, categorical_cols = select_features(labeled)

    # V4 position models are intentionally numeric-only.
    # School/college categoricals can be sparse by early year and can crash/overfit.
    # Position is handled by separate models.
    categorical_cols = []

    feature_report = {
        "model": "v4_position_specific_walkforward",
        "target": target,
        "year_col": year_col,
        "name_col": name_col,
        "pos_col": pos_col,
        "numeric_features_used": numeric_cols,
        "categorical_features_used": categorical_cols,
        "position_groups": sorted(labeled["v4_position_group"].dropna().unique().tolist()),
        "model_groups": sorted(labeled["v4_model_group"].dropna().unique().tolist()),
        "hard_rule": "Only explicit pre-draft/draft/combine fields are used. No PFF, NFL stats, All-Pro, Pro Bowl, games, started seasons, or AV fields used as features.",
    }
    (REPORTS / "prospect_projection_features_v4_2_dbfallback.json").write_text(
        json.dumps(feature_report, indent=2),
        encoding="utf-8"
    )

    log(f"Target: {target}")
    log(f"Year col: {year_col}")
    log(f"Name col: {name_col}")
    log(f"Position col: {pos_col}")
    log(f"Numeric features: {numeric_cols}")
    log(f"Categorical features: {categorical_cols}")
    log("")
    log("Position counts:")
    log(labeled["v4_model_group"].value_counts().to_string())

    all_preds = []
    all_model_year_rows = []

    for group, gdf in labeled.groupby("v4_model_group"):
        if len(gdf) < 120:
            log(f"Skipping {group}: only {len(gdf)} rows")
            continue

        log(f"\nTraining walk-forward position group: {group} rows={len(gdf):,}")

        pred_g, metrics_g = run_one_position_group(
            gdf,
            group,
            target,
            year_col,
            name_col,
            numeric_cols,
            categorical_cols,
        )

        if not pred_g.empty:
            all_preds.append(pred_g)

        if not metrics_g.empty:
            all_model_year_rows.append(metrics_g)

    if not all_preds:
        raise RuntimeError("No position-specific predictions created.")

    pred = pd.concat(all_preds, ignore_index=True, sort=False)

    # Global score for comparison, based on all predicted values across all positions.
    pred["v4_global_score_0_100"] = [
        percentile_score(x, pred["v4_projected_outcome_value"])
        for x in pred["v4_projected_outcome_value"]
    ]

    mae = mean_absolute_error(pred["v4_actual_outcome_value"], pred["v4_projected_outcome_value"])
    rmse = mean_squared_error(pred["v4_actual_outcome_value"], pred["v4_projected_outcome_value"]) ** 0.5
    r2 = r2_score(pred["v4_actual_outcome_value"], pred["v4_projected_outcome_value"])

    pred["v4_outlier_type"] = "near_expected"
    pred.loc[pred["v4_residual_actual_minus_projected"] >= mae * 2, "v4_outlier_type"] = "positive_outlier_exceeded_projection"
    pred.loc[pred["v4_residual_actual_minus_projected"] <= -mae * 2, "v4_outlier_type"] = "negative_outlier_missed_projection"

    pred["v4_explanation"] = pred.apply(explain, axis=1)

    out_path = PREDS / "prospect_projection_walkforward_v4_2_dbfallback.csv"
    pred.to_csv(out_path, index=False)

    if all_model_year_rows:
        position_year_metrics = pd.concat(all_model_year_rows, ignore_index=True, sort=False)
    else:
        position_year_metrics = pd.DataFrame()

    # Position-level performance metrics.
    pos_metrics = []
    for group, g in pred.groupby("v4_model_group"):
        if len(g) < 20:
            continue

        pos_metrics.append({
            "position_group": group,
            "rows": int(len(g)),
            "mae": float(mean_absolute_error(g["v4_actual_outcome_value"], g["v4_projected_outcome_value"])),
            "rmse": float(mean_squared_error(g["v4_actual_outcome_value"], g["v4_projected_outcome_value"]) ** 0.5),
            "r2": float(r2_score(g["v4_actual_outcome_value"], g["v4_projected_outcome_value"])),
            "avg_position_score": float(g["v4_position_score_0_100"].mean()),
        })

    pos_metrics_df = pd.DataFrame(pos_metrics).sort_values("position_group")
    pos_metrics_df.to_csv(REPORTS / "position_model_metrics_v4_2.csv", index=False)

    position_year_metrics.to_csv(REPORTS / "position_year_walkforward_counts_v4_2.csv", index=False)

    metrics = {
        "dataset_source": source,
        "dataset_path": str(data_path),
        "target": target,
        "year_col": year_col,
        "name_col": name_col,
        "pos_col": pos_col,
        "rows_predicted": int(len(pred)),
        "overall_mae": float(mae),
        "overall_rmse": float(rmse),
        "overall_r2": float(r2),
        "position_groups_modeled": sorted(pred["v4_position_group"].dropna().unique().tolist()),
        "model_groups_modeled": sorted(pred["v4_model_group"].dropna().unique().tolist()),
        "warning": "V4 is strict no-leakage and position-specific. It should be more honest and position-relevant than V1/V2.",
    }

    metrics_path = REPORTS / "prospect_projection_metrics_v4_2_dbfallback.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cols = []
    for c in [
        name_col, year_col, "position", "pos", "v4_position_group", "college", "school",
        "round", "pick", "overall_pick",
        "v4_projected_outcome_value",
        "v4_global_score_0_100",
        "v4_position_score_0_100",
        "v4_position_tier",
        "v4_actual_outcome_value",
        "v4_residual_actual_minus_projected",
        "v4_outlier_type",
        "v4_explanation",
    ]:
        if c and c in pred.columns and c not in cols:
            cols.append(c)

    pred[pred["v4_outlier_type"].eq("positive_outlier_exceeded_projection")][cols].to_csv(
        REPORTS / "positive_outliers_v4_2_dbfallback.csv",
        index=False
    )
    pred[pred["v4_outlier_type"].eq("negative_outlier_missed_projection")][cols].to_csv(
        REPORTS / "negative_outliers_v4_2_dbfallback.csv",
        index=False
    )

    print("")
    print("DONE")
    print(f"Predictions: {out_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Position metrics: {REPORTS / 'position_model_metrics_v4_2.csv'}")
    print(f"Features: {REPORTS / 'prospect_projection_features_v4_2_dbfallback.json'}")


if __name__ == "__main__":
    main()
