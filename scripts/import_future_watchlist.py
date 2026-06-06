#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

OUT_SITE = Path("site_data/future_prospects.csv")
OUT_DOCS = Path("docs/data/future_prospects.csv")

NAME_COLS = ["player", "player_name", "name", "full_name", "prospect"]
YEAR_COLS = ["draft_year", "year", "class", "draft_class"]
POS_COLS = ["position", "pos"]
COLLEGE_COLS = ["college", "school", "team", "college_name"]
RANK_COLS = ["watchlist_rank", "rank", "overall", "overall_rank", "big_board_rank"]
PICK_COLS = ["projected_pick", "pick", "draft_pick"]
ROUND_COLS = ["projected_round", "round", "draft_round"]

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for col in df.columns:
        cl = str(col).lower()
        for c in candidates:
            if c.lower() in cl:
                return col
    return None

def pos_group(pos):
    p = str(pos).upper().strip()
    if p in {"QB"}:
        return "QB"
    if p in {"RB", "HB", "FB"}:
        return "RB"
    if p in {"WR"}:
        return "WR"
    if p in {"TE"}:
        return "TE"
    if p in {"OT", "T", "LT", "RT"}:
        return "OT"
    if p in {"OG", "G", "LG", "RG", "C"}:
        return "IOL"
    if p in {"OL"}:
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
    return p or "UNK"

def pick_to_round(pick):
    if pd.isna(pick):
        return np.nan
    pick = float(pick)
    if pick <= 32: return 1
    if pick <= 64: return 2
    if pick <= 100: return 3
    if pick <= 140: return 4
    if pick <= 180: return 5
    if pick <= 220: return 6
    return 7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="private/future_watchlist.csv")
    parser.add_argument("--force-year", type=int, help="Use this draft year if source has no year column")
    args = parser.parse_args()

    src = Path(args.source).expanduser()
    if not src.exists():
        raise SystemExit(f"Missing source: {src}")

    incoming = pd.read_csv(src, low_memory=False)
    incoming.columns = [str(c).strip() for c in incoming.columns]

    name_col = find_col(incoming, NAME_COLS)
    year_col = find_col(incoming, YEAR_COLS)
    pos_col = find_col(incoming, POS_COLS)
    college_col = find_col(incoming, COLLEGE_COLS)
    rank_col = find_col(incoming, RANK_COLS)
    pick_col = find_col(incoming, PICK_COLS)
    round_col = find_col(incoming, ROUND_COLS)

    if not name_col:
        raise SystemExit("Could not find player/name column.")
    if not pos_col:
        raise SystemExit("Could not find position column.")
    if not year_col and not args.force_year:
        raise SystemExit("No draft_year column found. Rerun with --force-year 2027.")

    rows = []

    for _, r in incoming.iterrows():
        player = str(r.get(name_col, "")).strip()
        if not player or player.lower() == "nan":
            continue

        if year_col:
            year = pd.to_numeric(pd.Series([r.get(year_col)]), errors="coerce").iloc[0]
        else:
            year = args.force_year

        if pd.isna(year) or int(year) < 2027:
            continue

        position = str(r.get(pos_col, "")).strip().upper()
        college = str(r.get(college_col, "")).strip() if college_col else ""

        rank = pd.to_numeric(pd.Series([r.get(rank_col)]), errors="coerce").iloc[0] if rank_col else np.nan
        pick = pd.to_numeric(pd.Series([r.get(pick_col)]), errors="coerce").iloc[0] if pick_col else np.nan
        rnd = pd.to_numeric(pd.Series([r.get(round_col)]), errors="coerce").iloc[0] if round_col else np.nan

        if pd.isna(pick) and not pd.isna(rank):
            pick = rank
        if pd.isna(rnd) and not pd.isna(pick):
            rnd = pick_to_round(pick)

        rows.append({
            "player": player,
            "draft_year": int(year),
            "position": position,
            "position_group": pos_group(position),
            "college": college,
            "class_status": "future_watchlist",
            "watchlist_rank": rank,
            "projected_pick": pick,
            "projected_round": rnd,
            "projection_score": "",
            "ceiling_score": "",
            "floor_score": "",
            "starter_probability": "",
            "elite_probability": "",
            "bust_probability": "",
            "projection_tier": "Unscored future watchlist",
            "projection_explanation": "Future class watchlist player. Add college stats, PFF, and All-22 data as available.",
            "college_stats_status": "pending",
            "pff_status": "pending",
            "all22_status": "pending",
            "notes": str(r.get("notes", "")).strip() if "notes" in incoming.columns else "",
        })

    if not rows:
        raise SystemExit("No future rows imported.")

    new = pd.DataFrame(rows)

    existing = pd.DataFrame()
    if OUT_SITE.exists():
        existing = pd.read_csv(OUT_SITE, low_memory=False)

    combined = pd.concat([existing, new], ignore_index=True, sort=False)

    combined["_key"] = (
        combined["player"].astype(str).str.lower().str.strip()
        + "|"
        + combined["draft_year"].astype(str)
        + "|"
        + combined["position_group"].astype(str)
        + "|"
        + combined["college"].astype(str).str.lower().str.strip()
    )

    combined["_rank_sort"] = pd.to_numeric(combined["watchlist_rank"], errors="coerce").fillna(999999)

    combined = (
        combined.sort_values(["draft_year", "_rank_sort", "player"])
        .drop_duplicates("_key", keep="first")
        .drop(columns=["_key", "_rank_sort"])
    )

    # Remove sample placeholder if real rows exist.
    if len(combined) > 1:
        combined = combined[combined["player"].ne("Sample 2027 Player")].copy()

    combined.to_csv(OUT_SITE, index=False)
    combined.to_csv(OUT_DOCS, index=False)

    print(f"WROTE: {OUT_SITE}")
    print(f"WROTE: {OUT_DOCS}")
    print(f"Rows: {len(combined):,}")
    print("")
    print(combined.head(40).to_string(index=False))

if __name__ == "__main__":
    main()
