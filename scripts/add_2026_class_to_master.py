#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re

MASTER = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.csv")
BACKUP = Path("private/nflverse_draft_player_master_WITH_PFF_AND_COMBINE.before_2026_append.csv")

COMMON_SOURCES = [
    Path("data/jacklich_nfl_draft_prospects.csv"),
    Path("data/jacklich_prospect_master.csv"),
    Path("site_data/board_2026.csv"),
    Path("private/board_2026.csv"),
    Path("private/prospects_2026.csv"),
    Path("private/2026_prospects.csv"),
    Path("../nfl_player_dataset/raw/jacklich_nfl_draft_prospects.csv"),
    Path("../nfl_player_dataset/bronze/jacklich_nfl_draft_prospects.csv"),
    Path("../nfl_player_dataset/gold/jacklich_prospect_master.csv"),
]

NAME_CANDIDATES = [
    "player_name_clean", "player", "player_name", "name", "full_name",
    "prospect", "prospect_name", "display_name"
]

POS_CANDIDATES = [
    "position", "pos", "player_position"
]

COLLEGE_CANDIDATES = [
    "college", "school", "college_name", "team", "school_name"
]

YEAR_CANDIDATES = [
    "season", "draft_year", "year", "class", "draft_class"
]

RANK_CANDIDATES = [
    "rank", "overall_rank", "consensus_rank", "big_board_rank",
    "board_rank", "jacklich_rank", "draft_rank", "overall"
]


def norm_col(c):
    return str(c).strip()


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def normalize_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_pos(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


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


def choose_source(arg_source):
    if arg_source:
        p = Path(arg_source).expanduser()
        if not p.exists():
            raise SystemExit(f"Source not found: {p}")
        return p

    for p in COMMON_SOURCES:
        if p.exists():
            return p

    print("No source file found automatically.")
    print("Create one of these, or pass --source path/to/file.csv:")
    for p in COMMON_SOURCES:
        print(" ", p)
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="CSV source with 2026 prospects/board")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    if not MASTER.exists():
        raise SystemExit(f"Missing master file: {MASTER}")

    source = choose_source(args.source)

    print(f"MASTER: {MASTER}")
    print(f"SOURCE: {source}")

    master = pd.read_csv(MASTER, low_memory=False)
    master.columns = [norm_col(c) for c in master.columns]

    src = pd.read_csv(source, low_memory=False)
    src.columns = [norm_col(c) for c in src.columns]

    print(f"Master rows before: {len(master):,}")
    print(f"Source rows: {len(src):,}")
    print("Source columns:")
    print(src.columns.tolist())

    name_col = find_col(src, NAME_CANDIDATES)
    pos_col = find_col(src, POS_CANDIDATES)
    college_col = find_col(src, COLLEGE_CANDIDATES)
    year_col = find_col(src, YEAR_CANDIDATES)
    rank_col = find_col(src, RANK_CANDIDATES)

    if not name_col:
        raise SystemExit("Could not find player/name column in source.")
    if not pos_col:
        raise SystemExit("Could not find position column in source.")

    print("")
    print("Detected columns:")
    print(" name:", name_col)
    print(" pos:", pos_col)
    print(" college:", college_col)
    print(" year:", year_col)
    print(" rank:", rank_col)

    # Filter to 2026 if a year/class column exists.
    work = src.copy()
    if year_col:
        year_vals = pd.to_numeric(work[year_col], errors="coerce")
        work = work[year_vals.eq(args.year)].copy()
    else:
        print("No year column found; assuming entire source is 2026 class.")

    if work.empty:
        raise SystemExit("No 2026 rows found in source after filtering.")

    # Build append rows with master columns.
    append = pd.DataFrame(columns=master.columns)
    append["season"] = args.year if "season" in append.columns else np.nan

    if "draft_year" in append.columns:
        append["draft_year"] = args.year
    if "year" in append.columns:
        append["year"] = args.year

    # Name columns.
    names = work[name_col].map(normalize_name)

    for c in ["player_name_clean", "player_name", "name", "display_name"]:
        if c in append.columns:
            append[c] = names

    # Position.
    positions = work[pos_col].map(normalize_pos)
    for c in ["position", "pos", "player_position"]:
        if c in append.columns:
            append[c] = positions

    # College.
    if college_col:
        colleges = work[college_col].astype(str).replace("nan", "")
        for c in ["college", "school", "college_name", "school_name"]:
            if c in append.columns:
                append[c] = colleges

    # Rank proxy as projected pick/round.
    if rank_col:
        ranks = pd.to_numeric(work[rank_col], errors="coerce")
        if "pick" in append.columns:
            append["pick"] = ranks
        if "round" in append.columns:
            append["round"] = ranks.map(pick_to_round)

        # Preserve rank columns if master has any.
        for c in ["rank", "overall_rank", "consensus_rank", "big_board_rank", "board_rank", "jacklich_rank"]:
            if c in append.columns:
                append[c] = ranks

    # Copy matching source columns into master when names overlap.
    for c in work.columns:
        if c in append.columns and c not in ["season", "draft_year", "year"]:
            append[c] = work[c].values

    # Add source-specific rank columns if master already has them absent? We cannot add new columns safely
    # unless we add them to master too.
    extra_cols = []
    if rank_col:
        for new_col in ["projection_rank_2026", "projection_source_2026"]:
            if new_col not in master.columns:
                master[new_col] = np.nan
                append[new_col] = np.nan
                extra_cols.append(new_col)

        append["projection_rank_2026"] = pd.to_numeric(work[rank_col], errors="coerce").values
        append["projection_source_2026"] = source.name

    # Deduplicate against existing 2026 master rows.
    master_year_col = "season" if "season" in master.columns else ("draft_year" if "draft_year" in master.columns else "year")
    existing_2026 = master[pd.to_numeric(master[master_year_col], errors="coerce").eq(args.year)].copy()

    def key_frame(df):
        name_base = None
        for c in ["player_name_clean", "player_name", "name", "display_name"]:
            if c in df.columns:
                name_base = c
                break

        pos_base = None
        for c in ["position", "pos", "player_position"]:
            if c in df.columns:
                pos_base = c
                break

        college_base = None
        for c in ["college", "school", "college_name"]:
            if c in df.columns:
                college_base = c
                break

        n = df[name_base].map(normalize_name).str.lower() if name_base else ""
        p = df[pos_base].map(normalize_pos) if pos_base else ""
        s = df[college_base].astype(str).str.lower().str.strip() if college_base else ""

        return n.astype(str) + "|" + p.astype(str) + "|" + s.astype(str)

    existing_keys = set(key_frame(existing_2026)) if not existing_2026.empty else set()
    append_keys = key_frame(append)

    append = append[~append_keys.isin(existing_keys)].copy()

    # Also remove duplicate rows within append.
    append["_key"] = key_frame(append)
    append = append.drop_duplicates("_key").drop(columns=["_key"])

    print("")
    print(f"Existing 2026 rows in master: {len(existing_2026):,}")
    print(f"New 2026 rows to append: {len(append):,}")

    if append.empty:
        print("Nothing to append.")
        return

    if not BACKUP.exists():
        master.to_csv(BACKUP, index=False)
        print(f"Backup written: {BACKUP}")

    combined = pd.concat([master, append], ignore_index=True, sort=False)
    combined.to_csv(MASTER, index=False)

    print("")
    print("DONE")
    print(f"Master rows after: {len(combined):,}")
    print(f"Appended 2026 players: {len(append):,}")

    show_cols = []
    for c in ["player_name_clean", "player_name", "name", "season", "draft_year", "position", "college", "school", "round", "pick", "projection_rank_2026"]:
        if c in combined.columns and c not in show_cols:
            show_cols.append(c)

    check = combined[pd.to_numeric(combined[master_year_col], errors="coerce").eq(args.year)]
    print("")
    print("2026 sample:")
    print(check[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
