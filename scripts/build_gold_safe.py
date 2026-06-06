#!/usr/bin/env python3

from pathlib import Path
import re
import unicodedata
import pandas as pd

BASE = Path("nfl_player_dataset")
BRONZE = BASE / "bronze"
GOLD = BASE / "gold"
GOLD.mkdir(parents=True, exist_ok=True)

def log(x):
    print(x, flush=True)

def snake(x):
    x = str(x)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.replace("%", "pct").replace("/", "_").replace("+", "plus")
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_").lower()

def norm_name(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.lower()
    x = re.sub(r"\([^)]*\)", " ", x)
    x = re.sub(r"[^a-z0-9\s'-]", " ", x)
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    parts = [p.strip(".'-") for p in x.split()]
    parts = [p for p in parts if p and p not in suffixes]
    return " ".join(parts)

def norm_school(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.lower().replace("&", "and")
    x = re.sub(r"[^a-z0-9()\s.-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip(" .")
    fixes = {
        "ohio st": "ohio state",
        "ohio st.": "ohio state",
        "usc": "southern california",
        "ucf": "central florida",
        "tcu": "texas christian",
        "smu": "southern methodist",
        "byu": "brigham young",
        "ole miss": "mississippi",
        "miami fl": "miami (fl)",
        "miami florida": "miami (fl)",
        "miami oh": "miami (oh)",
        "miami ohio": "miami (oh)",
    }
    return fixes.get(x, x)

def read_csv(name):
    path = BRONZE / f"{name}.csv"
    log(f"reading: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [snake(c) for c in df.columns]
    log(f"  rows={len(df):,} cols={len(df.columns):,}")
    return df

def first_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None

def add_identity(df, name_choices, school_choices):
    df = df.copy()
    name_col = first_col(df, name_choices)
    school_col = first_col(df, school_choices)

    df["player_name_clean"] = df[name_col].astype(str) if name_col else ""
    df["name_norm"] = df[name_col].map(norm_name) if name_col else ""

    df["school_clean"] = df[school_col].astype(str) if school_col else ""
    df["school_norm"] = df[school_col].map(norm_school) if school_col else ""

    return df

def write(df, name):
    path = GOLD / f"{name}.csv"
    df.to_csv(path, index=False)
    log(f"WROTE: {path} rows={len(df):,}")
    return path

# -------------------------
# 1. NFLVERSE draft master
# -------------------------

draft = read_csv("nflverse_draft_picks")
players = read_csv("nflverse_players")

draft = add_identity(
    draft,
    ["player_name", "pfr_player_name", "name", "full_name", "display_name"],
    ["college", "college_univ", "school", "college_name"]
)

players = add_identity(
    players,
    ["display_name", "player_name", "football_name", "full_name", "name"],
    ["college_name", "college", "school", "college_univ"]
)

nfl_master = draft.copy()
nfl_master["player_bridge_status"] = "draft_only"

# Keep only safe player columns.
player_keep = [
    "gsis_id", "pfr_id", "espn_id", "sportradar_id",
    "display_name", "football_name", "first_name", "last_name",
    "position", "college_name", "birth_date", "draft_year",
    "draft_round", "draft_pick", "height", "weight",
    "player_name_clean", "name_norm", "school_clean", "school_norm"
]
player_keep = [c for c in player_keep if c in players.columns]
players_small = players[player_keep].drop_duplicates()

# Join only on strong IDs. No name/school fallback here.
join_done = False

if "pfr_player_id" in nfl_master.columns and "pfr_id" in players_small.columns:
    players_join = players_small.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    nfl_master = nfl_master.merge(
        players_join.add_prefix("player_"),
        left_on="pfr_player_id",
        right_on="player_pfr_id",
        how="left"
    )
    nfl_master.loc[nfl_master["player_pfr_id"].notna(), "player_bridge_status"] = "joined_pfr_id"
    join_done = True

elif "pfr_id" in nfl_master.columns and "pfr_id" in players_small.columns:
    players_join = players_small.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    nfl_master = nfl_master.merge(
        players_join.add_prefix("player_"),
        left_on="pfr_id",
        right_on="player_pfr_id",
        how="left"
    )
    nfl_master.loc[nfl_master["player_pfr_id"].notna(), "player_bridge_status"] = "joined_pfr_id"
    join_done = True

elif "gsis_id" in nfl_master.columns and "gsis_id" in players_small.columns:
    players_join = players_small.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    nfl_master = nfl_master.merge(
        players_join.add_prefix("player_"),
        left_on="gsis_id",
        right_on="player_gsis_id",
        how="left"
    )
    nfl_master.loc[nfl_master["player_gsis_id"].notna(), "player_bridge_status"] = "joined_gsis_id"
    join_done = True

if not join_done:
    log("No safe nflverse player ID join found. Wrote draft-only master.")

write(nfl_master, "nflverse_draft_player_master_SAFE")

# Free memory.
del draft
del players
del players_small

# -------------------------
# 2. Jacklich prospect master
# -------------------------

jack = read_csv("jacklich_nfl_draft_prospects")

jack = add_identity(
    jack,
    ["player", "player_name", "name"],
    ["college", "school", "team", "college_team"]
)

write(jack, "jacklich_prospect_master_SAFE")

# -------------------------
# 3. Exact join candidates
# -------------------------

nfl = nfl_master.copy()
jck = jack.copy()

nfl_year_col = first_col(nfl, ["season", "draft_year", "year"])
jack_year_col = first_col(jck, ["year", "draft_year", "season"])

# Only join real names. Never join blanks.
nfl = nfl[nfl["name_norm"].astype(str).str.len() > 0].copy()
jck = jck[jck["name_norm"].astype(str).str.len() > 0].copy()

if nfl_year_col and jack_year_col:
    nfl["_join_year"] = nfl[nfl_year_col].astype(str)
    jck["_join_year"] = jck[jack_year_col].astype(str)

    candidates = nfl.merge(
        jck,
        on=["name_norm", "_join_year"],
        how="inner",
        suffixes=("_nfl", "_jack")
    )
else:
    # If no year columns exist, avoid a dangerous all-years join.
    candidates = pd.DataFrame()
    log("Skipped NFL ↔ Jacklich candidates because no safe draft-year column was found.")

if not candidates.empty:
    keep = []
    for c in [
        "_join_year",
        "player_name_clean_nfl", "school_clean_nfl", "position_nfl",
        "round", "pick", "overall_pick", "team",
        "pfr_player_id", "pfr_id", "gsis_id",
        "player_id",
        "player_name_clean_jack", "school_clean_jack", "position_jack", "pos",
        "name_norm"
    ]:
        if c in candidates.columns and c not in keep:
            keep.append(c)

    candidates = candidates[keep].drop_duplicates()

write(candidates, "nflverse_jacklich_exact_candidates_SAFE")

summary = pd.DataFrame([
    {"file": "nflverse_draft_player_master_SAFE.csv", "rows": len(nfl_master)},
    {"file": "jacklich_prospect_master_SAFE.csv", "rows": len(jack)},
    {"file": "nflverse_jacklich_exact_candidates_SAFE.csv", "rows": len(candidates)},
])
write(summary, "build_summary_SAFE")

print("")
print("DONE")
print(f"Open this folder:")
print(f"  {GOLD.resolve()}")
