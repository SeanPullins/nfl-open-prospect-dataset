# NFL Open Prospect Dataset

This repository builds a public NFL player/prospect dataset for draft and historical performance modeling.

## Current dataset files

### `data/nflverse_draft_player_master_SAFE.csv`
Draft/player master file built from public nflverse data.

### `data/nflverse_draft_player_master_WITH_COMBINE.csv`
Draft/player master with nflverse combine measurements joined by stable player identifiers where available.

### `data/jacklich_prospect_master_SAFE.csv`
Prospect-level file from public Jack Lichtenstein draft data.

### `data/nflverse_jacklich_exact_candidates_SAFE.csv`
Exact-name/year candidate matches between nflverse draft data and Jack Lichtenstein prospect data.

## Not included

PFF-derived fields are intentionally not included in the public dataset unless redistribution rights are confirmed.

## Intended use

- Prospect modeling
- Draft outcome modeling
- Historical player-performance research
- Reproducible public-data pipeline development

## Data sources

- nflverse public data
- Jack Lichtenstein public NFL draft data

## Status

Early-stage research dataset. Joins should be reviewed before using for production modeling.
