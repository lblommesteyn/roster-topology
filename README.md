# NBA roster topology

Can player skill profiles plus observed player-player interactions recover a meaningful geometry
of NBA roster construction?

Short answer from this pipeline: **the skill geometry is real and interpretable, individual pair
"chemistry" is essentially unmeasurable, and a small league-wide complementarity structure does
exist but is far too weak to improve prediction of future lineups or rosters.** See `FINDINGS.md`.

## Data

All data comes from the public [pbpstats](https://api.pbpstats.com) API (`stats.nba.com` is not
reachable from this machine).

| file | contents |
|---|---|
| `data/raw/Player_<season>.json` | player-season totals: shot zones, rim finishing, assists, rebounding rates, steals/blocks, on-court ratings |
| `data/raw/teamlineups_<season>.json` | every 5-man lineup a team used in a season, with possessions, points scored and points allowed |
| `data/raw/Team_<season>.json` | team totals |

Seasons: 2018-19 through 2025-26 (regular season).

Two API limits shaped the design: every query truncates at 500 rows (so lineups are pulled
per team, which keeps ~100% of possessions), and per-game endpoints are rate-limited to roughly
one request per 3.5s, which made a 3,700-game stint pull impractical. The consequence is that
lineup observations are **season totals per lineup**, so there is no opponent or home/away
adjustment. Player and team effects absorb most of that, but it is a real limitation.

## Pipeline

```
python src/fetch.py               # player / team season totals
python src/fetch_team_lineups.py  # per-team lineup season totals (resumable cache)
python src/features.py            # player-season feature table + PCA/UMAP embeddings
python src/lineups.py             # flatten lineups into the modelling table
python src/run_all.py             # pair ridge model + bilinear interaction model + patterns
python src/reliability.py         # split-half and across-season replication of synergy
python src/validate.py 2022-23,2023-24,2024-25 2025-26   # chronological lineup prediction
python src/run_rosters.py         # next-season team prediction from prior information only
python src/run_moves.py           # do players improve after joining a better-fitting roster?
python src/run_topology.py        # 30 team topology figures + roster-structure metrics
python src/run_structure_all.py   # redundancy / connector metrics for all 240 team-seasons
python src/swap_cli.py --team OKC --season 2025-26 --out "Isaiah Hartenstein" --in "Nikola Jokic"
```

## Outputs

| file | contents |
|---|---|
| `out/player_features.csv` | 3,102 player-seasons x 41 skill features + PCA/UMAP coordinates |
| `out/pair_synergy.csv` | adjusted synergy for 10,101 pairs (offence, defence, net, shared possessions) |
| `out/patterns.csv` | 18 fit heuristics with bootstrap intervals |
| `out/validation.csv` | out-of-sample lineup prediction, 6 models x 6 chronological splits |
| `out/roster_forward_test.csv` | 209 team-seasons predicted from prior information only |
| `out/roster_structure_all.csv` | redundancy / roles / coverage / fragility / connector, 240 team-seasons |
| `out/player_moves.csv` | 739 team changes with before/after fit and impact |
| `out/final_log.txt` | full console log of the end-to-end run |
| `figs/topology_2025-26_*.png` | 30 team topology plots |

`tests/test_recovery.py` plants a known player effect and a known pair effect in synthetic
lineup data and checks the estimator finds both.

## Method

**Player representation.** 41 rate-normalized skill features per player-season (usage, shot
frequency by zone, rim finishing, 3PA rate, assisted vs unassisted share, passing, rebounding
rates by zone, steals/blocks/fouls), z-scored within season so league-wide drift does not
dominate, then PCA (12 components, 86% of variance) and UMAP. No discrete archetypes are imposed.

**Pair interaction effects.** Two possession-weighted ridge regressions over lineup rows:

```
points per 100 off poss  ~ season + sum(player offence) + sum(pair offence)
points allowed per 100   ~ season + sum(player defence) + sum(pair defence)
```

Player effects get a light penalty, pair effects a heavy one, so a pair coefficient only leaves
zero when the pair's shared possessions overcome the shrinkage. Net synergy = offence pair
coefficient minus defence pair coefficient.

**Structured interaction model.** `synergy(i,j) ~ z_i' M z_j` with `M` symmetric. Two fits are
implemented: a two-stage one (fit `M` to the estimated pair coefficients) and a **direct joint**
one (`src/direct.py`), which estimates player effects and `M` in a single ridge over lineup
outcomes, since the model is linear in `M`. The joint fit is the one to use: three times the
spread at equal replication. Either way this generalizes to players who have never shared a floor,
which is what the counterfactual tool needs.

**Topology.** Nodes are players (size = minutes, colour = PC1), edges are adjusted synergy.
Layout is MDS on synergy-derived target distances. A signed force-directed layout was tried first
and is kept in `src/topology.py` for reference: it collapses rosters onto a line, because signed
springs have no equilibrium length.

## Visualizations

```
python src/viz_all.py             # regenerate everything into viz/
python src/viz_all.py static 4d   # or only the named stages
```

Open `viz/index.html`. It links the league gallery, the 2D/3D/UMAP skill maps, per-team static,
interactive, animated and 4D views, the structure timelines, the counterfactual swaps, the real
transactions and the pair explanation panels. `VISUALIZATION_FINDINGS.md` says which views are
worth the screen space and which flatter the model.

### Win projection and league trends

```
python src/fetch_results.py   # game results -> actual wins
python src/backfill.py        # 1996-97 .. 2017-18 player totals, results, lineups
python src/box_value.py       # box-score value model, defined for all 30 seasons
python src/wins_long.py       # preseason feature table, 25 target seasons
python src/model_wins_long.py # leave-one-season-out backtest
python src/trends.py          # league trends, 1996-97 to 2025-26
python src/wins_viz.py        # charts
```

`PROJECTION_FINDINGS.md` has the results over **746 team-seasons (2001-02 to 2025-26)**: last
season's record plus talent plus continuity reaches MAE 6.91 wins and R² 0.49 against 7.81 for the
record alone; depth (players 4-9) correlates +0.58 with next-season wins against +0.47 for the top
three; no structural feature improves the projection. Trends over 30 seasons: the 3-point rate
doubled and the long midrange lost two thirds of its share, rotations grew from 9.4 to 11.1
players, and star minute concentration fell from 44% to 35%.

**Data reach.** Play-by-play starts in 1996-97, so nothing here can go back to 1990. Game results
start in 2000-01. Lineup data exists from 1996-97 but is only served per team per season, and the
rate limit puts a full backfill at roughly 15 hours, so lineup-derived quantities stay on the
2018-2026 window and player value before that comes from `box_value.py`.

### Roster organisms (3D blobs)

```
python src/fetch_monthly.py      # monthly player totals (one request per month)
python src/monthly.py            # monthly embeddings on the season PCA basis
python src/blob_all.py           # team viewers, gallery, season evolution, trade morphs, metrics
python src/blob_season.py        # within-season month-by-month blobs
python src/blob_gif.py           # morph GIFs
python src/blob_index.py         # viz/blobs.html
```

`src/blob.py` builds the occupancy field and extracts the surface; `src/blob_viz.py` renders it;
`src/blob_scenes.py` holds the viewers, evolution animations and field-interpolated trade morphs.
Findings in `BLOB_VIS_FINDINGS.md`: peak density tracks redundancy (r = 0.48), lobe count tracks
net rating (+0.25), and whole-field deformation tracks roster turnover (r = 0.54), while scalar
shape summaries do not.

Two things to keep straight when reading them:

- **Distance means different things in different views.** In the team graphs it is *fit*
  (MDS on synergy-derived distances), in the skill maps and 3D/4D views it is *skill similarity*
  (PCA). For a complementary pair those are close to opposites.
- **Edges use one league-wide scale** (99th percentile of absolute synergy, 1.45 pts/100), never a
  per-team scale, so a flat roster looks flat.

## Caveats

- Lineup observations are season totals: no opponent, home/away or in-season timing adjustment.
- Player effects are pooled across seasons in the multi-season fits, so ageing and role change
  are not modelled.
- Predicted lineup scores from `swap_cli.py` are uncalibrated sums of effects and should be read
  as rankings, not as expected net ratings.
