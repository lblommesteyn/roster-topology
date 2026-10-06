"""Preseason win projection across 2000-01 .. 2025-26, using only data available for all seasons.

Differences from `wins.py`, which covered 2020-21 .. 2025-26:

  * player value comes from `box_value` (a box-score model calibrated against the lineup-ridge
    value on the seasons where lineup data exists) rather than from the lineup ridge itself,
    because 5-man lineup data cannot be backfilled past 2018 at this API's rate limit
  * no prior net rating, for the same reason; the baseline is last season's actual wins
  * skill-space structure and blob shape are still available, since they need only player totals

Game results only exist from 2000-01, so that is where the target series starts.
"""
import io, json, os, sys, glob, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import wins as W, metrics as MT, blob as B

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")
VALUE_COL = "box_value"


def load():
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    bv = pd.read_parquet(os.path.join(PROC, "box_value.parquet"))
    players = players.merge(bv[["season", "pid", "box_value", "box_value_onoff"]],
                            on=["season", "pid"], how="left")
    return players, W.team_wins()


def season_features(players, wins, hist, target, value_col=VALUE_COL, top_n=10, min_minutes=300):
    """One row per team for `target`, from `hist` seasons only."""
    h = players[players.season.isin(hist)].sort_values("season")
    last = h.drop_duplicates("pid", keep="last").set_index("pid")
    prev = players[players.season == hist[-1]]
    prev_min = dict(zip(prev.pid, prev.minutes))
    prev_teams = prev.set_index("pid")["team"].to_dict()
    # value: average the player's recent seasons, weighted toward the most recent
    vals = {}
    for pid, g in h.groupby("pid"):
        v = g[value_col].values
        w = np.linspace(0.5, 1.0, len(v))
        vals[pid] = float(np.average(v, weights=w))
    repl = float(np.percentile(list(vals.values()), 20)) if vals else 0.0

    lg = h[h.season == hist[-1]]
    league_Z = lg[[f"pc{k+1}" for k in range(MT.DIMS)]].values
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
    thresh = np.median(np.linalg.norm(league_Z - km.cluster_centers_[km.labels_], axis=1)) * 1.5
    lo, hi = B.league_bounds(h, min_minutes)

    tgt = players[players.season == target]
    rows = []
    for team, grp in tgt.groupby("team"):
        pids = sorted(list(grp.pid), key=lambda p: -prev_min.get(p, 0))[:top_n]
        known = [p for p in pids if p in last.index]
        if len(known) < 5:
            continue
        r = last.loc[known].copy().reset_index(drop=False)
        r["minutes"] = W.project_minutes(prev_min, known)
        r["value"] = [vals.get(p, repl) for p in known]
        w = r.minutes.values
        v = np.sort(r.value.values)[::-1]
        prev_team_min = {p: m for p, m in prev_min.items() if prev_teams.get(p) == team}
        cont = (sum(m for p, m in prev_team_min.items() if p in set(pids)) /
                max(sum(prev_team_min.values()), 1))
        blob = B.team_blob_from(r, lo, hi, n=36) if hasattr(B, "team_blob_from") else None
        trow = wins[(wins.season == target) & (wins.team == team)]
        prow = wins[(wins.season == hist[-1]) & (wins.team == team)]
        if not len(trow):
            continue
        rows.append(dict(
            season=target, team=team, wins82=float(trow.wins82.iloc[0]),
            prior_wins82=float(prow.wins82.iloc[0]) if len(prow) else np.nan,
            talent_mw=float(np.average(r.value, weights=w)),
            star=float(v[:3].sum()), depth=float(v[3:9].sum()) if len(v) > 3 else 0.0,
            continuity=cont,
            redundancy=MT.redundancy(r), eff_roles=MT.effective_roles(r),
            coverage=MT.coverage(r, km.cluster_centers_, thresh),
            scarcity=MT.scarcity(r, league_Z), n_known=len(known)))
    return pd.DataFrame(rows)


def main(hist_window=5, value_col=VALUE_COL, out_name="preseason_features_long.csv"):
    players, wins = load()
    have_wins = set(wins.season.unique())
    seasons = sorted(players.season.unique())
    frames = []
    for i in range(1, len(seasons)):
        tgt = seasons[i]
        if tgt not in have_wins or seasons[i - 1] not in have_wins:
            continue
        hist = seasons[max(0, i - hist_window):i]
        f = season_features(players, wins, hist, tgt, value_col=value_col)
        if len(f):
            frames.append(f)
            print(f"  {tgt}: {len(f)} teams (history {hist[0]}..{hist[-1]})", flush=True)
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(OUT, out_name), index=False, encoding="utf-8")
    print(f"\n{len(tab)} team-seasons, {tab.season.nunique()} target seasons "
          f"({tab.season.min()} to {tab.season.max()})")
    return tab


if __name__ == "__main__":
    main()
