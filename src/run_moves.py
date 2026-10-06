"""Player-move test: when a player changes team, does joining a better-fitting roster raise
his measured impact? Fit is scored with information available before the move only.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B, validate as V, run_all as RA

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")
DIMS = 8


def fit_score(pid, roster, fit):
    """Minutes-weighted mean predicted synergy of pid with a roster (excluding himself)."""
    if pid not in fit.Z.index:
        return np.nan
    z = fit.Z.loc[pid].values[:DIMS]
    s, w = [], []
    for p, m in zip(roster.pid, roster.minutes):
        if p == pid or p not in fit.Z.index:
            continue
        s.append(float(z @ fit.M @ fit.Z.loc[p].values[:DIMS])); w.append(m)
    if not s:
        return np.nan
    return float(np.average(s, weights=w))


def main(min_minutes=400):
    players, stints = RA.load()
    seasons = sorted(stints.season.unique())
    per_season = {s: V.Fit(stints[stints.season == s].reset_index(drop=True), players)
                  for s in seasons}
    rows = []
    for i in range(1, len(seasons)):
        prev, cur = seasons[i - 1], seasons[i]
        hist = V.Fit(stints[stints.season.isin(seasons[:i])].reset_index(drop=True), players)
        a = players[(players.season == prev) & (players.minutes >= min_minutes)]
        b = players[(players.season == cur) & (players.minutes >= min_minutes)]
        movers = a.merge(b, on="pid", suffixes=("_a", "_b"))
        movers = movers[movers.team_a != movers.team_b]
        for r in movers.itertuples():
            old = a[(a.team == r.team_a)]
            new = b[(b.team == r.team_b)]
            fo, fn = fit_score(r.pid, old, hist), fit_score(r.pid, new, hist)
            va = per_season[prev].add_o["player"].get(r.pid, np.nan) - \
                 per_season[prev].add_d["player"].get(r.pid, np.nan)
            vb = per_season[cur].add_o["player"].get(r.pid, np.nan) - \
                 per_season[cur].add_d["player"].get(r.pid, np.nan)
            rows.append(dict(pid=r.pid, name=r.name_b, season=cur, old=r.team_a, new=r.team_b,
                             fit_old=fo, fit_new=fn, d_fit=(fn - fo) if fn == fn and fo == fo else np.nan,
                             val_prev=va, val_cur=vb, d_val=vb - va,
                             minutes=r.minutes_b))
    t = pd.DataFrame(rows).dropna(subset=["d_fit", "d_val"])
    t.to_csv(os.path.join(OUT, "player_moves.csv"), index=False, encoding="utf-8")
    print(f"player moves with before/after impact estimates: {len(t)}")
    print(t[["d_fit", "val_prev", "d_val"]].corr().round(3).to_string())
    # control for regression to the mean: does fit change explain impact change beyond prior value?
    X = np.column_stack([np.ones(len(t)), t.val_prev.values, t.d_fit.values])
    bb, *_ = np.linalg.lstsq(X, t.d_val.values, rcond=None)
    resid = t.d_val.values - X @ bb
    se = np.sqrt((resid ** 2).sum() / (len(t) - 3) * np.diag(np.linalg.inv(X.T @ X)))
    print(f"\nd_impact = {bb[0]:.2f} + {bb[1]:.3f}*prior_value + {bb[2]:.2f}*d_fit"
          f"   (d_fit t = {bb[2]/se[2]:.2f})")
    print("d_fit sd:", round(float(t.d_fit.std()), 3), "pts/100 per teammate")
    return t


if __name__ == "__main__":
    main()
