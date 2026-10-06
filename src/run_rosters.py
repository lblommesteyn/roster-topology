"""Newly-formed-roster test: predict a team's next-season net rating from prior-season information only.

For each season pair (t -> t+1): take the t+1 roster (who is on the team), but estimate every
player quantity from season <= t. Compare an additive player-value model against models that add
roster-structure / complementarity terms.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B, metrics as MT, validate as V, run_all as RA

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")
DIMS = 8


def roster_features(r, fit, players_prev, centroids, thresh, league_Z):
    """Structure summary of a roster using only prior-season information."""
    po, pdd = fit.add_o["player"], fit.add_d["player"]
    val = np.array([po.get(p, po.mean()) - pdd.get(p, pdd.mean()) for p in r.pid])
    m = r.minutes.values.astype(float)
    syn_b, syn_r = [], []
    ww = []
    for i, j in itertools.combinations(range(len(r)), 2):
        a, b = r.pid.iloc[i], r.pid.iloc[j]
        k = (a, b) if a < b else (b, a)
        if a in fit.Z.index and b in fit.Z.index:
            syn_b.append(float(fit.Z.loc[a].values[:DIMS] @ fit.M @ fit.Z.loc[b].values[:DIMS]))
        else:
            syn_b.append(0.0)
        syn_r.append(fit.ridge_pairs.get(k, 0.0))
        ww.append(m[i] * m[j])
    ww = np.array(ww)
    return dict(
        value_mw=float(np.average(val, weights=m)), value_sum=float(val.sum()),
        bilinear=float(np.average(syn_b, weights=ww)) if len(ww) else 0.0,
        ridgepair=float(np.average(syn_r, weights=ww)) if len(ww) else 0.0,
        redundancy=MT.redundancy(r), eff_roles=MT.effective_roles(r),
        coverage=MT.coverage(r, centroids, thresh), scarcity=MT.scarcity(r, league_Z),
        n=len(r))


def main(min_minutes=300):
    players, stints = RA.load()
    seasons = sorted(stints.season.unique())
    tn = RA.team_net(stints)
    from sklearn.cluster import KMeans
    rows = []
    for i in range(1, len(seasons)):
        prev, cur = seasons[:i], seasons[i]
        fit = V.Fit(stints[stints.season.isin(prev)].reset_index(drop=True), players)
        pp = players[players.season.isin(prev)]
        league_Z = pp[[f"pc{k+1}" for k in range(MT.DIMS)]].values
        km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
        centroids = km.cluster_centers_
        thresh = np.median(np.linalg.norm(league_Z - centroids[km.labels_], axis=1)) * 1.5
        cur_players = players[(players.season == cur) & (players.minutes >= min_minutes)]
        # only players with prior-season history can be evaluated with prior-only information
        known = set(fit.add_o["player"].index)
        for team, r in cur_players.groupby("team"):
            r = r[r.pid.isin(known)].sort_values("minutes", ascending=False).reset_index(drop=True)
            if len(r) < 7:
                continue
            f = roster_features(r, fit, pp, centroids, thresh, league_Z)
            f.update(season=cur, team=team,
                     known_share=float(r.minutes.sum() /
                                       cur_players[cur_players.team == team].minutes.sum()))
            rows.append(f)
    tab = pd.DataFrame(rows)
    # attach realised net rating
    import json, glob
    abbr = {}
    for fpath in glob.glob(os.path.join(ROOT, "data", "raw", "Player_*.json")):
        for x in json.load(open(fpath)):
            abbr[x["TeamAbbreviation"]] = str(x["TeamId"])
    tab["team_id"] = tab.team.map(abbr)
    tab = tab.merge(tn.rename(columns={"team": "team_id"})[["season", "team_id", "net"]],
                    on=["season", "team_id"], how="left")
    tab.to_csv(os.path.join(OUT, "roster_forward_test.csv"), index=False, encoding="utf-8")
    print(f"team-seasons predicted from prior information only: {len(tab)}")
    print(tab[["value_mw", "value_sum", "bilinear", "ridgepair", "redundancy", "eff_roles",
               "coverage", "scarcity", "net"]].corr()["net"].round(3).to_string())

    # incremental value over the additive baseline
    from numpy.linalg import lstsq
    y = tab.net.values
    def r2(cols):
        X = np.column_stack([np.ones(len(tab))] + [tab[c].values for c in cols])
        b = lstsq(X, y, rcond=None)[0]
        p = X @ b
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    base = ["value_mw", "value_sum"]
    print("\nin-sample R2  additive baseline:", round(r2(base), 3))
    for extra in ["bilinear", "ridgepair", "redundancy", "eff_roles", "coverage", "scarcity"]:
        print(f"  + {extra:12s}: {r2(base + [extra]):.3f}")
    # leave-one-season-out
    print("\nleave-one-season-out R2:")
    for cols, lab in [(base, "additive"), (base + ["redundancy", "eff_roles", "coverage"], "+structure"),
                      (base + ["bilinear"], "+bilinear")]:
        preds, ys = [], []
        for s in tab.season.unique():
            tr, teo = tab[tab.season != s], tab[tab.season == s]
            X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
            b = lstsq(X, tr.net.values, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(teo))] + [teo[c].values for c in cols])
            preds.append(Xt @ b); ys.append(teo.net.values)
        p = np.concatenate(preds); yy = np.concatenate(ys)
        print(f"  {lab:12s}: R2={1 - ((yy-p)**2).sum()/((yy-yy.mean())**2).sum():.3f} "
              f"corr={np.corrcoef(yy,p)[0,1]:.3f}")
    return tab


if __name__ == "__main__":
    main()
