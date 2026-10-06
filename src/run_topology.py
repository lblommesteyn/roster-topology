"""Team topology figures + roster-structure metrics, and whether structure predicts performance."""
import os, sys, json, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B, topology as TP, metrics as MT, run_all as RA

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT, FIGS = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out"), os.path.join(ROOT, "figs")


def main(season=None, dims=8):
    players, stints = RA.load()
    seasons = sorted(stints.season.unique())
    season = season or seasons[-1]
    pairs = pd.read_parquet(os.path.join(PROC, "pairs.parquet"))
    eff_o = pd.read_parquet(os.path.join(PROC, "player_eff_off.parquet"))["eff"].to_dict()
    eff_d = pd.read_parquet(os.path.join(PROC, "player_eff_def.parquet"))["eff"].to_dict()
    syn, M, Z = RA.direct_synergy(dims=6)

    # league manifold reference for coverage / scarcity
    lg = players[players.season == season]
    league_Z = lg[[f"pc{k+1}" for k in range(MT.DIMS)]].values
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
    centroids = km.cluster_centers_
    thresh = np.median(np.linalg.norm(league_Z - centroids[km.labels_], axis=1)) * 1.5

    tn = RA.team_net(stints)
    rows = []
    os.makedirs(FIGS, exist_ok=True)
    for team in sorted(players[players.season == season].team.dropna().unique()):
        r, W = TP.team_graph(team, season, players, syn)
        if len(r) < 6:
            continue
        abbr = team
        TP.draw(abbr, season, r, W, os.path.join(FIGS, f"topology_{season}_{abbr}.png"))
        ps = MT.pair_stats(r, syn)
        tid = None
        rows.append(dict(season=season, team=abbr, n=len(r),
                         redundancy=MT.redundancy(r), eff_roles=MT.effective_roles(r),
                         coverage=MT.coverage(r, centroids, thresh), scarcity=MT.scarcity(r, league_Z),
                         complementarity=ps["complementarity"], neg_share=ps["neg_share"],
                         fragility=ps["fragility"], connector=ps["connector"],
                         value_sum=sum(eff_o.get(p, 0) - eff_d.get(p, 0) for p in r.pid),
                         value_mw=float(np.average([eff_o.get(p, 0) - eff_d.get(p, 0) for p in r.pid],
                                                   weights=r.minutes))))
    tab = pd.DataFrame(rows)
    abbr_map = {}
    for f in glob.glob(os.path.join(ROOT, "data", "raw", "Player_*.json")):
        for x in json.load(open(f)):
            abbr_map[x["TeamAbbreviation"]] = str(x["TeamId"])
    tab["team_id"] = tab.team.map(abbr_map)
    tab = tab.merge(tn[tn.season == season][["team", "net"]], left_on="team_id", right_on="team", how="left",
                    suffixes=("", "_y")).drop(columns=["team_y"])
    tab.to_csv(os.path.join(OUT, f"roster_metrics_{season}.csv"), index=False, encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    num = ["redundancy", "eff_roles", "coverage", "scarcity", "complementarity", "neg_share",
           "fragility", "value_sum", "value_mw"]
    print("\ncorrelation with team net rating:")
    print(tab[num + ["net"]].corr()["net"].round(3).to_string())
    return tab


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
