"""Roster-structure metrics for every team-season (no figures): redundancy, connectors, coverage."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import metrics as MT, topology as TP, run_all as RA

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def main(min_minutes=300):
    players, stints = RA.load()
    syn, M, Z = RA.direct_synergy(dims=6)
    tn = RA.team_net(stints)
    import json, glob
    abbr = {}
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Player_*.json")):
        for x in json.load(open(f)):
            abbr[x["TeamAbbreviation"]] = str(x["TeamId"])
    rows = []
    for season in sorted(stints.season.unique()):
        lg = players[players.season == season]
        league_Z = lg[[f"pc{k+1}" for k in range(MT.DIMS)]].values
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
        thresh = np.median(np.linalg.norm(league_Z - km.cluster_centers_[km.labels_], axis=1)) * 1.5
        for team in sorted(lg.team.dropna().unique()):
            r, _ = TP.team_graph(team, season, players, syn, min_minutes=min_minutes)
            if len(r) < 6:
                continue
            ps = MT.pair_stats(r, syn)
            rows.append(dict(season=season, team=team, n=len(r),
                             redundancy=MT.redundancy(r), eff_roles=MT.effective_roles(r),
                             coverage=MT.coverage(r, km.cluster_centers_, thresh),
                             complementarity=ps["complementarity"], fragility=ps["fragility"],
                             connector=ps["connector"]))
    t = pd.DataFrame(rows)
    t["team_id"] = t.team.map(abbr)
    t = t.merge(tn.rename(columns={"team": "team_id"})[["season", "team_id", "net"]],
                on=["season", "team_id"], how="left")
    t.to_csv(os.path.join(OUT, "roster_structure_all.csv"), index=False, encoding="utf-8")
    print(len(t), "team-seasons")
    print("\nmost redundant rosters (players closest together in skill space):")
    print(t.nlargest(8, "redundancy")[["season", "team", "redundancy", "eff_roles", "net"]].to_string(index=False))
    print("\nmost fragile rosters (most dependent on one connector):")
    print(t.nlargest(8, "fragility")[["season", "team", "fragility", "connector", "net"]].to_string(index=False))
    print("\nplayers most often the connector:")
    print(t.connector.value_counts().head(12).to_string())
    print("\ncorrelation of structure with SAME-season net rating (in-sample, circular):")
    print(t[["redundancy", "eff_roles", "coverage", "complementarity", "fragility", "net"]]
          .corr()["net"].round(3).to_string())
    return t


if __name__ == "__main__":
    main()
