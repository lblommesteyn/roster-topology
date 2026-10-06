"""Counterfactual roster swap tool.

  python src/swap_cli.py --team OKC --season 2025-26 --out "Isaiah Hartenstein" --in "Nikola Jokic"

Reports the change in roster structure, complementarity, best predicted lineups and projected
lineup performance. Works for incoming players who never shared a floor with the roster,
because pair fit comes from the bilinear model over skill space.
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import bilinear as B, metrics as MT, swaps as SW, run_all as RA, topology as TP

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, FIGS = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "figs")
DIMS = 8


def context(season=None):
    players, stints = RA.load()
    seasons = sorted(stints.season.unique())
    season = season or seasons[-1]
    M_unused = None
    syn, M, Z = RA.direct_synergy(dims=6)
    eff_o = pd.read_parquet(os.path.join(PROC, "player_eff_off.parquet"))["eff"].to_dict()
    eff_d = pd.read_parquet(os.path.join(PROC, "player_eff_def.parquet"))["eff"].to_dict()
    lg = players[players.season == season]
    league_Z = lg[[f"pc{k+1}" for k in range(MT.DIMS)]].values
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
    thresh = np.median(np.linalg.norm(league_Z - km.cluster_centers_[km.labels_], axis=1)) * 1.5
    return dict(players=players, season=season, syn=syn, eff_o=eff_o, eff_d=eff_d,
                league_Z=league_Z, centroids=km.cluster_centers_, thresh=thresh, M=M, Z=Z)


def show(tag, d):
    print(f"  {tag:10s} redundancy {d['redundancy']:+.3f} | roles {d['eff_roles']:.2f} | "
          f"coverage {d['coverage']:.2f} | scarcity {d['scarcity']:.2f} | "
          f"complementarity {d['complementarity']:+.3f} | frag {d['fragility']:+.3f} "
          f"({d['connector']}) | value {d['player_value_mw']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--season", default=None)
    ap.add_argument("--out", dest="outgoing", required=True)
    ap.add_argument("--in", dest="incoming", required=True)
    ap.add_argument("--min-minutes", type=int, default=300)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    C = context(a.season)
    players, season = C["players"], a.season or C["season"]
    roster = SW.team_roster(players, a.team, season, a.min_minutes)
    if not len(roster):
        raise SystemExit(f"no roster for {a.team} {season}")
    cand = players[(players.season == season) & (players.name.str.lower() == a.incoming.lower())]
    if not len(cand):
        cand = players[players.name.str.lower() == a.incoming.lower()].sort_values("season")
    if not len(cand):
        raise SystemExit(f"unknown incoming player {a.incoming}")
    inc = cand.iloc[-1]
    before, after, delta, new = SW.replace_player(
        players, roster, a.outgoing, inc, C["syn"], C["league_Z"], C["centroids"],
        C["thresh"], C["eff_o"], C["eff_d"])
    print(f"\n{a.team} {season}: OUT {a.outgoing}  ->  IN {inc['name']} ({inc.season})\n")
    show("before", before); show("after", after)
    print("\n  deltas: " + ", ".join(f"{k} {v:+.3f}" for k, v in delta.items()))
    for tag, r in (("BEFORE", roster), ("AFTER", new)):
        best, worst = SW.predicted_top_lineup(r, C["syn"], C["eff_o"], C["eff_d"])
        print(f"\n  {tag} best predicted units:")
        for s, nm in best:
            print(f"    {s:+7.2f}  " + ", ".join(x.split()[-1] for x in nm))
    if a.figure:
        os.makedirs(FIGS, exist_ok=True)
        for tag, r in (("before", roster), ("after", new)):
            _, W = TP.team_graph(a.team, season, r.assign(team=a.team, season=season),
                                 C["syn"], min_minutes=0)
            TP.draw(f"{a.team} {tag} swap", season, r, W,
                    os.path.join(FIGS, f"swap_{a.team}_{season}_{tag}.png"))
        print(f"\n  figures written to figs/swap_{a.team}_{season}_*.png")


if __name__ == "__main__":
    main()
