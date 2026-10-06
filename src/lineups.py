"""Flatten lineup data into a modeling table.

Two sources, same schema:
  teamlineups_<season>.json  - season totals per (team, 5-man lineup)  [primary]
  gamelineups_<season>.json  - per-game stints with opponent + home    [optional, rate-limited]
"""
import json, os, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, PROC = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "proc")


def from_team_totals():
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "teamlineups_*.json"))):
        season = os.path.basename(p)[len("teamlineups_"):-len(".json")]
        for t in json.load(open(p)):
            opp = {r["EntityId"]: r for r in t["LineupOpponent"]}
            for r in t["Lineup"]:
                o = opp.get(r["EntityId"], {})
                op, dp = r.get("OffPoss", 0), r.get("DefPoss", 0) or o.get("OffPoss", 0)
                if op < 1 or dp < 1:
                    continue
                rows.append(dict(season=season, team=str(t["TeamId"]), opp="ALL", home=0,
                                 lineup=r["EntityId"], game="season",
                                 off_poss=op, def_poss=dp,
                                 pts=r.get("Points", 0),
                                 opp_pts=o.get("Points", r.get("OpponentPoints", 0))))
    return pd.DataFrame(rows)


def from_games():
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "gamelineups_*.json"))):
        season = os.path.basename(p)[len("gamelineups_"):-len(".json")]
        for g in json.load(open(p)):
            for side in ("Home", "Away"):
                tid = g["HomeTeamId"] if side == "Home" else g["AwayTeamId"]
                oid = g["AwayTeamId"] if side == "Home" else g["HomeTeamId"]
                for r in g[side]:
                    op, dp = r.get("OffPoss", 0), r.get("DefPoss", 0)
                    if op + dp == 0:
                        continue
                    rows.append(dict(season=season, team=str(tid), opp=str(oid),
                                     home=int(side == "Home"), lineup=r["EntityId"], game=g["GameId"],
                                     off_poss=op, def_poss=dp, pts=r.get("Points", 0),
                                     opp_pts=r.get("OpponentPoints", 0)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(PROC, exist_ok=True)
    df = from_team_totals()
    if len(df):
        df.to_parquet(os.path.join(PROC, "stints.parquet"))
        print("season-total lineup rows:", len(df))
        print(df.groupby("season").agg(lineups=("pts", "size"), offposs=("off_poss", "sum")))
        print("possession quantiles:", df.off_poss.quantile([.1, .5, .9, .99]).round(1).to_dict())
    g = from_games()
    if len(g):
        g.to_parquet(os.path.join(PROC, "stints_game.parquet"))
        print("game stint rows:", len(g))


if __name__ == "__main__":
    main()
