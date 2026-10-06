"""Fetch per-game, per-lineup stint totals (FullGame rows) for both teams."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEASONS = ["2023-24", "2024-25", "2025-26", "2022-23", "2021-22"]
KEEP = ["EntityId", "Minutes", "PlusMinus", "OffPoss", "DefPoss", "Points", "OpponentPoints",
        "Assists", "Turnovers", "OffRebounds", "DefRebounds", "FG3A", "FG3M", "FG2A", "FG2M",
        "AtRimFGA", "AtRimFGM", "FtPoints", "Fouls", "Steals", "Blocks", "SecondChancePoints"]

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            code = getattr(e, "code", None)
            time.sleep((8 if code == 503 else 2) * (i + 1))
    return None

def trim(rows):
    return [{k: r[k] for k in KEEP if k in r} for r in rows]

def one_game(g):
    gid = g["GameId"]
    time.sleep(0.15)
    d = get(f"https://api.pbpstats.com/get-game-stats?GameId={gid}&Type=Lineup")
    if not d or "stats" not in d:
        return None
    out = {"GameId": gid, "Date": g["Date"], "HomeTeamId": g["HomeTeamId"], "AwayTeamId": g["AwayTeamId"],
           "HomeAbbr": g["HomeTeamAbbreviation"], "AwayAbbr": g["AwayTeamAbbreviation"],
           "HomePoints": g["HomePoints"], "AwayPoints": g["AwayPoints"]}
    for side in ("Home", "Away"):
        out[side] = trim(d["stats"][side].get("FullGame", []))
    return out

def main():
    for s in SEASONS:
        p = os.path.join(RAW, f"gamelineups_{s}.json")
        if os.path.exists(p) and os.path.getsize(p) > 5000:
            print("skip", s); continue
        gl = get(f"https://api.pbpstats.com/get-games/nba?Season={s}&SeasonType=Regular%20Season")
        if not gl or not gl.get("results"):
            print("no games for", s); continue
        games = gl["results"]
        print(s, len(games), "games", flush=True)
        res = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            for i, r in enumerate(ex.map(one_game, games)):
                if r: res.append(r)
                if i % 100 == 0: print("  ", s, i, flush=True)
        print(s, "done", len(res), flush=True)
        json.dump(res, open(p, "w"))

if __name__ == "__main__":
    main()
