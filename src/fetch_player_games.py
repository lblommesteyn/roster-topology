"""Per-game player minutes (and a few counting stats) for within-season blob evolution."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE = os.path.join(RAW, "pg_cache")
SEASONS = ["2025-26"]
KEEP = ["EntityId", "Name", "Minutes", "OffPoss", "DefPoss", "Points", "FG3A", "FG2A",
        "AtRimFGA", "Assists", "OffRebounds", "DefRebounds", "Turnovers", "Blocks", "Steals"]


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep((8 if getattr(e, "code", None) == 503 else 2) * (i + 1))
    return None


def one(g):
    gid = g["GameId"]
    cp = os.path.join(CACHE, f"{gid}.json")
    if os.path.exists(cp) and os.path.getsize(cp) > 200:
        return json.load(open(cp))
    time.sleep(0.2)
    d = get(f"https://api.pbpstats.com/get-game-stats?GameId={gid}&Type=Player")
    if not d or "stats" not in d:
        return None
    out = {"GameId": gid, "Date": g["Date"]}
    for side, key in (("Home", "HomeTeamAbbreviation"), ("Away", "AwayTeamAbbreviation")):
        rows = d["stats"][side].get("FullGame", [])
        out[side] = {"team": g[key],
                     "players": [{k: r[k] for k in KEEP if k in r}
                                 for r in rows if r.get("EntityId") not in (None, "0")]}
    json.dump(out, open(cp, "w"))
    return out


def main():
    os.makedirs(CACHE, exist_ok=True)
    for s in SEASONS:
        p = os.path.join(RAW, f"playergames_{s}.json")
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            print("skip", s); continue
        gl = get(f"https://api.pbpstats.com/get-games/nba?Season={s}&SeasonType=Regular%20Season")
        games = (gl or {}).get("results", [])
        print(s, len(games), "games", flush=True)
        res = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            for i, r in enumerate(ex.map(one, games)):
                if r:
                    res.append(r)
                if i % 100 == 0:
                    print("  ", s, i, len(res), flush=True)
        print(s, "done", len(res), flush=True)
        if res:
            json.dump(res, open(p, "w"))


if __name__ == "__main__":
    main()
