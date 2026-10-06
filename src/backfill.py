"""Backfill 1996-97 .. 2017-18: player totals, game results, and per-team lineup totals.

1996-97 is the earliest season with play-by-play, and therefore the earliest season from which
5-man lineups can be reconstructed. Anything before that returns zero rows from this source.
"""
import io, json, os, sys, time, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE = os.path.join(RAW, "tl_cache")
SEASONS = [f"{y}-{str(y+1)[2:]}" for y in range(1996, 2018)]


def get(url, tries=5, timeout=180):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            time.sleep((10 if getattr(e, "code", None) == 503 else 3) * (i + 1))
    return None


def players_and_games():
    for s in SEASONS:
        p = os.path.join(RAW, f"Player_{s}.json")
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            d = get(f"https://api.pbpstats.com/get-totals/nba?Season={s}"
                    f"&SeasonType=Regular%20Season&Type=Player")
            rows = (d or {}).get("multi_row_table_data", [])
            print("players", s, len(rows), flush=True)
            if rows:
                json.dump(rows, io.open(p, "w", encoding="utf-8"))
            time.sleep(1)
        g = os.path.join(RAW, f"games_{s}.json")
        if not (os.path.exists(g) and os.path.getsize(g) > 1000):
            d = get(f"https://api.pbpstats.com/get-games/nba?Season={s}&SeasonType=Regular%20Season")
            rows = (d or {}).get("results", [])
            print("games", s, len(rows), flush=True)
            if rows:
                json.dump(rows, io.open(g, "w", encoding="utf-8"))
            time.sleep(1)


def team_ids(season):
    p = os.path.join(RAW, f"Player_{season}.json")
    if not os.path.exists(p):
        return []
    return sorted({str(r["TeamId"]) for r in json.load(io.open(p, encoding="utf-8"))})


def lineups():
    os.makedirs(CACHE, exist_ok=True)
    for s in SEASONS:
        out = os.path.join(RAW, f"teamlineups_{s}.json")
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print("skip lineups", s, flush=True); continue
        tids = team_ids(s)
        if not tids:
            print("no teams for", s, flush=True); continue
        rows = []
        for i, tid in enumerate(tids):
            cp = os.path.join(CACHE, f"{s}_{tid}.json")
            if os.path.exists(cp) and os.path.getsize(cp) > 500:
                rows.append(json.load(io.open(cp, encoding="utf-8"))); continue
            d = get(f"https://api.pbpstats.com/get-totals/nba?Season={s}"
                    f"&SeasonType=Regular%20Season&Type=Lineup&TeamId={tid}")
            r = {"TeamId": tid, "Lineup": (d or {}).get("multi_row_table_data", []),
                 "LineupOpponent": []}
            if r["Lineup"]:
                json.dump(r, io.open(cp, "w", encoding="utf-8"))
            rows.append(r)
            time.sleep(0.4)
        n = sum(len(r["Lineup"]) for r in rows)
        print("lineups", s, len(tids), "teams", n, "rows", flush=True)
        if n:
            json.dump(rows, io.open(out, "w", encoding="utf-8"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "players"):
        players_and_games()
    if what in ("all", "lineups"):
        lineups()
    print("BACKFILL DONE")
