"""Season-total lineup rows per team (league-wide endpoint truncates at 500 rows).

Caches one file per (season, team) so an interrupted run resumes cheaply.
"""
import json, glob, os, time, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE = os.path.join(RAW, "tl_cache")
SEASONS = ["2025-26","2024-25","2023-24","2022-23","2021-22","2020-21","2019-20","2018-19"]

def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep((8 if getattr(e, "code", None) == 503 else 2) * (i + 1))
    return None

def main():
    os.makedirs(CACHE, exist_ok=True)
    tids = sorted({str(r["TeamId"]) for r in json.load(open(glob.glob(os.path.join(RAW, "Player_*.json"))[0]))})
    print(len(tids), "teams", flush=True)
    for s in SEASONS:
        out = os.path.join(RAW, f"teamlineups_{s}.json")
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print("skip", s, flush=True); continue
        rows = []
        for i, tid in enumerate(tids):
            cp = os.path.join(CACHE, f"{s}_{tid}.json")
            if os.path.exists(cp) and os.path.getsize(cp) > 500:
                rows.append(json.load(open(cp))); continue
            d = get(f"https://api.pbpstats.com/get-totals/nba?Season={s}&SeasonType=Regular%20Season&Type=Lineup&TeamId={tid}")
            # Lineup rows already carry OpponentPoints/DefPoss, so no second request is needed
            r = {"TeamId": tid,
                 "Lineup": (d or {}).get("multi_row_table_data", []),
                 "LineupOpponent": []}
            if r["Lineup"]:
                json.dump(r, open(cp, "w"))
            rows.append(r)
            print("   ", s, i, len(r["Lineup"]), flush=True)
            time.sleep(0.4)
        n = sum(len(r["Lineup"]) for r in rows)
        print(s, "lineups", n, flush=True)
        if n:
            json.dump(rows, open(out, "w"))

if __name__ == "__main__":
    main()
