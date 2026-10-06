"""Fetch player + lineup season totals from the public pbpstats API."""
import json, os, sys, time, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE = "https://api.pbpstats.com/get-totals/nba"
SEASONS = ["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"]
TYPES = ["Player", "Lineup", "LineupOpponent", "Team", "TeamOpponent", "Opponent"]

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print("  retry", i, e); time.sleep(5 * (i + 1))
    raise RuntimeError(url)

def main():
    os.makedirs(RAW, exist_ok=True)
    for s in SEASONS:
        for t in TYPES:
            p = os.path.join(RAW, f"{t}_{s}.json")
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                print("skip", p); continue
            url = f"{BASE}?Season={s}&SeasonType=Regular%20Season&Type={t}"
            print("fetch", s, t, flush=True)
            try:
                d = get(url)
            except RuntimeError:
                print("  FAILED", s, t); continue
            rows = d.get("multi_row_table_data", [])
            print("   rows:", len(rows))
            if not rows: continue
            json.dump(rows, open(p, "w"))
            time.sleep(1)

if __name__ == "__main__":
    main()
