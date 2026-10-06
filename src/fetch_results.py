"""Game results per season -> actual team wins (the target for any projection)."""
import io, json, os, time, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEASONS = ["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"]


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            time.sleep((10 if getattr(e, "code", None) == 503 else 3) * (i + 1))
    return None


def main():
    for s in SEASONS:
        p = os.path.join(RAW, f"games_{s}.json")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print("skip", s); continue
        d = get(f"https://api.pbpstats.com/get-games/nba?Season={s}&SeasonType=Regular%20Season")
        rows = (d or {}).get("results", [])
        print(s, len(rows), "games", flush=True)
        if rows:
            json.dump(rows, io.open(p, "w", encoding="utf-8"))
        time.sleep(1.5)


if __name__ == "__main__":
    main()
