"""Month-by-month player totals: one request per month, thanks to FromDate/ToDate on get-totals."""
import io, json, os, time, urllib.request
from datetime import date

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MONTHS = {
    "2025-26": [("2025-10", "2025-10-01", "2025-10-31"), ("2025-11", "2025-11-01", "2025-11-30"),
                ("2025-12", "2025-12-01", "2025-12-31"), ("2026-01", "2026-01-01", "2026-01-31"),
                ("2026-02", "2026-02-01", "2026-02-28"), ("2026-03", "2026-03-01", "2026-03-31"),
                ("2026-04", "2026-04-01", "2026-04-30")],
}


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            time.sleep((10 if getattr(e, "code", None) == 503 else 3) * (i + 1))
    return None


def main():
    out = os.path.join(RAW, "monthly")
    os.makedirs(out, exist_ok=True)
    for season, months in MONTHS.items():
        for label, a, b in months:
            p = os.path.join(out, f"Player_{season}_{label}.json")
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                print("skip", label); continue
            url = (f"https://api.pbpstats.com/get-totals/nba?Season={season}"
                   f"&SeasonType=Regular%20Season&Type=Player&FromDate={a}&ToDate={b}")
            d = get(url)
            rows = (d or {}).get("multi_row_table_data", [])
            print(label, len(rows), "players", flush=True)
            if rows:
                json.dump(rows, io.open(p, "w", encoding="utf-8"))
            time.sleep(2)


if __name__ == "__main__":
    main()
