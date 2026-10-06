"""Within-season blob evolution from month-by-month player totals.

Two things move the organism across a season: who is in the rotation and for how long (exact), and
each player's own form and role (noisy). Measured month to month, a player's position wanders with
a standard deviation of about 0.21, 0.29 and 0.62 of the league spread on PC1, PC2 and PC3. Using
raw monthly positions would make every blob shimmer from sampling noise.

So monthly positions are shrunk toward the player's season position by the measured reliability
(1 - noise ratio): 0.79 on PC1, 0.71 on PC2, 0.38 on PC3. Minutes and membership are used exactly
as measured, because those are counted, not estimated.
"""
import os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, blob_scenes as BS

PROC = os.path.join(V.ROOT, "data", "proc")
SHRINK = {"pc1": 0.79, "pc2": 0.71, "pc3": 0.38}
MIN_MONTH_SHARE = 0.04          # share of the team's monthly minutes to count as rotation
PCS = [f"pc{k+1}" for k in range(6)]


def load(season="2025-26"):
    p = os.path.join(PROC, f"players_monthly_{season}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def shrunk_positions(monthly, season_players, season):
    """Blend each monthly position toward the player's season position."""
    s = season_players[season_players.season == season].drop_duplicates("pid").set_index("pid")
    out = monthly.copy()
    for c, k in SHRINK.items():
        base = out.pid.map(s[c])
        out[c] = np.where(base.notna(), base + k * (out[c] - base.fillna(0)), out[c])
    for c in PCS[3:]:
        base = out.pid.map(s[c]) if c in s.columns else None
        if base is not None:
            out[c] = np.where(base.notna(), base, out[c])
    for col in ("value", "leaning", "off_eff", "def_eff"):
        if col in s.columns:
            out[col] = out.pid.map(s[col]).fillna(0.0)
    return out


def month_rosters(ctx, monthly, team, min_share=MIN_MONTH_SHARE):
    out = []
    sub = monthly[monthly.team == team]
    for month, g in sub.groupby("month"):
        tot = g.minutes.sum()
        g = g[g.minutes / max(tot, 1) >= min_share]
        if len(g) < 5:
            continue
        r = g.sort_values("minutes", ascending=False).reset_index(drop=True)
        out.append((month, r))
    return out


def annotate(windows, ctx):
    """Name who entered and left the rotation between consecutive months."""
    nm = ctx.players.drop_duplicates("pid").set_index("pid")["name"]
    labelled, prev = [], None
    for month, r in windows:
        note = ""
        if prev is not None:
            gone = [p for p in prev.pid if p not in set(r.pid)]
            new = [p for p in r.pid if p not in set(prev.pid)]
            bits = []
            if new:
                bits.append("rotation in: " + ", ".join(V.surname(nm.get(p, p)) for p in new[:4]))
            if gone:
                bits.append("out: " + ", ".join(V.surname(nm.get(p, p)) for p in gone[:4]))
            note = "  |  ".join(bits)
        labelled.append((month, r, note))
        prev = r
    return labelled


def main(teams=None, season="2025-26"):
    ctx = V.Ctx()
    m = load(season)
    if m is None or not len(m):
        print("no monthly data; run src/fetch_monthly.py then src/monthly.py")
        return None
    m = shrunk_positions(m, ctx.players, season)
    teams = teams or ["OKC", "MIN", "GSW", "DAL", "MEM", "NYK"]
    rows = []
    for t in teams:
        w = annotate(month_rosters(ctx, m, t), ctx)
        if len(w) < 3:
            print("  too few months for", t); continue
        res = BS.window_evolution(ctx, t, w,
                                  os.path.join(V.VIZ, "blobs", f"season_{t}_{season}.html"))
        if res:
            df = res[1]; df.insert(0, "team", t); rows.append(df)
            print(f"  within-season blob: {t} ({len(w)} months)", flush=True)
    if rows:
        out = pd.concat(rows)
        out.to_csv(os.path.join(V.OUT, "blob_within_season.csv"), index=False, encoding="utf-8")
        print()
        print(out.round(3).to_string(index=False))
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or None)
