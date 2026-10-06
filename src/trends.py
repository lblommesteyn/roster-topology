"""How roster construction has changed, 1996-97 to 2025-26.

Everything here is descriptive: league-level aggregates per season, computed on the same skill
embedding used everywhere else. The embedding is z-scored WITHIN season, so a trend in a z-scored
feature would be meaningless; trends are therefore reported on raw rate stats and on quantities
that are comparable across seasons by construction (dispersion, concentration, shares).
"""
import os, sys, glob, json, io, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, metrics as MT

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, OUT = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "out")


def raw_league_rates():
    """Un-standardized league rates per season, straight from the raw player totals."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "Player_*.json"))):
        season = os.path.basename(p)[len("Player_"):-len(".json")]
        d = json.load(io.open(p, encoding="utf-8"))
        d = [x for x in d if (x.get("Minutes") or 0) >= 400]
        mins = np.array([x["Minutes"] for x in d], float)
        def wavg(key):
            v = np.array([x.get(key) or 0 for x in d], float)
            return float(np.average(v, weights=mins))
        rows.append(dict(
            season=season, players=len(d),
            fg3a_rate=wavg("FG3APct"), rim_freq=wavg("AtRimFrequency"),
            c3_freq=wavg("Corner3Frequency"), arc3_freq=wavg("Arc3Frequency"),
            lmid_freq=wavg("LongMidRangeFrequency"), smid_freq=wavg("ShortMidRangeFrequency"),
            ts=wavg("TsPct"), assisted3=wavg("Assisted3sPct"), assisted2=wavg("Assisted2sPct"),
            usage_top=float(np.mean(sorted([x.get("Usage") or 0 for x in d])[-30:])),
        ))
    return pd.DataFrame(rows)


def roster_shape_trends(ctx, seasons=None):
    """Per-season league averages of the structure metrics, plus talent concentration.

    Iterates the seasons present in the PLAYER table (30) rather than the lineup table (8).
    Metrics that need only skill space - redundancy, effective roles, coverage, spread - are valid
    for all of them. Complementarity and fragility use the interaction surface fit on 2018-2026
    lineups, so for earlier seasons they are an extrapolation and are not reported as trends.
    """
    rows = []
    for s in (seasons or ctx.seasons):
        per_team = []
        for t in ctx.teams(s):
            r = ctx.roster(t, s)
            if len(r) < 6:
                continue
            st = ctx.structure(r)
            w = r.minutes.values.astype(float)
            share = np.sort(w)[::-1] / w.sum()
            per_team.append(dict(
                redundancy=st["redundancy"], eff_roles=st["eff_roles"],
                complementarity=st["complementarity"], fragility=st["fragility"],
                top3_minutes_share=float(share[:3].sum()),
                rotation_size=float((w >= 0.04 * w.sum()).sum()),
                spread=float(np.mean(np.std(r[["pc1", "pc2", "pc3"]].values, axis=0))),
            ))
        d = pd.DataFrame(per_team).mean().to_dict()
        lg = ctx.players[(ctx.players.season == s) & (ctx.players.minutes >= ctx.min_minutes)]
        # box value is defined for all 30 seasons; the lineup-ridge value only for 2018 onward
        vcol = "box_value" if "box_value" in lg.columns else "value"
        # league-wide dispersion of playing styles, and how unequal value is
        # player value can be negative, so shares of a sum are unstable; use spread and gaps
        v = np.sort(lg[vcol].values)
        d.update(season=s,
                 league_style_spread=float(np.mean(np.std(lg[["pc1", "pc2", "pc3"]].values, axis=0))),
                 value_sd=float(v.std()),
                 star_gap=float(np.percentile(v, 95) - np.median(v)),
                 top30_mean_value=float(v[-30:].mean()))
        rows.append(d)
    return pd.DataFrame(rows)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def within_team_similarity(ctx, seasons=None):
    """Are team-mates becoming more alike, or less? Mean pairwise skill distance inside rosters."""
    rows = []
    for s in (seasons or ctx.seasons):
        vals = []
        for t in ctx.teams(s):
            r = ctx.roster(t, s)
            if len(r) < 6:
                continue
            Z = r[["pc1", "pc2", "pc3"]].values
            w = r.minutes.values.astype(float)
            D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
            iu = np.triu_indices(len(r), 1)
            ww = np.outer(w, w)[iu]
            vals.append(float(np.average(D[iu], weights=ww)))
        rows.append(dict(season=s, mean_pair_distance=float(np.mean(vals))))
    return pd.DataFrame(rows)


def main():
    ctx = V.Ctx()
    bv_path = os.path.join(ROOT, "data", "proc", "box_value.parquet")
    if os.path.exists(bv_path):
        bv = pd.read_parquet(bv_path)
        ctx.players = ctx.players.merge(bv[["season", "pid", "box_value"]],
                                        on=["season", "pid"], how="left")
    seasons = sorted(ctx.players.season.unique())
    print(f"seasons in the player table: {len(seasons)} ({seasons[0]} to {seasons[-1]})")
    rates = raw_league_rates()
    shape = roster_shape_trends(ctx, seasons)
    sim = within_team_similarity(ctx, seasons)
    tab = rates.merge(shape, on="season").merge(sim, on="season")
    tab.to_csv(os.path.join(OUT, "league_trends.csv"), index=False, encoding="utf-8")

    show = ["season", "fg3a_rate", "rim_freq", "lmid_freq", "ts", "assisted3",
            "mean_pair_distance", "league_style_spread", "eff_roles", "redundancy",
            "rotation_size", "top3_minutes_share", "star_gap"]
    print("=== league trends ===")
    print(tab[show].iloc[::3].round(3).to_string(index=False))

    print("\n=== trend strength (Spearman vs season order; |rho| > 0.74 is p < 0.05 at n=8) ===")
    from scipy.stats import spearmanr
    idx = np.arange(len(tab))
    out = []
    for c in show[1:]:
        rho, p = spearmanr(idx, tab[c].values)
        out.append((c, tab[c].iloc[0], tab[c].iloc[-1], rho, p))
    out.sort(key=lambda x: -abs(x[3]))
    for c, a, b, rho, p in out:
        flag = "TREND" if p < 0.05 else ("weak" if p < 0.15 else "no trend")
        print(f"  {c:22s} {a:8.3f} -> {b:8.3f}  ({b-a:+.3f})  rho {rho:+.2f}  p {p:.3f}  {flag}")
    return tab


if __name__ == "__main__":
    main()
