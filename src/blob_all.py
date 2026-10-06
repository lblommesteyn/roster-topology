"""Generate every blob deliverable."""
import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, blob as B, blob_viz as BV, blob_scenes as BS

OUT = os.path.join(V.VIZ, "blobs")

TRADES = [
    ("MIN", "2025-26", "Rudy Gobert", "LaMelo Ball", "swap the rim anchor for a lead guard"),
    ("GSW", "2025-26", "Al Horford", "Nikola Jokic", "the flattest roster meets a passing hub"),
    ("OKC", "2025-26", "Isaiah Hartenstein", "Draymond Green", "rim big out, passing big in"),
    ("MEM", "2025-26", "Javon Small", "Rudy Gobert", "a guard-heavy roster grows an interior lobe"),
    ("DAL", "2025-26", "Daniel Gafford", "Myles Turner", "double-big roster adds spacing"),
]
ROSTER_MORPHS = [
    ("MIN", "2021-22", "2022-23", "Gobert arrives"),
    ("NYK", "2023-24", "2024-25", "Towns and Bridges arrive"),
    ("PHX", "2022-23", "2023-24", "Durant and Beal era"),
    ("DAL", "2024-25", "2025-26", "post-Doncic roster"),
]
EVOLVE = ["OKC", "GSW", "BOS", "MIN", "DAL", "CLE", "MEM", "NYK"]


def main(stages=None):
    stages = stages or ["teams", "gallery", "evolution", "morphs", "metrics"]
    os.makedirs(OUT, exist_ok=True)
    ctx = V.Ctx()
    season = ctx.seasons[-1]
    t0 = time.time()

    if "teams" in stages:
        for t in ctx.teams(season):
            r = BS.single_team(ctx, t, season)
            if r:
                print("  blob:", t, flush=True)
        print(f"[{time.time()-t0:.0f}s] team viewers done", flush=True)

    if "gallery" in stages:
        BS.league_gallery(ctx, season,
                          path_html=os.path.join(V.VIZ, f"blob_gallery_{season}.html"),
                          path_png=os.path.join(V.VIZ, f"blob_gallery_{season}.png"))
        print(f"[{time.time()-t0:.0f}s] gallery done", flush=True)

    if "evolution" in stages:
        rows = []
        for t in EVOLVE:
            res = BS.season_evolution(ctx, t, os.path.join(OUT, f"evolution_{t}.html"))
            if res:
                df = res[1]; df.insert(0, "team", t); rows.append(df)
                print("  evolution:", t, flush=True)
        if rows:
            pd.concat(rows).to_csv(os.path.join(V.OUT, "blob_evolution_metrics.csv"),
                                   index=False, encoding="utf-8")
        print(f"[{time.time()-t0:.0f}s] evolution done", flush=True)

    if "morphs" in stages:
        for team, s, og, inc, note in TRADES:
            try:
                BS.trade_morph(ctx, team, s, og, inc,
                               os.path.join(OUT, f"morph_{team}_{V.surname(og)}_to_{V.surname(inc)}.html"),
                               title_note=note)
                print("  morph:", team, og, "->", inc, flush=True)
            except Exception as e:
                print("  SKIP morph", team, e, flush=True)
        for team, s0, s1, note in ROSTER_MORPHS:
            try:
                BS.roster_morph(ctx, team, s0, s1,
                                os.path.join(OUT, f"morph_{team}_{s0}_to_{s1}.html"), note=note)
                print("  roster morph:", team, s0, "->", s1, flush=True)
            except Exception as e:
                print("  SKIP roster morph", team, e, flush=True)
        print(f"[{time.time()-t0:.0f}s] morphs done", flush=True)

    if "metrics" in stages:
        lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
        rows = []
        for s in ctx.seasons:
            for t in ctx.teams(s):
                r = ctx.roster(t, s)
                if len(r) < 5:
                    continue
                bl = B.team_blob(ctx, r, lo, hi, n=44)
                if bl is None:
                    continue
                st = ctx.structure(r)
                rows.append(dict(season=s, team=t, net=ctx.net_rating(t, s),
                                 **bl["metrics"],
                                 complementarity=st["complementarity"],
                                 redundancy=st["redundancy"], eff_roles=st["eff_roles"],
                                 fragility=st["fragility"], connector=st["connector"]))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(V.OUT, "blob_metrics.csv"), index=False, encoding="utf-8")
        print(f"[{time.time()-t0:.0f}s] metrics done: {len(df)} team-seasons", flush=True)
        num = ["volume", "sphericity", "components", "peak_density", "density_ratio",
               "largest_share"]
        print("\ncorrelation of blob shape with the graph-based structure metrics:")
        print(df[num + ["complementarity", "redundancy", "eff_roles", "fragility", "net"]]
              .corr().loc[num, ["complementarity", "redundancy", "eff_roles", "fragility", "net"]]
              .round(3).to_string())
    print("BLOBS DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
