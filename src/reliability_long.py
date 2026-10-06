"""Is pair chemistry measurable? The central negative result, rerun on 30 seasons.

Three questions the 8-season version could not answer:

  1. Does adjusted pair synergy replicate any better with 3.75x the data?
  2. Does it replicate better in EARLIER eras? Rotations were shorter and star minute
     concentration far higher (44% of minutes to the top three in 1996-97 against 35% now), so
     pairs accumulated more shared possessions. Sample size is the obvious explanation for a null,
     and the old eras are the natural place to test it.
  3. Does the structured interaction surface replicate across disjoint seasons within each era?

Each era is analysed independently so that nothing is pooled across rule changes.
"""
import os, sys, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, synergy as S, validate as VD, bilinear as B, direct as D

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")

ERAS = [("1996-2001", ["1996-97", "1997-98", "1998-99", "1999-00", "2000-01"]),
        ("2001-2006", ["2001-02", "2002-03", "2003-04", "2004-05", "2005-06"]),
        ("2006-2011", ["2006-07", "2007-08", "2008-09", "2009-10", "2010-11"]),
        ("2011-2016", ["2011-12", "2012-13", "2013-14", "2014-15", "2015-16"]),
        ("2016-2021", ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]),
        ("2021-2026", ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])]


def sb(r):
    """Spearman-Brown: full-sample reliability implied by a half-sample correlation."""
    return 2 * r / (1 + r) if r > -1 else np.nan


def split_half_era(stints, players, seeds=(0, 1)):
    """Split lineup rows at random, estimate pair effects twice, compare."""
    out = []
    cnt = S.pair_counts(stints, "off")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        f = rng.integers(0, 2, len(stints))
        A = VD.Fit(stints[f == 0].reset_index(drop=True), players)
        C = VD.Fit(stints[f == 1].reset_index(drop=True), players)
        common = sorted(set(A.ridge_pairs) & set(C.ridge_pairs))
        if len(common) < 40:
            continue
        x = np.array([A.ridge_pairs[k] for k in common])
        y = np.array([C.ridge_pairs[k] for k in common])
        poss = np.array([cnt.get(k, 0) for k in common], float)
        ra, rb = A.raw_pairs, C.raw_pairs
        cr = sorted(set(ra) & set(rb))
        ci = A.add_o["player"].index.intersection(C.add_o["player"].index)
        out.append(dict(
            seed=seed, n_pairs=len(common),
            adj_r=float(np.corrcoef(x, y)[0, 1]),
            raw_r=float(np.corrcoef([ra[k] for k in cr], [rb[k] for k in cr])[0, 1]) if len(cr) > 40 else np.nan,
            off_r=float(np.corrcoef(A.add_o["player"][ci], C.add_o["player"][ci])[0, 1]),
            def_r=float(np.corrcoef(A.add_d["player"][ci], C.add_d["player"][ci])[0, 1]),
            median_poss=float(np.median(poss)), p90_poss=float(np.percentile(poss, 90)),
            x=x, y=y, poss=poss))
    return out


def surface_era(stints, players, dims=6):
    """Does the structured interaction surface replicate across DISJOINT seasons in this era?"""
    seasons = sorted(stints.season.unique())
    if len(seasons) < 4:
        return None
    a = [s for i, s in enumerate(seasons) if i % 2 == 0]
    b = [s for i, s in enumerate(seasons) if i % 2 == 1]
    res = []
    for blk in (a, b):
        d = stints[stints.season.isin(blk)].reset_index(drop=True)
        d = d.assign(players=d.lineup.str.split("-"))
        fo = D.fit(d, players, "off", dims=dims)
        fd = D.fit(d, players, "def", dims=dims)
        res.append((D.net_M(fo, fd), fo["Z"]))
    idx = res[0][1].index.intersection(res[1][1].index)
    if len(idx) < 50:
        return None
    s0 = D.surface(res[0][0], res[0][1].loc[idx], dims)
    s1 = D.surface(res[1][0], res[1][1].loc[idx], dims)
    iu = np.triu_indices(len(idx), 1)
    return dict(surface_r=float(np.corrcoef(s0[iu], s1[iu])[0, 1]),
                M_r=float(np.corrcoef(res[0][0].ravel(), res[1][0].ravel())[0, 1]),
                spread=float(s0[iu].std()), n_players=len(idx))


def main():
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    st = S.load_stints()
    rows, poolx, pooly, poolp = [], [], [], []
    for name, seasons in ERAS:
        sub = st[st.season.isin(seasons)].reset_index(drop=True)
        if not len(sub):
            print("no data for", name); continue
        res = split_half_era(sub, players)
        if not res:
            continue
        adj = float(np.mean([r["adj_r"] for r in res]))
        raw = float(np.nanmean([r["raw_r"] for r in res]))
        off = float(np.mean([r["off_r"] for r in res]))
        dfr = float(np.mean([r["def_r"] for r in res]))
        med = float(np.mean([r["median_poss"] for r in res]))
        p90 = float(np.mean([r["p90_poss"] for r in res]))
        surf = surface_era(sub, players)
        rows.append(dict(era=name, rows=len(sub), n_pairs=res[0]["n_pairs"],
                         adj_r=adj, adj_sb=sb(adj), raw_r=raw, off_r=off, def_r=dfr,
                         median_shared_poss=med, p90_shared_poss=p90,
                         surface_r=(surf or {}).get("surface_r", np.nan),
                         M_r=(surf or {}).get("M_r", np.nan),
                         surface_spread=(surf or {}).get("spread", np.nan)))
        for r in res:
            poolx.append(r["x"]); pooly.append(r["y"]); poolp.append(r["poss"])
        print(f"  {name}: adj r={adj:+.3f}  raw r={raw:+.3f}  player off r={off:+.3f}  "
              f"surface r={rows[-1]['surface_r']:+.3f}  median shared poss={med:.0f}", flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "reliability_by_era.csv"), index=False, encoding="utf-8")
    print("\n=== pair-synergy reliability by era ===")
    print(tab[["era", "rows", "n_pairs", "adj_r", "adj_sb", "raw_r", "off_r",
               "median_shared_poss", "surface_r", "M_r"]].round(3).to_string(index=False))

    # the decisive test: does reliability rise with shared possessions, pooling every era?
    x = np.concatenate(poolx); y = np.concatenate(pooly); p = np.concatenate(poolp)
    print("\n=== adjusted-pair reliability by shared possessions (all eras pooled) ===")
    qs = [0, 50, 75, 90, 97, 100]
    edges = np.percentile(p, qs)
    for lo, hi, a, b in zip(edges[:-1], edges[1:], qs[:-1], qs[1:]):
        m = (p >= lo) & (p < hi) if b < 100 else (p >= lo)
        if m.sum() > 200:
            print(f"  pct {a:3d}-{b:3d}  poss {int(lo):5d}-{int(hi):5d}  n={int(m.sum()):6d}  "
                  f"r={np.corrcoef(x[m], y[m])[0,1]:+.3f}")
    print(f"\n  overall: n={len(x)}, r={np.corrcoef(x, y)[0,1]:+.3f}, "
          f"Spearman-Brown {sb(float(np.corrcoef(x, y)[0,1])):+.3f}")

    if len(tab) > 2:
        from scipy.stats import pearsonr
        r1, p1 = pearsonr(tab.median_shared_poss, tab.adj_r)
        print(f"\n  across eras: correlation between median shared possessions and pair "
              f"reliability r={r1:+.2f} (p={p1:.2f}, n={len(tab)} eras)")
    return tab


if __name__ == "__main__":
    main()
