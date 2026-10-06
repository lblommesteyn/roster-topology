"""How much of measured pair synergy is signal? Split-half and across-season replication."""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B, validate as V

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")


def sb(r):
    """Spearman-Brown: reliability of the full sample from a half-sample correlation."""
    return 2 * r / (1 + r) if r > -1 else np.nan


def split_half(stints, players, seeds=(0, 1, 2), min_shared=0):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        f = rng.integers(0, 2, len(stints))
        A = V.Fit(stints[f == 0].reset_index(drop=True), players)
        C = V.Fit(stints[f == 1].reset_index(drop=True), players)
        common = sorted(set(A.ridge_pairs) & set(C.ridge_pairs))
        if len(common) < 30:
            continue
        cnt = S.pair_counts(stints, "off")
        x = np.array([A.ridge_pairs[k] for k in common])
        y = np.array([C.ridge_pairs[k] for k in common])
        poss = np.array([cnt.get(k, 0) for k in common])
        rows.append(pd.DataFrame({"x": x, "y": y, "poss": poss, "seed": seed}))
        # raw (unadjusted) pair net rating, for contrast
        ra, rb = A.raw_pairs, C.raw_pairs
        cr = sorted(set(ra) & set(rb))
        rows[-1].attrs["raw_r"] = np.corrcoef([ra[k] for k in cr], [rb[k] for k in cr])[0, 1]
        rows[-1].attrs["raw_n"] = len(cr)
        # player main effects, positive control
        ci = A.add_o["player"].index.intersection(C.add_o["player"].index)
        rows[-1].attrs["off_r"] = np.corrcoef(A.add_o["player"][ci], C.add_o["player"][ci])[0, 1]
        rows[-1].attrs["def_r"] = np.corrcoef(A.add_d["player"][ci], C.add_d["player"][ci])[0, 1]
    return rows


def report(stints, players, label=""):
    rows = split_half(stints, players)
    allr = []
    print(f"\n### split-half reliability {label} ({len(stints)} lineup-rows)")
    for d in rows:
        r = np.corrcoef(d.x, d.y)[0, 1]
        allr.append(r)
        print(f"  seed {d.seed.iloc[0]}: adjusted pair r={r:+.3f} (n={len(d)}, "
              f"Spearman-Brown {sb(r):+.3f}) | raw pair r={d.attrs['raw_r']:+.3f} "
              f"| player off r={d.attrs['off_r']:+.3f} def r={d.attrs['def_r']:+.3f}")
    big = pd.concat(rows)
    print(f"  mean adjusted-pair split-half r = {np.mean(allr):+.3f}")
    qs = big.poss.quantile([0.5, 0.8, 0.95]).values
    for lo, hi, name in [(0, qs[0], "low"), (qs[0], qs[1], "mid"), (qs[1], qs[2], "high"),
                         (qs[2], np.inf, "top 5% by shared possessions")]:
        sub = big[(big.poss >= lo) & (big.poss < hi)]
        if len(sub) > 40:
            print(f"    {name:28s} n={len(sub):5d} poss>={int(lo):5d}  r={np.corrcoef(sub.x, sub.y)[0,1]:+.3f}")
    return np.mean(allr), big


def bilinear_stability(stints, players, seeds=(0, 1)):
    print("\n### does the structured model z_i' M z_j replicate?")
    for seed in seeds:
        rng = np.random.default_rng(seed)
        f = rng.integers(0, 2, len(stints))
        A = V.Fit(stints[f == 0].reset_index(drop=True), players)
        C = V.Fit(stints[f == 1].reset_index(drop=True), players)
        idx = A.Z.index.intersection(C.Z.index)
        sa = B.synergy_matrix(A.M, A.Z.loc[idx]).values
        sc = B.synergy_matrix(C.M, C.Z.loc[idx]).values
        iu = np.triu_indices(len(idx), 1)
        print(f"  seed {seed}: surface r={np.corrcoef(sa[iu], sc[iu])[0,1]:+.3f} "
              f"| M entries r={np.corrcoef(A.M.ravel(), C.M.ravel())[0,1]:+.3f} "
              f"| spread {sa[iu].std():.3f} vs {sc[iu].std():.3f} pts/100")


if __name__ == "__main__":
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    st = S.load_stints()
    seasons = sorted(st.season.unique())
    report(st, players, label=f"pooled {seasons[0]}..{seasons[-1]}")
    bilinear_stability(st, players)
    if len(seasons) >= 2:
        print("\n### across-season replication (different seasons, same pair)")
        fits = {s: V.Fit(st[st.season == s].reset_index(drop=True), players) for s in seasons}
        for i in range(len(seasons) - 1):
            a, b = fits[seasons[i]], fits[seasons[i + 1]]
            c = sorted(set(a.ridge_pairs) & set(b.ridge_pairs))
            ra, rb = a.raw_pairs, b.raw_pairs
            cr = sorted(set(ra) & set(rb))
            print(f"  {seasons[i]} vs {seasons[i+1]}: adjusted r="
                  f"{np.corrcoef([a.ridge_pairs[k] for k in c], [b.ridge_pairs[k] for k in c])[0,1]:+.3f} "
                  f"(n={len(c)})  raw r={np.corrcoef([ra[k] for k in cr], [rb[k] for k in cr])[0,1]:+.3f}")


def direct_stability_by_size(stints, players, dims=6, seeds=(0,)):
    """Split-half replication of the DIRECT interaction surface vs amount of training data."""
    import direct as D
    seasons = sorted(stints.season.unique())
    print("\n### direct joint fit: surface replication vs sample size")
    for k in (1, 2, 4, len(seasons)):  # seasons of training data per half
        sub = stints[stints.season.isin(seasons[-k:])].reset_index(drop=True)
        rs, sds = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            f = rng.integers(0, 2, len(sub))
            out = []
            for h in (0, 1):
                d = sub[f == h].reset_index(drop=True)
                fo = D.fit(d, players, "off", dims=dims)
                fd = D.fit(d, players, "def", dims=dims)
                out.append((D.net_M(fo, fd), fo["Z"]))
            idx = out[0][1].index.intersection(out[1][1].index)
            s0 = D.surface(out[0][0], out[0][1].loc[idx], dims)
            s1 = D.surface(out[1][0], out[1][1].loc[idx], dims)
            iu = np.triu_indices(len(idx), 1)
            rs.append(np.corrcoef(s0[iu], s1[iu])[0, 1]); sds.append(s0[iu].std())
        print(f"  {k} season(s) (each half ~{len(sub)//2} lineup-rows): "
              f"split-half r={np.mean(rs):+.3f}  surface sd={np.mean(sds):.2f} pts/100")


def across_season_surface(stints, players, dims=6):
    """Cleanest replication test for the interaction surface: fit on disjoint SEASON blocks.

    A random split of lineup rows leaves the same team-seasons in both halves, so any team-level
    effect the model fails to absorb is shared by both halves and inflates agreement. Disjoint
    seasons do not share team context.
    """
    import direct as D
    seasons = sorted(stints.season.unique())
    splits = {
        "odd vs even seasons": ([s for i, s in enumerate(seasons) if i % 2],
                                [s for i, s in enumerate(seasons) if not i % 2]),
        "first half vs second half": (seasons[:len(seasons) // 2], seasons[len(seasons) // 2:]),
    }
    print("\n### interaction surface: replication across DISJOINT SEASONS")
    for lab, (A, C) in splits.items():
        out = []
        for blk in (A, C):
            d = stints[stints.season.isin(blk)].reset_index(drop=True)
            fo = D.fit(d, players, "off", dims=dims); fd = D.fit(d, players, "def", dims=dims)
            out.append((D.net_M(fo, fd), fo["Z"]))
        idx = out[0][1].index.intersection(out[1][1].index)
        s0 = D.surface(out[0][0], out[0][1].loc[idx], dims)
        s1 = D.surface(out[1][0], out[1][1].loc[idx], dims)
        iu = np.triu_indices(len(idx), 1)
        print(f"  {lab}: surface r={np.corrcoef(s0[iu], s1[iu])[0,1]:+.3f} "
              f"| M entries r={np.corrcoef(out[0][0].ravel(), out[1][0].ravel())[0,1]:+.3f} "
              f"| spread {s0[iu].std():.2f} vs {s1[iu].std():.2f} pts/100 (n={len(idx)} players)")
