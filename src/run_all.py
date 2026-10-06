"""End-to-end pipeline: features -> synergy -> patterns -> topology -> metrics -> validation."""
import os, sys, itertools, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B, topology as TP, metrics as MT, swaps as SW

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT, FIGS = (os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out"), os.path.join(ROOT, "figs"))
for d in (OUT, FIGS):
    os.makedirs(d, exist_ok=True)

TEAM_NAMES = {}


def load():
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    stints = pd.read_parquet(os.path.join(PROC, "stints.parquet"))
    stints["players"] = stints.lineup.str.split("-")
    for f in glob.glob(os.path.join(ROOT, "data", "raw", "Player_*.json")):
        for r in json.load(open(f)):
            TEAM_NAMES[str(r["TeamId"])] = r["TeamAbbreviation"]
    return players, stints


def direct_synergy(dims=6):
    """Synergy from the directly fitted interaction surface.

    Per-pair ridge coefficients are deliberately NOT blended in: their split-half reliability is
    ~0.06, so using them would inject noise. The structured surface replicates at ~0.42.
    """
    M = np.load(os.path.join(PROC, "M_direct.npy"))
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    Z = B.player_z(players, dims=dims)

    def f(a, b):
        if a in Z.index and b in Z.index:
            return float(Z.loc[a].values[:dims] @ M @ Z.loc[b].values[:dims])
        return 0.0
    return f, M, Z


def synergy_fn(ridge_pairs, M, Z, dims=8, blend=True):
    """Pair synergy: measured ridge coefficient when the pair has history, bilinear otherwise."""
    Zi = Z
    def f(a, b):
        k = (a, b) if a < b else (b, a)
        if blend and k in ridge_pairs:
            return ridge_pairs[k]
        if a in Zi.index and b in Zi.index:
            return float(Zi.loc[a].values[:dims] @ M @ Zi.loc[b].values[:dims])
        return 0.0
    return f


def team_net(stints):
    g = stints.groupby(["season", "team"], as_index=False).agg(
        off_poss=("off_poss", "sum"), def_poss=("def_poss", "sum"),
        pts=("pts", "sum"), opp_pts=("opp_pts", "sum"))
    g["net"] = 100 * (g.pts / g.off_poss - g.opp_pts / g.def_poss)
    return g


def main():
    players, stints = load()
    seasons = sorted(stints.season.unique())
    print("seasons with lineups:", seasons)

    # ---- 2. pair interaction effects
    pairs = S.main(seasons)
    pairs = pd.read_parquet(os.path.join(PROC, "pairs.parquet"))
    eff_o = pd.read_parquet(os.path.join(PROC, "player_eff_off.parquet"))["eff"].to_dict()
    eff_d = pd.read_parquet(os.path.join(PROC, "player_eff_def.parquet"))["eff"].to_dict()

    name = players.drop_duplicates("pid").set_index("pid")["name"].to_dict()
    pairs["n1"] = pairs.p1.map(name); pairs["n2"] = pairs.p2.map(name)
    pairs.to_csv(os.path.join(OUT, "pair_synergy.csv"), index=False, encoding="utf-8")

    # ---- 3. bilinear model + interpretable patterns
    pf = players[players.season.isin(seasons)]
    Z = B.player_z(pf, dims=8)
    M, coef, iu, rfit, matched, _ = B.fit_M(pairs, Z, dims=8)
    print("bilinear fit corr (in-sample):", round(rfit, 3), "pairs used:", len(matched))
    np.save(os.path.join(PROC, "M.npy"), M)

    import patterns as PT
    zcols = [c for c in players.columns if c.startswith("z_")]
    feats = pf.groupby("pid")[zcols].mean()
    use = pairs[pairs.p1.isin(feats.index) & pairs.p2.isin(feats.index)].reset_index(drop=True)
    tab, X = PT.fit(use, feats)
    tab.to_csv(os.path.join(OUT, "patterns.csv"), encoding="utf-8")
    print("\n=== interaction patterns (pts/100 per sd of interaction) ===")
    print(tab.round(3).to_string())
    return dict(players=players, stints=stints, pairs=pairs, M=M, Z=Z,
                eff_o=eff_o, eff_d=eff_d, feats=feats, seasons=seasons)


if __name__ == "__main__":
    main()
