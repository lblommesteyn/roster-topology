"""Ridge-regularized pair interaction effects on lineup-stint data.

Offense rows:  pts/100 off poss ~ season + home + sum(player_off) + opp_team_def + sum(pair_off)
Defense rows:  opp_pts/100 def poss ~ season + home + sum(player_def) + opp_team_off + sum(pair_def)

Player effects carry a light ridge penalty, pair effects a heavy one, so a pair coefficient only
moves away from zero when the pair has enough possessions to overcome the shrinkage. That is the
"two good players together != complementarity" separation: player quality is absorbed by the main
effects, the pair term is what is left over.
"""
import os, sys, itertools, json
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")

LAM_PLAYER = 30.0     # ridge on player main effects (possession units)
LAM_PAIR = 250.0      # heavy shrinkage on pair effects
LAM_CTX = 5.0
MIN_PAIR_POSS = 400   # pairs below this are not given their own coefficient


def load_stints(seasons=None):
    df = pd.read_parquet(os.path.join(PROC, "stints.parquet"))
    if seasons:
        df = df[df.season.isin(seasons)]
    df["players"] = df.lineup.str.split("-")
    return df.reset_index(drop=True)


def pair_counts(df, side="off"):
    poss = df["off_poss" if side == "off" else "def_poss"].values
    cnt = {}
    for pl, w in zip(df.players.values, poss):
        if w <= 0:
            continue
        for a, b in itertools.combinations(sorted(pl), 2):
            cnt[(a, b)] = cnt.get((a, b), 0) + w
    return cnt


def build(df, side, pairs, min_poss_row=1):
    """Design matrix for one side."""
    wcol, ycol = ("off_poss", "pts") if side == "off" else ("def_poss", "opp_pts")
    d = df[df[wcol] >= min_poss_row].reset_index(drop=True)
    y = 100.0 * d[ycol].values / d[wcol].values
    w = d[wcol].values.astype(float)

    players = sorted({p for pl in d.players for p in pl})
    pidx = {p: i for i, p in enumerate(players)}
    teams = sorted(set(d.team) | set(d.opp))
    tidx = {t: i for i, t in enumerate(teams)}
    seasons = sorted(d.season.unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    pairidx = {p: i for i, p in enumerate(pairs)}

    nP, nT, nS, nPr = len(players), len(teams), len(seasons), len(pairs)
    off_p, off_t, off_s, off_pr, off_home = 0, nP, nP + nT, nP + nT + nS, nP + nT + nS + nPr
    ncol = off_home + 1

    use_opp = d.opp.nunique() > 1
    use_home = d.home.nunique() > 1
    rows, cols, vals = [], [], []
    for i, (pl, tm, op, hm, ss) in enumerate(zip(d.players, d.team, d.opp, d.home, d.season)):
        for p in pl:
            rows.append(i); cols.append(pidx[p]); vals.append(1.0)
        if use_opp:
            rows.append(i); cols.append(off_t + tidx[op]); vals.append(1.0)   # opponent unit
        rows.append(i); cols.append(off_s + sidx[ss]); vals.append(1.0)     # season intercept
        if use_home:
            rows.append(i); cols.append(off_home); vals.append(1.0 if hm else -1.0)
        for a, b in itertools.combinations(sorted(pl), 2):
            j = pairidx.get((a, b))
            if j is not None:
                rows.append(i); cols.append(off_pr + j); vals.append(1.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(d), ncol))

    # possession weights -> sqrt weighting of rows
    sw = np.sqrt(w / w.mean())
    Xw = sp.diags(sw) @ X
    yw = y * sw

    lam = np.concatenate([
        np.full(nP, LAM_PLAYER), np.full(nT, LAM_CTX), np.full(nS, 1e-6),
        np.full(nPr, LAM_PAIR), np.array([1e-6])])
    R = sp.diags(np.sqrt(lam))
    A = sp.vstack([Xw, R]).tocsr()
    b = np.concatenate([yw, np.zeros(ncol)])
    sol = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=3000)[0]

    out = dict(
        player=pd.Series(sol[off_p:off_p + nP], index=players),
        team=pd.Series(sol[off_t:off_t + nT], index=teams),
        season=pd.Series(sol[off_s:off_s + nS], index=seasons),
        pair=(pd.Series(sol[off_pr:off_pr + nPr], index=pd.MultiIndex.from_tuples(pairs))
              if nPr else pd.Series(dtype=float)),
        home=sol[off_home], sol=sol, cols=(pidx, tidx, sidx, pairidx, off_t, off_s, off_pr, off_home),
        n=len(d))
    pred = X @ sol
    ss_res = np.sum(w * (y - pred) ** 2); ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    out["r2_in"] = 1 - ss_res / ss_tot
    return out


def main(seasons=None):
    df = load_stints(seasons)
    print("stints:", len(df), "seasons:", sorted(df.season.unique()))
    res = {}
    for side in ("off", "def"):
        cnt = pair_counts(df, side)
        pairs = sorted([p for p, c in cnt.items() if c >= MIN_PAIR_POSS])
        print(side, "pairs kept:", len(pairs), "of", len(cnt))
        r = build(df, side, pairs)
        r["poss"] = pd.Series({p: cnt[p] for p in pairs})
        print("  weighted in-sample R2:", round(r["r2_in"], 4), " home:", round(r["home"], 3))
        res[side] = r

    # net synergy: offense pair effect minus defense pair effect (defense y is points allowed)
    o, dd = res["off"]["pair"], res["def"]["pair"]
    common = o.index.intersection(dd.index)
    net = (o.loc[common] - dd.loc[common]).rename("net")
    out = pd.DataFrame({"off": o.loc[common], "def_": dd.loc[common], "net": net})
    out["poss"] = res["off"]["poss"].reindex(common).values
    out.index.names = ["p1", "p2"]
    out.reset_index().to_parquet(os.path.join(PROC, "pairs.parquet"))
    for k in ("off", "def"):
        res[k]["player"].rename("eff").to_frame().to_parquet(os.path.join(PROC, f"player_eff_{k}.parquet"))
    print("\nsynergy spread (pts/100):", out.net.describe()[["std", "min", "max"]].round(2).to_dict())
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or None)
