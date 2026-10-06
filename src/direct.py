"""Direct joint fit: lineup outcome ~ player effects + sum over pairs of z_i' M z_j.

The model is linear in M, so player main effects and the whole interaction surface are estimated
in ONE ridge instead of the two-stage (pair coefficients, then fit M). Far better powered: the
interaction surface costs d(d+1)/2 parameters shared across every pair, not one per pair.
"""
import os, sys, itertools
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, bilinear as B

LAM_PLAYER, LAM_M, LAM_S = 30.0, 5.0, 1e-6


def lineup_basis(players_list, Z, dims):
    """For one lineup: sum over its 10 pairs of the symmetric outer-product basis."""
    iu = np.triu_indices(dims)
    zs = [Z.loc[p].values[:dims] for p in players_list if p in Z.index]
    acc = np.zeros((dims, dims))
    for a, b in itertools.combinations(zs, 2):
        acc += np.outer(a, b) + np.outer(b, a)
    acc /= 2
    return acc[iu]


def fit(stints, players, side="off", dims=6, lam_m=LAM_M):
    wcol, ycol = ("off_poss", "pts") if side == "off" else ("def_poss", "opp_pts")
    d = stints[stints[wcol] >= 1].reset_index(drop=True)
    if "players" not in d:
        d["players"] = d.lineup.str.split("-")
    Z = B.player_z(players[players.season.isin(set(d.season))], dims=dims)
    y = 100.0 * d[ycol].values / d[wcol].values
    w = d[wcol].values.astype(float)

    pl = sorted({p for x in d.players for p in x})
    pidx = {p: i for i, p in enumerate(pl)}
    seasons = sorted(d.season.unique()); sidx = {s: i for i, s in enumerate(seasons)}
    nb = dims * (dims + 1) // 2
    nP, nS = len(pl), len(seasons)
    o_s, o_b = nP, nP + nS
    ncol = o_b + nb

    rows, cols, vals = [], [], []
    Bfeat = np.zeros((len(d), nb))
    for i, (x, ss) in enumerate(zip(d.players, d.season)):
        for p in x:
            rows.append(i); cols.append(pidx[p]); vals.append(1.0)
        rows.append(i); cols.append(o_s + sidx[ss]); vals.append(1.0)
        Bfeat[i] = lineup_basis(x, Z, dims)
    bs = Bfeat.std(axis=0) + 1e-9
    Bfeat = Bfeat / bs
    for j in range(nb):
        nz = np.nonzero(Bfeat[:, j])[0]
        rows.extend(nz.tolist()); cols.extend([o_b + j] * len(nz)); vals.extend(Bfeat[nz, j].tolist())
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(d), ncol))
    sw = np.sqrt(w / w.mean())
    lam = np.concatenate([np.full(nP, LAM_PLAYER), np.full(nS, LAM_S), np.full(nb, lam_m)])
    A = sp.vstack([sp.diags(sw) @ X, sp.diags(np.sqrt(lam))]).tocsr()
    b = np.concatenate([y * sw, np.zeros(ncol)])
    sol = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=4000)[0]

    iu = np.triu_indices(dims)
    M = np.zeros((dims, dims)); M[iu] = sol[o_b:] / bs; M = (M + M.T) / 2
    return dict(player=pd.Series(sol[:nP], index=pl), M=M, Z=Z, dims=dims,
                season=pd.Series(sol[o_s:o_s + nS], index=seasons), basis_scale=bs)


def net_M(fo, fd):
    return fo["M"] - fd["M"]


def surface(M, Z, dims):
    V = Z.values[:, :dims]
    return V @ M @ V.T
