"""synergy(i,j) ~ z_i^T M z_j  with M symmetric, fit to estimated pair coefficients.

Generalizes to pairs that never shared a floor, which is what the counterfactual tool needs.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def player_z(players, dims=8, season_map=None):
    """pid -> latent vector (PCA dims), averaged over that player's seasons in scope."""
    cols = [f"pc{k+1}" for k in range(dims)]
    g = players.groupby("pid")[cols].mean()
    return g


def sym_features(za, zb):
    """Symmetric bilinear basis: vec of (z_a z_b^T + z_b z_a^T)/2 upper triangle."""
    d = za.shape[1]
    iu = np.triu_indices(d)
    out = np.einsum("ni,nj->nij", za, zb)
    out = (out + np.transpose(out, (0, 2, 1))) / 2
    return out[:, iu[0], iu[1]], iu


def fit_M(pairs, Z, dims=8, alpha=50.0, weight_cap=4000):
    """Ridge fit of M to pair net coefficients, weighted by shared possessions."""
    have = Z.index
    m = pairs[pairs.p1.isin(have) & pairs.p2.isin(have)].copy()
    za = Z.loc[m.p1].values[:, :dims]
    zb = Z.loc[m.p2].values[:, :dims]
    F, iu = sym_features(za, zb)
    y = m.net.values
    w = np.minimum(m.poss.values, weight_cap)
    sw = np.sqrt(w / w.mean())
    A = np.vstack([F * sw[:, None], np.sqrt(alpha) * np.eye(F.shape[1])])
    b = np.concatenate([y * sw, np.zeros(F.shape[1])])
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    M = np.zeros((dims, dims)); M[iu] = coef; M = (M + M.T) / 2
    np.fill_diagonal(M, np.diag(M))  # diagonal counted once in the basis
    pred = F @ coef
    r = np.corrcoef(pred, y)[0, 1]
    return M, coef, iu, r, m, pred


def synergy_matrix(M, Z, dims=8):
    V = Z.values[:, :dims]
    return pd.DataFrame(V @ M @ V.T, index=Z.index, columns=Z.index)
