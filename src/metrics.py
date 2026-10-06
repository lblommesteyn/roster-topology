"""Roster-structure metrics: redundancy, coverage, complementarity, fragility, diversity, scarcity."""
import os, itertools, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
DIMS = 5


def _z(roster):
    return roster[[f"pc{k+1}" for k in range(DIMS)]].values


def redundancy(roster):
    """Minutes-weighted mean nearest-neighbour closeness in skill space (higher = more duplicated roles)."""
    Z, m = _z(roster), roster.minutes.values.astype(float)
    if len(Z) < 3:
        return np.nan
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return float(np.average(-nn, weights=m))


def effective_roles(roster):
    """exp(entropy) of the minutes-weighted skill covariance spectrum: how many independent roles."""
    Z, m = _z(roster), roster.minutes.values.astype(float)
    if len(Z) < 3:
        return np.nan
    mu = np.average(Z, axis=0, weights=m)
    C = np.cov((Z - mu).T, aweights=m)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-9, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def coverage(roster, centroids, thresh):
    """Share of league role-centroids that some rotation player is close to."""
    Z = _z(roster)
    if len(Z) < 3:
        return np.nan
    D = np.linalg.norm(centroids[:, None] - Z[None], axis=2)
    return float((D.min(axis=1) < thresh).mean())


def scarcity(roster, league_Z, k=25):
    """Mean rarity of the roster's players in the league skill manifold (kNN distance)."""
    Z, m = _z(roster), roster.minutes.values.astype(float)
    D = np.linalg.norm(Z[:, None] - league_Z[None], axis=2)
    D.sort(axis=1)
    return float(np.average(D[:, 1:k + 1].mean(axis=1), weights=m))


def pair_stats(roster, syn):
    """Minutes-weighted complementarity and fragility from the signed synergy graph."""
    ids, m = list(roster.pid), roster.minutes.values.astype(float)
    n = len(ids)
    if n < 3:
        return dict(complementarity=np.nan, neg_share=np.nan, fragility=np.nan, connector=None)
    W = np.zeros((n, n)); wt = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        W[i, j] = W[j, i] = syn(ids[i], ids[j])
        wt[i, j] = wt[j, i] = m[i] * m[j]
    iu = np.triu_indices(n, 1)
    comp = float(np.average(W[iu], weights=wt[iu]))
    neg = float(np.average((W[iu] < 0).astype(float), weights=wt[iu]))
    # fragility: how much the team's average complementarity falls if its best connector leaves
    base = comp
    drops = []
    for k in range(n):
        keep = [x for x in range(n) if x != k]
        sub = W[np.ix_(keep, keep)]; sw = wt[np.ix_(keep, keep)]
        iu2 = np.triu_indices(len(keep), 1)
        drops.append(base - float(np.average(sub[iu2], weights=sw[iu2])))
    drops = np.array(drops)
    return dict(complementarity=comp, neg_share=neg, fragility=float(drops.max()),
                connector=roster.name.iloc[int(drops.argmax())])


def lineup_diversity(stints, team, season):
    s = stints[(stints.team == team) & (stints.season == season)]
    p = s.off_poss.values.astype(float)
    if p.sum() <= 0:
        return np.nan
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
