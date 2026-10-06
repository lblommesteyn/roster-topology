"""Which player-type interactions predict pair synergy? Hypothesis-driven + reproducibility test."""
import os, itertools
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")

# Each hypothesis is a symmetric function of two players' standardized skill vectors.
HYPOTHESES = {
    "rim_pressure x spacing":      lambda a, b: a.z_f_rim * b.z_fg3a_pct,
    "spacing x spacing":           lambda a, b: a.z_fg3a_pct * b.z_fg3a_pct,
    "two non-shooters":            lambda a, b: np.minimum(a.z_fg3a_pct, 0) * np.minimum(b.z_fg3a_pct, 0),
    "creator x creator (usage)":   lambda a, b: a.z_usage * b.z_usage,
    "creator x off-ball shooter":  lambda a, b: a.z_usage * b.z_c3_assisted,
    "creator x rim-running big":   lambda a, b: a.z_usage * (b.z_f_rim * b.z_rim_assisted),
    "passer x movement shooter":   lambda a, b: a.z_ast100 * b.z_assisted3,
    "two bigs (rim-heavy)":        lambda a, b: np.maximum(a.z_f_rim, 0) * np.maximum(b.z_f_rim, 0),
    "two small guards":            lambda a, b: np.minimum(a.z_f_rim, 0) * np.minimum(b.z_f_rim, 0),
    "oreb x oreb":                 lambda a, b: a.z_oreb_fg * b.z_oreb_fg,
    "rim protect x perimeter stl": lambda a, b: a.z_blk100 * b.z_stl100,
    "two rim protectors":          lambda a, b: a.z_blk100 * b.z_blk100,
    "dreb x dreb":                 lambda a, b: a.z_dreb_fg * b.z_dreb_fg,
    "turnover-prone x turnover":   lambda a, b: a.z_tov100 * b.z_tov100,
    "isolation x isolation":       lambda a, b: a.z_unast2_share * b.z_unast2_share,
    "shot-creation x efficiency":  lambda a, b: a.z_usage * b.z_ts,
    "midrange x midrange":         lambda a, b: a.z_f_lmid * b.z_f_lmid,
    "foul-prone x foul-prone":     lambda a, b: a.z_foul100 * b.z_foul100,
}


def design(pairs, feats):
    """Symmetrized hypothesis features for each pair (order-independent)."""
    A = feats.loc[pairs.p1].reset_index(drop=True)
    Bb = feats.loc[pairs.p2].reset_index(drop=True)
    cols = {}
    for name, f in HYPOTHESES.items():
        cols[name] = 0.5 * (np.asarray(f(A, Bb), float) + np.asarray(f(Bb, A), float))
    X = pd.DataFrame(cols)
    return (X - X.mean()) / X.std(ddof=0).replace(0, 1)


def fit(pairs, feats, alpha=25.0, weight_cap=4000, n_boot=300, seed=0):
    X = design(pairs, feats)
    y = pairs.net.values
    w = np.minimum(pairs.poss.values, weight_cap).astype(float)
    sw = np.sqrt(w / w.mean())
    names = list(X.columns)
    Xv = X.values

    def solve(idx):
        A = np.vstack([Xv[idx] * sw[idx, None], np.sqrt(alpha) * np.eye(Xv.shape[1])])
        b = np.concatenate([y[idx] * sw[idx], np.zeros(Xv.shape[1])])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    coef = solve(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boot = np.array([solve(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"coef": coef, "lo": lo, "hi": hi,
                        "z": coef / boot.std(axis=0)}, index=names)
    return out.sort_values("coef"), X
