"""A player-value metric available for all 30 seasons.

The lineup-ridge value used everywhere else needs 5-man lineup data, which this source only serves
per team per season and rate-limits hard enough that backfilling 1996-2018 would take most of a
day. Player totals, however, exist for all 30 seasons.

So: learn the mapping from box-score profile to lineup-ridge value on the seasons where both
exist, then apply it backwards. Two versions are fit.

  * `box_value`       individual production only. Deliberately excludes on-court team ratings, so
                      it cannot smuggle team quality into a projection whose baseline is "last
                      season's record".
  * `box_value_onoff` adds the player's on-court offensive and defensive ratings. Fits the target
                      better, but those ratings are mostly a property of the team, so it is only
                      reported as a sensitivity.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import features as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out")

# the WITHIN-SEASON z-scored features, so a player is rated against his own era's league. Using
# the raw rates instead makes the 1990s look like a different planet to a model trained on the
# 2020s, and it compressed every early-season prediction toward zero.
BOX = ["z_" + c for c in F.SKILL_COLS]
EXTRA = ["minutes"]
ONOFF = ["on_ortg", "on_drtg"]


def design(df, cols, mu=None, sd=None):
    X = df[cols].astype(float).copy()
    X["minutes"] = np.log1p(X["minutes"]) if "minutes" in X else 0
    if mu is None:
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    Z = ((X - mu) / sd).clip(-5, 5).fillna(0.0)
    return Z.values, mu, sd


def fit_ridge(X, y, alpha=25.0):
    X1 = np.column_stack([np.ones(len(X)), X])
    P = np.eye(X1.shape[1]) * alpha
    P[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + P, X1.T @ y)


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def main(min_minutes=400):
    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    eff_o = pd.read_parquet(os.path.join(PROC, "player_eff_off.parquet"))["eff"]
    eff_d = pd.read_parquet(os.path.join(PROC, "player_eff_def.parquet"))["eff"]
    ridge_val = (eff_o - eff_d).dropna()

    # the training set is the player-seasons that have a lineup-ridge value
    lab = players[players.pid.isin(ridge_val.index) & (players.minutes >= min_minutes)].copy()
    lab["y"] = lab.pid.map(ridge_val)
    lab = lab.dropna(subset=["y"])
    # one row per player-season; the ridge value is a career-in-sample value, so average the
    # player's seasons to avoid the same y appearing many times with different X
    lab = lab.groupby("pid", as_index=False).agg(
        {**{c: "mean" for c in BOX + EXTRA + ONOFF}, "y": "first"})
    print(f"training rows (players with a lineup-ridge value): {len(lab)}")

    results = {}
    for name, cols in (("box_value", BOX + EXTRA), ("box_value_onoff", BOX + EXTRA + ONOFF)):
        X, mu, sd = design(lab, cols)
        y = lab.y.values
        # 5-fold CV by player to see how well the mapping generalises
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(lab))
        pred = np.zeros(len(lab))
        for f in range(5):
            m = fold == f
            beta = fit_ridge(X[~m], y[~m])
            pred[m] = predict(beta, X[m])
        r = float(np.corrcoef(pred, y)[0, 1])
        mae = float(np.abs(pred - y).mean())
        print(f"  {name:16s} cross-validated r = {r:.3f}, MAE {mae:.2f} pts/100 "
              f"(target sd {y.std():.2f})")
        beta = fit_ridge(X, y)
        results[name] = (beta, mu, sd, cols, r)

    # apply to every player-season in the 30-season table
    for name, (beta, mu, sd, cols, r) in results.items():
        X, _, _ = design(players, cols, mu, sd)
        players[name] = predict(beta, X)
    out = players[["season", "pid", "name", "team", "minutes", "box_value", "box_value_onoff"]]
    out.to_parquet(os.path.join(PROC, "box_value.parquet"))
    print(f"\nwrote box value for {len(out)} player-seasons, "
          f"{out.season.nunique()} seasons ({out.season.min()} to {out.season.max()})")

    # sanity: the top of the list should be recognisable
    for s in ["2025-26", "2015-16", "2005-06", "1997-98"]:
        x = out[(out.season == s) & (out.minutes >= 1200)].nlargest(6, "box_value")
        print(f"  {s}: " + ", ".join(f"{n.split()[-1]} {v:.1f}"
                                     for n, v in zip(x.name, x.box_value)))
    return out


if __name__ == "__main__":
    main()
