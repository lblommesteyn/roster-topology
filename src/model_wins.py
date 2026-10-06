"""Backtest: what predicts next-season wins, before anyone plays.

Leave-one-season-out over every target season. Each model is fit on the other seasons and scored
on the held-out one, so no model ever sees its own season. Errors are reported in wins on an
82-game pace, which is the unit anyone actually cares about.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")

TALENT = ["talent_mw", "star", "depth"]
STRUCT = ["redundancy", "eff_roles", "coverage", "scarcity"]
SHAPE = ["volume", "sphericity", "components", "peak_density"]

MODELS = {
    "league average (41 wins)":          [],
    "last season's wins":                ["prior_wins82"],
    "last season's net rating":          ["prior_net"],
    "talent only":                       TALENT,
    "talent + continuity":               TALENT + ["continuity"],
    "talent + continuity + structure":   TALENT + ["continuity"] + STRUCT,
    "talent + continuity + complementarity": TALENT + ["continuity", "complementarity"],
    "talent + continuity + blob shape":  TALENT + ["continuity"] + SHAPE,
    "everything":                        TALENT + ["continuity", "complementarity"] + STRUCT + SHAPE,
    "last season + talent":              ["prior_wins82"] + TALENT,
    "last season + talent + structure":  ["prior_wins82"] + TALENT + ["continuity"] + STRUCT,
}


def fit_predict(train, test, cols, ridge=1.0):
    y = train.wins82.values
    if not cols:
        return np.full(len(test), y.mean())
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    Xtr = ((train[cols] - mu) / sd).fillna(0).values
    Xte = ((test[cols] - mu) / sd).fillna(0).values
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte = np.column_stack([np.ones(len(Xte)), Xte])
    P = np.eye(Xtr.shape[1]) * ridge
    P[0, 0] = 0.0
    beta = np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ y)
    return Xte @ beta


def loso(tab, cols, ridge=1.0):
    pred = np.zeros(len(tab))
    for s in tab.season.unique():
        m = (tab.season == s).values
        pred[m] = fit_predict(tab[~m], tab[m], cols, ridge)
    return pred


def score(y, p):
    err = y - p
    ss = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                r2=float(ss), corr=float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan)


def main():
    tab = pd.read_csv(os.path.join(OUT, "preseason_features.csv"))
    tab = tab.dropna(subset=["wins82", "prior_wins82"]).reset_index(drop=True)
    y = tab.wins82.values
    print(f"{len(tab)} team-seasons, {tab.season.nunique()} target seasons "
          f"({tab.season.min()} to {tab.season.max()})")
    print(f"wins spread: sd {y.std():.1f}, range {y.min():.0f}-{y.max():.0f}\n")

    rows, preds = [], {}
    for name, cols in MODELS.items():
        p = loso(tab, cols)
        preds[name] = p
        rows.append(dict(model=name, n_features=len(cols), **score(y, p)))
    res = pd.DataFrame(rows).sort_values("mae")
    print("=== leave-one-season-out, error in wins (82-game pace) ===")
    print(res.round(3).to_string(index=False))

    # is the best structural model actually better than last season's wins?
    rng = np.random.default_rng(0)
    base = preds["last season's wins"]
    print("\n=== paired bootstrap vs 'last season's wins' (MAE improvement, wins) ===")
    for name in ["talent only", "talent + continuity", "talent + continuity + structure",
                 "talent + continuity + complementarity", "everything",
                 "last season + talent", "last season + talent + structure"]:
        d = np.abs(y - base) - np.abs(y - preds[name])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {name:42s} {d.mean():+5.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"P(better)={float((bs > 0).mean()):.2f}")

    # standardized coefficients on the full sample, to read direction and size
    cols = MODELS["everything"]
    mu, sd = tab[cols].mean(), tab[cols].std().replace(0, 1)
    X = np.column_stack([np.ones(len(tab)), ((tab[cols] - mu) / sd).fillna(0).values])
    P = np.eye(X.shape[1]); P[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + P, X.T @ y)
    co = pd.Series(beta[1:], index=cols).sort_values(key=np.abs, ascending=False)
    print("\n=== standardized coefficients, wins per 1 sd (full sample, ridge=1) ===")
    for k, v in co.items():
        print(f"  {k:18s} {v:+6.2f}")

    tab["pred_best"] = preds[res.iloc[0].model]
    tab["pred_talent"] = preds["talent + continuity"]
    tab.to_csv(os.path.join(OUT, "preseason_backtest.csv"), index=False, encoding="utf-8")
    res.to_csv(os.path.join(OUT, "preseason_model_scores.csv"), index=False)

    print("\n=== biggest misses of the best model ===")
    tab["err"] = tab.wins82 - tab.pred_best
    w = tab.reindex(tab.err.abs().sort_values(ascending=False).index).head(8)
    print(w[["season", "team", "wins82", "pred_best", "prior_wins82", "err"]].round(1).to_string(index=False))
    return res, tab


if __name__ == "__main__":
    main()
