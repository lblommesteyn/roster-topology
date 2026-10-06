"""Backtest the preseason projection on 25 seasons, and check what the larger sample changes."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import model_wins as MW

OUT = os.path.join(os.path.dirname(__file__), "..", "out")

TALENT = ["talent_mw", "star", "depth"]
STRUCT = ["redundancy", "eff_roles", "coverage", "scarcity"]
MODELS = {
    "league average":                  [],
    "last season's wins":              ["prior_wins82"],
    "talent only":                     TALENT,
    "talent + continuity":             TALENT + ["continuity"],
    "talent + continuity + structure":  TALENT + ["continuity"] + STRUCT,
    "last season + talent":            ["prior_wins82"] + TALENT,
    "last season + talent + continuity": ["prior_wins82"] + TALENT + ["continuity"],
    "last season + talent + cont + structure": ["prior_wins82"] + TALENT + ["continuity"] + STRUCT,
}


def run(tab, label, models=MODELS):
    y = tab.wins82.values
    rows, preds = [], {}
    for name, cols in models.items():
        p = MW.loso(tab, cols)
        preds[name] = p
        rows.append(dict(model=name, k=len(cols), **MW.score(y, p)))
    res = pd.DataFrame(rows).sort_values("mae")
    print(f"\n=== {label}: n={len(tab)}, {tab.season.nunique()} seasons ===")
    print(res.round(3).to_string(index=False))
    return res, preds


def main():
    tab = pd.read_csv(os.path.join(OUT, "preseason_features_long.csv"))
    tab = tab.dropna(subset=["wins82", "prior_wins82"]).reset_index(drop=True)
    res, preds = run(tab, "25 seasons, box-score value")
    y = tab.wins82.values

    rng = np.random.default_rng(0)
    base = preds["last season's wins"]
    print("\n=== paired bootstrap vs 'last season's wins' (MAE gain, wins) ===")
    for name in [m for m in MODELS if m not in ("league average", "last season's wins")]:
        d = np.abs(y - base) - np.abs(y - preds[name])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        print(f"  {name:42s} {d.mean():+5.2f}  95% CI [{np.percentile(bs,2.5):+.2f}, "
              f"{np.percentile(bs,97.5):+.2f}]  P(better)={float((bs>0).mean()):.2f}")

    # Each structural model is compared against ITS OWN control: the same model with the structure
    # block removed. Comparing against a different baseline would credit structure with a gain that
    # actually comes from the other features.
    print("\n=== does roster structure add anything? (each vs its own control) ===")
    for name, ctrl in [("talent + continuity + structure", "talent + continuity"),
                       ("last season + talent + cont + structure",
                        "last season + talent + continuity")]:
        d = np.abs(y - preds[ctrl]) - np.abs(y - preds[name])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        print(f"  {name:42s} {d.mean():+5.2f}  95% CI [{np.percentile(bs,2.5):+.2f}, "
              f"{np.percentile(bs,97.5):+.2f}]  P(better)={float((bs>0).mean()):.2f}  "
              f"(vs {ctrl})")

    # era split: has predictability changed over 25 years?
    print("\n=== by era, best model MAE ===")
    best = res.iloc[0].model
    tab["pred"] = preds[best]
    tab["era"] = pd.cut(tab.season.str[:4].astype(int),
                        [2000, 2007, 2014, 2021, 2026],
                        labels=["2001-07", "2008-14", "2015-21", "2022-26"])
    for era, g in tab.groupby("era", observed=True):
        mae_m = float(np.abs(g.wins82 - g.pred).mean())
        mae_b = float(np.abs(g.wins82 - g.prior_wins82).mean())
        print(f"  {era}: n={len(g):3d}  model MAE {mae_m:.2f}   last-season MAE {mae_b:.2f}"
              f"   wins sd {g.wins82.std():.1f}")

    # feature correlations on the big sample
    feats = TALENT + ["continuity", "prior_wins82"] + STRUCT
    c = tab[feats + ["wins82"]].corr()["wins82"].drop("wins82").sort_values(key=np.abs,
                                                                           ascending=False)
    print("\n=== correlation with next-season wins (n=%d) ===" % len(tab))
    for k, v in c.items():
        print(f"  {k:16s} {v:+.3f}")

    tab.to_csv(os.path.join(OUT, "preseason_backtest_long.csv"), index=False, encoding="utf-8")
    res.to_csv(os.path.join(OUT, "preseason_model_scores_long.csv"), index=False)

    # head-to-head on the overlap era: box value vs lineup-ridge value
    short = pd.read_csv(os.path.join(OUT, "preseason_backtest.csv"))
    ov = sorted(set(short.season) & set(tab.season))
    a = short[short.season.isin(ov)]
    b = tab[tab.season.isin(ov)]
    print(f"\n=== head-to-head on {len(ov)} overlapping seasons ===")
    print(f"  lineup-ridge value model : MAE {np.abs(a.wins82 - a.pred_best).mean():.2f} (n={len(a)})")
    print(f"  box-score value model    : MAE {np.abs(b.wins82 - b.pred).mean():.2f} (n={len(b)})")
    print(f"  last season's wins       : MAE {np.abs(b.wins82 - b.prior_wins82).mean():.2f}")
    return res, tab


if __name__ == "__main__":
    main()
