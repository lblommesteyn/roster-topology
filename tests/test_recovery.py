"""Synthetic recovery: the ridge must find planted player effects and a planted pair effect."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import synergy as S


def make(seed=0, n=2000, bonus=6.0):
    rng = np.random.default_rng(seed)
    pl = [f"p{i}" for i in range(12)]
    true = {p: rng.normal(0, 3) for p in pl}
    rows = []
    for _ in range(n):
        c = sorted(rng.choice(pl, 5, replace=False))
        op = int(rng.integers(20, 120))
        mu = 110 + sum(true[p] for p in c) + (bonus if ("p0" in c and "p1" in c) else 0)
        rows.append(dict(season="S", team="T", opp="ALL", home=0, lineup="-".join(c), game="g",
                         off_poss=op, def_poss=op, pts=rng.normal(mu, 12) * op / 100,
                         opp_pts=rng.normal(110, 12) * op / 100))
    df = pd.DataFrame(rows); df["players"] = df.lineup.str.split("-")
    return df, true


def test_recovery():
    df, true = make()
    cnt = S.pair_counts(df, "off")
    prs = sorted([k for k, v in cnt.items() if v >= 400])
    r = S.build(df, "off", prs)
    est = r["player"]
    assert np.corrcoef(est.values, [true[p] for p in est.index])[0, 1] > 0.8
    pp = r["pair"]
    assert pp.loc[("p0", "p1")] > 3 * pp.drop(("p0", "p1")).std()


if __name__ == "__main__":
    test_recovery(); print("ok")
