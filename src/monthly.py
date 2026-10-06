"""Month-by-month player embeddings, projected onto the season model's own PCA basis.

Monthly rows are standardized with the SEASON's means and standard deviations and then pushed
through the PCA fitted on season data. Refitting either per month would make the axes mean
something different in November than in March, and the organism would appear to move when only the
coordinate system had.
"""
import io, json, os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import features as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, PROC = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "proc")
MIN_MONTH_MINUTES = 60


def season_basis(season):
    """Refit the season-level standardization and PCA exactly as features.py does."""
    rows = []
    for d in json.load(io.open(os.path.join(RAW, f"Player_{season}.json"), encoding="utf-8")):
        if (d.get("Minutes") or 0) < F.MIN_MIN:
            continue
        rows.append(F.player_row(d, season))
    df = pd.DataFrame(rows)
    X = df[F.SKILL_COLS].astype(float).fillna(0.0)
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    Z = ((X - mu) / sd).clip(-4, 4).fillna(0.0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=12, random_state=0).fit(Z.values)
    return dict(mu=mu, sd=sd, pca=pca, season_df=df)


def monthly_frame(season, basis=None, min_minutes=MIN_MONTH_MINUTES):
    basis = basis or season_basis(season)
    files = sorted(glob.glob(os.path.join(RAW, "monthly", f"Player_{season}_*.json")))
    out = []
    for f in files:
        month = os.path.basename(f)[len(f"Player_{season}_"):-len(".json")]
        rows = []
        for d in json.load(io.open(f, encoding="utf-8")):
            if (d.get("Minutes") or 0) < min_minutes:
                continue
            r = F.player_row(d, season)
            r["month"] = month
            rows.append(r)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        X = df[F.SKILL_COLS].astype(float).fillna(0.0)
        Z = ((X - basis["mu"]) / basis["sd"]).clip(-4, 4).fillna(0.0)
        P = basis["pca"].transform(Z.values)
        for k in range(P.shape[1]):
            df[f"pc{k+1}"] = P[:, k]
        for c in F.SKILL_COLS:
            df["z_" + c] = Z[c].values
        out.append(df)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def main(season="2025-26"):
    basis = season_basis(season)
    df = monthly_frame(season, basis)
    if not len(df):
        print("no monthly data"); return
    df = df.sort_values(["month", "minutes"], ascending=[True, False]).reset_index(drop=True)
    df.to_parquet(os.path.join(PROC, f"players_monthly_{season}.parquet"))
    print(f"{len(df)} player-months, {df.month.nunique()} months, {df.pid.nunique()} players")
    print(df.groupby("month").agg(players=("pid", "size"), minutes=("minutes", "sum")).to_string())
    # sanity: a player's monthly position should sit near his season position
    season_df = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    s = season_df[season_df.season == season].set_index("pid")
    m = df[df.minutes > 300].groupby("pid")[["pc1", "pc2", "pc3"]].mean()
    common = m.index.intersection(s.index)
    d = np.linalg.norm(m.loc[common].values - s.loc[common, ["pc1", "pc2", "pc3"]].values, axis=1)
    print(f"median distance between monthly-average and season position: {np.median(d):.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
