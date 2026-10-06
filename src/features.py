"""Player-season feature table + continuous embeddings (PCA / UMAP)."""
import json, os, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, PROC = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "proc")
MIN_MIN = 400  # rotation-player cut

def per100(d, col, poss):
    return 100.0 * d.get(col, 0) / max(poss, 1)

def player_row(d, season):
    """One player-season (or player-month) row of features, from a raw pbpstats totals record."""
    mins = d.get("Minutes", 0)
    op, dp = d.get("OffPoss", 0), d.get("DefPoss", 0)
    fga = d.get("FG2A", 0) + d.get("FG3A", 0)
    if True:
        if True:
            r = dict(
                season=season, pid=str(d["EntityId"]), name=d["Name"], team=d.get("TeamAbbreviation"),
                minutes=mins, off_poss=op, def_poss=dp, games=d.get("GamesPlayed", 0),
                # --- load / creation
                usage=d.get("Usage", 0), fga100=per100(d, "FG2A", op) + per100(d, "FG3A", op),
                pts100=per100(d, "Points", op), ast100=per100(d, "Assists", op),
                astpts100=per100(d, "AssistPoints", op), tov100=per100(d, "Turnovers", op),
                badpass100=per100(d, "BadPassTurnovers", op), lostball100=per100(d, "LostBallTurnovers", op),
                fouldrawn100=per100(d, "FoulsDrawn", op), ftpts100=per100(d, "FtPoints", op),
                # --- shot profile
                f_rim=d.get("AtRimFrequency", 0), f_smid=d.get("ShortMidRangeFrequency", 0),
                f_lmid=d.get("LongMidRangeFrequency", 0), f_c3=d.get("Corner3Frequency", 0),
                f_a3=d.get("Arc3Frequency", 0), fg3a_pct=d.get("FG3APct", 0),
                avg2dist=d.get("Avg2ptShotDistance", 0) or 0,
                # --- efficiency / finishing
                ts=d.get("TsPct", 0), efg=d.get("EfgPct", 0),
                rim_acc=d.get("AtRimAccuracy", 0), rim_blk_pct=d.get("AtRimPctBlocked", 0),
                fg3_pct=d.get("NonHeaveFg3Pct", 0), mid_acc=d.get("ShortMidRangeAccuracy", 0),
                shotq=d.get("ShotQualityAvg", 0),
                # --- self-creation vs catch-and-shoot
                assisted2=d.get("Assisted2sPct", 0), assisted3=d.get("Assisted3sPct", 0),
                rim_assisted=d.get("AtRimPctAssisted", 0), c3_assisted=d.get("Corner3PctAssisted", 0),
                unast2_share=(d.get("PtsUnassisted2s", 0) / max(d.get("Points", 1), 1)),
                # --- passing mix
                ast3_share=(d.get("ThreePtAssists", 0) / max(d.get("Assists", 1), 1)),
                # --- rebounding
                oreb_fg=d.get("OffFGReboundPct", 0), dreb_fg=d.get("DefFGReboundPct", 0),
                oreb_rim=d.get("OffAtRimReboundPct", 0), dreb_rim=d.get("DefAtRimReboundPct", 0),
                dreb_3=d.get("DefThreePtReboundPct", 0), self_oreb=d.get("SelfORebPct", 0),
                putback100=per100(d, "PtsPutbacks", op),
                # --- defense / events
                stl100=per100(d, "Steals", dp), blk100=per100(d, "Blocks", dp),
                foul100=per100(d, "Fouls", dp), charge100=per100(d, "Charge Fouls Drawn", dp),
                # --- on-court impact (kept out of the skill embedding, used as value prior)
                on_ortg=d.get("OnOffRtg", np.nan), on_drtg=d.get("OnDefRtg", np.nan),
                plusminus=d.get("PlusMinus", 0),
            )
    return r


def build_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "Player_*.json"))):
        season = os.path.basename(p)[len("Player_"):-len(".json")]
        for d in json.load(open(p)):
            mins = d.get("Minutes", 0)
            if mins is None or mins < MIN_MIN:
                continue
            rows.append(player_row(d, season))
    return pd.DataFrame(rows)

SKILL_COLS = ["usage","fga100","pts100","ast100","astpts100","tov100","badpass100","lostball100",
    "fouldrawn100","ftpts100","f_rim","f_smid","f_lmid","f_c3","f_a3","fg3a_pct","avg2dist","ts","efg",
    "rim_acc","rim_blk_pct","fg3_pct","mid_acc","shotq","assisted2","assisted3","rim_assisted",
    "c3_assisted","unast2_share","ast3_share","oreb_fg","dreb_fg","oreb_rim","dreb_rim","dreb_3",
    "self_oreb","putback100","stl100","blk100","foul100","charge100"]

def main():
    os.makedirs(PROC, exist_ok=True)
    df = build_rows()
    # players traded mid-season appear once per team: keep the row with most minutes
    df = df.sort_values("minutes", ascending=False).drop_duplicates(["season", "pid"]).reset_index(drop=True)
    X = df[SKILL_COLS].astype(float).fillna(0.0)
    # z-score within season so league-wide drift (3pt boom) does not dominate the geometry
    Z = X.copy()
    for s, idx in df.groupby("season").groups.items():
        sub = X.loc[idx]
        Z.loc[idx] = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1)
    Z = Z.clip(-4, 4).fillna(0.0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=12, random_state=0).fit(Z.values)
    P = pca.transform(Z.values)
    print("PCA explained var:", np.round(pca.explained_variance_ratio_, 3),
          "cum:", round(pca.explained_variance_ratio_.sum(), 3))
    for k in range(6):
        load = pd.Series(pca.components_[k], index=SKILL_COLS).sort_values()
        print(f"\nPC{k+1} ({pca.explained_variance_ratio_[k]:.1%})  -: "
              + ", ".join(load.index[:5]) + "  |  +: " + ", ".join(load.index[-5:][::-1]))
    for k in range(12):
        df[f"pc{k+1}"] = P[:, k]
    import umap
    U = umap.UMAP(n_neighbors=25, min_dist=0.25, random_state=0).fit_transform(Z.values)
    df["umap1"], df["umap2"] = U[:, 0], U[:, 1]
    for c in SKILL_COLS:
        df["z_" + c] = Z[c].values
    df.to_parquet(os.path.join(PROC, "players.parquet"))
    print("\nplayer-seasons:", len(df), "seasons:", sorted(df.season.unique()))

if __name__ == "__main__":
    main()
