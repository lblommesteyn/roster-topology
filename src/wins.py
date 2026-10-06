"""Preseason win projection, backtested.

The framing is a projection made after free agency and before tip-off: you know who is on the
roster, you know nothing about how the season goes. Concretely, for target season t+1 every input
comes from seasons <= t:

  * player value          ridge lineup effects fit on stints from seasons <= t only
  * minutes               PROJECTED from prior-season minutes, never the realised ones
  * skill positions       prior-season PCA coordinates
  * complementarity       interaction surface M fit on seasons <= t only

The one thing taken from season t+1 is roster membership, which is what a preseason projection
legitimately knows. It is not perfectly clean: a player acquired at the February deadline is in
the membership set. Weighting strictly by PRIOR minutes keeps such players near zero weight, and
the leak is measured in section "membership sensitivity" of the report.

Targets are win rate, expressed on an 82-game pace, because 2019-20 and 2020-21 were short.
"""
import io, json, os, sys, glob, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, validate as VD, synergy as S, metrics as MT, run_all as RA, blob as B

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, PROC, OUT = (os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "proc"),
                  os.path.join(ROOT, "out"))
TEAM_MINUTES = 240 * 82          # player-minutes available to a team over an 82-game season
DIMS = 6


# ----------------------------------------------------------------- targets
def team_wins():
    """Actual wins, losses and win rate per team-season."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "games_*.json"))):
        season = os.path.basename(p)[len("games_"):-len(".json")]
        rec = {}
        for g in json.load(io.open(p, encoding="utf-8")):
            h, a = g["HomeTeamAbbreviation"], g["AwayTeamAbbreviation"]
            hw = g["HomePoints"] > g["AwayPoints"]
            for t, won in ((h, hw), (a, not hw)):
                r = rec.setdefault(t, [0, 0])
                r[0] += int(won); r[1] += 1
        for t, (w, n) in rec.items():
            rows.append(dict(season=season, team=t, wins=w, games=n, win_rate=w / n,
                             wins82=82 * w / n))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- projections
def project_minutes(prev_minutes, roster_pids, replacement=550.0, cap=2600.0):
    """Split a team's 19,680 player-minutes using prior-season minutes as the prior.

    A player with no prior season gets a replacement-level allocation rather than zero, so rookies
    and returnees from abroad do not silently vanish from the roster.
    """
    raw = np.array([min(prev_minutes.get(p, replacement), cap) for p in roster_pids], float)
    raw = np.maximum(raw, 120.0)
    if raw.sum() <= 0:
        return raw
    return raw / raw.sum() * TEAM_MINUTES


def build_features(seasons_hist, target_season, ctx, wins, top_n=10, verbose=False,
                   interaction=None):
    """One row per team for `target_season`, using only information from `seasons_hist`.

    `interaction` is an optional (M, Z) pair for the complementarity surface. Refitting it for
    every target season costs minutes each; it is refit once per multi-season block instead, which
    is still strictly prior-only.
    """
    st = ctx.stints[ctx.stints.season.isin(seasons_hist)].reset_index(drop=True)
    fit = VD.Fit(st, ctx.players)
    po, pdd = fit.add_o["player"], fit.add_d["player"]
    val = {p: float(po.get(p, po.mean()) - pdd.get(p, pdd.mean())) for p in set(po.index) | set(pdd.index)}
    repl = float(np.percentile(list(val.values()), 20)) if val else 0.0

    # most recent prior-season profile for each player
    hist = ctx.players[ctx.players.season.isin(seasons_hist)].sort_values("season")
    last = hist.drop_duplicates("pid", keep="last").set_index("pid")
    prev = ctx.players[ctx.players.season == seasons_hist[-1]]
    prev_min = dict(zip(prev.pid, prev.minutes))

    # complementarity surface from prior seasons only
    M, Zd = interaction if interaction is not None else (fit.M_dir, fit.Zd)

    lg = hist[hist.season == seasons_hist[-1]]
    league_Z = lg[[f"pc{k+1}" for k in range(MT.DIMS)]].values
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=14, n_init=10, random_state=0).fit(league_Z)
    thresh = np.median(np.linalg.norm(league_Z - km.cluster_centers_[km.labels_], axis=1)) * 1.5
    lo, hi = B.league_bounds(hist, ctx.min_minutes)

    tgt = ctx.players[ctx.players.season == target_season]
    prev_teams = ctx.players[ctx.players.season == seasons_hist[-1]].set_index("pid")["team"].to_dict()
    rows = []
    for team, grp in tgt.groupby("team"):
        # roster membership: the top prior-minutes players listed on this team next season
        pids = list(grp.pid)
        pids = sorted(pids, key=lambda p: -prev_min.get(p, 0))[:top_n]
        known = [p for p in pids if p in last.index]
        if len(known) < 5:
            continue
        r = last.loc[known].copy().reset_index()
        r["minutes"] = project_minutes(prev_min, known)
        r["value"] = [val.get(p, repl) for p in known]
        w = r.minutes.values

        # talent
        order = np.argsort(-r.value.values)
        vals = r.value.values[order]
        talent_mw = float(np.average(r.value, weights=w))
        star = float(vals[:3].sum())
        depth = float(vals[3:9].sum()) if len(vals) > 3 else 0.0

        # continuity: share of last season's minutes played by players still here
        prev_team_min = {p: m for p, m in prev_min.items() if prev_teams.get(p) == team}
        cont = (sum(m for p, m in prev_team_min.items() if p in set(pids)) /
                max(sum(prev_team_min.values()), 1))

        # structure in skill space
        syn_vals, ww = [], []
        for i, j in itertools.combinations(range(len(r)), 2):
            a, b = r.pid.iloc[i], r.pid.iloc[j]
            if a in Zd.index and b in Zd.index:
                syn_vals.append(float(Zd.loc[a].values[:DIMS] @ M @ Zd.loc[b].values[:DIMS]))
            else:
                syn_vals.append(0.0)
            ww.append(w[i] * w[j])
        comp = float(np.average(syn_vals, weights=ww)) if ww else 0.0
        ps = MT.pair_stats(r, lambda a, b: (float(Zd.loc[a].values[:DIMS] @ M @ Zd.loc[b].values[:DIMS])
                                            if a in Zd.index and b in Zd.index else 0.0))
        blob = B.team_blob(ctx, r, lo, hi, n=40)
        bm = blob["metrics"] if blob else {}

        prow = wins[(wins.season == seasons_hist[-1]) & (wins.team == team)]
        trow = wins[(wins.season == target_season) & (wins.team == team)]
        if not len(trow):
            continue
        tid = ctx.abbr_to_id.get(team)
        pnet = ctx.team_net[(ctx.team_net.season == seasons_hist[-1]) & (ctx.team_net.team == tid)]
        rows.append(dict(
            season=target_season, team=team,
            wins82=float(trow.wins82.iloc[0]),
            prior_wins82=float(prow.wins82.iloc[0]) if len(prow) else np.nan,
            prior_net=float(pnet.net.iloc[0]) if len(pnet) else np.nan,
            talent_mw=talent_mw, star=star, depth=depth, continuity=cont,
            redundancy=MT.redundancy(r), eff_roles=MT.effective_roles(r),
            coverage=MT.coverage(r, km.cluster_centers_, thresh),
            scarcity=MT.scarcity(r, league_Z),
            complementarity=comp, fragility=ps["fragility"],
            volume=bm.get("volume", np.nan), sphericity=bm.get("sphericity", np.nan),
            components=bm.get("components", np.nan), peak_density=bm.get("peak_density", np.nan),
            n_known=len(known)))
    return pd.DataFrame(rows)


HIST_WINDOW = 5      # seasons of history used for player value; a 1997 season should not inform 2020
M_BLOCK = 5          # refit the interaction surface once per this many target seasons


def main(min_hist=2, hist_window=HIST_WINDOW):
    import direct as D
    ctx = V.Ctx()
    wins = team_wins()
    wins.to_csv(os.path.join(OUT, "team_wins.csv"), index=False)
    seasons = ctx.seasons
    frames = []
    inter, inter_block = None, None
    for i in range(min_hist, len(seasons)):
        hist, tgt = seasons[max(0, i - hist_window):i], seasons[i]
        block = (i - min_hist) // M_BLOCK
        if inter_block != block:
            st = ctx.stints[ctx.stints.season.isin(hist)].reset_index(drop=True)
            st = st.assign(players=st.lineup.str.split("-"))
            print(f"  fitting interaction surface on {hist[0]}..{hist[-1]}", flush=True)
            fo = D.fit(st, ctx.players, "off", dims=DIMS)
            fd = D.fit(st, ctx.players, "def", dims=DIMS)
            inter, inter_block = (D.net_M(fo, fd), fo["Z"]), block
        print(f"building {tgt} from {hist[0]}..{hist[-1]}", flush=True)
        f = build_features(hist, tgt, ctx, wins, interaction=inter)
        if len(f):
            frames.append(f)
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(OUT, "preseason_features.csv"), index=False, encoding="utf-8")
    print(f"\n{len(tab)} team-seasons projected")
    return tab


if __name__ == "__main__":
    main()
