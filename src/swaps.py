"""Counterfactual roster swaps: replace_player(team, outgoing, incoming) -> structural + predicted deltas."""
import os, itertools
import numpy as np, pandas as pd
import metrics as MT

DIMS = 8


def team_roster(players, team, season, min_minutes=300):
    r = players[(players.team == team) & (players.season == season) & (players.minutes >= min_minutes)]
    return r.sort_values("minutes", ascending=False).reset_index(drop=True)


def roster_summary(roster, syn, league_Z, centroids, cov_thresh, eff_o, eff_d):
    ps = MT.pair_stats(roster, syn)
    val = sum(eff_o.get(p, 0) - eff_d.get(p, 0) for p in roster.pid)
    mw = np.average([eff_o.get(p, 0) - eff_d.get(p, 0) for p in roster.pid], weights=roster.minutes)
    return dict(
        redundancy=MT.redundancy(roster), eff_roles=MT.effective_roles(roster),
        coverage=MT.coverage(roster, centroids, cov_thresh), scarcity=MT.scarcity(roster, league_Z),
        complementarity=ps["complementarity"], neg_share=ps["neg_share"],
        fragility=ps["fragility"], connector=ps["connector"],
        player_value_sum=val, player_value_mw=mw)


def predicted_top_lineup(roster, syn, eff_o, eff_d, k=5, top=3):
    """Best predicted 5-man units: additive player value + all pair synergies."""
    out = []
    ids, names = list(roster.pid), list(roster.name)
    val = {p: eff_o.get(p, 0) - eff_d.get(p, 0) for p in ids}
    for combo in itertools.combinations(range(len(ids)), k):
        s = sum(val[ids[i]] for i in combo)
        s += sum(syn(ids[i], ids[j]) for i, j in itertools.combinations(combo, 2))
        out.append((s, [names[i] for i in combo]))
    out.sort(reverse=True, key=lambda x: x[0])
    return out[:top], out[-top:]


def replace_player(players, roster, outgoing_name, incoming_row, syn, league_Z, centroids,
                   cov_thresh, eff_o, eff_d):
    """Return (before, after, deltas). incoming_row is a player-season row (pd.Series)."""
    before = roster_summary(roster, syn, league_Z, centroids, cov_thresh, eff_o, eff_d)
    out_idx = roster.index[roster.name == outgoing_name]
    if len(out_idx) == 0:
        raise ValueError(f"{outgoing_name} not on roster; roster is: "
                         + ", ".join(roster.name))
    new = roster.drop(index=out_idx[0]).copy()
    inc = incoming_row.copy()
    inc["minutes"] = roster.loc[out_idx[0], "minutes"]   # incoming inherits the vacated minutes
    new = pd.concat([new, pd.DataFrame([inc])], ignore_index=True)
    after = roster_summary(new, syn, league_Z, centroids, cov_thresh, eff_o, eff_d)
    delta = {k: (after[k] - before[k]) for k in before if isinstance(before[k], float)}
    return before, after, delta, new
