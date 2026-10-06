"""Shared data layer for the visualization set.

Coordinate systems used throughout, and why:

* **Skill space** (`pc1..pc3`): one common space for the whole league across all seasons, so a
  player's position means the same thing in 2018-19 and 2025-26. Everything that compares players,
  teams or seasons uses this. PC1 runs perimeter spacing to rim-and-offensive-glass (a play-style
  axis, NOT size: Markkanen sits at +0.4 and Jokic at +1.9 while Gobert is at +9.9), PC2 runs
  off-ball to on-ball load, and PC3 runs steals/self-creation to efficient scoring volume.
* **Synergy space** (per-team MDS on synergy-derived distances): shows a single roster's *shape*,
  where distance means "fits together". It is only defined up to rotation and reflection, so raw
  MDS layouts are NOT comparable between seasons. `aligned_team_layouts` Procrustes-aligns a
  team's consecutive seasons so animations show real change rather than arbitrary spin.
"""
import os, sys, itertools, json, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_all as RA, topology as TP, metrics as MT

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, OUT, FIGS = (os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "out"),
                   os.path.join(ROOT, "figs"))
VIZ = os.path.join(ROOT, "viz")
DIMS = 6
MIN_MINUTES = 300

# Diverging scale for synergy/role colour; readable in both light and dark backgrounds.
POS, NEG = "#1b7f4f", "#c0392b"


def ensure_dirs():
    for d in (VIZ, os.path.join(VIZ, "teams"), os.path.join(VIZ, "evolution"),
              os.path.join(VIZ, "swaps"), os.path.join(VIZ, "pairs")):
        os.makedirs(d, exist_ok=True)


def league_edge_scale(ctx, q=0.99, sample_season=None):
    """One synergy scale for every picture.

    Normalising edge colour/width per team makes a flat roster look as dramatic as a polarised
    one, which is the single easiest way to over-read these graphs. Every view uses this league
    value instead.
    """
    import itertools as _it
    rows = []
    for s in ([sample_season] if sample_season else ctx.seasons):
        d = ctx.players[(ctx.players.season == s) & (ctx.players.minutes >= ctx.min_minutes)]
        ids = [p for p in d.pid if p in ctx.Z.index]
        Zs = np.array([ctx.Z.loc[p].values[:DIMS] for p in ids])
        if len(Zs) < 5:
            continue
        S = Zs @ ctx.M @ Zs.T
        iu = np.triu_indices(len(ids), 1)
        rows.append(np.abs(S[iu]))
    return float(np.quantile(np.concatenate(rows), q)) if rows else 1.0


class Ctx:
    """Everything the visualizations need, loaded once."""

    def __init__(self, min_minutes=MIN_MINUTES):
        self.players, self.stints = RA.load()
        self.syn, self.M, self.Z = RA.direct_synergy(dims=DIMS)
        self.eff_o = pd.read_parquet(os.path.join(PROC, "player_eff_off.parquet"))["eff"]
        self.eff_d = pd.read_parquet(os.path.join(PROC, "player_eff_def.parquet"))["eff"]
        self.value = (self.eff_o - self.eff_d).dropna()
        self.pairs = pd.read_parquet(os.path.join(PROC, "pairs.parquet"))
        self.seasons = sorted(self.stints.season.unique())
        self.min_minutes = min_minutes
        self.players["value"] = self.players.pid.map(self.value).fillna(0.0)
        self.players["off_eff"] = self.players.pid.map(self.eff_o).fillna(0.0)
        self.players["def_eff"] = self.players.pid.map(self.eff_d).fillna(0.0)
        # offence-minus-defence leaning: eff_d is points ALLOWED, so defensive contribution is -eff_d
        self.players["leaning"] = self.players.off_eff + self.players.def_eff
        self._add_roles()
        self.team_net = RA.team_net(self.stints)
        self._edge_scale = None
        self.abbr_to_id = {}
        for f in glob.glob(os.path.join(ROOT, "data", "raw", "Player_*.json")):
            for x in json.load(open(f)):
                self.abbr_to_id[x["TeamAbbreviation"]] = str(x["TeamId"])

    # ---------------------------------------------------------------- roles
    def _add_roles(self, k=7, seed=0):
        """Data-driven role families, named from whichever skills their centroid is extreme in."""
        from sklearn.cluster import KMeans
        cols = [f"pc{i+1}" for i in range(4)]
        X = self.players[cols].values
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        self.players["role_id"] = km.labels_
        # The clusters are data-driven; only the naming is rule-based, so that the legend reads in
        # basketball language instead of "high z_oreb_fg / low z_f_rim".
        prof = self.players.groupby("role_id").agg(
            size=("z_f_rim", "mean"), glass=("z_oreb_fg", "mean"), shoot=("z_fg3a_pct", "mean"),
            usage=("z_usage", "mean"), pas=("z_ast100", "mean"), blk=("z_blk100", "mean"))
        prof["big"] = prof[["size", "glass", "blk"]].mean(axis=1)
        order = list(prof.sort_values("big", ascending=False).index)
        names = {}
        bigs, mids, smalls = order[:2], order[2:4], order[4:]
        # bigs: the more perimeter-oriented one is the stretch big
        b = sorted(bigs, key=lambda r: -prof.loc[r, "shoot"])
        names[b[0]], names[b[1]] = "stretch big", "rim big"
        # forwards: higher on-ball load creates, the other connects or spaces
        m = sorted(mids, key=lambda r: -prof.loc[r, "usage"])
        names[m[0]] = "forward creator"
        names[m[1]] = "3-and-D wing" if prof.loc[m[1], "shoot"] > 0 else "connecting forward"
        # guards: lead / secondary playmaker / off-ball
        g = sorted(smalls, key=lambda r: -prof.loc[r, "usage"])
        names[g[0]] = "lead guard"
        rest = sorted(g[1:], key=lambda r: -prof.loc[r, "pas"])
        names[rest[0]] = "secondary playmaker"
        for extra in rest[1:]:
            names[extra] = "off-ball guard" if "off-ball guard" not in names.values() else "movement shooter"
        self.role_name = names
        self.role_order = [names[r] for r in order]
        self.players["role"] = self.players.role_id.map(names)

    # ------------------------------------------------------------- rosters
    def roster(self, team, season, min_minutes=None):
        mm = self.min_minutes if min_minutes is None else min_minutes
        r = self.players[(self.players.team == team) & (self.players.season == season) &
                         (self.players.minutes >= mm)]
        return r.sort_values("minutes", ascending=False).reset_index(drop=True)

    def teams(self, season):
        return sorted(self.players[self.players.season == season].team.dropna().unique())

    # ------------------------------------------------------------- synergy
    def syn_matrix(self, roster):
        ids = list(roster.pid)
        n = len(ids)
        W = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            W[i, j] = W[j, i] = self.syn(ids[i], ids[j])
        return W

    @property
    def edge_scale(self):
        if self._edge_scale is None:
            self._edge_scale = league_edge_scale(self)
        return self._edge_scale

    def net_rating(self, team, season):
        tid = self.abbr_to_id.get(team)
        row = self.team_net[(self.team_net.season == season) & (self.team_net.team == tid)]
        return float(row.net.iloc[0]) if len(row) else np.nan

    def structure(self, roster):
        ps = MT.pair_stats(roster, self.syn)
        return dict(redundancy=MT.redundancy(roster), eff_roles=MT.effective_roles(roster),
                    complementarity=ps["complementarity"], fragility=ps["fragility"],
                    connector=ps["connector"], neg_share=ps["neg_share"])


# ------------------------------------------------------------------ layouts
def procrustes(A, B):
    """Rotate/reflect B onto A (both centred), returning the aligned B."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(B0.T @ A0)
    return B0 @ (U @ Vt)


def aligned_team_layouts(ctx, team, seasons=None, seed=0):
    """MDS synergy layouts per season, Procrustes-aligned to the previous season on shared players.

    Without this, consecutive seasons spin and flip arbitrarily and an animation looks like chaos
    even when the roster barely changed.
    """
    seasons = seasons or ctx.seasons
    out, prev = {}, None
    for s in seasons:
        r = ctx.roster(team, s)
        if len(r) < 4:
            continue
        pos = TP.synergy_mds(ctx.syn_matrix(r), seed=seed)
        if prev is not None:
            shared = [p for p in r.pid if p in prev[0]]
            if len(shared) >= 3:
                idx_new = [list(r.pid).index(p) for p in shared]
                idx_old = [prev[0].index(p) for p in shared]
                R = _best_rotation(prev[1][idx_old], pos[idx_new])
                pos = (pos - pos[idx_new].mean(0)) @ R + prev[1][idx_old].mean(0)
        out[s] = (r, pos)
        prev = (list(r.pid), pos)
    return out


def _best_rotation(target, source):
    t0, s0 = target - target.mean(0), source - source.mean(0)
    U, _, Vt = np.linalg.svd(s0.T @ t0)
    return U @ Vt


def edge_list(roster, W, thresh_frac=0.0):
    """(i, j, weight) for the upper triangle, optionally dropping the weakest edges."""
    n = len(roster)
    iu = np.triu_indices(n, 1)
    w = W[iu]
    keep = np.abs(w) >= thresh_frac * (np.abs(w).max() + 1e-12)
    return [(int(i), int(j), float(x)) for i, j, x, k in zip(iu[0], iu[1], w, keep) if k]


SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def surname(nm):
    parts = [x for x in nm.split() if x.lower().strip(".") not in SUFFIX]
    return parts[-1] if parts else nm.split()[-1]


def short_labels(names):
    last = [surname(n) for n in names]
    dup = {x for x in last if last.count(x) > 1}
    lab = [f"{n.split()[0][0]}. {l}" if l in dup else l for n, l in zip(names, last)]
    dup2 = {x for x in lab if lab.count(x) > 1}
    return [f"{n.split()[0]} {l}" if x in dup2 else x for n, l, x in zip(names, last, lab)]
