"""Chronological validation: do interaction terms beat an additive lineup model?

Every model is calibrated (intercept + slope) by CROSS-FITTING inside the training period:
effects are fit on one half of the training lineups and calibrated on the other half, so the
calibration is never read off predictions that saw their own outcome.
"""
import os, sys, itertools, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S
import bilinear as B
import direct as D

DIMS = 8


def agg_lineups(df, min_poss=50):
    g = df.groupby(["season", "team", "lineup"], as_index=False).agg(
        off_poss=("off_poss", "sum"), def_poss=("def_poss", "sum"),
        pts=("pts", "sum"), opp_pts=("opp_pts", "sum"))
    g = g[(g.off_poss >= min_poss) & (g.def_poss >= min_poss)].copy()
    g["net"] = 100 * (g.pts / g.off_poss - g.opp_pts / g.def_poss)
    g["poss"] = g.off_poss + g.def_poss
    g["players"] = g.lineup.str.split("-")
    return g.reset_index(drop=True)


def raw_pair_net(stints, prior=800.0):
    acc = {}
    for pl, op, dp, pts, opp in zip(stints.players, stints.off_poss, stints.def_poss,
                                    stints.pts, stints.opp_pts):
        for a, b in itertools.combinations(sorted(pl), 2):
            r = acc.setdefault((a, b), [0, 0, 0, 0])
            r[0] += op; r[1] += dp; r[2] += pts; r[3] += opp
    out = {}
    for k, (op, dp, pts, opp) in acc.items():
        if op < 50 or dp < 50:
            continue
        out[k] = 100 * (pts / op - opp / dp) * (op / (op + prior))
    return out


class Fit:
    """Everything estimated from one block of stints."""

    def __init__(self, stints, players, dims=DIMS):
        self.stints = stints
        cnt_o = S.pair_counts(stints, "off"); cnt_d = S.pair_counts(stints, "def")
        self.pairs_o = sorted([p for p, c in cnt_o.items() if c >= S.MIN_PAIR_POSS])
        self.pairs_d = sorted([p for p, c in cnt_d.items() if c >= S.MIN_PAIR_POSS])
        self.add_o = S.build(stints, "off", [])
        self.add_d = S.build(stints, "def", [])
        self.pr_o = S.build(stints, "off", self.pairs_o)
        self.pr_d = S.build(stints, "def", self.pairs_d)
        self.ridge_pairs = {}
        common = self.pr_o["pair"].index.intersection(self.pr_d["pair"].index)
        for k in common:
            self.ridge_pairs[tuple(k)] = float(self.pr_o["pair"].loc[k] - self.pr_d["pair"].loc[k])
        self.raw_pairs = raw_pair_net(agg_lineups(stints, min_poss=1))
        seasons = set(stints.season.unique())
        pf = players[players.season.isin(seasons)]
        self.Z = B.player_z(pf, dims=dims)
        pdf = pd.DataFrame([{"p1": a, "p2": b, "net": v, "poss": cnt_o.get((a, b), 0)}
                            for (a, b), v in self.ridge_pairs.items()])
        if len(pdf) > 50:
            self.M, _, _, self.rfit, _, _ = B.fit_M(pdf, self.Z, dims=dims)
        else:
            self.M, self.rfit = np.zeros((dims, dims)), np.nan
        self.dims = dims
        self.ddims = 6
        self._players = players
        self._direct_fit = None

    @property
    def dir_o(self):
        self._ensure_direct(); return self._direct_fit[0]

    @property
    def dir_d(self):
        self._ensure_direct(); return self._direct_fit[1]

    @property
    def M_dir(self):
        self._ensure_direct(); return self._direct_fit[2]

    @property
    def Zd(self):
        self._ensure_direct(); return self._direct_fit[0]["Z"]

    def _ensure_direct(self):
        """Direct joint fit is expensive, so it is only computed when a model asks for it."""
        if self._direct_fit is None:
            fo = D.fit(self.stints, self._players, "off", dims=self.ddims)
            fd = D.fit(self.stints, self._players, "def", dims=self.ddims)
            self._direct_fit = (fo, fd, D.net_M(fo, fd))

    def _direct(self, lineups):
        po, pdd = self.dir_o["player"], self.dir_d["player"]
        mo, md = po.mean(), pdd.mean()
        out = np.zeros(len(lineups))
        for i, pl in enumerate(lineups.players.values):
            v = sum(po.get(p, mo) for p in pl) - sum(pdd.get(p, md) for p in pl)
            zs = [self.Zd.loc[p].values[:self.ddims] for p in pl if p in self.Zd.index]
            for a, b in itertools.combinations(zs, 2):
                v += float(a @ self.M_dir @ b)
            out[i] = v
        return out, np.zeros(len(lineups), bool)

    def _direct_add(self, lineups):
        """Direct-fit player effects with the interaction surface switched off (control)."""
        po, pdd = self.dir_o["player"], self.dir_d["player"]
        mo, md = po.mean(), pdd.mean()
        out = np.array([sum(po.get(p, mo) for p in pl) - sum(pdd.get(p, md) for p in pl)
                        for pl in lineups.players.values])
        return out, np.zeros(len(lineups), bool)

    def _base(self, lineups, pairmodel):
        po, pdf_ = (self.add_o["player"], self.add_d["player"]) if pairmodel is None \
            else (self.pr_o["player"], self.pr_d["player"])
        mo, md = po.mean(), pdf_.mean()
        out = np.zeros(len(lineups)); newp = np.zeros(len(lineups), bool)
        for i, pl in enumerate(lineups.players.values):
            v = sum(po.get(p, mo) for p in pl) - sum(pdf_.get(p, md) for p in pl)
            for a, b in itertools.combinations(sorted(pl), 2):
                if pairmodel == "ridge":
                    if (a, b) in self.ridge_pairs:
                        v += self.ridge_pairs[(a, b)]
                    else:
                        newp[i] = True
                elif pairmodel == "raw":
                    v += self.raw_pairs.get((a, b), 0.0)
                elif pairmodel == "bilinear":
                    if a in self.Z.index and b in self.Z.index:
                        v += float(self.Z.loc[a].values[:self.dims] @ self.M
                                   @ self.Z.loc[b].values[:self.dims])
            out[i] = v
        return out, newp

    def predict(self, lineups, model):
        if model == "direct joint (M on)":
            return self._direct(lineups)
        if model == "direct joint (M off)":
            return self._direct_add(lineups)
        pm = {"additive": None, "additive+rawpair": "raw", "additive+ridgepair": "ridge",
              "additive+bilinear": "bilinear"}[model]
        return self._base(lineups, pm)


MODELS = ["additive", "additive+rawpair", "additive+ridgepair", "additive+bilinear",
          "direct joint (M off)", "direct joint (M on)"]


def wmetrics(y, p, w):
    w = np.asarray(w, float)
    mu = np.average(y, weights=w)
    mse = np.average((y - p) ** 2, weights=w)
    return dict(rmse=float(np.sqrt(mse)), r2=float(1 - mse / np.average((y - mu) ** 2, weights=w)),
                corr=float(np.corrcoef(y, p)[0, 1]), n=int(len(y)))


def wls(p, y, w):
    A = np.vstack([np.ones_like(p), p]).T
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


def run(train_seasons, test_season, players=None, min_poss=50, seed=0, verbose=True):
    df = S.load_stints()
    players = players if players is not None else pd.read_parquet(os.path.join(PROC, "players.parquet"))
    tr = df[df.season.isin(train_seasons)].reset_index(drop=True)
    te = agg_lineups(df[df.season == test_season], min_poss=min_poss)
    if verbose:
        print(f"train {train_seasons}: {len(tr)} lineup-rows | test {test_season}: {len(te)} lineups")

    # --- cross-fit calibration inside the training period
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(tr))
    cal_pred = {m: [] for m in MODELS}
    cal_y, cal_w = [], []
    for f in (0, 1):
        fit_f = Fit(tr[fold == f].reset_index(drop=True), players)
        held = agg_lineups(tr[fold != f], min_poss=min_poss)
        if len(held) < 30:
            continue
        for m in MODELS:
            cal_pred[m].append(fit_f.predict(held, m)[0])
        cal_y.append(held.net.values); cal_w.append(held.poss.values)
    cal_y = np.concatenate(cal_y); cal_w = np.concatenate(cal_w)
    coefs = {m: wls(np.concatenate(cal_pred[m]), cal_y, cal_w) for m in MODELS}
    if verbose:
        print("  cross-fit calibration slopes:",
              {m: round(float(c[1]), 3) for m, c in coefs.items()})

    full = Fit(tr, players)
    if verbose:
        print(f"  pairs with own coefficient: {len(full.ridge_pairs)} | "
              f"bilinear fit corr: {full.rfit:.3f}")

    y, w = te.net.values, te.poss.values
    res = {"mean": wmetrics(y, np.full(len(y), np.average(y, weights=w)), w)}
    preds, newp = {}, None
    for m in MODELS:
        p, np_ = full.predict(te, m)
        if m == "additive+ridgepair":
            newp = np_
        b = coefs[m]
        preds[m] = b[0] + b[1] * p
        res[m] = wmetrics(y, preds[m], w)
    out = pd.DataFrame(res).T
    if verbose:
        print(f"\n=== test {test_season} ===")
        print(out.round(4).to_string())
        if newp is not None and newp.sum() > 20:
            sub = {m: wmetrics(y[newp], preds[m][newp], w[newp]) for m in MODELS}
            print(f"\n-- lineups containing >=1 pair with no training history (n={int(newp.sum())}) --")
            print(pd.DataFrame(sub).T.round(4).to_string())
    return out, full, preds, te


if __name__ == "__main__":
    run(sys.argv[1].split(","), sys.argv[2])
