"""Team roster topology: signed-force layout where complementarity attracts and redundancy repels."""
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC, FIGS = os.path.join(ROOT, "data", "proc"), os.path.join(ROOT, "figs")


def signed_layout(W, seed=0, iters=900, size=None):
    """Signed force-directed layout (kept for comparison; unstable - it collapses rosters
    onto a line because the signed springs have no equilibrium length)."""
    rng = np.random.default_rng(seed)
    n = len(W)
    sd = np.abs(W[np.triu_indices(n, 1)]).std() + 1e-9
    Wn = np.clip(W / sd, -3, 3); np.fill_diagonal(Wn, 0.0)
    pos = rng.normal(scale=1.0, size=(n, 2))
    for t in range(iters):
        step = 0.05 * (1 - t / iters) + 0.004
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-3
        u = d / dist[:, :, None]
        rep = (0.55 / dist)[:, :, None] * u
        np.einsum("iij->ij", rep)[...] = 0
        att = (0.3 * Wn * (dist - 1.0))[:, :, None] * (-u)
        f = (rep + att).sum(axis=1) - 0.05 * pos
        pos += step * f
    return pos - pos.mean(axis=0)


def synergy_mds(W, seed=0):
    """Layout by MDS on synergy-derived target distances: complementary pairs sit close,
    redundant/clashing pairs sit far apart. Stable where signed springs are not."""
    from sklearn.manifold import MDS
    n = len(W)
    iu = np.triu_indices(n, 1)
    sd = np.abs(W[iu]).std() + 1e-9
    Wn = np.clip(W / sd, -2.5, 2.5)
    D = 1.0 - 0.32 * Wn
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.12, None); np.fill_diagonal(D, 0.0)
    pos = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              n_init=6, max_iter=600, normalized_stress=False).fit_transform(D)
    pos -= pos.mean(axis=0)
    return pos


def team_graph(team, season, roster, syn_lookup, min_minutes=300):
    r = roster[(roster.team == team) & (roster.season == season) & (roster.minutes >= min_minutes)]
    r = r.sort_values("minutes", ascending=False)
    ids = list(r.pid)
    n = len(ids)
    W = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        s = syn_lookup(ids[i], ids[j])
        W[i, j] = W[j, i] = s
    return r.reset_index(drop=True), W


def draw(team_abbr, season, r, W, path, scale=1.0):
    pos = synergy_mds(W)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
    vmax = np.abs(W).max() + 1e-9
    for i, j in itertools.combinations(range(len(r)), 2):
        w = W[i, j]
        if abs(w) < 0.15 * vmax:
            continue
        ax.plot(*zip(pos[i], pos[j]), lw=0.4 + 3.4 * abs(w) / vmax,
                color=("#1b7f4f" if w > 0 else "#c0392b"), alpha=0.35 + 0.4 * abs(w) / vmax, zorder=1)
    s = 110 + 850 * (r.minutes / r.minutes.max())
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=s, c=r.pc1, cmap="coolwarm",
                    vmin=-6, vmax=6, edgecolor="k", linewidth=0.7, zorder=2)
    SUFFIX = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"}

    def surname(nm):
        parts = [x for x in nm.split() if x.lower().strip(".") not in
                 {s.strip(".") for s in SUFFIX}]
        return parts[-1] if parts else nm.split()[-1]

    last = [surname(nm) for nm in r.name]
    dup = {x for x in last if last.count(x) > 1}
    lab = [f"{nm.split()[0][0]}. {l}" if l in dup else l for nm, l in zip(r.name, last)]
    dup2 = {x for x in lab if lab.count(x) > 1}
    lab = [f"{nm.split()[0]} {l}" if x in dup2 else x
           for nm, l, x in zip(r.name, last, lab)]
    for k, l in enumerate(lab):
        ax.annotate(l, pos[k], textcoords="offset points",
                    xytext=(0, -(np.sqrt(s.iloc[k]) / 2 + 7)), fontsize=7.2,
                    ha="center", va="top", zorder=3)
    pad = 0.18 * (pos.max() - pos.min())
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad * 1.2, pos[:, 1].max() + pad * 0.6)
    ax.set_title(f"{team_abbr}  {season}   (green = complementary, red = redundant/clashing)", fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="PC1  spacing <-> rim and glass")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
