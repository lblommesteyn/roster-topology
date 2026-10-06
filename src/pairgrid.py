"""Pair-fit matrices: the thing a blob cannot show.

The occupancy field that makes a blob smooth throws pair identity away, which is why blob shape
correlates only |r| <= 0.27 with complementarity. A sorted matrix keeps it: one cell per pair,
ordered by hierarchical clustering on the fit values so that groups of mutually-compatible players
appear as blocks.

Cells carry the MODEL fit (the interaction surface), not the measured pair coefficient. That is
deliberate: measured per-pair synergy has a split-half reliability of 0.06 across 30 seasons, so
printing it in a grid would be printing noise in a form that looks authoritative. Where a pair has
a lot of shared history the measured value is shown as a small annotation for comparison, which is
usually a lesson in how far apart the two are.
"""
import os, sys, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

OUT, FIGS = os.path.join(V.ROOT, "out"), os.path.join(V.VIZ, "pairgrid")
# green = fits, red = redundant: the same convention as the topology graphs
CMAP = "RdYlGn"


def order_by_cluster(W):
    """Hierarchical leaf order, so mutually-compatible groups sit together."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    n = len(W)
    if n < 3:
        return list(range(n))
    D = W.max() - W
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    try:
        Z = linkage(squareform(D, checks=False), method="average")
        return list(leaves_list(Z))
    except Exception:
        return list(range(n))


def measured_lookup(ctx):
    d = {}
    for r in ctx.pairs.itertuples():
        d[(r.p1, r.p2)] = (r.net, r.poss)
    return d


def draw(ctx, team, season, path=None, ax=None, meas=None, min_poss=3000, annotate=True):
    r = ctx.roster(team, season)
    if len(r) < 5:
        return None
    W = ctx.syn_matrix(r)
    order = order_by_cluster(W)
    r2 = r.iloc[order].reset_index(drop=True)
    W2 = W[np.ix_(order, order)]
    n = len(r2)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(0.62 * n + 4.2, 0.62 * n + 3.4), dpi=130)
    lim = ctx.edge_scale
    M = W2.copy().astype(float)
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    labels = [V.surname(x) for x in r2.name]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="#999")
                    continue
                v = W2[i, j]
                txt = f"{v:+.1f}"
                col = "#111" if abs(v) < 0.75 * lim else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.6, color=col)
                if meas is not None and i < j:
                    a, b = r2.pid.iloc[i], r2.pid.iloc[j]
                    k = (a, b) if a < b else (b, a)
                    if k in meas and meas[k][1] >= min_poss:
                        ax.text(j, i - 0.32, f"obs {meas[k][0]:+.1f}", ha="center", va="center",
                                fontsize=5.2, color="#555")
    st = ctx.structure(r)
    ax.set_title(f"{team} {season}   how every pair fits\n"
                 f"net {ctx.net_rating(team, season):+.1f}  |  "
                 f"avg complementarity {st['complementarity']:+.3f}  |  "
                 f"connector {V.surname(st['connector'])}", fontsize=10.5)
    if own:
        cb = plt.colorbar(im, ax=ax, shrink=0.72)
        cb.set_label("predicted fit, pts/100 (green = complementary, red = redundant)", fontsize=9)
        fig.tight_layout()
        if path:
            fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        return path
    return im


def gallery(ctx, season, path, ncol=6):
    teams = [t for t in ctx.teams(season) if len(ctx.roster(t, season)) >= 6]
    nrow = int(np.ceil(len(teams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.4 * nrow), dpi=110)
    axes = np.atleast_2d(axes)
    im = None
    for k, t in enumerate(teams):
        ax = axes[k // ncol, k % ncol]
        out = draw(ctx, t, season, ax=ax, annotate=False)
        if out is not None:
            im = out
        ax.set_xticklabels([]); ax.set_yticklabels([])
        st = ctx.structure(ctx.roster(t, season))
        ax.set_title(f"{t}   comp {st['complementarity']:+.3f}", fontsize=9)
    for k in range(len(teams), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"Pair fit inside every roster, {season}   "
                 f"(players ordered by clustering; green blocks are mutually compatible groups)",
                 fontsize=13, y=1.0)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.35, pad=0.01)
        cb.set_label("predicted fit, pts/100")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def model_vs_measured(ctx, season, path, min_poss=2000):
    """The lesson that belongs next to any pair grid: what the model says vs what was observed."""
    meas = measured_lookup(ctx)
    xs, ys, labs = [], [], []
    for t in ctx.teams(season):
        r = ctx.roster(t, season)
        for i, j in itertools.combinations(range(len(r)), 2):
            a, b = r.pid.iloc[i], r.pid.iloc[j]
            k = (a, b) if a < b else (b, a)
            if k in meas and meas[k][1] >= min_poss:
                xs.append(ctx.syn(a, b)); ys.append(meas[k][0])
                labs.append(f"{V.surname(r.name.iloc[i])}/{V.surname(r.name.iloc[j])}")
    if len(xs) < 20:
        return None
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=130)
    ax.scatter(xs, ys, s=26, alpha=0.6, color="#2b6cb0", edgecolor="none")
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
    lim = max(np.abs(xs).max(), 1e-6) * 1.15
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#bbb", lw=1)
    r_ = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel("model fit (interaction surface), pts/100")
    ax.set_ylabel("observed pair effect, pts/100")
    ax.set_title(f"Model fit vs observed pair effect, {season}\n"
                 f"n={len(xs)} pairs with {min_poss}+ shared possessions, r = {r_:.2f}; "
                 f"the observed axis is ~15x wider and does not replicate", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path, r_, len(xs)


def main(season=None):
    ctx = V.Ctx()
    os.makedirs(FIGS, exist_ok=True)
    season = season or ctx.seasons[-1]
    meas = measured_lookup(ctx)
    for t in ctx.teams(season):
        draw(ctx, t, season, path=os.path.join(FIGS, f"pairgrid_{season}_{t}.png"), meas=meas)
    print("wrote per-team pair grids to viz/pairgrid/")
    gallery(ctx, season, os.path.join(V.VIZ, f"pairgrid_gallery_{season}.png"))
    print(f"wrote viz/pairgrid_gallery_{season}.png")
    out = model_vs_measured(ctx, season, os.path.join(V.VIZ, f"pairgrid_model_vs_observed_{season}.png"))
    if out:
        print(f"model vs observed: r={out[1]:.2f} over {out[2]} well-sampled pairs")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
