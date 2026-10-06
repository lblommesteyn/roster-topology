"""Static team topology images and the 30-team league gallery."""
import os, sys, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, topology as TP

COLOR_FIELDS = {
    "pc1": ("PC1   perimeter spacing <-> rim and offensive glass", "coolwarm", (-6, 6)),
    "pc2": ("PC2   off-ball <-> on-ball load", "PuOr_r", (-6, 6)),
    "value": ("estimated value (pts/100)", "RdYlGn", (-6, 6)),
    "leaning": ("offence <-> defence leaning", "BrBG", (-4, 4)),
}


def draw_team(ctx, team, season, ax=None, color="pc1", label=True, edge_frac=0.16,
              title=None, pos=None, legend=True, vmax=None):
    r = ctx.roster(team, season)
    if len(r) < 4:
        return None
    W = ctx.syn_matrix(r)
    if pos is None:
        pos = TP.synergy_mds(W)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=130)
    vmax = vmax or ctx.edge_scale          # league-wide scale, never per team
    for i, j, w in V.edge_list(r, W, edge_frac):
        f = min(abs(w) / vmax, 1.4)
        ax.plot(*zip(pos[i], pos[j]), lw=0.4 + 3.6 * f,
                color=(V.POS if w > 0 else V.NEG), alpha=min(0.9, 0.30 + 0.45 * f), zorder=1)
    lab, cmap, (lo, hi) = COLOR_FIELDS[color]
    s = 90 + 900 * (r.minutes / r.minutes.max())
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=s, c=r[color], cmap=cmap, vmin=lo, vmax=hi,
                    edgecolor="k", linewidth=0.7, zorder=2)
    if label:
        for k, l in enumerate(V.short_labels(list(r.name))):
            ax.annotate(l, pos[k], textcoords="offset points",
                        xytext=(0, -(np.sqrt(s.iloc[k]) / 2 + 7)), fontsize=7.2,
                        ha="center", va="top", zorder=3)
    st = ctx.structure(r)
    net = ctx.net_rating(team, season)
    ax.set_title(title or f"{team}  {season}    net {net:+.1f}   "
                          f"complementarity {st['complementarity']:+.3f}", fontsize=10)
    pad = 0.2 * (pos.max() - pos.min())
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad * 1.25, pos[:, 1].max() + pad * 0.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    if own:
        cb = plt.colorbar(sc, ax=ax, shrink=0.72); cb.set_label(lab, fontsize=9)
        if legend:
            ax.legend(handles=[Line2D([], [], color=V.POS, lw=3, label="fits together"),
                               Line2D([], [], color=V.NEG, lw=3, label="redundant / clashing")],
                      loc="lower right", fontsize=8, frameon=False)
        fig.tight_layout()
        return fig, ax, sc
    return sc


def league_gallery(ctx, season, color="pc1", path=None, sort_by="complementarity"):
    """All 30 rosters on one sheet, ordered so the shape differences are comparable."""
    teams = ctx.teams(season)
    rows = []
    for t in teams:
        r = ctx.roster(t, season)
        if len(r) < 4:
            continue
        st = ctx.structure(r)
        rows.append((t, st[sort_by] if sort_by in st else 0.0))
    rows.sort(key=lambda x: -x[1])
    n = len(rows)
    ncol, nrow = 6, int(np.ceil(n / 6))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow), dpi=105)
    axes = np.atleast_2d(axes)
    sc = None
    for k, (t, val) in enumerate(rows):
        ax = axes[k // ncol, k % ncol]
        net = ctx.net_rating(t, season)
        sc = draw_team(ctx, t, season, ax=ax, color=color, label=True, edge_frac=0.28,
                       title=f"{t}   net {net:+.1f}   comp {val:+.3f}") or sc
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    lab = COLOR_FIELDS[color][0]
    fig.suptitle(f"NBA roster topology, {season}   "
                 f"(ordered by average complementarity; green edges fit, red edges clash)",
                 fontsize=15, y=0.997)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.35, pad=0.01); cb.set_label(lab)
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def main(season=None, color="pc1"):
    ctx = V.Ctx(); V.ensure_dirs()
    season = season or ctx.seasons[-1]
    out = os.path.join(V.VIZ, "teams")
    for t in ctx.teams(season):
        res = draw_team(ctx, t, season, color=color)
        if res is None:
            continue
        fig = res[0]
        fig.savefig(os.path.join(out, f"topology_{season}_{t}.png"))
        plt.close(fig)
    print("wrote per-team images to viz/teams/")
    league_gallery(ctx, season, color=color,
                   path=os.path.join(V.VIZ, f"league_gallery_{season}.png"))
    print(f"wrote viz/league_gallery_{season}.png")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
