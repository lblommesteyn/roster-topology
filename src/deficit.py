"""Coverage-deficit maps: what a roster does NOT have.

Every other view in this project draws the mass a roster occupies. Roster construction is mostly a
question about the holes, so this one draws league density minus team density on the two skill
axes that carry the structure (PC1 spacing-to-rim/glass, PC2 off-ball-to-on-ball load).

Both fields are minutes-weighted and normalised to sum to one, so the difference reads as
"share of minutes this roster is missing, relative to how the league allocates minutes". Positive
(warm) is a hole; negative (cool) is a surplus, which is the same thing as redundancy.

Deliberately 2D: measured over 240 team-seasons, the third dimension adds ~0.02 of correlation
with the redundancy metric (0.452 vs 0.433) while costing occlusion and readability.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

OUT, FIGS = os.path.join(V.ROOT, "out"), os.path.join(V.VIZ, "deficit")
GRID = 120
BW = 1.15                       # kernel width in skill-space units
# warm = missing, cool = surplus. Deliberately NOT the green/red used for pair fit elsewhere.
CMAP = "PuOr_r"


def bounds(players, min_minutes=300, pad=1.5, q=1.0):
    d = players[players.minutes >= min_minutes]
    lo = np.array([np.percentile(d.pc1, q), np.percentile(d.pc2, q)]) - pad
    hi = np.array([np.percentile(d.pc1, 100 - q), np.percentile(d.pc2, 100 - q)]) + pad
    return lo, hi


def grid(lo, hi, n=GRID):
    gx = np.linspace(lo[0], hi[0], n)
    gy = np.linspace(lo[1], hi[1], n)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1)
    return gx, gy, G.reshape(-1, 2)


def density(df, pts, bw=BW, weight="minutes"):
    """Minutes-weighted kernel density, normalised to sum to 1 over the grid."""
    P = df[["pc1", "pc2"]].values
    w = df[weight].values.astype(float)
    w = w / max(w.sum(), 1e-9)
    f = np.zeros(len(pts))
    for k in range(len(df)):
        f += w[k] * np.exp(-((pts - P[k]) ** 2).sum(1) / (2 * bw ** 2))
    s = f.sum()
    return f / s if s > 0 else f


def deficit_field(ctx, team, season, n=GRID, bw=BW):
    lg = ctx.players[(ctx.players.season == season) &
                     (ctx.players.minutes >= ctx.min_minutes)]
    r = ctx.roster(team, season)
    if len(r) < 5:
        return None
    lo, hi = bounds(ctx.players[ctx.players.season == season], ctx.min_minutes)
    gx, gy, pts = grid(lo, hi, n)
    fl = density(lg, pts, bw)
    ft = density(r, pts, bw)
    d = (fl - ft).reshape(n, n)
    return dict(gx=gx, gy=gy, deficit=d, league=fl.reshape(n, n), team=ft.reshape(n, n),
                roster=r, lo=lo, hi=hi, league_df=lg)


def describe_hole(res, k=3, min_sep=2.0):
    """Name the biggest gaps by the league players who live there."""
    d = res["deficit"]
    gx, gy = res["gx"], res["gy"]
    lg = res["league_df"]
    flat = np.argsort(d.ravel())[::-1]
    picks = []
    for idx in flat:
        i, j = np.unravel_index(idx, d.shape)
        p = np.array([gx[i], gy[j]])
        if any(np.linalg.norm(p - q) < min_sep for q, _, _ in picks):
            continue
        dist = np.linalg.norm(lg[["pc1", "pc2"]].values - p, axis=1)
        # name the gap after the most RECOGNISABLE player who lives there (most minutes among the
        # nearest dozen), not the literally closest, who is often a 400-minute end-of-bench name
        near = lg.iloc[np.argsort(dist)[:12]].sort_values("minutes", ascending=False)
        picks.append((p, float(d[i, j]), list(near.name)))
        if len(picks) >= k:
            break
    return picks


def surplus_players(res, k=3):
    """Which of a team's own players sit in its most over-covered region."""
    r = res["roster"]
    gx, gy = res["gx"], res["gy"]
    d = res["deficit"]
    vals = []
    for _, p in r.iterrows():
        i = int(np.argmin(np.abs(gx - p.pc1)))
        j = int(np.argmin(np.abs(gy - p.pc2)))
        vals.append((float(d[i, j]), p["name"], p["minutes"]))
    vals.sort()
    return vals[:k]


def draw(ctx, team, season, path=None, ax=None, label=True, show_holes=True):
    res = deficit_field(ctx, team, season)
    if res is None:
        return None
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=130)
    d = res["deficit"] * 100
    lim = float(np.abs(d).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    im = ax.pcolormesh(res["gx"], res["gy"], d.T, cmap=CMAP, norm=norm, shading="gouraud")
    ax.contour(res["gx"], res["gy"], d.T, levels=[0], colors="#555", linewidths=0.6, alpha=0.6)
    r = res["roster"]
    ax.scatter(r.pc1, r.pc2, s=28 + 210 * (r.minutes / r.minutes.max()),
               facecolor="white", edgecolor="#111", linewidth=1.1, zorder=3)
    if label:
        for _, p in r.iterrows():
            ax.annotate(V.surname(p["name"]), (p.pc1, p.pc2), fontsize=7.4, zorder=4,
                        xytext=(0, 8), textcoords="offset points", ha="center", color="#111")
    if show_holes:
        for p, val, names in describe_hole(res, k=2):
            ax.annotate(f"gap: like {V.surname(names[0])}",
                        p, fontsize=7.6, color="#7a3b00", ha="center", zorder=5,
                        xytext=(0, -13), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#c07a30", lw=0.6,
                                  alpha=0.85))
    ax.set_xlabel("PC1   perimeter spacing  <->  rim and offensive glass", fontsize=9)
    ax.set_ylabel("PC2   off-ball  <->  on-ball load", fontsize=9)
    net = ctx.net_rating(team, season)
    holes = describe_hole(res, k=1)
    sur = surplus_players(res, k=1)
    bits = []
    if holes:
        bits.append(f"biggest gap: a {V.surname(holes[0][2][0])} type")
    if sur:
        bits.append(f"most duplicated: {V.surname(sur[0][1])}")
    ax.set_title(f"{team} {season}   net {net:+.1f}\n" + "   |   ".join(bits), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if own:
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("minutes share missing (warm)  <->  surplus (cool), x100", fontsize=9)
        fig.tight_layout()
        if path:
            fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        return path, res
    return im, res


def gallery(ctx, season, path, ncol=6):
    teams = [t for t in ctx.teams(season) if len(ctx.roster(t, season)) >= 6]
    nrow = int(np.ceil(len(teams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow), dpi=110)
    axes = np.atleast_2d(axes)
    im = None
    for k, t in enumerate(teams):
        ax = axes[k // ncol, k % ncol]
        out = draw(ctx, t, season, ax=ax, label=False, show_holes=False)
        if out:
            im = out[0]
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=10)
    for k in range(len(teams), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"What each roster is missing, {season}   "
                 f"(warm = under-covered role, cool = surplus; "
                 f"x = spacing to rim/glass, y = off-ball to on-ball)", fontsize=13, y=1.0)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.35, pad=0.01)
        cb.set_label("missing  <->  surplus")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def table(ctx, season):
    """Deficit summary per team, plus what the biggest hole looks like."""
    rows = []
    for t in ctx.teams(season):
        res = deficit_field(ctx, t, season)
        if res is None:
            continue
        d = res["deficit"] * 100
        holes = describe_hole(res, k=1)
        sur = surplus_players(res, k=2)
        rows.append(dict(season=season, team=t, net=ctx.net_rating(t, season),
                         max_gap=float(d.max()), max_surplus=float(d.min()),
                         total_abs_mismatch=float(np.abs(d).sum() / d.size),
                         gap_like=", ".join(V.surname(n) for n in holes[0][2][:2]) if holes else "",
                         surplus_players=", ".join(f"{V.surname(n)}" for _, n, _ in sur)))
    return pd.DataFrame(rows)


def main(season=None):
    ctx = V.Ctx()
    os.makedirs(FIGS, exist_ok=True)
    season = season or ctx.seasons[-1]
    for t in ctx.teams(season):
        draw(ctx, t, season, path=os.path.join(FIGS, f"deficit_{season}_{t}.png"))
    print(f"wrote per-team deficit maps to viz/deficit/")
    gallery(ctx, season, os.path.join(V.VIZ, f"deficit_gallery_{season}.png"))
    print(f"wrote viz/deficit_gallery_{season}.png")
    tab = table(ctx, season)
    tab.to_csv(os.path.join(OUT, f"deficit_{season}.csv"), index=False, encoding="utf-8")
    print()
    print(tab.sort_values("max_gap", ascending=False).round(2).to_string(index=False))
    return tab


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
