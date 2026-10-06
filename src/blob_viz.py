"""Render roster blobs: interactive 3D organisms on a dark stage.

The blob surface is the primary object. Player markers are drawn faintly inside it as nuclei, and
a second, tighter isosurface is drawn as an inner core so that minutes concentration reads as a
glow rather than as a separate chart.
"""
import os, sys
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, blob as B

BG = "#0a0c10"
FG = "#e8e8e6"
GRID_LINE = "rgba(255,255,255,0.055)"

# offence (warm) to defence (cool); reads on a dark stage
LEAN_SCALE = [[0.0, "#2b6cb0"], [0.28, "#4aa3c7"], [0.5, "#dfe6ea"],
              [0.72, "#e8964a"], [1.0, "#c8492e"]]
DENSE_SCALE = [[0.0, "#1d2b3a"], [0.45, "#2f6f8f"], [0.75, "#5fc2c9"], [1.0, "#eafff6"]]

LIGHT = dict(ambient=0.42, diffuse=0.92, specular=0.32, roughness=0.55, fresnel=0.22)
LIGHTPOS = dict(x=120, y=180, z=140)

# ranges are the league 5th-95th percentile of each scalar, so the gradient uses its full span
COLOR_FIELDS = {
    "leaning": ("defence-leaning  <->  offence-leaning", LEAN_SCALE, (-3.5, 6.0)),
    "pc1": ("perimeter spacing  <->  rim and offensive glass", LEAN_SCALE, (-4.0, 7.5)),
    "z_fg3a_pct": ("rim pressure  <->  3pt volume", LEAN_SCALE, (-1.8, 1.6)),
    "value": ("estimated value", LEAN_SCALE, (-3.8, 5.2)),
}


def scene(lo, hi, title=None, show_axes=True):
    ax = lambda t, r: dict(title=dict(text=t, font=dict(color="rgba(232,232,230,0.55)", size=10)),
                           range=r, backgroundcolor=BG, gridcolor=GRID_LINE, zerolinecolor=GRID_LINE,
                           showbackground=True, color="rgba(232,232,230,0.35)",
                           showticklabels=False, showgrid=show_axes, zeroline=False)
    return dict(xaxis=ax("spacing -> rim and glass", [lo[0], hi[0]]),
                yaxis=ax("off-ball -> on-ball", [lo[1], hi[1]]),
                zaxis=ax("disruption -> efficient scoring", [lo[2], hi[2]]),
                aspectmode="cube", bgcolor=BG,
                camera=dict(eye=dict(x=1.18, y=1.18, z=0.68)))


def mesh_trace(s, color_by="leaning", opacity=0.72, name="roster", showscale=True,
               intensity=None, cmin=None, cmax=None, scale=None, colorbar_x=1.02):
    lab, dflt_scale, (lo, hi) = COLOR_FIELDS.get(color_by, COLOR_FIELDS["leaning"])
    inten = s["color"] if intensity is None else intensity
    return go.Mesh3d(
        x=s["verts"][:, 0], y=s["verts"][:, 1], z=s["verts"][:, 2],
        i=s["faces"][:, 0], j=s["faces"][:, 1], k=s["faces"][:, 2],
        intensity=inten, colorscale=scale or dflt_scale,
        cmin=lo if cmin is None else cmin, cmax=hi if cmax is None else cmax,
        opacity=opacity, name=name, showscale=showscale, hoverinfo="skip",
        lighting=LIGHT, lightposition=LIGHTPOS, flatshading=False,
        colorbar=dict(title=dict(text=lab, side="right", font=dict(color=FG, size=11)),
                      tickfont=dict(color=FG, size=9), thickness=12, len=0.55, x=colorbar_x,
                      outlinewidth=0))


def core_trace(ctx, roster, s, lo, hi, n=B.GRID, q=0.82, opacity=0.5):
    """Inner isosurface: where the roster's minutes are actually concentrated."""
    axes, pts = B.make_grid(lo, hi, n)
    inner = B.surface(s["field"], axes, iso=q * float(s["field"].max()))
    if inner is None:
        return None
    dens = B.sample_at(s["field"], axes, inner["verts"])
    return go.Mesh3d(
        x=inner["verts"][:, 0], y=inner["verts"][:, 1], z=inner["verts"][:, 2],
        i=inner["faces"][:, 0], j=inner["faces"][:, 1], k=inner["faces"][:, 2],
        intensity=dens, colorscale=DENSE_SCALE, opacity=opacity, showscale=False,
        name="core", hoverinfo="skip", lighting=dict(ambient=0.75, diffuse=0.5, specular=0.1),
        lightposition=LIGHTPOS, flatshading=False)


_SHELL_CACHE = {}


def league_shell(ctx, season, lo, hi, n=B.GRID, opacity=0.055):
    """A faint envelope of where the whole league lives, so a team's size and position read.

    Cached: the envelope is a 400-player field over the whole grid, and recomputing it once per
    team turns a 30-team run into a very slow one.
    """
    key = (season, n, float(lo[0]), float(hi[0]))
    s = _SHELL_CACHE.get(key)
    if s is None:
        d = ctx.players[(ctx.players.season == season) & (ctx.players.minutes >= ctx.min_minutes)]
        axes, pts = B.make_grid(lo, hi, n)
        f = B.occupancy(d, pts, base_h=1.5)
        s = B.surface(f, axes, iso_q=0.16)
        _SHELL_CACHE[key] = s
    if s is None:
        return None
    return go.Mesh3d(x=s["verts"][:, 0], y=s["verts"][:, 1], z=s["verts"][:, 2],
                     i=s["faces"][:, 0], j=s["faces"][:, 1], k=s["faces"][:, 2],
                     color="#8fa3b8", opacity=opacity, showscale=False, hoverinfo="skip",
                     name="league", lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0),
                     flatshading=False)


def nuclei_trace(roster, size_scale=1.0, labels=True, label_top=9):
    """Player markers inside the blob: nuclei, deliberately understated."""
    sz = 3.5 + 11 * np.sqrt(roster.minutes / max(roster.minutes.max(), 1)) * size_scale
    txt = None
    if labels:
        keep = set(roster.nlargest(label_top, "minutes").index)
        short = V.short_labels(list(roster.name))
        txt = [t if i in keep else "" for i, t in zip(roster.index, short)]
    hover = [f"<b>{r['name']}</b><br>{int(r['minutes'])} min"
             f"<br>value {r['value']:+.2f} pts/100"
             f"<br>rim/glass {r['pc1']:+.1f} | on-ball load {r['pc2']:+.1f}"
             f" | scoring efficiency {r['pc3']:+.1f}"
             for _, r in roster.iterrows()]
    return go.Scatter3d(
        x=roster.pc1, y=roster.pc2, z=roster.pc3,
        mode="markers+text" if labels else "markers",
        text=txt, textposition="top center",
        textfont=dict(size=9, color="rgba(232,232,230,0.62)"),
        hovertext=hover, hoverinfo="text",
        marker=dict(size=sz, color="rgba(255,255,255,0.55)",
                    line=dict(color="rgba(0,0,0,0.45)", width=0.5)),
        name="players", showlegend=False)


def subtitle(team, season, roster, ctx, m):
    st = ctx.structure(roster)
    return (f"<span style='color:#9aa3ad;font-size:12px'>"
            f"net {ctx.net_rating(team, season):+.1f} &nbsp;|&nbsp; "
            f"volume {m['volume']:.0f} &nbsp;|&nbsp; compactness {m['sphericity']:.2f}"
            f" &nbsp;|&nbsp; lobes {m['components']} &nbsp;|&nbsp; "
            f"peak density {m['peak_density']:.2f} &nbsp;|&nbsp; "
            f"connector {st['connector']}</span>")


def team_figure(ctx, team, season, color_by="leaning", lo=None, hi=None, n=B.GRID,
                labels=True, core=True):
    roster = ctx.roster(team, season)
    if len(roster) < 4:
        return None
    if lo is None:
        lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    s = B.team_blob(ctx, roster, lo, hi, n=n, color_by=color_by)
    if s is None:
        return None
    fig = go.Figure()
    shell = league_shell(ctx, season, lo, hi, n=n)
    if shell is not None:
        fig.add_trace(shell)
    fig.add_trace(mesh_trace(s, color_by))
    if core:
        c = core_trace(ctx, roster, s, lo, hi, n=n)
        if c is not None:
            fig.add_trace(c)
    fig.add_trace(nuclei_trace(roster, labels=labels))
    m = s["metrics"]
    fig.update_layout(
        title=dict(text=f"<b style='color:{FG}'>{team} {season}</b><br>"
                        + subtitle(team, season, roster, ctx, m),
                   x=0.02, y=0.96, font=dict(color=FG, size=20)),
        scene=scene(lo, hi), paper_bgcolor=BG, plot_bgcolor=BG,
        height=820, margin=dict(l=0, r=0, t=95, b=10),
        font=dict(color=FG),
        annotations=[dict(text="the surface is the roster's occupancy of skill space: lobes are role"
                               " concentrations, necks are connectors, a fat centre is redundancy",
                          showarrow=False, xref="paper", yref="paper", x=0.02, y=0.035,
                          font=dict(size=11, color="rgba(232,232,230,0.5)"))])
    return fig, s


def write(fig, path):
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path
