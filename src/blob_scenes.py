"""The blob deliverables: single-team viewers, league gallery, season evolution, trade morphs.

The trade morph is the one that matters most. Rather than cross-fading two pictures, it
interpolates the *field*:

    f_t(x) = f_keep(x) + (1 - t) * f_out(x) + t * f_in(x)

and re-runs marching cubes at every step, so the surface genuinely deforms: the outgoing player's
lobe deflates, the incoming player's lobe inflates, and the topology can change mid-morph as
necks form or break. Vertices are coloured by the signed change in occupancy against the starting
field, so the regions being lost and gained are visible on the surface itself.
"""
import os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, blob as B, blob_viz as BV

OUTDIR = os.path.join(V.VIZ, "blobs")
# the neutral mid-tone has to stay visible against the dark stage, otherwise the parts of the
# roster that are NOT changing disappear and the morph looks like a lobe floating in space
GAIN_SCALE = [[0.0, "#e05a43"], [0.38, "#9a6a63"], [0.5, "#8b95a1"],
              [0.62, "#5aab8c"], [1.0, "#2ee08f"]]


def _dirs():
    os.makedirs(OUTDIR, exist_ok=True)


# --------------------------------------------------------------------- viewers
def single_team(ctx, team, season, path=None, color_by="leaning"):
    res = BV.team_figure(ctx, team, season, color_by=color_by)
    if res is None:
        return None
    fig, s = res
    path = path or os.path.join(OUTDIR, f"blob_{season}_{team}.html")
    BV.write(fig, path)
    return path, s["metrics"]


def league_gallery(ctx, season, path_html=None, path_png=None, ncol=6, n=40):
    """Thirty organisms side by side, all in the same box and on the same scales."""
    teams = [t for t in ctx.teams(season) if len(ctx.roster(t, season)) >= 5]
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    nrow = int(np.ceil(len(teams) / ncol))
    specs = [[{"type": "scene"} for _ in range(ncol)] for _ in range(nrow)]
    rows = []
    for t in teams:
        r = ctx.roster(t, season)
        s = B.team_blob(ctx, r, lo, hi, n=n)
        rows.append((t, r, s))
    rows.sort(key=lambda x: -x[2]["metrics"]["peak_density"] if x[2] else 0)
    titles = [f"{t}  net {ctx.net_rating(t, season):+.1f}  dens {s['metrics']['peak_density']:.1f}"
              for t, r, s in rows]
    fig = make_subplots(rows=nrow, cols=ncol, specs=specs, subplot_titles=titles,
                        horizontal_spacing=0.004, vertical_spacing=0.028)
    for k, (t, r, s) in enumerate(rows):
        rr, cc = k // ncol + 1, k % ncol + 1
        fig.add_trace(BV.mesh_trace(s, "leaning", opacity=0.9, showscale=(k == 0),
                                    colorbar_x=1.005), row=rr, col=cc)
        sid = "scene" if k == 0 else f"scene{k+1}"
        fig.layout[sid].update(BV.scene(lo, hi, show_axes=False))
        fig.layout[sid].update(dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.28, y=1.28, z=0.72))))
    for a in fig.layout.annotations:
        a.font = dict(size=10, color="rgba(232,232,230,0.75)")
    fig.update_layout(
        title=dict(text=f"<b>Thirty organisms, {season}</b>  "
                        f"<span style='font-size:12px;color:#9aa3ad'>ordered by peak density "
                        f"(how concentrated the roster is in one region of skill space)</span>",
                   x=0.01, y=0.995, font=dict(color=BV.FG, size=20)),
        paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
        height=300 * nrow + 90, margin=dict(l=0, r=0, t=95, b=0), showlegend=False)
    if path_html:
        fig.write_html(path_html, include_plotlyjs="cdn")
    if path_png:
        fig.write_image(path_png, width=1900, height=300 * nrow + 90, scale=1)
    return fig, rows


# ----------------------------------------------------------------- evolution
def _frame_traces(ctx, roster, lo, hi, n, color_by, base_field=None):
    s = B.team_blob(ctx, roster, lo, hi, n=n, color_by=color_by)
    if s is None:
        return None, None
    mesh = BV.mesh_trace(s, color_by, opacity=0.78)
    nuc = BV.nuclei_trace(roster)
    return [mesh, nuc], s


def season_evolution(ctx, team, path, color_by="leaning", n=48, events=None):
    """One frame per season: the organism grows, splits and re-forms as the roster turns over."""
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    seasons = [s for s in ctx.seasons if len(ctx.roster(team, s)) >= 5]
    frames, first, rowsm = [], None, []
    for s in seasons:
        r = ctx.roster(team, s)
        tr, blob = _frame_traces(ctx, r, lo, hi, n, color_by)
        if tr is None:
            continue
        m = blob["metrics"]
        rowsm.append(dict(season=s, **{k: m[k] for k in
                                       ("volume", "sphericity", "components", "peak_density")}))
        note = (events or {}).get(s, "")
        frames.append(go.Frame(data=tr, name=s, layout=go.Layout(title=dict(
            text=f"<b style='color:{BV.FG}'>{team} {s}</b><br>"
                 + BV.subtitle(team, s, r, ctx, m)
                 + (f"<br><span style='color:#7fd4b0;font-size:12px'>{note}</span>" if note else "")))))
        if first is None:
            first = tr
    if not frames:
        return None
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title, scene=BV.scene(lo, hi),
        paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
        height=840, margin=dict(l=0, r=0, t=110, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
                          bgcolor="rgba(255,255,255,0.08)", font=dict(color=BV.FG),
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=1200, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.13, len=0.82, y=0.06,
                      currentvalue=dict(prefix="season: ", font=dict(color=BV.FG)),
                      font=dict(color=BV.FG), bgcolor="rgba(255,255,255,0.15)",
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate",
                                                       frame=dict(duration=500, redraw=True))])
                             for f in frames])])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path, pd.DataFrame(rowsm)


def window_evolution(ctx, team, windows, path, color_by="leaning", n=48):
    """One frame per within-season window. `windows` is a list of (label, roster_df, note)."""
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    frames, rowsm = [], []
    for label, roster, note in windows:
        if roster is None or len(roster) < 5:
            continue
        tr, blob = _frame_traces(ctx, roster, lo, hi, n, color_by)
        if tr is None:
            continue
        m = blob["metrics"]
        rowsm.append(dict(window=label, **{k: m[k] for k in
                                           ("volume", "sphericity", "components", "peak_density")}))
        frames.append(go.Frame(data=tr, name=label, layout=go.Layout(title=dict(
            text=f"<b style='color:{BV.FG}'>{team}</b>  "
                 f"<span style='color:#9aa3ad;font-size:14px'>{label}</span><br>"
                 f"<span style='color:#9aa3ad;font-size:12px'>volume {m['volume']:.0f} | "
                 f"compactness {m['sphericity']:.2f} | lobes {m['components']} | "
                 f"peak density {m['peak_density']:.2f}</span>"
                 + (f"<br><span style='color:#7fd4b0;font-size:12px'>{note}</span>" if note else "")))))
    if not frames:
        return None
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title, scene=BV.scene(lo, hi),
        paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
        height=840, margin=dict(l=0, r=0, t=115, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
                          bgcolor="rgba(255,255,255,0.08)", font=dict(color=BV.FG),
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.13, len=0.82, y=0.06,
                      currentvalue=dict(prefix="", font=dict(color=BV.FG)),
                      font=dict(color=BV.FG), bgcolor="rgba(255,255,255,0.15)",
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate",
                                                       frame=dict(duration=400, redraw=True))])
                             for f in frames])])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path, pd.DataFrame(rowsm)


# --------------------------------------------------------------------- morph
def trade_morph(ctx, team, season, outgoing, incoming, path, steps=22, n=48,
                inc_season=None, title_note=""):
    """Deform one roster into another by interpolating the occupancy field itself."""
    roster = ctx.roster(team, season)
    if outgoing not in set(roster.name):
        raise ValueError(f"{outgoing} not on {team} {season}: {', '.join(roster.name)}")
    cand = ctx.players[ctx.players.name.str.lower() == incoming.lower()]
    if inc_season:
        cand = cand[cand.season == inc_season]
    if not len(cand):
        raise ValueError(f"unknown incoming player {incoming}")
    inc = cand.sort_values("season").iloc[-1].copy()
    idx = roster.index[roster.name == outgoing][0]
    inc["minutes"] = roster.loc[idx, "minutes"]

    keep = roster.drop(index=idx)
    out_row = roster.loc[[idx]]
    in_row = pd.DataFrame([inc])
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    axes, pts = B.make_grid(lo, hi, n)

    # every piece is normalised against the ORIGINAL roster's totals, so f_keep + f_out is
    # exactly the original field and the interpolation hands mass from one player to the other
    total, nref = float(roster.minutes.sum()), len(roster)
    ref_h = float(np.mean(B.bandwidths(roster)))

    def part(df):
        if not len(df):
            return np.zeros((n, n, n))
        w = B.norm_weights(df, ref_total=total, ref_n=nref)
        h = B.bandwidths(roster)[:len(df)] * 0 + ref_h
        return B.occupancy(df, pts, weights=w, ref_h=h)

    f_keep, f_out, f_in = part(keep), part(out_row), part(in_row)
    col_keep = B.scalar_field(roster, pts, column="leaning")
    col_new = B.scalar_field(pd.concat([keep, in_row], ignore_index=True), pts, column="leaning")

    f0 = f_keep + f_out
    f1 = f_keep + f_in
    level = B.ISO_Q * float(max(f0.max(), f1.max()))
    dmax = max(float(np.abs(f1 - f0).max()) * 0.55, 1e-3)
    frames = []
    for si in range(steps + 1):
        t = si / steps
        f = f_keep + (1 - t) * f_out + t * f_in
        s = B.surface(f, axes, iso=level)
        if s is None:
            continue
        delta = B.sample_at(f - f0, axes, s["verts"])
        mesh = BV.mesh_trace(s, "leaning", opacity=0.8, intensity=delta,
                             cmin=-dmax, cmax=dmax, scale=GAIN_SCALE, showscale=(si == 0))
        mesh.colorbar = dict(title=dict(text="lost  <->  gained", side="right",
                                        font=dict(color=BV.FG, size=11)),
                             tickfont=dict(color=BV.FG, size=9), thickness=12, len=0.5,
                             x=1.02, outlinewidth=0)
        cur = pd.concat([keep, out_row if t < 0.5 else in_row], ignore_index=True)
        nuc = BV.nuclei_trace(cur)
        nuc.marker.size = nuc.marker.size * (1.0 if t < 0.5 else 1.0)
        m = B.blob_metrics(cur, f, axes, level)
        phase = ("original roster" if si == 0 else
                 f"{outgoing} out" if t < 0.5 else
                 f"{incoming} in" if t < 1.0 else "new roster")
        frames.append(go.Frame(data=[mesh, nuc], name=f"{t:.2f}", layout=go.Layout(title=dict(
            text=f"<b style='color:{BV.FG}'>{team} {season}</b>  "
                 f"<span style='color:#9aa3ad;font-size:14px'>{outgoing} &#8594; {incoming}</span>"
                 f"<br><span style='color:#9aa3ad;font-size:12px'>{phase} &nbsp;|&nbsp; "
                 f"volume {m['volume']:.0f} &nbsp;|&nbsp; compactness {m['sphericity']:.2f}"
                 f" &nbsp;|&nbsp; lobes {m['components']}</span>"
                 + (f"<br><span style='color:#7fd4b0;font-size:12px'>{title_note}</span>"
                    if title_note else "")))))
    if not frames:
        return None
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title, scene=BV.scene(lo, hi),
        paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
        height=840, margin=dict(l=0, r=0, t=115, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
                          bgcolor="rgba(255,255,255,0.08)", font=dict(color=BV.FG),
                          buttons=[dict(label="morph", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.16, len=0.79, y=0.06,
                      currentvalue=dict(prefix="morph ", font=dict(color=BV.FG)),
                      font=dict(color=BV.FG), bgcolor="rgba(255,255,255,0.15)",
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                             for f in frames])],
        annotations=[dict(text="green = volume gained by the incoming player, "
                               "red = volume vacated by the outgoing player",
                          showarrow=False, xref="paper", yref="paper", x=0.02, y=0.015,
                          font=dict(size=11, color="rgba(232,232,230,0.55)"))])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path


def roster_morph(ctx, team, s0, s1, path, steps=22, n=48, note=""):
    """Morph between two real rosters (a whole offseason of change), same field interpolation."""
    r0, r1 = ctx.roster(team, s0), ctx.roster(team, s1)
    if len(r0) < 5 or len(r1) < 5:
        return None
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    axes, pts = B.make_grid(lo, hi, n)
    f0 = B.occupancy(r0, pts); f1 = B.occupancy(r1, pts)
    c0 = B.scalar_field(r0, pts, "leaning"); c1 = B.scalar_field(r1, pts, "leaning")
    level = B.ISO_Q * float(max(f0.max(), f1.max()))
    stay = set(r0.pid) & set(r1.pid)
    frames = []
    for si in range(steps + 1):
        t = si / steps
        f = (1 - t) * f0 + t * f1
        s = B.surface(f, axes, iso=level)
        if s is None:
            continue
        delta = B.sample_at(f1 - f0, axes, s["verts"])
        mesh = BV.mesh_trace(s, "leaning", opacity=0.8, intensity=delta,
                             cmin=-0.6, cmax=0.6, scale=GAIN_SCALE, showscale=(si == 0))
        mesh.colorbar = dict(title=dict(text="lost  <->  gained", side="right",
                                        font=dict(color=BV.FG, size=11)),
                             tickfont=dict(color=BV.FG, size=9), thickness=12, len=0.5,
                             x=1.02, outlinewidth=0)
        cur = r0 if t < 0.5 else r1
        m = B.blob_metrics(cur, f, axes, level)
        frames.append(go.Frame(data=[mesh, BV.nuclei_trace(cur)], name=f"{t:.2f}",
                               layout=go.Layout(title=dict(
                                   text=f"<b style='color:{BV.FG}'>{team}</b>  "
                                        f"<span style='color:#9aa3ad;font-size:14px'>{s0} "
                                        f"&#8594; {s1}</span><br>"
                                        f"<span style='color:#9aa3ad;font-size:12px'>"
                                        f"{'before' if t < 0.5 else 'after'} &nbsp;|&nbsp; "
                                        f"volume {m['volume']:.0f} &nbsp;|&nbsp; "
                                        f"compactness {m['sphericity']:.2f} &nbsp;|&nbsp; "
                                        f"lobes {m['components']} &nbsp;|&nbsp; "
                                        f"{len(stay)} players stay</span>"
                                        + (f"<br><span style='color:#7fd4b0;font-size:12px'>"
                                           f"{note}</span>" if note else "")))))
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title, scene=BV.scene(lo, hi),
        paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
        height=840, margin=dict(l=0, r=0, t=115, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
                          bgcolor="rgba(255,255,255,0.08)", font=dict(color=BV.FG),
                          buttons=[dict(label="morph", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.16, len=0.79, y=0.06,
                      currentvalue=dict(prefix="morph ", font=dict(color=BV.FG)),
                      font=dict(color=BV.FG), bgcolor="rgba(255,255,255,0.15)",
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                             for f in frames])])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path
