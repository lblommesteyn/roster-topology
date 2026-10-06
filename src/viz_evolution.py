"""Roster evolution over time: season snapshots, animated transitions, metric timelines.

Season-to-season MDS layouts are only defined up to rotation and reflection, so each season is
Procrustes-aligned to the previous one on the players the two seasons share (see
`viz_common.aligned_team_layouts`). Without that the animation spins wildly even when the roster
is unchanged, which looks like signal and is not.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, viz_plotly as P, viz_static as VS

METRICS = ["complementarity", "redundancy", "eff_roles", "coverage", "fragility"]
NICE = {"complementarity": "complementarity (avg fit)", "redundancy": "redundancy (higher = more duplicated)",
        "eff_roles": "effective roles", "coverage": "coverage of league skill space",
        "fragility": "fragility (dependence on one connector)", "net": "net rating"}


# ------------------------------------------------------------------ snapshots
def snapshots(ctx, team, path, color="pc1"):
    lay = V.aligned_team_layouts(ctx, team)
    seasons = list(lay)
    if len(seasons) < 2:
        return None
    n = len(seasons)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    for k, s in enumerate(seasons):
        r, pos = lay[s]
        VS.draw_team(ctx, team, s, ax=axes[k], color=color, edge_frac=0.2, pos=pos,
                     title=f"{s}   net {ctx.net_rating(team, s):+.1f}   "
                           f"comp {ctx.structure(r)['complementarity']:+.3f}")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"{team}: roster topology by season (layouts aligned across seasons)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def animate_gif(ctx, team, path, color="pc1", hold=3, fps=1.4):
    """GIF of a team's topology morphing season to season."""
    import imageio.v2 as imageio
    lay = V.aligned_team_layouts(ctx, team)
    frames = []
    tmp = os.path.join(V.VIZ, "_tmp_frame.png")
    for s, (r, pos) in lay.items():
        res = VS.draw_team(ctx, team, s, color=color, pos=pos, edge_frac=0.16,
                           title=f"{team}  {s}    net {ctx.net_rating(team, s):+.1f}")
        if res is None:
            continue
        fig = res[0]
        fig.savefig(tmp, bbox_inches="tight")
        plt.close(fig)
        img = imageio.imread(tmp)
        frames.extend([img] * hold)
    if not frames:
        return None
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)
    if os.path.exists(tmp):
        os.remove(tmp)
    return path


# ------------------------------------------------------------------ animated HTML
def animated_team(ctx, team, path):
    """Plotly animation with a season slider; play morphs the roster through time."""
    lay = V.aligned_team_layouts(ctx, team)
    seasons = list(lay)
    if len(seasons) < 2:
        return None
    frames, first_traces = [], None
    for s in seasons:
        r, pos = lay[s]
        W = ctx.syn_matrix(r)
        traces = P.edge_traces(pos, V.edge_list(r, W, 0.12), vmax=ctx.edge_scale)
        field, label, scale, lo, hi = P.COLOR_OPTS[0]
        traces.append(go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers+text",
            text=V.short_labels(list(r.name)), textposition="bottom center",
            textfont=dict(size=10), hovertext=P.hover_text(r), hoverinfo="text",
            marker=dict(size=P.node_sizes(r.minutes), color=r[field], colorscale=scale,
                        cmin=lo, cmax=hi, line=dict(color="#222", width=1),
                        colorbar=dict(title=dict(text=label, side="right"), thickness=14, len=0.7)),
            showlegend=False, name="players"))
        st = ctx.structure(r)
        frames.append(go.Frame(data=traces, name=s,
                               layout=go.Layout(title=dict(
                                   text=f"<b>{team} {s}</b>  net {ctx.net_rating(team, s):+.1f} | "
                                        f"complementarity {st['complementarity']:+.3f} | "
                                        f"connector {st['connector']}"))))
        if first_traces is None:
            first_traces = traces
    # every frame must have the same number of traces, or plotly leaves stale edges on screen
    nmax = max(len(f.data) for f in frames)
    for f in frames:
        pad = [go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False)
               for _ in range(nmax - len(f.data))]
        f.data = list(f.data[:-1]) + pad + [f.data[-1]]
    first = list(frames[0].data)
    fig = go.Figure(data=first, frames=frames)
    allpos = np.vstack([lay[s][1] for s in seasons])
    pad = 0.2 * (allpos.max() - allpos.min())
    fig.update_layout(
        title=frames[0].layout.title,
        xaxis=dict(visible=False, range=[allpos[:, 0].min() - pad, allpos[:, 0].max() + pad]),
        yaxis=dict(visible=False, scaleanchor="x",
                   range=[allpos[:, 1].min() - pad, allpos[:, 1].max() + pad]),
        plot_bgcolor="white", height=760, margin=dict(l=20, r=20, t=110, b=90),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=-0.06, xanchor="left",
                          buttons=[
                              dict(label="play", method="animate",
                                   args=[None, dict(frame=dict(duration=900, redraw=True),
                                                    transition=dict(duration=400),
                                                    fromcurrent=True)]),
                              dict(label="pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.85, y=-0.04, currentvalue=dict(prefix="season: "),
                      steps=[dict(label=s, method="animate",
                                  args=[[s], dict(mode="immediate",
                                                  frame=dict(duration=500, redraw=True),
                                                  transition=dict(duration=300))])
                             for s in seasons])],
        annotations=[dict(text="layouts are Procrustes-aligned between seasons, so movement means "
                               "a change in fit, not an arbitrary rotation",
                          showarrow=False, xref="paper", yref="paper", x=0, y=1.045,
                          font=dict(size=11, color="#555"))])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path


# ------------------------------------------------------------------ timelines
def metric_timelines(ctx, path):
    """One page: each structure metric over time, with a team dropdown against the league band."""
    tab = pd.read_csv(os.path.join(V.OUT, "roster_structure_all.csv"))
    teams = sorted(tab.team.unique())
    seasons = sorted(tab.season.unique())
    fig = go.Figure()
    # league median / quartile band per metric, drawn once
    for m in METRICS + ["net"]:
        g = tab.groupby("season")[m]
        fig.add_trace(go.Scatter(x=seasons, y=g.median().reindex(seasons), mode="lines",
                                 line=dict(color="#999", dash="dot"), name="league median",
                                 visible=(m == METRICS[0]), legendgroup="lg"))
        fig.add_trace(go.Scatter(
            x=list(seasons) + list(seasons)[::-1],
            y=list(g.quantile(.75).reindex(seasons)) + list(g.quantile(.25).reindex(seasons))[::-1],
            fill="toself", fillcolor="rgba(150,150,150,0.16)", line=dict(width=0),
            hoverinfo="skip", name="league middle 50%", visible=(m == METRICS[0])))
    nbase = len(fig.data)
    for m in METRICS + ["net"]:
        sub = tab[tab.team == teams[0]].set_index("season").reindex(seasons)
        fig.add_trace(go.Scatter(x=seasons, y=sub[m], mode="lines+markers",
                                 line=dict(color="#1f77b4", width=3), marker=dict(size=9),
                                 name=teams[0], visible=(m == METRICS[0]),
                                 hovertext=[f"{s}<br>{m}: {v:.3f}<br>connector: {c}"
                                            for s, v, c in zip(seasons, sub[m], sub.connector)],
                                 hoverinfo="text"))
    mlist = METRICS + ["net"]
    metric_buttons = []
    for k, m in enumerate(mlist):
        vis = [False] * len(fig.data)
        vis[2 * k] = vis[2 * k + 1] = True
        vis[nbase + k] = True
        metric_buttons.append(dict(label=NICE[m], method="update",
                                   args=[{"visible": vis}, {"yaxis.title.text": NICE[m]}]))
    team_buttons = []
    for t in teams:
        sub = tab[tab.team == t].set_index("season").reindex(seasons)
        team_buttons.append(dict(label=t, method="restyle",
                                 args=[{"y": [sub[m] for m in mlist],
                                        "name": [t] * len(mlist),
                                        "hovertext": [[f"{s}<br>{m}: {v:.3f}<br>connector: {c}"
                                                       for s, v, c in zip(seasons, sub[m], sub.connector)]
                                                      for m in mlist]},
                                       list(range(nbase, nbase + len(mlist)))]))
    fig.update_layout(
        title=dict(text="<b>Roster structure over time</b>", x=0.02, y=0.97, yanchor="top"),
        updatemenus=[dict(buttons=metric_buttons, x=0.0, y=1.12, xanchor="left", yanchor="top"),
                     dict(buttons=team_buttons, x=0.42, y=1.12, xanchor="left", yanchor="top")],
        height=620, margin=dict(l=60, r=30, t=150, b=50), plot_bgcolor="white",
        yaxis=dict(title=NICE[METRICS[0]], gridcolor="#eee"), xaxis=dict(gridcolor="#eee"),
        annotations=[dict(text="metric (left menu) and team (right menu) against the league band",
                          showarrow=False, xref="paper", yref="paper", x=0, y=1.035,
                          font=dict(size=11, color="#555"))])
    fig.write_html(path, include_plotlyjs="cdn")
    return path


GIF_TEAMS = ["OKC", "GSW", "BOS", "MIN", "DAL", "CLE"]


def main(teams=None):
    ctx = V.Ctx(); V.ensure_dirs()
    ev = os.path.join(V.VIZ, "evolution")
    metric_timelines(ctx, os.path.join(V.VIZ, "structure_timelines.html"))
    print("wrote viz/structure_timelines.html", flush=True)
    show = teams or ctx.teams(ctx.seasons[-1])
    for t in show:
        try:
            snapshots(ctx, t, os.path.join(ev, f"snapshots_{t}.png"))
            animated_team(ctx, t, os.path.join(ev, f"animated_{t}.html"))
            if t in GIF_TEAMS:
                animate_gif(ctx, t, os.path.join(ev, f"evolution_{t}.gif"))
            print("  evolution:", t, flush=True)
        except Exception as e:
            print("  SKIP", t, e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
