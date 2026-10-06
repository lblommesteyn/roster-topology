"""The 4D views: three skill-space axes, rotatable, with season as the fourth (animated) axis.

Two scales:
  * per team  - players in skill space, synergy edges, a season slider, hover on players
  * league    - one moving node per team (its minutes-weighted skill centroid) through 8 seasons

Skill-space coordinates are used rather than per-team MDS, because MDS is only defined up to
rotation and would make every season jump for no reason.
"""
import os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, viz_plotly as P

AX = ("PC1 spacing - rim/glass", "PC2 off-ball - on-ball load",
      "PC3 disruption - efficient scoring")


def _ranges(ctx):
    d = ctx.players[ctx.players.minutes >= ctx.min_minutes]
    return [[d.pc1.min() - 1, d.pc1.max() + 1],
            [d.pc2.min() - 1, d.pc2.max() + 1],
            [d.pc3.min() - 1, d.pc3.max() + 1]]


def team_4d(ctx, team, path, ghost=True):
    seasons = [s for s in ctx.seasons if len(ctx.roster(team, s)) >= 4]
    if len(seasons) < 2:
        return None
    rng = _ranges(ctx)
    field, label, scale, lo, hi = P.COLOR_OPTS[3]      # value
    frames = []
    for s in seasons:
        r = ctx.roster(team, s)
        pos = r[["pc1", "pc2", "pc3"]].values
        W = ctx.syn_matrix(r)
        data = []
        if ghost:
            lg = ctx.players[(ctx.players.season == s) &
                             (ctx.players.minutes >= ctx.min_minutes)]
            data.append(go.Scatter3d(
                x=lg.pc1, y=lg.pc2, z=lg.pc3, mode="markers", hoverinfo="skip",
                marker=dict(size=2.2, color="#bbb", opacity=0.35), showlegend=False,
                name="league"))
        data += P.edge_traces(pos, V.edge_list(r, W, 0.12), dims=3, vmax=ctx.edge_scale)
        data.append(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
            text=V.short_labels(list(r.name)), textposition="top center",
            textfont=dict(size=9),
            hovertext=P.hover_text(r), hoverinfo="text",
            marker=dict(size=P.node_sizes(r.minutes, 6, 22), color=r[field], colorscale=scale,
                        cmin=lo, cmax=hi, line=dict(color="#222", width=1),
                        colorbar=dict(title=dict(text=label, side="right"), thickness=14, len=0.7)),
            showlegend=False, name="players"))
        st = ctx.structure(r)
        frames.append(go.Frame(data=data, name=s, layout=go.Layout(title=dict(
            text=f"<b>{team} {s}</b>   net {ctx.net_rating(team, s):+.1f} | "
                 f"complementarity {st['complementarity']:+.3f} | connector {st['connector']}"))))
    nmax = max(len(f.data) for f in frames)
    for f in frames:                     # pad so every frame has the same trace count
        pad = [go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False)
               for _ in range(nmax - len(f.data))]
        f.data = list(f.data[:-1]) + pad + [f.data[-1]]
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title,
        scene=dict(xaxis=dict(title=AX[0], range=rng[0]), yaxis=dict(title=AX[1], range=rng[1]),
                   zaxis=dict(title=AX[2], range=rng[2]), aspectmode="cube",
                   camera=dict(eye=dict(x=1.55, y=1.55, z=0.95))),
        height=820, margin=dict(l=0, r=0, t=95, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left",
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=1100, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.84, y=0.03, currentvalue=dict(prefix="season: "),
                      steps=[dict(label=s, method="animate",
                                  args=[[s], dict(mode="immediate",
                                                  frame=dict(duration=400, redraw=True))])
                             for s in seasons])],
        annotations=[dict(text="drag to rotate | grey cloud = the rest of the league that season | "
                               "edges = fit between team-mates | size = minutes",
                          showarrow=False, xref="paper", yref="paper", x=0, y=1.03,
                          font=dict(size=11, color="#555"))])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path


def league_4d(ctx, path):
    """One node per team: its minutes-weighted position in skill space, moving through seasons."""
    tab = pd.read_csv(os.path.join(V.OUT, "roster_structure_all.csv"))
    rows = []
    for s in ctx.seasons:
        for t in ctx.teams(s):
            r = ctx.roster(t, s)
            if len(r) < 5:
                continue
            w = r.minutes.values.astype(float)
            c = np.average(r[["pc1", "pc2", "pc3"]].values, axis=0, weights=w)
            st = ctx.structure(r)
            m = tab[(tab.season == s) & (tab.team == t)]
            rows.append(dict(season=s, team=t, x=c[0], y=c[1], z=c[2],
                             net=ctx.net_rating(t, s),
                             comp=st["complementarity"], redundancy=st["redundancy"],
                             eff_roles=st["eff_roles"], connector=st["connector"],
                             value=float(np.average(r.value, weights=w))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(V.OUT, "team_centroids.csv"), index=False, encoding="utf-8")
    frames = []
    for s in ctx.seasons:
        sub = d[d.season == s]
        frames.append(go.Frame(name=s, data=[go.Scatter3d(
            x=sub.x, y=sub.y, z=sub.z, mode="markers+text", text=sub.team,
            textposition="top center", textfont=dict(size=10),
            hovertext=[f"<b>{r.team} {r.season}</b><br>net {r.net:+.1f}"
                       f"<br>complementarity {r.comp:+.3f}<br>redundancy {r.redundancy:+.2f}"
                       f"<br>effective roles {r.eff_roles:.2f}<br>connector {r.connector}"
                       for r in sub.itertuples()],
            hoverinfo="text",
            marker=dict(size=10 + 1.6 * (sub.net - d.net.min()), color=sub.comp,
                        colorscale="RdYlGn", cmin=-0.06, cmax=0.10,
                        line=dict(color="#222", width=1),
                        colorbar=dict(title=dict(text="complementarity", side="right"),
                                      thickness=14, len=0.7)),
            showlegend=False)],
            layout=go.Layout(title=dict(text=f"<b>League roster shape, {s}</b>  "
                                             "(position = minutes-weighted skill centroid, "
                                             "size = net rating, colour = complementarity)"))))
    fig = go.Figure(data=list(frames[0].data), frames=frames)
    fig.update_layout(
        title=frames[0].layout.title,
        scene=dict(xaxis=dict(title=AX[0], range=[d.x.min() - .5, d.x.max() + .5]),
                   yaxis=dict(title=AX[1], range=[d.y.min() - .5, d.y.max() + .5]),
                   zaxis=dict(title=AX[2], range=[d.z.min() - .5, d.z.max() + .5]),
                   aspectmode="cube", camera=dict(eye=dict(x=1.55, y=1.55, z=0.95))),
        height=820, margin=dict(l=0, r=0, t=95, b=70),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left",
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=1200, redraw=True),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.84, y=0.03, currentvalue=dict(prefix="season: "),
                      steps=[dict(label=s, method="animate",
                                  args=[[s], dict(mode="immediate",
                                                  frame=dict(duration=400, redraw=True))])
                             for s in ctx.seasons])])
    fig.write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path


def main(teams=None):
    ctx = V.Ctx(); V.ensure_dirs()
    d4 = os.path.join(V.VIZ, "4d"); os.makedirs(d4, exist_ok=True)
    league_4d(ctx, os.path.join(V.VIZ, "four_d_league.html"))
    print("wrote viz/four_d_league.html")
    for t in (teams or ctx.teams(ctx.seasons[-1])):
        team_4d(ctx, t, os.path.join(d4, f"four_d_{t}.html"))
    print("wrote per-team 4D pages to viz/4d/")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
