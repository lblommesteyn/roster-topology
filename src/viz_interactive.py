"""Interactive per-team topology HTML (hover on players, colour dropdown, signed edges)."""
import os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, viz_plotly as P, topology as TP


def partner_notes(ctx, r, W):
    """Best and worst fitting team-mate for each player, for the hover card."""
    names = list(r.name)
    out = {}
    for i in range(len(r)):
        w = W[i].copy(); w[i] = np.nan
        if np.all(np.isnan(w)):
            continue
        b, s = int(np.nanargmax(w)), int(np.nanargmin(w))
        out[i] = (f"best fit: {names[b]} ({w[b]:+.2f})<br>"
                  f"worst fit: {names[s]} ({w[s]:+.2f})")
    return out


def team_figure(ctx, team, season, pos=None, title=None):
    r = ctx.roster(team, season)
    if len(r) < 4:
        return None
    W = ctx.syn_matrix(r)
    pos = TP.synergy_mds(W) if pos is None else pos
    edges = V.edge_list(r, W, 0.10)
    fig = go.Figure()
    for t in P.edge_traces(pos, edges, vmax=ctx.edge_scale):
        fig.add_trace(t)
    node_idx = len(fig.data)
    field, label, scale, lo, hi = P.COLOR_OPTS[0]
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text",
        text=V.short_labels(list(r.name)), textposition="bottom center",
        textfont=dict(size=10),
        hovertext=P.hover_text(r, extra=partner_notes(ctx, r, W)), hoverinfo="text",
        marker=dict(size=P.node_sizes(r.minutes), color=r[field], colorscale=scale,
                    cmin=lo, cmax=hi, line=dict(color="#222", width=1),
                    colorbar=dict(title=dict(text=label, side="right"), thickness=14, len=0.7)),
        name="players", showlegend=False))
    st = ctx.structure(r)
    net = ctx.net_rating(team, season)
    fig.update_layout(
        title=dict(text=title or (f"<b>{team} {season}</b>  net {net:+.1f} | "
                                  f"complementarity {st['complementarity']:+.3f} | "
                                  f"connector {st['connector']}"), x=0.02, xanchor="left"),
        updatemenus=[P.color_menu(r, node_idx)],
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white", height=720, margin=dict(l=20, r=20, t=110, b=20),
        legend=dict(orientation="h", y=-0.04),
        annotations=[dict(text="distance = fit (MDS on synergy); node size = minutes; "
                               "green edge = complementary, red = redundant",
                          showarrow=False, xref="paper", yref="paper", x=0, y=1.055,
                          font=dict(size=11, color="#555"))])
    return fig


def main(season=None):
    ctx = V.Ctx(); V.ensure_dirs()
    season = season or ctx.seasons[-1]
    out = os.path.join(V.VIZ, "teams")
    made = []
    for t in ctx.teams(season):
        fig = team_figure(ctx, t, season)
        if fig is None:
            continue
        p = os.path.join(out, f"team_{season}_{t}.html")
        fig.write_html(p, include_plotlyjs="cdn", full_html=True)
        made.append(t)
    print(f"wrote {len(made)} interactive team pages to viz/teams/")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
