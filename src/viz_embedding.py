"""League-wide player embedding maps, 2D and 3D, with team highlighting and nearest neighbours."""
import os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, viz_plotly as P

NDIM = 6


def neighbours(df, k=4):
    """k nearest players in skill space, for the hover card."""
    X = df[[f"pc{i+1}" for i in range(NDIM)]].values
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.inf)
    names = df.name.values
    out = {}
    for i in range(len(df)):
        idx = np.argsort(D[i])[:k]
        out[i] = "closest: " + ", ".join(f"{names[j]}" for j in idx)
    return out


def _axis_titles(mode):
    if mode == "umap":
        return "UMAP 1", "UMAP 2", None
    return ("PC1   perimeter spacing <-> rim and offensive glass",
            "PC2   off-ball <-> on-ball load",
            "PC3   steals and self-creation <-> efficient scoring volume")


def build(ctx, season, mode="pc", dims=2, label_top=28):
    df = ctx.players[(ctx.players.season == season) &
                     (ctx.players.minutes >= ctx.min_minutes)].reset_index(drop=True)
    if mode == "umap":
        xs, ys, zs = df.umap1.values, df.umap2.values, None
    else:
        xs, ys, zs = df.pc1.values, df.pc2.values, df.pc3.values
    nb = neighbours(df)
    hover = P.hover_text(df, extra=nb)
    field, label, scale, lo, hi = P.COLOR_OPTS[0]
    Scatter = go.Scatter3d if dims == 3 else go.Scatter
    common = dict(mode="markers", hovertext=hover, hoverinfo="text", name="league")
    coords = dict(x=xs, y=ys) if dims == 2 else dict(x=xs, y=ys, z=zs)
    fig = go.Figure()
    fig.add_trace(Scatter(
        **coords, **common,
        marker=dict(size=P.node_sizes(df.minutes, 5, 17) if dims == 2
                    else P.node_sizes(df.minutes, 3, 11),
                    color=df[field], colorscale=scale, cmin=lo, cmax=hi, opacity=0.78,
                    line=dict(color="#333", width=0.5),
                    colorbar=dict(title=dict(text=label, side="right"), thickness=14, len=0.68))))
    # labels for the highest-minute players, so the map is readable without hovering
    top = df.nlargest(label_top, "minutes")
    tcoords = dict(x=top.pc1 if mode != "umap" else top.umap1,
                   y=top.pc2 if mode != "umap" else top.umap2)
    if dims == 3:
        tcoords["z"] = top.pc3
    fig.add_trace(Scatter(**tcoords, mode="text", text=[V.surname(n) for n in top.name],
                          textposition="top center", textfont=dict(size=9, color="#333"),
                          hoverinfo="skip", showlegend=False, name="labels"))
    # one highlight trace, driven by the team dropdown
    teams = ctx.teams(season)
    first = df[df.team == teams[0]]
    hcoords = dict(x=first.pc1 if mode != "umap" else first.umap1,
                   y=first.pc2 if mode != "umap" else first.umap2)
    if dims == 3:
        hcoords["z"] = first.pc3
    fig.add_trace(Scatter(**hcoords, mode="markers",
                          hovertext=P.hover_text(first), hoverinfo="text",
                          marker=dict(size=P.node_sizes(first.minutes, 12, 30) if dims == 2
                                      else P.node_sizes(first.minutes, 6, 16),
                                      color="rgba(0,0,0,0)",
                                      line=dict(color="#111", width=2.5)),
                          name=f"highlight: {teams[0]}"))
    buttons = []
    for t in teams:
        sub = df[df.team == t]
        args = {"x": [sub.pc1.tolist() if mode != "umap" else sub.umap1.tolist()],
                "y": [sub.pc2.tolist() if mode != "umap" else sub.umap2.tolist()],
                "hovertext": [P.hover_text(sub)],
                "marker.size": [P.node_sizes(sub.minutes, 12, 30).tolist() if dims == 2
                                else P.node_sizes(sub.minutes, 6, 16).tolist()],
                "name": [f"highlight: {t}"]}
        if dims == 3:
            args["z"] = [sub.pc3.tolist()]
        buttons.append(dict(label=t, method="restyle", args=[args, [2]]))
    xt, yt, zt = _axis_titles(mode)
    layout = dict(
        title=dict(text=f"<b>League skill map {season}</b>  "
                        f"({'UMAP' if mode=='umap' else 'PCA'}, {dims}D)", x=0.02, y=0.985,
                   yanchor="top"),
        updatemenus=[P.color_menu(df, 0, x=0.0, y=1.10),
                     dict(buttons=buttons, direction="down", showactive=True,
                          x=0.30, y=1.10, xanchor="left", yanchor="top")],
        height=830, margin=dict(l=20, r=20, t=170, b=30), plot_bgcolor="white",
        annotations=[dict(text="colour = skill dimension (left menu) | ring = selected team "
                               "(right menu) | size = minutes | hover for nearest comparables",
                          showarrow=False, xref="paper", yref="paper", x=0, y=1.022,
                          font=dict(size=11, color="#555"))])
    if dims == 3:
        layout["scene"] = dict(xaxis_title=xt, yaxis_title=yt, zaxis_title=zt,
                               aspectmode="cube")
    else:
        layout["xaxis"] = dict(title=xt, zeroline=True, gridcolor="#eee")
        layout["yaxis"] = dict(title=yt, zeroline=True, gridcolor="#eee")
    fig.update_layout(**layout)
    return fig


def main(season=None):
    ctx = V.Ctx(); V.ensure_dirs()
    season = season or ctx.seasons[-1]
    for mode, dims, name in (("pc", 2, "embedding_2d"), ("pc", 3, "embedding_3d"),
                             ("umap", 2, "embedding_umap")):
        fig = build(ctx, season, mode=mode, dims=dims)
        p = os.path.join(V.VIZ, f"{name}_{season}.html")
        fig.write_html(p, include_plotlyjs="cdn")
        print("wrote", os.path.relpath(p, V.ROOT))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
