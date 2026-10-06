"""Plotly helpers shared by the interactive views."""
import numpy as np

# Plotly cannot vary line width inside one trace, so edges are binned into a few width classes.
WIDTH_BINS = [(0.00, 0.25, 1.0), (0.25, 0.50, 2.2), (0.50, 0.75, 3.8), (0.75, 1.01, 5.6)]
POS, NEG = "#1b7f4f", "#c0392b"


def edge_traces(pos, edges, vmax=None, name_prefix="", dims=2, visible=True, opacity_scale=1.0):
    """Line traces for a signed graph, binned by |weight|."""
    import plotly.graph_objects as go
    if not edges:
        return []
    vmax = vmax or max(abs(w) for *_, w in edges) + 1e-12
    traces = []
    for sign, col, lab in ((1, POS, "fits together"), (-1, NEG, "redundant / clashing")):
        for lo, hi, width in WIDTH_BINS:
            xs, ys, zs = [], [], []
            for i, j, w in edges:
                if np.sign(w) != sign:
                    continue
                f = abs(w) / vmax
                if not (lo <= f < hi):
                    continue
                xs += [pos[i, 0], pos[j, 0], None]
                ys += [pos[i, 1], pos[j, 1], None]
                if dims == 3:
                    zs += [pos[i, 2], pos[j, 2], None]
            if not xs:
                continue
            kw = dict(x=xs, y=ys, mode="lines", hoverinfo="skip", visible=visible,
                      line=dict(color=col, width=width),
                      opacity=min(1.0, (0.22 + 0.5 * (lo + hi) / 2) * opacity_scale),
                      name=f"{name_prefix}{lab}", legendgroup=lab,
                      showlegend=(lo == WIDTH_BINS[-1][0]))
            traces.append(go.Scatter3d(z=zs, **kw) if dims == 3 else go.Scatter(**kw))
    return traces


def hover_text(r, ctx=None, extra=None):
    """Rich hover card per player."""
    out = []
    for k, row in r.reset_index(drop=True).iterrows():
        t = (f"<b>{row['name']}</b><br>{row.get('team','')} {row.get('season','')}"
             f"<br>{int(row['minutes'])} min | value {row['value']:+.2f} pts/100"
             f"<br>role: {row.get('role','')}"
             f"<br>PC1 rim/glass {row['pc1']:+.1f} | PC2 on-ball load {row['pc2']:+.1f}"
             f" | PC3 scoring eff {row['pc3']:+.1f}")
        if extra is not None and k in extra:
            t += "<br>" + extra[k]
        out.append(t)
    return out


def node_sizes(minutes, lo=8, hi=34):
    m = np.asarray(minutes, float)
    if m.max() <= 0:
        return np.full(len(m), lo)
    return lo + (hi - lo) * np.sqrt(m / m.max())


COLOR_OPTS = [
    ("pc1", "PC1  spacing - rim/glass", "RdBu_r", -6, 6),
    ("pc2", "PC2  off-ball - on-ball load", "PuOr_r", -6, 6),
    ("pc3", "PC3  disruption - efficient scoring", "PRGn", -6, 6),
    ("value", "estimated value", "RdYlGn", -6, 6),
    ("leaning", "offence - defence leaning", "BrBG", -4, 4),
]


def color_menu(r, trace_index, x=0.0, y=1.12):
    """Dropdown that restyles one trace's marker colour between the interpretable scalars."""
    buttons = []
    for field, label, scale, lo, hi in COLOR_OPTS:
        buttons.append(dict(label=label, method="restyle",
                            args=[{"marker.color": [r[field].tolist()],
                                   "marker.colorscale": [scale],
                                   "marker.cmin": [lo], "marker.cmax": [hi],
                                   "marker.colorbar.title.text": [label]}, [trace_index]]))
    return dict(buttons=buttons, direction="down", showactive=True, x=x, y=y,
                xanchor="left", yanchor="top", pad=dict(t=2, b=2))
