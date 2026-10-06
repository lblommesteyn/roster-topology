"""Render a trade morph to an animated GIF (a quick-look artifact that needs no browser)."""
import os, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, blob as B, blob_viz as BV, blob_scenes as BS


def morph_gif(ctx, team, season, outgoing, incoming, path, steps=18, n=44, spin=False, fps=9):
    import imageio.v2 as imageio
    roster = ctx.roster(team, season)
    cand = ctx.players[ctx.players.name.str.lower() == incoming.lower()].sort_values("season")
    inc = cand.iloc[-1].copy()
    idx = roster.index[roster.name == outgoing][0]
    inc["minutes"] = roster.loc[idx, "minutes"]
    keep, out_row, in_row = roster.drop(index=idx), roster.loc[[idx]], pd.DataFrame([inc])
    lo, hi = B.league_bounds(ctx.players, ctx.min_minutes)
    axes, pts = B.make_grid(lo, hi, n)
    total, nref = float(roster.minutes.sum()), len(roster)
    ref_h = float(np.mean(B.bandwidths(roster)))

    def part(df):
        if not len(df):
            return np.zeros((n, n, n))
        w = B.norm_weights(df, ref_total=total, ref_n=nref)
        return B.occupancy(df, pts, weights=w, ref_h=np.full(len(df), ref_h))

    fk, fo, fi = part(keep), part(out_row), part(in_row)
    f0, f1 = fk + fo, fk + fi
    level = B.ISO_Q * float(max(f0.max(), f1.max()))
    dmax = max(float(np.abs(f1 - f0).max()) * 0.55, 1e-3)
    tmp = os.path.join(V.VIZ, "blobs", "_gif_frame.png")
    frames = []
    seq = list(np.linspace(0, 1, steps)) + [1.0] * 3 + list(np.linspace(1, 0, steps // 2))
    for si, t in enumerate(seq):
        f = fk + (1 - t) * fo + t * fi
        s = B.surface(f, axes, iso=level)
        if s is None:
            continue
        d = B.sample_at(f - f0, axes, s["verts"])
        mesh = BV.mesh_trace(s, "leaning", opacity=0.9, intensity=d, cmin=-dmax, cmax=dmax,
                             scale=BS.GAIN_SCALE, showscale=False)
        # the light is fixed while the camera can move, so a GIF needs more ambient than the
        # interactive view or half the frames come out black
        mesh.lighting = dict(ambient=0.72, diffuse=0.65, specular=0.18, roughness=0.6, fresnel=0.1)
        cur = pd.concat([keep, out_row if t < 0.5 else in_row], ignore_index=True)
        fig = go.Figure(data=[mesh, BV.nuclei_trace(cur, labels=False)])
        sc = BV.scene(lo, hi)
        if spin:
            ang = 2 * np.pi * si / max(len(seq), 1) * 0.6
            sc["camera"] = dict(eye=dict(x=1.35 * np.cos(ang), y=1.35 * np.sin(ang), z=0.72))
        fig.update_layout(scene=sc, paper_bgcolor=BV.BG, plot_bgcolor=BV.BG,
                          font=dict(color=BV.FG), height=760, width=900,
                          margin=dict(l=0, r=0, t=56, b=0),
                          title=dict(text=f"<b>{team} {season}</b>  {outgoing} &#8594; {incoming}",
                                     x=0.02, font=dict(color=BV.FG, size=17)))
        fig.write_image(tmp)
        frames.append(imageio.imread(tmp))
    if not frames:
        return None
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    imageio.mimsave(path, [f[:h, :w] for f in frames], duration=1000.0 / fps, loop=0)
    if os.path.exists(tmp):
        os.remove(tmp)
    return path


def main():
    ctx = V.Ctx()
    out = os.path.join(V.VIZ, "blobs")
    for team, season, og, inc in [("MIN", "2025-26", "Rudy Gobert", "LaMelo Ball"),
                                  ("GSW", "2025-26", "Al Horford", "Nikola Jokic")]:
        p = morph_gif(ctx, team, season, og, inc,
                      os.path.join(out, f"morph_{team}_{V.surname(og)}_to_{V.surname(inc)}.gif"))
        print("  gif:", p, flush=True)


if __name__ == "__main__":
    main()


def _lobe_meshes(s, color_by, opacity):
    """One Mesh3d per connected lobe.

    Plotly sorts BETWEEN traces by depth but cannot correctly sort the front and back faces of a
    single non-convex transparent mesh, which is what produced the banding when the blob turned
    edge-on. Each lobe on its own is close to convex, so splitting the surface fixes most of it.
    """
    from scipy import ndimage
    field, axes, level = s["field"], s["axes"], s["level"]
    lab, n = ndimage.label(field >= level)
    out = []
    floor = float(field.min())
    for L in range(1, n + 1):
        masked = np.where(lab == L, field, floor)
        sub = B.surface(masked, axes, iso=level)
        if sub is None or len(sub["verts"]) < 8:
            continue
        sub["color"] = B.sample_at(s["_colorfield"], axes, sub["verts"])
        m = BV.mesh_trace(sub, color_by, opacity=opacity, showscale=False)
        m.lighting = dict(ambient=0.62, diffuse=0.78, specular=0.24, roughness=0.55, fresnel=0.15)
        out.append((float(np.mean(sub["verts"], axis=0)[0]), m))
    return [m for _, m in sorted(out, key=lambda x: x[0])]


def rotate_gif(ctx, team, season, path, frames=36, n=62, fps=12, color_by="leaning",
               labels=True, opacity=0.6, label_top=None):
    """A full 360 orbit around one team's organism, translucent, with names that stay readable.

    Two rendering traps this works around:
      * the light source is fixed while the camera orbits, so the interactive default ambient
        leaves roughly half the frames nearly black
      * player names drawn as Scatter3d text are depth-tested and disappear inside the surface, so
        they are drawn as scene annotations instead, which render above the mesh at every angle
    """
    import imageio.v2 as imageio
    roster = ctx.roster(team, season)
    if len(roster) < 4:
        return None
    lo, hi = B.bounds_including(ctx.players, roster, ctx.min_minutes)
    s = B.team_blob(ctx, roster, lo, hi, n=n, color_by=color_by)
    if s is None:
        return None
    axes, pts = B.make_grid(lo, hi, n)
    s["_colorfield"] = B.scalar_field(roster, pts, column=color_by)
    meshes = _lobe_meshes(s, color_by, opacity)
    m = s["metrics"]

    keep = roster if label_top is None else roster.nlargest(label_top, "minutes")
    keep = keep.sort_values("pc2").reset_index(drop=True)
    names = dict(zip(keep.pid, V.short_labels(list(keep.name))))
    # rosters pile up in a small region of skill space, so labels collide. Alternate the leader
    # direction by rank to fan them apart instead of stacking them on one another.
    anns = []
    for rank, p in enumerate(keep.itertuples()):
        if p.pid not in names:
            continue
        ay = [-26, 24, -42, 40][rank % 4]
        anns.append(dict(x=float(p.pc1), y=float(p.pc2), z=float(p.pc3), text=names[p.pid],
                         showarrow=True, arrowhead=0, arrowwidth=0.7,
                         arrowcolor="rgba(242,242,240,0.45)", ax=0, ay=ay,
                         font=dict(size=10, color="#f2f2f0"),
                         bgcolor="rgba(10,12,16,0.72)", borderpad=2))

    tmp = os.path.join(V.VIZ, "blobs", "_rot_frame.png")
    imgs = []
    for k in range(frames):
        ang = 2 * np.pi * k / frames
        nuc = BV.nuclei_trace(roster, labels=False)
        nuc.marker.color = "rgba(255,255,255,0.95)"
        fig = go.Figure(data=list(meshes) + [nuc])
        sc = BV.scene(lo, hi)
        sc["camera"] = dict(eye=dict(x=1.45 * np.cos(ang), y=1.45 * np.sin(ang), z=0.62))
        sc["annotations"] = anns
        # 3D axis titles rotate with the scene and are unreadable at most angles, so they are
        # blanked here and replaced by a fixed key drawn in paper coordinates below
        for a in ("xaxis", "yaxis", "zaxis"):
            sc[a] = dict(sc[a]); sc[a]["title"] = dict(text="")
        fig.update_layout(
            scene=sc, paper_bgcolor=BV.BG, plot_bgcolor=BV.BG, font=dict(color=BV.FG),
            height=820, width=980, margin=dict(l=0, r=0, t=76, b=0),
            title=dict(text=f"<b>{team} {season}</b>  "
                            f"<span style='font-size:13px;color:#9aa3ad'>"
                            f"net {ctx.net_rating(team, season):+.1f} | volume {m['volume']:.0f} | "
                            f"lobes {m['components']} | peak density {m['peak_density']:.2f}</span>",
                       x=0.02, font=dict(color=BV.FG, size=19)),
            annotations=[
                dict(text="<b>axes</b>", xref="paper", yref="paper", x=0.012, y=0.145,
                     showarrow=False, font=dict(size=13, color="#e8e8e6"), xanchor="left"),
                dict(text="horizontal:  perimeter spacing &#8594; rim and offensive glass",
                     xref="paper", yref="paper", x=0.012, y=0.105, showarrow=False,
                     font=dict(size=12.5, color="#b9c0c8"), xanchor="left"),
                dict(text="depth:  off-ball &#8594; on-ball load",
                     xref="paper", yref="paper", x=0.012, y=0.068, showarrow=False,
                     font=dict(size=12.5, color="#b9c0c8"), xanchor="left"),
                dict(text="vertical:  disruption &#8594; efficient scoring",
                     xref="paper", yref="paper", x=0.012, y=0.031, showarrow=False,
                     font=dict(size=12.5, color="#b9c0c8"), xanchor="left"),
                dict(text="colour:  defence-leaning &#8594; offence-leaning",
                     xref="paper", yref="paper", x=0.988, y=0.031, showarrow=False,
                     font=dict(size=12.5, color="#b9c0c8"), xanchor="right")])
        fig.write_image(tmp)
        imgs.append(imageio.imread(tmp))
    h = min(i.shape[0] for i in imgs); w = min(i.shape[1] for i in imgs)
    imageio.mimsave(path, [i[:h, :w] for i in imgs], duration=1000.0 / fps, loop=0)
    if os.path.exists(tmp):
        os.remove(tmp)
    return path
