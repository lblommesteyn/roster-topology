"""Turn a roster into a 3D organism.

Each player is an influence centre in learned skill space (PC1/PC2/PC3). A roster becomes a scalar
occupancy field

    f(x) = sum_i  w_i * exp( -||x - z_i||^2 / (2 h_i^2) )

with w_i the player's share of team minutes and h_i a bandwidth that grows slightly with minutes,
so a heavy-minutes player occupies more of the volume. Marching cubes then extracts the level set
f(x) = tau as a closed surface: the team blob.

The metaball formulation is what makes the shape organic rather than a point cloud. Two players in
similar roles fuse into one fat lobe (redundancy), a lone specialist grows his own protrusion
(role concentration), and a player sitting between two clusters creates a narrow neck joining them
(a connector). When nobody bridges two clusters, the surface separates into disconnected
components, which is exactly what a split roster should look like.

A second field carries an interpretable scalar (offence-defence leaning by default), evaluated at
the surface vertices, so the blob's colour gradient means something.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

GRID = 56                 # samples per axis; 56^3 is smooth enough and stays fast in a browser
PAD = 2.2                 # padding around the league cloud, in skill-space units
BASE_H = 1.35             # base metaball radius
ISO_Q = 0.34              # iso level as a fraction of the field's own max


def league_bounds(players, min_minutes=300, pad=PAD, q=1.0):
    """Box around the league, clipped at the 1st/99th percentile.

    A couple of extreme players (Gobert sits at PC1 ~ 12) would otherwise stretch the box until
    every team blob looked like a speck in an empty cube.
    """
    d = players[players.minutes >= min_minutes]
    lo = np.array([np.percentile(d.pc1, q), np.percentile(d.pc2, q),
                   np.percentile(d.pc3, q)]) - pad
    hi = np.array([np.percentile(d.pc1, 100 - q), np.percentile(d.pc2, 100 - q),
                   np.percentile(d.pc3, 100 - q)]) + pad
    return lo, hi


def bounds_including(players, roster, min_minutes=300, pad=PAD, q=1.0, margin=0.8):
    """League box, expanded so every player on THIS roster is inside it.

    `league_bounds` clips at the 1st/99th percentile so a couple of extremes do not stretch the box
    until every blob is a speck. The side effect is that a player past the clip is dropped from the
    scene entirely: Luka Doncic at PC2 = 9.09 fell outside a 8.99 ceiling and vanished from the
    figure, surface contribution included. Always widen to contain the roster being drawn.
    """
    lo, hi = league_bounds(players, min_minutes, pad, q)
    P = roster[["pc1", "pc2", "pc3"]].values
    lo = np.minimum(lo, P.min(axis=0) - margin)
    hi = np.maximum(hi, P.max(axis=0) + margin)
    return lo, hi


def make_grid(lo, hi, n=GRID):
    axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    return axes, np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def bandwidths(roster, base=BASE_H):
    """Heavier-minutes players get a slightly larger radius; nobody disappears entirely."""
    w = roster.minutes.values.astype(float)
    share = w / max(w.max(), 1.0)
    return base * (0.72 + 0.55 * np.sqrt(share))


def norm_weights(roster, ref_total=None, ref_n=None):
    """Player weights on one shared scale.

    Splitting a roster into pieces (kept / outgoing / incoming) only interpolates correctly if all
    pieces are normalised by the SAME totals, otherwise a one-player piece gets re-normalised to
    weight 1 and the morph barely moves. That was the first version's bug.
    """
    w = roster.minutes.values.astype(float)
    total = float(ref_total) if ref_total else max(w.sum(), 1e-9)
    n = float(ref_n) if ref_n else max(len(roster), 1)
    return w / total * n


def occupancy(roster, gridpts, base_h=BASE_H, weights=None, ref_h=None):
    """Scalar occupancy field of a set of players."""
    P = roster[["pc1", "pc2", "pc3"]].values
    w = norm_weights(roster) if weights is None else np.asarray(weights, float)
    h = bandwidths(roster, base_h) if ref_h is None else np.asarray(ref_h, float)
    flat = gridpts.reshape(-1, 3)
    field = np.zeros(len(flat))
    for k in range(len(roster)):
        d2 = ((flat - P[k]) ** 2).sum(axis=1)
        field += w[k] * np.exp(-d2 / (2 * h[k] ** 2))
    return field.reshape(gridpts.shape[:3])


def scalar_field(roster, gridpts, column="leaning", base_h=BASE_H, eps=1e-6):
    """Minutes-and-proximity weighted average of a player attribute at every grid point."""
    P = roster[["pc1", "pc2", "pc3"]].values
    vals = roster[column].values.astype(float)
    w = norm_weights(roster)
    h = bandwidths(roster, base_h)
    flat = gridpts.reshape(-1, 3)
    num = np.zeros(len(flat)); den = np.full(len(flat), eps)
    for k in range(len(roster)):
        kern = w[k] * np.exp(-((flat - P[k]) ** 2).sum(axis=1) / (2 * h[k] ** 2))
        num += kern * vals[k]; den += kern
    return (num / den).reshape(gridpts.shape[:3])


def surface(field, axes, iso=None, iso_q=ISO_Q, step=1):
    """Marching-cubes surface of the occupancy field, in skill-space coordinates."""
    from skimage import measure
    level = iso if iso is not None else iso_q * float(field.max())
    if not np.isfinite(level) or level <= field.min() or level >= field.max():
        return None
    verts, faces, normals, _ = measure.marching_cubes(field, level=level, step_size=step)
    # marching cubes returns index space; map back to skill-space units
    out = np.empty_like(verts)
    for k in range(3):
        a = axes[k]
        out[:, k] = a[0] + verts[:, k] * (a[1] - a[0])
    return dict(verts=out, faces=faces, normals=normals, level=level)


def sample_at(field, axes, pts):
    """Trilinear sample of a grid field at arbitrary points (for vertex colouring)."""
    from scipy.ndimage import map_coordinates
    idx = np.empty((3, len(pts)))
    for k in range(3):
        a = axes[k]
        idx[k] = (pts[:, k] - a[0]) / (a[1] - a[0])
    return map_coordinates(field, idx, order=1, mode="nearest")


def components(field, level):
    """How many disconnected pieces the blob has, and the share of volume in the largest."""
    from scipy import ndimage
    mask = field >= level
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0.0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return int(n), float(sizes.max() / sizes.sum())


def blob_metrics(roster, field, axes, level):
    """Shape statistics that map onto the roster-structure ideas."""
    from scipy import ndimage
    mask = field >= level
    vox = float(mask.sum())
    cell = np.prod([a[1] - a[0] for a in axes])
    volume = vox * cell
    # surface area via marching cubes, for a compactness (sphericity) measure
    s = surface(field, axes, iso=level)
    area = np.nan
    if s is not None:
        from skimage import measure
        area = float(measure.mesh_surface_area(s["verts"], s["faces"]))
    spher = np.nan
    if area and area > 0 and volume > 0:
        spher = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / area
    n_comp, largest = components(field, level)
    # peak density = how concentrated the roster is in one spot (redundancy)
    peak = float(field.max())
    return dict(volume=volume, area=area, sphericity=spher, components=n_comp,
                largest_share=largest, peak_density=peak,
                density_ratio=float(peak / max(field[mask].mean(), 1e-9)) if vox else np.nan)


def team_blob(ctx, roster, lo=None, hi=None, n=GRID, color_by="leaning", iso_q=ISO_Q,
              base_h=BASE_H):
    """Everything needed to draw one team: mesh, vertex colours, metrics."""
    if lo is None or hi is None:
        lo, hi = league_bounds(ctx.players, ctx.min_minutes)
    axes, pts = make_grid(lo, hi, n)
    field = occupancy(roster, pts, base_h=base_h)
    s = surface(field, axes, iso_q=iso_q)
    if s is None:
        return None
    level = s["level"]
    col = scalar_field(roster, pts, column=color_by, base_h=base_h)
    s["color"] = sample_at(col, axes, s["verts"])
    s["density"] = sample_at(field, axes, s["verts"])
    s["metrics"] = blob_metrics(roster, field, axes, level)
    s["field"] = field
    s["axes"] = axes
    return s
