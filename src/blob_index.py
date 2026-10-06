"""viz/blobs.html: the dark-stage index for the organism views."""
import os, sys, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

CSS = """
:root { --bg:#0a0c10; --fg:#e8e8e6; --muted:#8b949e; --line:#1e242c; --pos:#2ecc8f; --neg:#c0392b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:44px 18px 90px; }
h1 { font-size:34px; margin:0 0 4px; letter-spacing:-0.02em; font-weight:650; }
h2 { font-size:19px; margin:46px 0 8px; padding-bottom:7px; border-bottom:1px solid var(--line);
  font-weight:600; }
p.lede { color:var(--muted); margin:0 0 6px; max-width:74ch; }
.note { color:var(--muted); font-size:13.5px; margin:6px 0 16px; max-width:82ch; }
.hero { display:block; width:100%; border-radius:10px; border:1px solid var(--line); margin:10px 0 4px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:8px; }
a.card { display:block; padding:10px 12px; border:1px solid var(--line); border-radius:9px;
  text-decoration:none; color:var(--fg); background:#0e1218; }
a.card:hover { border-color:var(--pos); background:#111820; }
a.card small { display:block; color:var(--muted); font-size:12px; margin-top:2px; }
.chips a { display:inline-block; margin:0 6px 6px 0; padding:5px 11px; border:1px solid var(--line);
  border-radius:20px; font-size:13px; text-decoration:none; color:var(--fg); background:#0e1218; }
.chips a:hover { border-color:var(--pos); }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:6px 9px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
code { background:#151b23; padding:1px 6px; border-radius:4px; font-size:12.5px; }
.key { display:flex; gap:22px; flex-wrap:wrap; color:var(--muted); font-size:13px; margin:8px 0 0; }
.key b { color:var(--fg); font-weight:600; }
"""


def main():
    ctx = V.Ctx()
    season = ctx.seasons[-1]
    blobs = os.path.join(V.VIZ, "blobs")

    def rel(p):
        return os.path.relpath(p, V.VIZ).replace("\\", "/")

    def cards(items):
        return ('<div class="grid">' +
                "".join(f'<a class="card" href="{h}">{t}<small>{s}</small></a>'
                        for t, s, h in items) + "</div>")

    def chips(items):
        return '<div class="chips">' + "".join(f'<a href="{h}">{t}</a>' for t, h in items) + "</div>"

    teams = sorted(glob.glob(os.path.join(blobs, f"blob_{season}_*.html")))
    evo = sorted(glob.glob(os.path.join(blobs, "evolution_*.html")))
    seasonw = sorted(glob.glob(os.path.join(blobs, f"season_*_{season}.html")))
    morphs = sorted(glob.glob(os.path.join(blobs, "morph_*.html")))
    gifs = sorted(glob.glob(os.path.join(blobs, "morph_*.gif")))
    rots = sorted(glob.glob(os.path.join(blobs, "rotate_*.gif")))

    mpath = os.path.join(V.OUT, "blob_metrics.csv")
    table = ""
    if os.path.exists(mpath):
        m = pd.read_csv(mpath)
        cur = m[m.season == season].sort_values("peak_density", ascending=False)
        rows = "".join(
            f"<tr><td><b>{r.team}</b></td><td>{r.net:+.1f}</td><td>{r.volume:.0f}</td>"
            f"<td>{r.sphericity:.2f}</td><td>{int(r.components)}</td>"
            f"<td>{r.peak_density:.2f}</td><td>{r.connector}</td></tr>" for r in cur.itertuples())
        table = (f"<table><tr><th>team</th><th>net</th><th>volume</th><th>compactness</th>"
                 f"<th>lobes</th><th>peak density</th><th>connector</th></tr>{rows}</table>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roster organisms</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Roster organisms</h1>
<p class="lede">Every NBA roster as a 3D body in learned skill space. Each player is a metaball
whose radius grows with his minutes; the team is the surface where their combined occupancy field
crosses a threshold. Shape is roster construction: lobes are role concentrations, necks are
connectors, a fat dense centre is redundancy, and a detached satellite is a player nobody overlaps
with.</p>
<img class="hero" src="blob_gallery_{season}.png" alt="thirty roster organisms">
<div class="key">
  <span><b>position</b> where the roster sits in skill space</span>
  <span><b>volume</b> how much of the space it covers</span>
  <span><b>lobes</b> distinct role clusters</span>
  <span><b>colour</b> defence-leaning (blue) to offence-leaning (orange)</span>
  <span><b>axes</b> spacing-to-rim/glass, off-ball-to-on-ball, disruption-to-efficient-scoring</span>
  <span><b>inner glow</b> where the minutes concentrate</span>
</div>

<h2>Start here</h2>
{cards([("League gallery", f"all 30 organisms, {season}", f"blob_gallery_{season}.html"),
        ("Trade morph", "MIN: Gobert becomes LaMelo Ball", rel(os.path.join(blobs, "morph_MIN_Gobert_to_Ball.html"))),
        ("Season evolution", "OKC across 8 seasons", rel(os.path.join(blobs, "evolution_OKC.html"))),
        ("Within a season", f"OKC month by month, {season}", rel(os.path.join(blobs, f"season_OKC_{season}.html")))])}

<h2>Single-team viewers, {season}</h2>
<p class="note">Rotate, zoom, hover a nucleus for the player. The faint grey envelope is the whole
league that season, so you can see how much of the space a roster actually occupies.</p>
{chips([(os.path.basename(p).split("_")[-1][:-5], rel(p)) for p in teams])}

<h2>Season evolution</h2>
<p class="note">One frame per season, 2018-19 to {season}. The organism grows, splits and re-forms
as the roster turns over. Play it and watch Oklahoma City go from a single dense ball of
interchangeable guards to a wide three-lobed body.</p>
{chips([(os.path.basename(p)[len("evolution_"):-5], rel(p)) for p in evo])}

<h2>Within a season, month by month</h2>
<p class="note">Built from monthly player totals. Minutes and rotation membership are exact;
each player's position is shrunk toward his season position, because raw monthly form is noisy
enough to make a blob shimmer for no reason. Annotations name who entered and left the rotation.</p>
{chips([(os.path.basename(p)[len("season_"):-len(f"_{season}.html")], rel(p)) for p in seasonw])}

<h2>Trade morphs</h2>
<p class="note">Not two pictures side by side: the occupancy field itself is interpolated and the
surface is re-extracted at every step, so the outgoing player's lobe deflates while the incoming
player's inflates, and the topology can change mid-morph. Green is volume gained, red is volume
vacated.</p>
{chips([(os.path.basename(p)[len("morph_"):-5].replace("_", " "), rel(p)) for p in morphs])}
{("<p class='note'>Quick-look animations, no browser interaction needed:</p>" + chips([(os.path.basename(p)[:-4].replace("_", " "), rel(p)) for p in gifs])) if gifs else ""}

<h2>Rotations</h2>
<p class="note">A full orbit around one organism. The surface is drawn opaque here: a non-convex
semi-transparent mesh self-sorts badly and bands when it turns edge-on, and the orbit conveys the
form better than see-through does anyway.</p>
{chips([(os.path.basename(p)[len("rotate_"):-4].replace("_", " "), rel(p)) for p in rots]) if rots else ""}

<h2>Shape numbers, {season}</h2>
<p class="note">Peak density tracks the graph-based redundancy measure at r = 0.48, and lobe count
correlates with net rating at +0.25: more separated role clusters, better team. Full reasoning in
<code>BLOB_VIS_FINDINGS.md</code>.</p>
{table}
</div></body></html>"""
    p = os.path.join(V.VIZ, "blobs.html")
    open(p, "w", encoding="utf-8").write(html)
    print("wrote viz/blobs.html")
    return p


if __name__ == "__main__":
    main()
