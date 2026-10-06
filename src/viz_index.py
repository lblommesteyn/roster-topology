"""Build viz/index.html: one page linking every view, with the reading guide."""
import os, sys, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

CSS = """
:root { --bg:#fbfbfa; --fg:#1a1a1a; --muted:#666; --line:#e3e3e0; --pos:#1b7f4f; --neg:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181a; --fg:#ececeb; --muted:#a2a2a0; --line:#2c2f33; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:40px 16px 80px; }
h1 { font-size:30px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:20px; margin:44px 0 6px; padding-bottom:6px; border-bottom:1px solid var(--line); }
p.lede { color:var(--muted); margin:0 0 8px; }
.note { color:var(--muted); font-size:13.5px; margin:4px 0 14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:8px; }
a.card { display:block; padding:9px 11px; border:1px solid var(--line); border-radius:8px;
  text-decoration:none; color:var(--fg); background:transparent; }
a.card:hover { border-color:var(--pos); }
a.card small { display:block; color:var(--muted); font-size:12px; }
.chips a { display:inline-block; margin:0 5px 5px 0; padding:4px 9px; border:1px solid var(--line);
  border-radius:20px; font-size:13px; text-decoration:none; color:var(--fg); }
.chips a:hover { border-color:var(--pos); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
code { background:rgba(127,127,127,.13); padding:1px 5px; border-radius:4px; font-size:13px; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
"""


def cards(items):
    return ('<div class="grid">' +
            "".join(f'<a class="card" href="{h}">{t}<small>{s}</small></a>' for t, s, h in items) +
            "</div>")


def chips(items):
    return '<div class="chips">' + "".join(f'<a href="{h}">{t}</a>' for t, h in items) + "</div>"


def main():
    ctx = V.Ctx(); V.ensure_dirs()
    season = ctx.seasons[-1]
    teams = ctx.teams(season)
    tab = pd.read_csv(os.path.join(V.OUT, "roster_structure_all.csv"))
    cur = tab[tab.season == season].sort_values("complementarity", ascending=False)

    def exists(p):
        return os.path.exists(os.path.join(V.VIZ, p))

    top = cards([
        ("Roster organisms", "3D blobs: the metaphor view", "blobs.html"),
        ("Win projection backtest", "does any of this predict wins?", "wins/backtest.png"),
        ("What rosters are missing", "coverage-deficit maps", f"deficit_gallery_{season}.png"),
        ("Pair fit matrices", "who fits with whom, per team", f"pairgrid_gallery_{season}.png"),
        ("League trends", "what changed, 2018-19 to 2025-26", "wins/league_trends.png"),
        ("League gallery", f"all 30 rosters, {season}", f"league_gallery_{season}.png"),
        ("4D league view", "team shapes moving through 8 seasons", "four_d_league.html"),
        ("Skill map (2D)", "every rotation player, team highlight", f"embedding_2d_{season}.html"),
        ("Skill map (3D)", "rotatable PC1/PC2/PC3", f"embedding_3d_{season}.html"),
        ("UMAP map", "non-linear view of the same space", f"embedding_umap_{season}.html"),
        ("Structure timelines", "metrics per team over 8 seasons", "structure_timelines.html"),
    ])
    team_rows = []
    for t in cur.team:
        bits = []
        if exists(f"teams/team_{season}_{t}.html"):
            bits.append(f'<a href="teams/team_{season}_{t}.html">interactive</a>')
        if exists(f"teams/topology_{season}_{t}.png"):
            bits.append(f'<a href="teams/topology_{season}_{t}.png">static</a>')
        if exists(f"4d/four_d_{t}.html"):
            bits.append(f'<a href="4d/four_d_{t}.html">4D</a>')
        if exists(f"evolution/animated_{t}.html"):
            bits.append(f'<a href="evolution/animated_{t}.html">animated</a>')
        if exists(f"evolution/snapshots_{t}.png"):
            bits.append(f'<a href="evolution/snapshots_{t}.png">by season</a>')
        if exists(f"evolution/evolution_{t}.gif"):
            bits.append(f'<a href="evolution/evolution_{t}.gif">gif</a>')
        r = cur[cur.team == t].iloc[0]
        cls = "pos" if r.complementarity > 0 else "neg"
        team_rows.append(
            f"<tr><td><b>{t}</b></td><td class='{cls}'>{r.complementarity:+.3f}</td>"
            f"<td>{r.net:+.1f}</td><td>{r.connector}</td><td>{' · '.join(bits)}</td></tr>")

    swaps = sorted(glob.glob(os.path.join(V.VIZ, "swaps", "swap_*.png")))
    trades = sorted(glob.glob(os.path.join(V.VIZ, "swaps", "trade_*.png")))
    pairs = sorted(glob.glob(os.path.join(V.VIZ, "pairs", "pair_*.png")))

    def rel(p):
        return os.path.relpath(p, V.VIZ).replace("\\", "/")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA Roster Topology</title><style>{CSS}</style></head><body><div class="wrap">
<h1>NBA roster topology</h1>
<p class="lede">Player skill embeddings, pair complementarity and roster shape, 2018-19 to {season}.
Every view is generated from the pipeline in <code>src/</code>; the statistical caveats are in
<code>FINDINGS.md</code> and the reading guide for these views is in
<code>VISUALIZATION_FINDINGS.md</code>.</p>

<h2>Start here</h2>
{top}

<h2>How to read a roster graph</h2>
<p class="note">
Each node is a player: <b>size</b> = minutes, <b>colour</b> = a skill dimension (PC1 runs
perimeter spacing to rim-and-offensive-glass). Each edge is the model's estimate of how well
two players fit:
<span class="pos">green = complementary</span>, <span class="neg">red = redundant or clashing</span>,
thickness = strength. In the static and interactive team views, <b>distance is fit</b>: the layout
is MDS on synergy-derived distances, so players who fit sit close together. In the 3D and 4D views
the axes are skill dimensions instead, which stay comparable across seasons.
</p>
<p class="note">
A <b>healthy</b> roster looks like a loose ring of well-separated nodes with green edges crossing
the middle. <b>Redundancy</b> looks like a tight red knot. A <b>bridge player</b> sits between two
otherwise unconnected groups with green edges to both, and is named in the connector column below.
</p>

<h2>Teams, {season}</h2>
<table><tr><th>team</th><th>complementarity</th><th>net</th><th>connector</th><th>views</th></tr>
{''.join(team_rows)}</table>

<h2>Roster organisms</h2>
<p class="note">The same rosters rendered as 3D bodies rather than graphs: each player is a
metaball, the team is the isosurface of their combined occupancy of skill space. Lobes are role
concentrations, a fat dense centre is redundancy, a detached satellite is a player nobody overlaps
with. Trades are shown as true shape morphs. See <code>viz/blobs.html</code> and
<code>BLOB_VIS_FINDINGS.md</code>.</p>
<div class="chips"><a href="blobs.html">open the organism gallery</a></div>

<h2>What each roster is missing</h2>
<p class="note">League density minus team density on the two structural skill axes: warm is a role
the roster is short of, cool is a surplus. Gaps are named after the league player who lives there.
Read it as a diagnostic, not a quality metric: a bigger gap correlates with <em>more</em> wins
(+0.20 over 776 team-seasons), because narrow specialised rosters are usually the good ones. See
<code>BETTER_VIEWS.md</code>.</p>
<div class="chips"><a href="deficit_gallery_{season}.png">deficit gallery</a></div>

<h2>Pair fit matrices</h2>
<p class="note">One cell per pair, clustered so compatible groups form blocks. This is the view the
blobs cannot produce, because a density field discards pair identity. Cells show the model fit;
observed pair effects are annotated only where a pair has 3,000+ shared possessions, and the two
correlate at just r = 0.09.</p>
<div class="chips"><a href="pairgrid_gallery_{season}.png">pair-fit gallery</a>
<a href="pairgrid_model_vs_observed_{season}.png">model vs observed</a></div>

<h2>Counterfactual swaps</h2>
<p class="note">Before and after a hypothetical roster change, including players who have never
shared a floor with the roster: which team-mates gain and lose, where the newcomer lands in skill
space, and how the structure metrics move.</p>
{chips([(os.path.basename(p)[5:-4].replace('_', ' '), rel(p)) for p in swaps])}

<h2>Real transactions</h2>
<p class="note">Both panels are real seasons: departures ringed grey, arrivals ringed orange.</p>
{chips([(os.path.basename(p)[6:-4].replace('_', ' '), rel(p)) for p in trades])}

<h2>Why a pair fits, or does not</h2>
<p class="note">Each panel decomposes one pair's predicted fit over the eigen-modes of the
interaction matrix, shows both players' skill profiles, lists comparable pairs elsewhere in the
league, and prints how many possessions the pair actually shared, with the uncertainty on the
measured estimate.</p>
{chips([(os.path.basename(p)[5:-4].replace('_', ' '), rel(p)) for p in pairs])}

<h2>Caveat worth keeping in view</h2>
<p class="note">Complementarity is real but small: about 0.45 points per 100 possessions per pair,
against 3.07 for individual player value, and roughly 17 points per 100 of noise in a single
lineup's season. These pictures make fit look like the main story because fit is all they draw.
It is not the main story; talent is.</p>
</div></body></html>"""
    p = os.path.join(V.VIZ, "index.html")
    open(p, "w", encoding="utf-8").write(html)
    print("wrote viz/index.html")
    return p


if __name__ == "__main__":
    main()
