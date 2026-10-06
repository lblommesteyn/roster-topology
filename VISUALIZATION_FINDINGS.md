# Which roster-topology views actually teach you something

A reading guide to `viz/`. Open `viz/index.html` first. The statistical results these pictures sit
on top of are in `FINDINGS.md`; the short version is that complementarity is real but small
(0.45 points per 100 possessions per pair against 3.07 for individual value), and these views draw
only the fit, which makes fit look like the whole story. It is not.

---

## The four views that are worth the screen space

**1. The league gallery (`viz/league_gallery_<season>.png`).** Thirty rosters on one sheet, ordered
by average complementarity, all on the same edge scale. This is the single view that makes the
concept land, because the shapes genuinely differ and the ordering is not arbitrary:

- **A healthy roster is a loose ring**: well-separated nodes, green edges crossing the middle, few
  thick red lines. Charlotte, the Lakers and Cleveland in 2025-26 look like this.
- **Redundancy is a tight red knot**: nodes piled together with thick red edges between them.
  Golden State and Memphis in 2025-26 are the clearest examples in the league.
- The ordering is *not* the standings. Oklahoma City is sixth by complementarity and first by net
  rating by a distance. That gap is the honest summary of the whole project.

**2. Season snapshots per team (`viz/evolution/snapshots_<TEAM>.png`).** Eight panels, Procrustes-
aligned so they are comparable. Oklahoma City's is the best single narrative in the set: the
Westbrook-Adams roster, the shapeless middle years, the 2021-22 tangle of thick red edges at
net -7.2, then the clean, spread structure of 2024-25 and 2025-26 at +12.5 and +13.2. Minnesota's
shows Gobert arriving and immediately becoming the hub of the green edges.

**3. Pair explanation panels (`viz/pairs/`).** These do the translation work: skill profiles side by
side, the fit decomposed over the interaction modes, comparable pairs elsewhere in the league, and
the measured shared-possession estimate with its uncertainty. Two of them are worth reading in full:

- *Kalkbrenner + Missi* (-6.56): both on the same pole of the dominant mode, which penalises
  doubling up. The seven most similar pairs in the league are all other two-traditional-big
  combinations. This is what "redundancy" means mechanically.
- *Murray + Jokic*: measured pair effect **+1.39 over 22,656 shared possessions**, model estimate
  **+0.08**. The single best teaching example in the set: the observed number is seventeen times
  the structural estimate, and it is the observed number that does not replicate.

**4. The counterfactual swap panels (`viz/swaps/swap_*.png`).** The "who gains and loses" bar chart
is the most actionable element anywhere in this project, because it names individuals. Swapping
Gobert for LaMelo Ball in Minnesota moves complementarity from +0.091 to -0.005, and the bar chart
says exactly where the damage lands: DiVincenzo -1.64, Hyland -1.31, Conley -1.07, all the guards
who were being covered for. McDaniels is the only real gainer.

---

## The views that look better than they are

**The 3D and 4D team views.** Rotation, a season slider and hover all work, and they are the most
fun to demo, but depth ambiguity makes it genuinely hard to tell which edge connects which pair.
Everything they show is legible in the 2D view plus the season slider. Keep them for the
"where does this player sit in the league" question, which the grey league cloud answers well, and
use the 2D animated view for anything about fit. The goal anticipated this: the 2D-plus-time
version is the one to reach for.

**The league 4D view (team centroids over time).** Team skill centroids barely move, because
averaging twelve players washes out nearly everything that matters. It reads as a jittering cloud.
Its one real use is spotting a team that jumps in PC1, which means a genuine change in how much
the roster lives at the rim and on the offensive glass rather than spacing the floor.

**UMAP.** Prettier clusters than PCA, and the clusters are an artifact of the projection. The league
is a continuum, as PCA shows honestly, and UMAP's apparent groups invite exactly the archetype
thinking the analysis says does not hold. Kept for completeness, not recommended.

**Role-family colouring.** k-means on the skill embedding produces clusters that mix size and role:
one cluster contains both Dyson Daniels and Onyeka Okongwu. The rule-based names on top of them are
a convenience for legends, not a finding. The continuous colourings (PC1, PC2, value) are faithful;
the role labels are a simplification.

---

## Three ways these pictures can mislead, and what was done about each

**1. Per-team edge normalisation.** The first version scaled edge colour and width by the strongest
edge *within each team*, so a roster with nothing interesting happening looked exactly as dramatic
as a polarised one, and the gallery was uncomparable across panels. Every view now uses one
league-wide scale (the 99th percentile of absolute synergy over all rotation pairs, 1.45 points per
100). Teams with genuinely flat structure now correctly look flat.

**2. Arbitrary rotation in season-to-season layouts.** MDS layouts are defined only up to rotation
and reflection. Animating consecutive seasons straight from MDS makes the roster spin violently
even when nine of twelve players are unchanged, which reads as dramatic upheaval and is pure
artifact. Each season is now Procrustes-aligned to the previous one on the players they share, so
movement in the animation means a change in fit.

**3. Distance means two different things.** In the team graphs distance is *fit* (MDS on synergy).
In the embedding maps and the 3D/4D views distance is *skill similarity* (PCA). Those are close to
opposites for a good pairing: Gobert and Edwards are far apart in skill space precisely because
they fit. The PCA axes are play-style, not physical: PC1 runs perimeter spacing to rim-and-glass,
so a stretch big lands near the wings rather than next to the centres. Every view states which space it is in, and the index page says it once more, because
this is the easiest thing in the whole set to misread.

---

## What the visuals answer well, and what they cannot

Answered well: which players occupy the same functional space, which rosters duplicate roles,
who the connector is, where a hypothetical addition would land, and who on the roster gains or
loses from a swap.

Not answered, and no picture here can fix it: whether any of this predicts wins. The graphs are
drawn from a model whose pair-level estimates replicate at r = 0.06 and whose structural surface,
though real, is about a seventh the size of individual talent. A convincing-looking green edge is a
league-wide tendency applied to two players, not a measured fact about those two. The most
important honest detail in the set is in the Minnesota trade panel: Gobert's arrival **raised**
complementarity from +0.054 to +0.073 while the team's net rating **fell** from +2.4 to +0.5.
