# Rosters as organisms: what the blob actually encodes

Open `viz/blobs.html`. Every NBA roster, 2018-19 to 2025-26, rendered as a 3D body in learned
skill space: each player is a metaball whose radius grows with his minutes, the team is the
isosurface where their combined occupancy field crosses a threshold, and marching cubes turns that
into a mesh. This document answers the question the visuals were built for.

**Primary question: can roster construction be visualized as a smooth evolving 3D organism whose
shape changes meaningfully through a season and during trades?**

Yes for construction and for trades, with one measured caveat about what the shape can and cannot
tell you. The blob's *local* structure is genuinely informative: peak density tracks redundancy at
r = 0.48, lobe count tracks team net rating at +0.25, and whole-field deformation tracks actual
roster turnover at r = 0.54. Its *scalar summaries* are much weaker, and fragility barely shows up
at all.

---

## 1. Which encodings capture which roster property

Measured over 210 team-seasons, correlating each blob shape statistic against the graph-based
structure metrics from the earlier analysis:

| shape encoding | redundancy | complementarity | fragility | net rating |
|---|---|---|---|---|
| **peak density** (how high the field piles up) | **+0.48** | -0.20 | -0.06 | -0.23 |
| **compactness** (sphericity) | **+0.38** | -0.27 | -0.16 | -0.27 |
| **lobe count** (disconnected components) | -0.32 | +0.10 | +0.14 | **+0.25** |
| **volume** (space occupied) | -0.26 | +0.11 | -0.04 | +0.02 |
| largest-lobe share | +0.22 | -0.19 | -0.16 | -0.27 |
| density ratio (peak / mean) | -0.19 | +0.14 | +0.08 | +0.11 |

Read that as four separate verdicts.

**Redundancy is captured well, and by the most obvious encoding.** Peak density is exactly the
"two players occupying the same functional space" idea made geometric: when metaballs overlap,
the field piles up, and the surface bulges into a single fat mass. At r = 0.48 against an
independently computed redundancy metric, this is the strongest correspondence in the set. Visually
it is unmistakable in the gallery: New York, Washington, Brooklyn, Philadelphia and Golden State,
the five densest organisms, are all single compact balls. San Antonio, Houston and Denver, the
loosest, are sprawling multi-lobed bodies with satellites.

**Diversity reads as lobe count, and lobes are the only encoding that likes good teams.** More
separated role clusters correlates +0.25 with net rating, and Oklahoma City at +13.2 is a
three-lobed organism. Compactness runs the other way at -0.27: a tidy sphere is a worse team. The
visual instinct that a "healthy" roster should look like a clean, symmetric shape is exactly
backwards.

**Complementarity is captured poorly.** No shape statistic exceeds |r| = 0.27. This is not a
rendering failure, it is the geometry: complementarity is a property of *pairs*, and the occupancy
field deliberately throws pair identity away in favour of a smooth density. The blob shows you
where a roster sits and how it is distributed; it cannot show you that these two particular players
fit. For that, the graph views in `viz/teams/` remain the right tool.

**Fragility is essentially invisible**, |r| <= 0.16 for everything tried. The intended encoding was
"a narrow neck between two lobes means one connector holds the roster together", and necks do form,
but whether a neck is narrow depends as much on the bandwidth and iso level as on the roster. This
is the one encoding from the original concept that did not survive measurement.

---

## 2. Roster change: the whole field, not its summary

For all 210 season-to-season transitions, against the actual share of minutes that turned over:

| how change is measured | correlation with real turnover |
|---|---|
| **mean absolute field difference** | **+0.54** |
| **volume overlap (IoU) between the two bodies** | **-0.42** |
| change in compactness | +0.16 |
| change in volume | +0.13 |
| change in lobe count | +0.13 |
| change in peak density | +0.12 |

The lesson is sharp: **compare the fields, not their summaries.** A roster can turn over half its
minutes and end up with a blob of nearly the same volume, compactness and lobe count, because
those are one-number descriptions of a 3D object. The field difference sees the body move. This is
also why the morph views colour vertices by the signed field change rather than plotting a
before/after bar of shape statistics.

Field difference also correlates +0.29 with the absolute change in team net rating: teams whose
organism deformed most had the largest swings in results, in either direction.

---

## 3. The trade morph is the view that works best

Interpolating the field, `f_t = f_keep + (1-t) f_out + t f_in`, and re-extracting the surface at
every step gives a genuine deformation rather than a cross-fade, and the topology changes mid-morph.

The clearest example is Minnesota swapping Rudy Gobert for LaMelo Ball. At t = 0 the organism has
**two components**: the main body, plus a detached satellite far out on the rim-and-glass axis that
is Gobert on his own, because no other Timberwolf occupies anything like his space. As the morph
runs, that satellite deflates and disappears, while a bright green lobe inflates on the guard side
and **fuses with the main body**: one component. The picture states the roster consequence more
directly than any table: the team stops having an isolated interior specialist and becomes a single
guard-heavy mass.

The colouring matters here. Vertices are coloured by occupancy change against the starting field,
so red is volume being vacated and green is volume being gained, on the surface itself. A viewer
can see *where* in role space the team is trading from and to, which the numeric swap panels
(`viz/swaps/`) cannot show.

---

## 4. Within a season the organism moves, but you must stop it shimmering

Monthly player totals (one API request per month) give exact minutes and exact rotation membership,
plus a monthly skill profile. The profile is the problem. Measured month to month for heavy-minutes
players, a player's position wanders with a standard deviation of **0.21, 0.29 and 0.62** of the
league spread on PC1, PC2 and PC3. Rendering raw monthly positions makes every blob shimmer, and
the shimmer looks exactly like roster change.

The fix is to shrink each monthly position toward the player's season position by the measured
reliability (0.79 / 0.71 / 0.38) while using minutes and membership exactly as counted. The
organism then deforms only when the rotation really changes.

What survives that filter is real. Dallas runs a 7-man rotation in October and a 12-man rotation in
April, and the blob correspondingly inflates from volume 193 to 341 and splits from one lobe to
three: the visual signature of a team whose season has stopped mattering. Memphis shows the
opposite in March, a single wide body at volume 275 with one lobe, a full rotation all occupying
overlapping space.

---

## 5. Three things that go wrong when building this

**Splitting a roster into pieces silently rescales it.** The first morph barely moved. The cause
was that the occupancy function normalised weights within whatever set it was given, so a one-player
"incoming" piece was renormalised to the same total mass as the twelve-player "kept" piece, making
the interpolation a rounding error. Every piece must be normalised against the *original* roster's
totals so that `f_keep + f_out` reproduces the original field exactly. Worth checking explicitly:
the sum of the parts should equal the whole.

**The iso level decides how organic the result looks.** At the first setting (55% of the field's
own maximum) blobs were small tight ellipsoids with players stranded outside their own team's
surface. Lowering it to 34% and widening the metaball radius produced bodies that actually enclose
their players and grow lobes. This is a real modelling choice, not just cosmetics: it sets how
close two players must be before they fuse into one mass, which is precisely the redundancy
threshold.

**A league-sized box makes every team look like a speck.** Two extreme players (Rudy Gobert sits at
PC1 ~ 12) stretched the bounding box until each organism occupied a few percent of the frame.
Clipping the box at the 1st and 99th percentile fixed the framing, and drawing a faint league
envelope behind the team gives back the sense of scale that clipping removes: you can see how much
of the league's space one roster actually occupies.

---

## 6. What the axes mean

PC1 runs perimeter spacing to rim-and-offensive-glass, PC2 runs off-ball to on-ball load, PC3 runs
disruption and self-creation to efficient scoring volume. PC1 is **not** a size axis, even though
its extreme is Rudy Gobert: Markkanen sits at +0.4 and Jokic at +1.9. A team whose blob stretches
along PC1 has rim-and-glass players and floor spacers, which may or may not mean it is tall.

---

## 6. Honest limits

The blob is a picture of **where a roster's minutes sit in skill space**. It is not a picture of
quality, and it is not a picture of fit between particular players. The colour gradient carries an
independent scalar (offence-defence leaning by default), and the underlying complementarity model
is weak enough, about 0.45 points per 100 possessions per pair against 3.07 for individual talent,
that a beautiful organism and a good team are only loosely related. The measured correlation
between lobe count and net rating is +0.25: real, worth knowing, and nowhere near a prediction.

The right use is exploratory: see which rosters pile up in one region, see which have an isolated
specialist floating off the main body, and watch what a trade does to the shape. Then check the
numbers.
