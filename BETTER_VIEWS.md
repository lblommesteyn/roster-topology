# Two views that work better than the blobs

Built after measuring that the 3D organism, whatever its merits as an explainer, is a poor
instrument. Over 240 team-seasons the third dimension adds almost nothing:

| | vs redundancy | vs net rating |
|---|---|---|
| 3D peak density | +0.452 | -0.222 |
| **2D peak density (PC1 x PC2 only)** | +0.433 | **-0.297** |

The two agree at r = 0.87 and the 2D version is *better* against net rating. PC3 is 8.6% of
variance and is "disruption to efficient scoring", not a structural axis, so the z-axis buys
~0.02 of correlation in exchange for occlusion, depth ambiguity and 69 MB of HTML.

---

## 1. Coverage-deficit maps (`viz/deficit/`, gallery `viz/deficit_gallery_<season>.png`)

Every other view draws the mass a roster occupies. This one draws **league density minus team
density**: warm is a role the roster is short of, cool is a surplus, which is the same thing as
redundancy. Both fields are minutes-weighted and normalised to sum to one, so it reads as "share of
minutes this roster is missing relative to how the league spends minutes".

Each gap is named after the highest-minute league player who lives in that region, which turns an
abstract hole into a shopping list:

| team, 2025-26 | biggest gaps | most duplicated |
|---|---|---|
| MIN | a McCollum type, a Smith type | Reid, McDaniels |
| GSW | a Durant type, a Gillespie type | Horford, Post |
| OKC | a Maxey type, a Murray type | Joe, Wallace |
| DAL | a Murphy type, a LaRavia type | Flagg, Thompson |

**Read it as a diagnostic, not a quality metric.** Measured over 776 team-seasons (2000-01 to
2025-26), a bigger gap goes with *more* wins, not fewer:

| | vs same-season wins | within-season standardized |
|---|---|---|
| max gap | +0.17 | +0.20 |
| total mismatch | +0.27 | +0.28 |
| max surplus | -0.01 | |

That sign is not a bug. A team with a large hole is a team concentrating its minutes narrowly, and
narrow, specialised rosters are usually good ones; a team that covers every region of skill space
evenly is usually a rebuilding team spreading minutes across a dozen interchangeable players. So
the map answers "what type of player is missing", never "is this roster good".

---

## 2. Pair-fit matrices (`viz/pairgrid/`, gallery `viz/pairgrid_gallery_<season>.png`)

The one thing a blob structurally cannot show. The occupancy field that makes a blob smooth throws
pair identity away, which is exactly why blob shape correlates only |r| <= 0.27 with
complementarity. A matrix keeps it: one cell per pair, players ordered by hierarchical clustering
on the fit values, so mutually-compatible groups appear as green blocks.

Golden State 2025-26 reads immediately: Porzingis-Horford at -1.0 and Post-Horford at -0.8 (three
bigs competing for the same space), against Curry-Payton +0.6 and Richard-Porzingis +0.6.

**Cells carry the model fit, not the measured pair effect.** That is deliberate. Measured per-pair
synergy has a split-half reliability of 0.06 across 30 seasons, so printing it in a grid would be
printing noise in a form that looks authoritative. Where a pair has 3,000+ shared possessions the
observed value is annotated in small grey text, and the gap is the lesson: Green-Podziemski is
-0.3 by model and -3.9 observed; Green-Curry is +0.5 by model and +2.9 observed. Across 501
well-sampled pairs in 2025-26 the two correlate at **r = 0.09** (`viz/pairgrid_model_vs_observed_
2025-26.png`), and it is the observed number, not the model number, that fails to replicate.

---

## What to use when

| question | view |
|---|---|
| what is this roster short of? | coverage-deficit map |
| who duplicates whom? | deficit map (cool regions) or the pair matrix |
| does this specific pair fit? | pair matrix |
| which groups on this roster cohere? | pair matrix, clustered order |
| explaining the concept to someone | the blob gallery and the trade morph |
| how much did a roster change? | the field-difference view, not any scalar summary |

The blobs keep their job as the explainer: the trade morph, where Gobert's detached satellite
deflates and a LaMelo lobe inflates and fuses, still states a roster consequence faster than any
table. They are just not the tool to answer a question with.
