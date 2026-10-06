# What the data says about NBA roster construction

8 seasons (2018-19 to 2025-26), 835 players, 3,102 player-seasons, 108,228 team-lineup rows,
3.72M possessions. Negative results are reported as prominently as positive ones, because most
of what this project tested turned out to be negative.

**In one paragraph.** The skill geometry is real and falls out of the data unprompted. Chemistry
between two specific players is not measurable: adjusted pair synergy has a split-half reliability
of 0.06, while the raw pair net rating that looks so repeatable (0.33) is just player quality in
disguise. A league-wide complementarity structure over skill *types* does exist, replicates across
disjoint seasons at r = 0.32, is orthogonal to player quality, and recovers textbook basketball
(two traditional centres bad, passing big next to a rim finisher good) without being told any of
it. But it is about 15% the size of individual talent, which is small enough that it does not
improve prediction of future lineups at all, and improves team-season prediction only
suggestively.

---

## 1. The skill manifold is real, continuous, and two-dimensional to first order

41 standardized skill features per player-season reduce to 12 PCA components holding 86% of the
variance, but the first two hold **51%** on their own, and they are the two axes every scout
already talks about:

| axis | variance | what it actually measures | low end | high end |
|---|---|---|---|---|
| PC1 (27.6%) | perimeter spacing vs rim and offensive glass | off reb +0.91, rim rate +0.86, putbacks +0.86, blocks +0.72 against 2pt shot distance -0.70, 3PA rate -0.63 | Trae Young, Fred VanVleet, Damian Lillard | Rudy Gobert, DeAndre Jordan, Jalen Duren |
| PC2 (23.1%) | off-ball vs on-ball load | usage +0.80, turnovers +0.83, fouls drawn +0.78 against corner-3 rate -0.79 | Nicolas Batum, Dean Wade, AJ Green | Luka Doncic, Nikola Jokic, Giannis Antetokounmpo |
| PC3 (7.8%) | disruption vs efficient scoring volume | points/100 +0.56, TS% +0.54, eFG +0.46 against steals -0.42, unassisted 2s -0.42 | Jamal Shead, Rob Dillingham, Yves Missi | Victor Wembanyama, Joel Embiid, Stephen Curry |

Nothing was labelled by hand, and the labels must come from the loadings rather than from the
exemplars. **PC1 is a play-style axis, not a size axis**: Lauri Markkanen sits at +0.4 and Nikola
Jokic at +1.9, indistinguishable from wings, while Rudy Gobert is at +9.9. What separates them is
living at the rim and crashing the offensive glass versus shooting from distance, not height. An
earlier version of this table called PC1 "size" and PC3 "stretch vs ground-bound"; both were read
off the tallest names at each extreme and both were wrong.

---

## 2. Individual pair chemistry is, for practical purposes, not measurable

This is the strongest result in the project and it is negative.

Adjusted pair synergy (a ridge coefficient per pair, on top of both players' main effects) was
estimated for 10,101 pairs with at least 400 shared possessions. Its **split-half reliability is
r = 0.06** (Spearman-Brown corrected, about 0.11) and its season-to-season replication is
r = 0.02 to 0.05. That is noise.

| quantity | split-half r | across-season r |
|---|---|---|
| adjusted pair synergy | **+0.06** | +0.02 to +0.05 |
| raw pair net rating (shrunk) | +0.33 | +0.38 |
| player offensive main effect | +0.32 | about +0.30 |
| player defensive main effect | +0.28 | about +0.27 |

The contrast in that table is the whole point. **Raw pair net rating looks highly repeatable
(r = 0.33) and is almost entirely player quality and team context, not chemistry.** Two good
players keep looking good together because they are good. Once individual quality is partialled
out, essentially nothing reproducible is left at the level of a specific duo.

**Replicated across 30 seasons and six independent eras.** The original test used 8 seasons. With
the full 1996-97 to 2025-26 lineup history (340,333 lineup rows, 13.3M possessions) the result is
the same in every era, and the obvious defence of the null is now dead:

| era | adjusted pair r | raw pair r | player effect r | median shared possessions |
|---|---|---|---|---|
| 1996-2001 | +0.093 | +0.422 | +0.420 | 2,100 |
| 2001-2006 | +0.042 | +0.338 | +0.515 | 2,011 |
| 2006-2011 | +0.040 | +0.352 | +0.490 | 2,092 |
| 2011-2016 | +0.068 | +0.348 | +0.401 | 1,901 |
| 2016-2021 | +0.051 | +0.311 | +0.390 | 1,821 |
| 2021-2026 | +0.063 | +0.354 | +0.367 | 1,846 |

**Sample size is not the explanation.** Older eras had shorter rotations and far more concentrated
minutes (44% of minutes to a team's top three in 1996-97 against 35% now), so pairs accumulated
*more* shared possessions, and reliability did not improve: across the six eras the correlation
between median shared possessions and pair reliability is +0.12 (p = 0.82).

Nor does it improve within eras. Pooling all 35,038 split-half pair comparisons and binning by how
much the pair actually played together:

| shared possessions | n pairs | split-half r |
|---|---|---|
| 805 - 1,944 | 17,509 | +0.041 |
| 1,944 - 3,184 | 8,766 | +0.043 |
| 3,184 - 5,264 | 5,259 | +0.068 |
| 5,264 - 8,487 | 2,452 | +0.092 |
| **8,487 - 19,677** | 1,052 | **+0.040** |

Reliability drifts up slightly through the middle of the range and then falls back at the very top.
Pairs with 8,000+ shared possessions, the most-measured duos in thirty years of basketball, are no
more reproducible than pairs with 1,000. Overall r = +0.059, Spearman-Brown corrected +0.111.

**Implication: a claim about a specific duo's "chemistry" read off on-off or pair net rating is
almost certainly noise, and would not replicate even in a second sample of that same duo's own
possessions.**

---

## 3. But a weak league-wide complementarity structure does exist

Individual pairs are hopeless; the *regularity across pairs* is not. Modelling synergy as
`z_i' M z_j`, a symmetric interaction over the skill embedding, shares 21 to 36 parameters across
every pair in the league instead of spending one parameter per pair.

Fitting M **jointly** with the player effects in a single ridge on lineup outcomes, rather than the
two-stage route of estimating pair coefficients first and then fitting M to them, is decisively
better: the two-stage surface has a spread of 0.15 points per 100 possessions, the direct joint fit
has 0.45, three times larger, at equal or better replication.

The surface replicates on the test that matters. A random split of lineup rows leaves the same
team-seasons on both sides, so any unabsorbed team effect is shared and agreement is inflated.
Splitting into **disjoint seasons** removes that. Measured within each of six independent eras over
30 seasons:

| era | surface r (disjoint seasons) |
|---|---|
| 1996-2001 | +0.61 |
| 2001-2006 | +0.40 |
| 2006-2011 | +0.55 |
| 2011-2016 | **+0.66** |
| 2016-2021 | +0.42 |
| 2021-2026 | +0.23 |

Median across eras **+0.48**, which is meaningfully higher than the +0.32 the original 8-season
window suggested. The league-wide regularity over skill *types* is more reproducible than first
reported, even as chemistry between named players stays at zero. The one soft spot is the most
recent era (+0.23), the same window the original estimate came from.

The interaction surface is also **orthogonal to player quality**: correlation with the sum of the
two players' individual values is -0.00, with their product -0.02. It is measuring fit, not talent
stacking, which is the separation the whole exercise was after.

Magnitude, in context:

| quantity | sd (points per 100 possessions) |
|---|---|
| individual player value | 3.07 |
| pair complementarity | 0.45 |
| whole-lineup complementarity (10 pairs) | about 1.4 |
| noise in a held-out lineup's season net rating | about 17 |

**Fit is real and it is about 15% the size of talent at the pair level.**

---

## 4. Of 18 canonical "fit" heuristics, one survives

Each heuristic was written as a symmetric function of the two players' standardized skills and
fitted to the measured pair coefficients, with bootstrap intervals.

| heuristic | effect (pts/100 per sd) | 95% CI | verdict |
|---|---|---|---|
| **creator x creator (usage x usage)** | **-0.052** | [-0.082, -0.019] | **negative, survives** |
| two small guards | +0.033 | [-0.002, +0.070] | suggestive |
| defensive rebounder x defensive rebounder | -0.028 | [-0.059, +0.002] | suggestive |
| **rim pressure x spacing** | **+0.001** | [-0.075, +0.073] | **dead null** |
| spacing x spacing | -0.030 | [-0.098, +0.041] | null |
| two bigs (rim-heavy) | -0.012 | [-0.044, +0.020] | null |
| two non-shooters | +0.009 | [-0.020, +0.037] | null |
| creator x rim-running big | -0.004 | [-0.032, +0.020] | null |
| rim protector x perimeter stealer | -0.000 | [-0.028, +0.026] | null |
| passer x movement shooter | +0.017 | [-0.011, +0.044] | null |
| isolation x isolation | -0.005 | [-0.033, +0.030] | null |
| 7 others | | | null |

The most surprising line is **rim pressure x spacing at +0.001**. The single most repeated idea in
modern basketball analysis, that rim attackers and shooters make each other better, is
indistinguishable from zero *as an interaction*, once both players' main effects are removed. The
likely reason: its benefit is already inside each player's main effect, since every shooter's
value is measured in a league where he already plays next to rim pressure.

The one survivor, **two high-usage creators fit worse together**, is the ball-dominance tax, and it
is small: two players each one standard deviation high in usage cost about 0.05 points per 100
relative to their individual values.

---

## 5. What the learned interaction surface encodes

Eigendecomposition of the jointly fitted M (net offence minus defence), mapped back to skills:

- **-0.17, the dominant mode, like-with-like loses.** Interior self-creation, midrange, offensive
  rebounding and getting blocked at the rim, against catch-and-shoot 3s, defensive rebounding and
  steals. **Doubling up on either pole is penalised, mixing them pays.** This is spacing
  complementarity, discovered without being told it exists.
- **+0.078, like-with-like gains.** True shooting, shot quality, foul drawing, 3-point volume.
  **Efficient shot-quality players compound with each other**, and inefficient midrange-heavy
  players compound negatively. Efficiency is not a substitutable resource.
- **-0.063, like-with-like loses.** Scoring usage against passing and playmaking. **Two finishers
  clash, a scorer plus a distributor gains.**

Ranking real 2025-26 rotation players by predicted fit gives an entirely recognisable answer that
nobody supplied to the model:

- **Worst fits:** Deandre Ayton x Ryan Kalkbrenner, Kalkbrenner x Luke Kornet, Rudy Gobert x
  Kalkbrenner, Gobert x Ayton, Jalen Duren x Ayton. Every one is two traditional non-shooting
  centres.
- **Best fits:** Draymond Green with essentially every rim centre (Kalkbrenner, Duren, Kornet,
  Gobert, Ayton), then creators with rim-runners (Doncic x Kalkbrenner, Giddey x Kalkbrenner,
  Wembanyama x Kalkbrenner).

The model rediscovers that the valuable big-man skill next to a rim finisher is passing and
low-usage defensive presence, not more rim finishing.

---

## 5b. Redundancy and connectors

Roster-structure metrics computed for all 240 team-seasons from skill geometry and minutes alone
(no fitted effects, so these are not circular):

- **Connector players** (the player whose removal costs the roster the most average
  complementarity) are overwhelmingly either lead creators or passing bigs: Luka Doncic, Jarrett
  Allen and Rudy Gobert lead with 6 team-seasons each, then Jokic, Adebayo, Antetokounmpo,
  Gilgeous-Alexander and Royce O'Neale. **The players who hold a roster's fit together are not the
  same list as the players who score the most**; O'Neale and Allen are on it because they are the
  compatible piece next to everyone.
- **Fragility** (dependence on one connector) peaks at 2018-19 Atlanta, whose whole structure ran
  through rookie Trae Young, followed by 2019-20 Portland and 2018-19 Utah.
- Both redundancy (r = -0.23) and effective role count (r = -0.30) correlate negatively with
  same-season net rating. Read that carefully: duplicated roles are mildly bad, but so is maximal
  skill dispersion. Good rosters are neither five copies of one player nor five unrelated
  specialists. These are weak associations, and the out-of-sample test in the next section is the
  one that counts.

---

## 6. Does any of it improve prediction? Not for lineups; maybe for whole rosters

**Future lineups: no.** Six chronological splits, each training on all prior seasons and predicting
held-out lineups in the next season, every model calibrated by cross-fitting inside the training
period so none is penalised for scale. Weighted out-of-sample R2, mean over the six splits:

| model | mean R2 | mean corr |
|---|---|---|
| direct joint fit, interaction **on** | **0.0658** | 0.216 |
| additive player effects only | 0.0654 | 0.217 |
| additive + per-pair ridge coefficients | 0.0641 | **0.218** |
| direct joint fit, interaction **off** (its own control) | 0.0624 | 0.212 |
| additive + two-stage bilinear surface | 0.0589 | 0.207 |
| additive + raw pair net rating | 0.0568 | 0.202 |

Two things to read there. The interaction surface beats **its own control**, 0.0658 vs 0.0624, in
five of six splits, so the structure is doing real work. But it ties the plain additive baseline
(+0.0004, noise). And **raw pair net rating is the worst model tested**, below even the mean-only
baseline in places: feeding observed pair results into a lineup prediction actively hurts, which is
the practical cost of the noise documented in section 2.

The arithmetic explains the tie. A lineup's fit advantage has a spread of about 1.4 points per 100
against about 17 points per 100 of noise in a season-sized sample of that lineup's possessions.
Fit is real and it is on the order of 1% of the variance you are trying to predict.

**New rosters: directionally yes, not established.** Predicting a team's next-season net rating
from prior-season information only (209 team-seasons, leave-one-season-out):

| model | R2 | corr |
|---|---|---|
| additive player value | 0.445 | 0.667 |
| + roster structure (redundancy, roles, coverage) | 0.466 | 0.683 |
| + complementarity term | **0.474** | **0.688** |

Whole-season team samples are ~8,000 possessions, so noise averages down and the small fit signal
becomes visible where it was invisible at lineup level. But bootstrapping over team-seasons, the
improvement from the complementarity term has a 95% interval of [-0.25, +1.62] in mean squared
error, i.e. **P(genuinely better) = 0.93, with zero inside the interval**. Suggestive, not proven;
it needs more seasons to settle. Worth noting that an earlier run of this same test on 90
team-seasons came out *negative*, which is itself a lesson about how much data these questions
need.

Beware the in-sample version of this table: structure metrics correlate strongly with team results
when the player and pair effects were fitted on the same possessions (complementarity r = 0.37,
effective roles r = -0.54 for 2025-26). That is circular, and it is how such metrics usually get
published.

**Player moves: no.** For 739 players who changed teams between consecutive seasons, the change in
predicted fit between old and new roster does not predict the change in their measured impact
(t = 0.68, controlling for regression to the mean, which is itself strong at -0.85).

---

## 7. Method notes worth keeping

- **Signed force-directed layout does not work for roster topology.** Positive-attract /
  negative-repel springs have no equilibrium length, so every roster collapses onto a line. MDS on
  synergy-derived target distances is stable and produced the 30 team figures in `figs/`.
- **Random split-half reliability overstates the interaction surface**, because the same
  team-seasons sit on both sides of the split. Season-disjoint splits are the honest test
  (r = 0.32 rather than 0.41).
- **Fit the interaction surface jointly, not in two stages.** Regressing estimated pair
  coefficients onto skill interactions throws away most of the signal, because those coefficients
  are about 94% noise.
- **In-sample correlation between roster-structure metrics and team results is meaningless** when
  the player and pair effects were estimated on the same possessions.
- **Calibrate on cross-fit predictions.** Calibrating on training lineups whose outcomes the model
  already saw produces a slope near 1 and out-of-sample R2 below zero; the honest cross-fit slope
  is about 0.4 to 0.6, meaning raw predictions must be shrunk by roughly half.

---

## 8. Bottom line for roster construction

1. **Buy talent.** Individual value has about 7x the spread of fit and predicts next season far
   better than anything structural measured here.
2. **Fit is real but small, and it is a property of skill types, not of particular people.** Model
   it as a league-wide regularity over skill space; do not trust a number attached to one duo.
3. **The one fit effect large enough to plan around is redundancy at the extremes:** two
   traditional centres, or two ball-dominant creators. Both are penalised, and both are visible
   from skill profiles alone, before the two players have ever shared a floor, which is exactly
   where a counterfactual tool is useful.
4. **The famous rim-pressure-and-spacing pairing is not detectable as an interaction.** Its value
   is already priced into what each player is individually worth.
5. **Judge fit at roster scale, not lineup scale.** Fit is invisible against single-lineup noise
   and only starts to show over a full team-season. Any tool that ranks five-man units by measured
   chemistry is ranking noise.
6. **Never feed raw pair net ratings into a projection.** It was the worst of the six models
   tested, below the additive baseline on every split.

---

## Failed approaches, recorded

| approach | outcome |
|---|---|
| per-pair adjusted synergy coefficients | split-half r = 0.06; abandoned as a per-pair estimate |
| two-stage bilinear (pair coefficients, then M) | 3x smaller spread than the joint fit; superseded |
| raw pair net rating as a fit signal | replicates well but is player quality, not fit |
| 18 hand-built fit heuristics | 17 null, 1 survives |
| signed force-directed topology layout | degenerate, collapses to a line; replaced with MDS |
| interaction terms for future-lineup prediction | tie the additive baseline; only beat their own control |
| fit change as a predictor of improvement after a move | null, t = 0.68 over 739 moves |
| per-game stint data (opponent and home/away adjusted) | abandoned: API rate limits made a 3,700-game pull impractical, so lineup observations are season totals |
