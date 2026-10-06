# Predicting wins before anyone plays, and what has actually changed about rosters

Backtested on **746 team-seasons, 25 target seasons (2001-02 to 2025-26)**, with actual win totals
as the target. Charts in `viz/wins/`, tables in `out/preseason_*_long.csv` and
`out/league_trends.csv`.

**How far back the data goes.** 1990 is not reachable: NBA play-by-play begins in 1996-97, and
every lineup, possession and shot-zone quantity here is derived from it. Before that the API
returns zero rows. Within what exists:

| quantity | earliest season | why |
|---|---|---|
| player totals with shot zones | 1996-97 | full play-by-play coverage |
| game results (wins) | 2000-01 | earlier seasons return no game list |
| 5-man lineup data | 1996-97 in principle | but only served per team per season, and the rate limit puts a full backfill at ~15 hours |

So the win backtest runs 2001-02 onward on a box-score value metric, and the trend analysis runs
on all 30 seasons. Lineup-derived quantities (pair synergy, the interaction surface) stay on the
2018-2026 window.

**The larger sample changed three of the earlier conclusions.** They are flagged as REVISED below.

---

## 1. Can you project wins before tip-off? Yes, but the earlier margin was overstated

Leave-one-season-out over 25 seasons. Player value comes from a box-score model calibrated against
the lineup-ridge value; minutes are projected from prior seasons, never realised.

| model | features | MAE (wins) | RMSE | R² |
|---|---|---|---|---|
| **last season + talent + continuity** | 5 | **6.91** | 8.72 | **0.49** |
| last season + talent + continuity + structure | 9 | 6.94 | 8.75 | 0.49 |
| last season + talent | 4 | 7.16 | 9.04 | 0.46 |
| talent + continuity | 4 | 7.46 | 9.30 | 0.42 |
| talent + continuity + structure | 8 | 7.47 | 9.30 | 0.42 |
| last season's wins | 1 | 7.81 | 9.83 | 0.36 |
| talent only | 3 | 7.86 | 9.78 | 0.36 |
| league average | 0 | 10.11 | 12.24 | 0.00 |

**REVISED.** On six seasons, talent alone beat last season's record by 0.90 wins. On twenty-five it
does not beat it at all (-0.05, 95% CI [-0.48, +0.37], P(better) = 0.41). The small-sample result
was optimistic.

What survives, and is now on much firmer ground, is that **roster information and last season's
record are complements, not substitutes**. Combining them gives +0.89 wins of MAE over the record
alone (95% CI [+0.59, +1.19], P(better) = 1.00), and the best model reaches R² = 0.49.

**Box-score value is not the bottleneck.** Head-to-head on the six seasons where both exist, the
projection built on the box-score metric scores MAE 7.26 against 7.22 for the one built on real
lineup-ridge value. Backfilling exact lineup data would buy almost nothing.

**Predictability is stable across eras**, with the present looking slightly harder:

| era | n | model MAE | last-season MAE |
|---|---|---|---|
| 2001-07 | 202 | 6.98 | 8.49 |
| 2008-14 | 206 | 6.84 | 8.46 |
| 2015-21 | 210 | 6.59 | 8.24 |
| 2022-26 | 120 | 7.51 | 9.02 |

---

## 2. What makes good teams good? Talent and continuity; structure still adds nothing

Correlations with next-season wins over 738 team-seasons:

| feature | r |
|---|---|
| last season's wins | +0.60 |
| **depth (value of players 4 through 9)** | **+0.58** |
| minutes-weighted player value | +0.56 |
| star value (top 3) | +0.47 |
| continuity | +0.40 |
| role scarcity | +0.31 |
| redundancy | -0.11 |
| effective roles | +0.03 |
| coverage | -0.01 |

**Depth still edges out star power** (+0.58 vs +0.47), now on four times the sample. This is the
most robust non-obvious result in the project.

**Roster structure is dead, and the bigger sample kills it more cleanly.** Comparing each
structural model against its own control, i.e. the identical model with the structure block
removed:

| model | gain over its own control | 95% CI | P(better) |
|---|---|---|---|
| talent + continuity + structure | **-0.01** | [-0.08, +0.07] | 0.39 |
| last season + talent + continuity + structure | **-0.02** | [-0.07, +0.03] | 0.21 |

Redundancy, effective roles, coverage and scarcity add nothing to a win projection, and the
confidence intervals are now tight enough to say that is a real zero rather than an underpowered
test.

---

## 3. Where the model fails, and why

The seven worst misses over 25 seasons:

| season | team | projected | actual | miss | what happened |
|---|---|---|---|---|---|
| 2007-08 | BOS | 36.9 | 66 | **+29.1** | Garnett and Allen arrive |
| 2024-25 | NOP | 49.6 | 21 | -28.6 | injuries |
| 2007-08 | MIA | 41.1 | 15 | -26.1 | Wade injured, roster collapse |
| 2011-12 | CHA | 34.6 | 8.7 | -25.9 | teardown |
| 2010-11 | CLE | 44.2 | 19 | -25.2 | LeBron leaves |
| 2025-26 | IND | 44.1 | 19 | -25.1 | injuries |
| 2010-11 | CHI | 37.7 | 62 | **+24.3** | Rose leap, Thibodeau arrives |

The failure modes are consistent across three decades and all sit outside the model's inputs:

- **Injuries.** A roster is projected healthy and then loses its best player for the year
  (NOP 2024-25, MIA 2007-08, IND 2025-26). Unforecastable from prior-season data.
- **Superstar arrival or departure**, where the model's minutes prior is anchored to the old
  roster (BOS 2007-08, CLE 2010-11).
- **Young-player leaps and coaching changes** (CHI 2010-11, SAS 2025-26). The model has **no age
  curve**, so it treats a 21-year-old's prior season as its best guess for his next one, which
  systematically underrates ascending teams and overrates ageing ones.

Adding player age remains the single highest-value improvement available, and it is now clear from
25 seasons rather than inferred from 6.

---

## 4. Honest caveat on the projection setup

The one thing taken from the target season is **roster membership**: which players are listed with
which team. That is what a preseason projection legitimately knows, but the implementation gets it
from the data itself, so a player acquired at the February deadline appears in the membership set,
and a player who misses the entire season with an injury does not appear at all. The second case
flatters the continuity feature, which is computed as the share of prior minutes retained.

Measured sensitivity on the six-season sample: replacing continuity with a coarse binary (did the
team retain more than 60% of its minutes) drops its gain from +0.41 wins to +0.15, so roughly a
third to a half of continuity's edge is likely this artifact. On 25 seasons continuity is still
worth +0.30 wins on top of last season's record plus talent, so it is not purely artifact, but the
clean version of this number needs opening-night rosters, which this data source does not carry.

---

## 5. Roster trends, 1996-97 to 2025-26

Spearman correlation against season order, n = 30. **This is where the longer window changes the
story most: three conclusions from the eight-season version were artifacts of a short window.**

**What changed, and it is dramatic:**

| | 1996-97 | 2025-26 | rho | p |
|---|---|---|---|---|
| 3-point attempt rate | 0.207 | 0.414 | **+0.97** | <0.001 |
| minutes share of the top 3 | 0.442 | 0.351 | **-0.95** | <0.001 |
| assisted 3-pointers | 0.662 | 0.839 | **+0.93** | <0.001 |
| rotation size | 9.41 | 11.07 | **+0.90** | <0.001 |
| true shooting | 0.533 | 0.582 | **+0.89** | <0.001 |
| long midrange rate | 0.211 | 0.067 | **-0.89** | <0.001 |
| league-wide spread of playing styles | 2.502 | 2.642 | **+0.70** | <0.001 |
| redundancy (local crowding) | -3.497 | -3.195 | +0.68 | <0.001 |
| within-roster skill distance | 6.041 | 6.102 | +0.55 | 0.002 |
| shots at the rim | 0.384 | 0.298 | -0.39 | 0.032 |
| star value gap | 2.74 | 2.36 | -0.17 | 0.376 |
| effective roles per roster | 3.05 | 3.16 | -0.12 | 0.511 |

**The 3-point rate doubled** (20.7% to 41.4%) and **the long midrange lost two thirds of its share**
(21.1% to 6.7%). True shooting rose almost five points. None of that is news, but the 30-year
window shows the midrange peaked around 2001-03 and then collapsed, rather than declining steadily.

Three REVISIONS of the eight-season findings:

1. **Rotations have genuinely expanded**, from 9.4 to 11.1 players (rho +0.90). The eight-season
   window said "no trend" (rho -0.17) because the whole move happened before 2018.
2. **The league has become MORE stylistically diverse, not less** (rho +0.70). The short window said
   flat. Teams converged on a shot profile while the *players* spread further apart in skill space.
3. **Star minute concentration fell steadily and substantially**, from 44.2% to 35.1% of minutes
   going to a team's top three (rho -0.95). The eight-season version caught only the tail of this.

The rim tells a more interesting story over 30 years than over 8: rim frequency fell sharply in the
late 1990s, recovered through the mid-2000s, and has fallen again since 2015. Over the full window
the trend is weak (rho -0.39) even though the last decade looks monotone.

**What genuinely has not changed:** the number of effective roles per roster (rho -0.12) and the
gap between star and median player value (rho -0.17). Teams use more players, spread wider in
style, and distribute minutes more evenly, but the *number of distinct jobs* on a roster has been
constant for thirty years.

---

## 6. Practical summary

1. **Combine talent with last year's record, do not choose between them.** Together they reach
   MAE 6.91 and R² 0.49; the record alone gives 7.81, and talent alone 7.86.
2. **Weight depth at least as heavily as stars.** Players four through nine correlate +0.58 with
   next season's wins against +0.47 for the top three, over 738 team-seasons.
3. **Do not pay for roster structure in a projection.** Redundancy, coverage, complementarity and
   blob shape all fail to improve it, and adding them together makes it worse.
4. **The biggest available gain is an age curve**, not a better fit model: the residuals are
   dominated by injuries (unforecastable here) and young-player leaps (forecastable, and currently
   not modelled at all).
5. **Roster construction has evolved in some ways and not others.** Rotations got deeper, minutes
   got flatter and players spread further apart in style; the number of distinct roles per roster
   did not move at all. Beware short windows: three trend conclusions flipped when the sample went
   from 8 seasons to 30.
