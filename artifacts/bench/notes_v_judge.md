# v_judge: choosing the live method after the ActionSA ward complaint

Judge's re-run of the three variants (v_ward, v_asa, v_spatial) and of their combination
(`src/saelection/bench/m_vbest.py`). Live check pinned to `IEC_LGE2026_GP_20261105_1518.csv`
(9 snapshots, 773 counted VDs in the model's VD set, all 29 tasks, IEC ward list
`data/processed/vd_ward_iec_2026.csv`, md5 efcdc2f2...). Benchmark: both cases, all orders,
`n_random 3`. Files: `artifacts/bench/live_v_judge.csv` (11 methods x 29 tasks) and
`artifacts/bench/results_v_judge.csv` (hybrid_cf, hybrid_asa, spatial_rake_half, asa_spatial_pull2, asa_rake_half, asa_rake_half2, 54 rows each).

## Verdict

**Winner: `m_vbest:asa_rake_half`.** It is v_asa's ActionSA baseline fix (`hybrid_asa`: low-base
floor 0.3 between LGE2021 and PE2024, plus a district level) with v_spatial's local pull
(`spatial_rake_half`: kriging of out-of-fold residuals, with ward and two-scale spatial terms,
raked to the pulled full model's municipal totals) on top. The two fixes act on different
things (municipal and area-type level vs. within-municipality pattern), and their gains add up:

* Live check against `hybrid_cf`: group 1.638 -> 1.568, random 0.520 -> 0.445, time 0.122 -> 0.111,
  muni 2.048 -> 1.888. VD MAE is better on every task kind. Group cov90 is 0.902 (hybrid_cf
  0.892), muni cov90 0.896, VD cov90 0.87-0.91.
* Benchmark, random + real2026 orders: province MAE 0.242 (hybrid_cf 0.251), muni 0.720 (0.743),
  ward 1.690 (1.768), VD 2.990 (3.076), CRPS 0.326 (0.335), ward Brier 0.057 (0.064). Over all orders it
  is also the best of the six benchmarked methods on every accuracy metric. On CRPS it ties
  asa_rake_half2 (0.840 vs 0.839).
* ActionSA error of the held-out group aggregate (pp), group / muni / random / time:
  hybrid_cf 1.74 / 4.33 / 0.34 / 0.11, asa_rake_half 1.47 / 4.05 / 0.17 / 0.05.

`asa_rake_half2` (district after municipality) ties on the live check (muni cov90 0.917) but
loses on the benchmark (LGE2021 muni MAE 1.387 vs 1.312): fitted after the municipality, the
district level absorbs almost nothing. So the district stays first, as in hybrid_asa.

**Rule checks.** No candidate worsens live group MAE against hybrid_cf, and all are better on
it. The winner's lowest coverage is 0.86 (muni VD cov90), above the 0.8 floor.

**What it does not do: it does not "fix" the flagged wards, and it should not.** In the six
flagged wards it moves ActionSA by 0.4-1.9 pp, all downwards (table below). The counted VDs
in those wards were already the wards' ActionSA strongholds in LGE2021. Copying their share
to the uncounted ward-mates is 1.7x worse than the model on live hold-outs, and a stronger
ward-mate pull than the fitted one does not help either (section "The user's diagnosis").
Two of the six wards (79700061, 79900057) have no counted VD in the model at all: their only
reported VD is outside the model's 2026 VD set, and only `live.py` can fix that.

## 1. Leakage check

Code review of `m_vward.py`, `m_vasa.py`, `m_vspatial.py` (and `m_swing.py`, which they all call),
plus two empirical tests (scratch scripts `leak.py`, `oofcheck.py`):

* Every read of `case.Y` in the four modules is `case.Y[counted]` or `case.Y[counted, :K]`
  (m_swing 433/622/628/758/845; m_vward 110; m_vasa 214, which only `_live_gamma` uses and the
  winner does not; m_vspatial 180/304). m_swing 1003 reads `case.Y[donors]` with
  `donors = counted & ~broken`, and only `hybrid_live` uses it.
* **Scramble test** (live task random_s0): `case.Y` of every uncounted VD is replaced with
  random integers, and each method is re-run on a copy of the case. Every module cache is
  keyed by `id(case)`, so the copy gets fresh caches. The mean and all 200 draws are
  bit-identical (max diff 0) for hybrid_cf, hybrid_asa, spatial_pull, spatial_rake_half,
  ward_pull, asa_spatial_pull, asa_rake_half and asa_ward_pull.
* **Own-result test for the cross-fitted residuals.** One counted VD's result is changed
  (x3, +500 ActionSA) and the out-of-fold residuals are recomputed. The base prediction its
  residual is measured against does not move (change 0), while other VDs' out-of-fold
  predictions do (up to 0.2-0.66 on the alr scale), so the test has power. This holds for
  m_vspatial (random VD folds), m_vward (ward-blocked folds) and m_vbest (m_vspatial folds on
  the hybrid_asa base).
* **Design points, none of them a leak.**
  * m_vspatial fits its covariance on cross-fold pairs only. With base = full hybrid_cf, a
    counted VD's out-of-fold prediction has seen its ward-mates in other folds, just as an
    uncounted VD's base has. So the kriged pull does not re-count what the base's ward EB
    level already uses.
  * m_vward uses ward-blocked folds and a ward-free base.
  * m_vasa's re-basing uses only previous elections, municipality and contiguity.
  * m_vasa replaces `m_swing._groups_for` at import with a version that adds a 'dist'
    grouping and delegates everything else unchanged. hybrid_cf gives identical numbers with
    or without that patch.
  * m_vbest swaps `m_vspatial._base_mean` and `_base_draws` inside a context manager and
    restores them afterwards (checked).
  * `m_vbest.asa_hybrid` reproduces `m_vasa.hybrid_asa` exactly (means and draws, diff 0).
* One harness-level caveat that applies to every method equally: in `live_check`, a held-out VD's
  `registered` is the IEC electorate from the results file, `max(registered, valid + spoilt)`.
  This is pre-count information except for the rare VD with more ballots than voters.

## 2. Live check (pinned 15:18, 29 tasks, mean over each task kind)

group_mae / vd_mae / vd_leader_acc / group_cov90 / vd_cov90, seconds per projection on a
machine at load 7-10.

| method | group | muni | random | time |
|---|---|---|---|---|
| m_swing:hybrid_cf (live now) | 1.638 / 3.598 / .845 / .892 / .873 | 2.048 / 4.434 / .801 / .896 / .848 | 0.520 / 3.367 / .831 / 1.00 / .895 | 0.122 / 3.330 / .868 / 1.00 / .902 |
| m_vward:ward_pull | 1.607 / 3.575 / .864 / .931 / .907 | 2.041 / 4.432 / .802 / .917 / .884 | 0.521 / 3.375 / .833 / 1.00 / .926 | 0.195 / 3.247 / .862 / 1.00 / .943 |
| m_vasa:hybrid_asa | 1.599 / 3.573 / .846 / .873 / .865 | 1.904 / 4.351 / .812 / .854 / .840 | 0.478 / 3.326 / .827 / 1.00 / .890 | 0.098 / 3.260 / .856 / 1.00 / .912 |
| m_vasa:vasa_floor_dist2 | 1.599 / 3.574 / .846 / .863 / .865 | 1.908 / 4.346 / .812 / .917 / .842 | 0.478 / 3.325 / .827 / 1.00 / .890 | 0.098 / 3.258 / .856 / 1.00 / .911 |
| m_vspatial:spatial_pull | 1.615 / 3.595 / .846 / .892 / .871 | 2.035 / 4.432 / .800 / .896 / .846 | 0.482 / 3.348 / .827 / 1.00 / .898 | 0.115 / 3.317 / .868 / 1.00 / .899 |
| m_vspatial:spatial_rake_half | 1.605 / 3.584 / .855 / .922 / .885 | 2.035 / 4.442 / .800 / .917 / .872 | 0.471 / 3.310 / .831 / .944 / .912 | 0.148 / 3.270 / .868 / 1.00 / .904 |
| m_vbest:asa_spatial_pull | 1.577 / 3.565 / .848 / .863 / .863 | 1.889 / 4.351 / .812 / .854 / .840 | 0.445 / 3.306 / .833 / 1.00 / .894 | 0.092 / 3.237 / .862 / 1.00 / .915 |
| m_vbest:asa_spatial_pull2 | 1.577 / 3.565 / .848 / .863 / .865 | 1.889 / 4.351 / .812 / .896 / .844 | 0.445 / 3.306 / .833 / 1.00 / .894 | 0.092 / 3.237 / .862 / 1.00 / .910 |
| **m_vbest:asa_rake_half** | **1.568** / 3.545 / .853 / .902 / .886 | **1.888** / 4.357 / .814 / .896 / .865 | **0.445** / 3.272 / .833 / 1.00 / .906 | 0.111 / 3.185 / .862 / 1.00 / .906 |
| m_vbest:asa_rake_half2 | 1.567 / 3.548 / .854 / .892 / .885 | 1.890 / 4.351 / .811 / .917 / .864 | 0.445 / 3.271 / .833 / 1.00 / .908 | 0.111 / 3.184 / .862 / 1.00 / .911 |
| m_vbest:asa_ward_pull | 1.550 / 3.530 / .858 / .922 / .905 | 1.907 / 4.344 / .813 / .875 / .881 | 0.498 / 3.303 / .831 / 1.00 / .933 | 0.168 / 3.152 / .862 / 1.00 / .940 |

Seconds per projection: hybrid_cf 1.5-4, hybrid_asa 1.5-3.7, spatial_pull 4-7, asa_spatial_pull2
2.4-3.1, spatial_rake_half 7.5-13, asa_rake_half 8-15, ward_pull and asa_ward_pull 7-19.

My re-runs reproduce every number the three agents reported for their own best methods and for
hybrid_cf (the v_ward agent's "28 tasks" is 29 in its CSV too).

Mean |ActionSA error| of the held-out group aggregate, pp (group / muni / random / time):

| hybrid_cf | spatial_pull | spatial_rake_half | ward_pull | hybrid_asa | asa_spatial_pull2 | asa_rake_half | asa_ward_pull |
|---|---|---|---|---|---|---|---|
| 1.74 / 4.33 / 0.34 / 0.11 | 1.72 / 4.34 / 0.38 / 0.11 | 1.64 / 4.34 / 0.30 / 0.20 | 1.61 / 4.30 / 0.24 / 0.31 | 1.59 / 4.02 / 0.18 / 0.03 | 1.57 / 4.04 / 0.18 / 0.01 | 1.47 / 4.05 / 0.17 / 0.05 | 1.34 / 3.99 / 0.15 / 0.14 |

**How consistent the gains are.** Out of 29 tasks, these methods beat hybrid_cf's group MAE on:

| method | tasks better than hybrid_cf |
|---|---|
| spatial_pull | 23 |
| spatial_rake_half | 22 |
| ward_pull | 20 |
| asa_rake_half | 18 |
| hybrid_asa | 16 |
| asa_spatial_pull2 | 16 |

The pull is a small, steady gain. The ActionSA re-basing (floor + district) is a large but
uneven gain. Per-task change in group MAE, asa_rake_half vs hybrid_cf, runs from -0.79 to
+0.68 pp:

* Better: muni:GT485 -0.79, GT481 -0.75, other:JHB -0.72, GT484 -0.39, township:JHB -0.34.
* Worse: muni:GT423 +0.68 (ActionSA error +2.5 -> +4.1), GT422 +0.25, township:GT423 +0.25,
  suburb:EKU +0.24, muni:EKU +0.21.

asa_rake_half inherits that spread: mean -0.093 pp, median -0.043, largest loss +0.68.

## 3. Benchmark (harness, both cases, all orders, n_random 3)

All orders. Columns: prov_mae / prov_max / muni_mae / ward_acc / ward_mae / vd_mae /
vd_turnout / prov_cov90 / prov_crps / muni_cov90 / ward_brier / seconds.

| method | prov | max | muni | ward_acc | ward | vd | turnout | cov90 | crps | muni_cov | brier | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_cf | 0.877 | 1.968 | 1.298 | 0.951 | 2.389 | 3.628 | 5.475 | 0.944 | 0.872 | 0.847 | 0.099 | 8.2 |
| hybrid_asa | 0.875 | 1.967 | 1.254 | 0.951 | 2.365 | 3.614 | 5.468 | 0.938 | 0.845 | 0.846 | 0.093 | 8.2 |
| spatial_rake_half | 0.871 | 1.952 | 1.294 | 0.952 | 2.325 | 3.555 | 5.385 | 0.969 | 0.854 | 0.894 | 0.087 | 10.9 |
| asa_spatial_pull2 | 0.872 | 1.952 | 1.280 | 0.951 | 2.371 | 3.610 | 5.465 | 0.938 | 0.850 | 0.848 | 0.094 | 10.0 |
| asa_rake_half2 | 0.872 | 1.952 | 1.280 | 0.952 | 2.322 | 3.553 | 5.381 | 0.969 | 0.839 | 0.895 | 0.087 | 11.1 |
| **asa_rake_half** | **0.868** | 1.952 | **1.251** | **0.953** | **2.302** | **3.541** | **5.380** | 0.972 | 0.840 | 0.893 | **0.087** | 12.5 |

Random + real2026 orders only. Columns: prov_mae / prov_max / muni_mae / ward_acc / ward_mae /
vd_mae / prov_cov90 / prov_crps / muni_cov90 / ward_brier.

| method | prov | max | muni | ward_acc | ward | vd | cov90 | crps | muni_cov | brier |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_cf | 0.251 | 0.558 | 0.743 | 0.963 | 1.768 | 3.076 | 0.967 | 0.335 | 0.851 | 0.064 |
| hybrid_asa | 0.252 | 0.555 | 0.725 | 0.963 | 1.764 | 3.075 | 0.972 | 0.329 | 0.850 | 0.063 |
| spatial_rake_half | 0.242 | 0.526 | 0.738 | 0.965 | 1.693 | 2.991 | 1.000 | 0.342 | 0.910 | 0.057 |
| asa_spatial_pull2 | 0.243 | 0.524 | 0.721 | 0.964 | 1.746 | 3.053 | 0.967 | 0.342 | 0.852 | 0.062 |
| asa_rake_half2 | 0.243 | 0.524 | 0.721 | 0.965 | 1.691 | 2.990 | 1.000 | 0.332 | 0.911 | 0.057 |
| **asa_rake_half** | **0.242** | **0.523** | **0.720** | **0.965** | **1.690** | **2.990** | 1.000 | **0.326** | 0.907 | **0.057** |

By case, prov / muni / ward MAE / prov cov90 / CRPS:

| method | LGE2021 | PE2024 |
|---|---|---|
| hybrid_cf | 1.013 / 1.404 / 2.558 / 0.938 / 0.891 | 0.742 / 1.192 / 2.220 / 0.951 / 0.852 |
| hybrid_asa | 1.010 / 1.307 / 2.516 / 0.920 / 0.886 | 0.739 / 1.201 / 2.213 / 0.957 / 0.805 |
| spatial_rake_half | 1.010 / 1.407 / 2.487 / 0.975 / 0.822 | 0.732 / 1.181 / 2.164 / 0.963 / 0.885 |
| asa_rake_half | 1.007 / 1.312 / 2.445 / 0.981 / 0.809 | 0.729 / 1.189 / 2.158 / 0.963 / 0.871 |

Ward MAE by order, hybrid_cf -> asa_rake_half; it is better in every order:

| order | hybrid_cf | asa_rake_half |
|---|---|---|
| random (3 seeds) | 1.83 / 1.79 / 1.80 | 1.77 / 1.71 / 1.71 |
| real2026 | 1.618 | 1.537 |
| small_first | 2.486 | 2.388 |
| DA_strongholds_first | 3.928 | 3.797 |
| ANC_strongholds_first | 3.082 | 3.013 |

Ward Brier is 0.142-0.145 vs 0.152-0.163 under the stronghold-first orders.

The hybrid_cf, hybrid_asa and spatial_rake_half rows reproduce the agents' results files
exactly. The benchmark tests the district level and the pull, not the ActionSA floor: in both
cases the floor is a no-op.

## 4. Flagged wards (full live case at 15:18; ActionSA % of valid votes)

Ward share = (the ward's VDs in the case: counted at actual, uncounted projected + counted VDs
outside the model's VD set at actual) / valid. [5-95% from the draws.]

| ward | VDs in case (counted) | counted share | hybrid_cf | asa_rake_half (winner) | asa_rake_half2 | spatial_rake_half | ward_pull | hybrid_asa |
|---|---|---|---|---|---|---|---|---|
| 79700053 | 8 (1) | 48.5 | 29.3 [20.2-40.2] | 27.8 [19.9-36.8] | 27.8 [19.9-37.1] | 29.3 | 29.3 | 28.1 |
| 79700061 | 5 (0; 1 outside set) | 42.6 | 18.7 [12.2-26.5] | 17.0 [11.2-24.6] | 17.0 [11.4-24.7] | 17.9 | 16.9 | 17.6 |
| 79900057 | 3 (0; 1 outside set) | 49.9 | 25.5 [17.6-37.0] | 24.0 [15.9-34.3] | 24.0 [15.7-34.2] | 25.0 | 24.4 | 24.4 |
| 79700024 | 4 (1) | 33.6 | 16.2 [11.2-21.5] | 15.8 [11.5-20.8] | 15.8 [11.5-21.2] | 15.8 | 15.4 | 16.2 |
| 74801006 | 2 (1) | 29.3 | 6.0 [5.0-7.1] | 5.4 [4.4-6.5] | 5.4 [4.5-6.5] | 6.0 | 6.0 | 5.4 |
| 79700099 | 7 (2) | 23.7 | 24.9 [18.4-32.7] | 23.0 [17.7-30.4] | 23.0 [17.6-29.7] | 24.1 | 23.8 | 23.4 |

Province ActionSA over the model's VD set: 18.4% under every method (18.39 for the winner).

Per VD, history -> forecast (hybrid_cf / winner):

* **79700053.** The counted 33000499 had ActionSA 37.6% in LGE2021 and polled 48.5%. Its
  uncounted ward-mates had 2.4-7.5% in LGE2021 and are forecast 8.8-31.0% (hybrid_cf) and 8.5-27.2% (winner). The exception is
  33000488 (23.7% in 2021), forecast 49.3 / 46.0%.
* **79700024.** The counted 33010591 had 19.3% and polled 33.6%. Its look-alike 33010861
  (19.3%) is forecast 33.8 / 31.7%. Two DA strongholds (DA 80-81% in PE2024, ActionSA
  0.0-2.0%) are forecast 0.7-6.6%.
* **74801006.** ActionSA did not stand in this municipality in 2021. The uncounted VD
  33250384 (1823 valid votes in 2021) had 0.0% ActionSA in PE2024 and is forecast 1.3 / 0.8%.
  Its counted neighbour went from 6.1% (PE2024) to 29.3%.

Reading: the wards look under-forecast because the reporting order put each ward's ActionSA
stronghold first. The forecasts move with each VD's own history.

## 5. The user's diagnosis, tested

Scratch scripts `wardmate.py` and `wardmate2/3.py`, pooled over the live hold-outs random_s0-2
and time. The test set is the 473 held VDs that have at least one counted ward-mate.

* **Copying the counted ward-mates' share** gives ActionSA VD MAE 5.8-7.4 pp, against
  3.4-4.4 pp for hybrid_cf. Over all parties it is 6.2-6.7 pp against 3.1-3.6.
* **Partial pull**, `model + lam x (ward-mates' share - model)`, ActionSA VD MAE:

  | lam | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 1 |
  |---|---|---|---|---|---|---|
  | hybrid_cf | 4.01 | 3.95 | 4.01 | 4.16 | 4.63 | 6.70 |
  | asa_rake_half (winner) | 3.73 | 3.72 | 3.81 | 3.99 | 4.48 | 6.70 |

  A one-sided pull (only upwards, lam 0.1-0.3) gives 3.72-3.85 on the winner (identical for asa_rake_half2).
  So the winner already holds the useful part of the ward-mate signal.
* **What the user sees is real but small.** Where the counted ward-mates are at least 10 pp
  above the forecast (53 VDs, average gap 17.8 pp), the model under-forecasts ActionSA by
  2.7 pp (hybrid_cf) and 3.1 pp (winner). So about 15% of the gap is real. Where the
  ward-mates are 10 pp or more below the forecast (42 VDs), the winner's bias is 0.0 pp. That residual
  is noise at VD level, so no global pull strength captures it without losing elsewhere.

## 6. Combinations implemented (`src/saelection/bench/m_vbest.py`)

All of these import from the variants' modules; nothing is copied.

* **Re-based case.** `asa_case` = `m_vasa._rebased_case(floor 0.3)` + `_add_district`.
* **Base models.** `m_vward._hybrid_base` provides both hybrid_asa and its no-ward version,
  through configurable pooling levels.
* **Kriging, raking and draws.** `_asa_base()` points `m_vspatial._base_mean` and
  `_base_draws` at those base models for the length of one call. `m_vspatial._spatial` and
  `_raked` then run unchanged.
* **Methods.**
  * `asa_spatial_pull(2)`: hybrid_asa + 5 km kriging. Draws are hybrid_asa's, re-centred.
    The "2" versions put the district level after the municipality (vasa_floor_dist2's draw
    order).
  * `asa_rake_half(2)`: spatial_rake_half on the hybrid_asa base.
  * `asa_ward_pull`: m_vward's ward_pull on the hybrid_asa base.
  * `asa_hybrid`: a check that the parts reproduce hybrid_asa.

## 7. Risks and recommendations

* **Tuning risk.** v_asa chose the ActionSA floor (0.3) and the district level on this same
  snapshot, out of 24 variants. The benchmark cannot test the floor, because ActionSA is never
  in both predictor elections there; it only tests the district level and the pull. Its
  per-task effect on the live check is uneven. The Lesedi (GT423) municipality hold-out loses
  0.68 pp, with ActionSA over-forecast by 4.1 pp. Emfuleni (GT421, 4 counted VDs) moves from
  5.8% to 7.7% ActionSA on little data of its own. Recommended: re-run this live check on the next
  snapshot before relying on the floor. I kept to the pinned snapshot here.
* **Province intervals are wide on the benchmark.** prov_cov90 is 0.97 overall and 1.00 on
  random + real2026, inherited from spatial_rake_half's kriging-posterior draws. Live group
  cov90 (0.90) and VD cov90 (0.87-0.91) are on target.
* **Narrow intervals off a zero base.** In 74801006, a VD with a 0.0% ActionSA base gets a
  4.4-6.5% ward interval although its neighbour went from 6% to 29%. The log-ratio draws
  barely move a near-zero base. This is a known weakness of every candidate; v_asa's neighbour
  floor (`vasa_fnb_dist`) lifts such VDs but was worse on the muni hold-outs.
* **Pipeline fix outside this bake-off.** 35 counted VDs at 15:18 are not in the model's VD
  set. `live.py` adds their votes to the ward totals without modelling them, so they cannot
  inform their uncounted ward-mates (79700061, 79900057). Map new stations to a 2026 VD, or
  feed them in as counted pseudo-VDs, in `live.py`.
* **Run time.** About 8-15 s per projection at load 8-10, against 1.5-4 s for hybrid_cf. That
  comes from 5 out-of-fold refits for each of two bases plus conditional simulation. It is
  fine for 15-minute snapshots. If speed matters, `asa_spatial_pull2` (2.5-3 s) keeps most of
  the accuracy, but its group cov90 is 0.863 and its ward gain is smaller.
* **Do not use** the unraked pulls (`ward_sp`, `spatial_nw`) for a new party: they are biased
  in the LGE2021 backtest.
