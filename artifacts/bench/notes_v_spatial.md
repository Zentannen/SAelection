# v_spatial: spatial residual pull on top of hybrid_cf

Code: `src/saelection/bench/m_vspatial.py`. Live check: `artifacts/bench/live_v_spatial.csv`
(snapshot `IEC_LGE2026_GP_20261105_1518.csv` pinned, all 29 tasks, 7 methods). Benchmark:
`artifacts/bench/results_v_spatial.csv` (both cases, all orders, `--n-random 3`; 6 methods incl.
`hybrid_cf`; runs were split by method / case and concatenated, rows are identical to one
harness run). `vd_ward_iec_2026.csv` exists, so uncounted VDs carry the IEC ward.

## Bottom line

* **Best: `m_vspatial:spatial_rake_half`.** Province and municipalities exactly as
  `spatial_pull` (hybrid_cf + spatial kriging; province MAE 0.877 -> 0.871), ward / VD pattern
  from a ward-free hybrid_cf + ward / two-scale spatial kriging (`spatial_nw`), blended 50/50
  in log-ratio and raked to the municipal totals. Benchmark: ward MAE 2.389 -> 2.325, ward Brier
  0.099 -> 0.087, VD MAE 3.628 -> 3.555, CRPS 0.872 -> 0.854, muni cov90 0.847 -> 0.894 (prov
  cov90 0.944 -> 0.969, a little wide). Live check: group 1.638 -> 1.605, muni 2.048 -> 2.035,
  random 0.520 -> 0.471, time 0.122 -> **0.148 (worse)**; VD MAE better on group / random / time,
  group cov90 0.892 -> 0.922.
* **Safe alternative: `m_vspatial:spatial_pull`** (hybrid_cf unchanged + kriged shift, hybrid_cf's
  draws re-centred): better than hybrid_cf on every live kind (1.615 / 2.035 / 0.482 / 0.115)
  and every benchmark accuracy metric (prov 0.871, muni 1.294, ward 2.375), CRPS 0.872 -> 0.878,
  but its ward-level gain is 5x smaller.
* **The user's flagged wards do not move (< 1.5 pp) under any variant, and the data say they
  should not.** The pull is driven by what the counted VDs show *relative to their own
  history*, not by their level. The counted VDs in those wards are predicted well from their
  own history (e.g. 33000499: LGE2021 37.6% ActionSA, now 48.5%, hybrid_cf in-sample 47.8%,
  out-of-fold residual -0.69 on the log-ratio scale, i.e. it did *worse* than its history
  implies), so their residuals are ~0 or negative. Pulling uncounted VDs toward ward-mates'
  *level* is much worse: on the live hold-outs (held VDs with >= 1 counted ward-mate) ActionSA
  VD MAE is 4.39 (hybrid_cf) vs 7.41 (ward-mates' level) vs 5.25 (50/50) on `time`, 4.38 / 7.41 /
  4.98, 3.35 / 5.81 / 3.98, 3.94 / 6.22 / 4.37 on the three random tasks; corr(actual - hybrid_cf,
  mates' level - hybrid_cf) is only 0.11-0.24. Two of the wards (79700061, 79900057) have no
  counted VD in the model's VD set at all: their counted VDs (33000523, 32980620) are new 2026
  stations outside the 2026 VD boundary set (a pipeline issue: map them to a VD).
* What the residuals do show: hybrid_cf's out-of-fold residuals are spatially correlated
  (ActionSA: correlation 0.5-0.75 below 1 km, 0.2 at 1-2 km, 0.1 at 2-4 km, ~0 beyond 8 km;
  MK similar plus a long-range part). With hybrid_cf's ward EB level kept, the *same-ward*
  covariance of cross-fitted residuals is zero or negative (hybrid_cf's ward effect already
  uses the ward-mates), but neighbours across ward boundaries still correlate (touching,
  different ward: ActionSA 0.20, MK 0.26). spatial_pull uses exactly that remainder.

## Method (details in the module docstring)

1. Base: hybrid_cf point projection, either as is (`base='full'`) or without its ward level in
   both parts (`base='nw'`).
2. Out-of-fold residuals of counted VDs: 5 random VD folds, base refitted without the fold,
   r = alr(Y) - alr(base_oof) per party (log-ratio vs ABSTAIN). No VD's own result enters the
   prediction its residual is measured against.
3. Covariance of r by moments from **cross-fold pairs only** (r_i comes from a base that saw
   r_j's VD, as an uncounted VD's base sees its counted neighbours; this removes the double
   counting of signal the base's ward / cell / municipality effects already use):
   Cov = tw [same ward] + sum_c ts_c k_c(d), k_c = exponential x Wendland taper (hard cap R_c,
   positive definite); ranges by grid search pooled over parties, tw, ts >= 0 by weighted
   NNLS; nugget s2 = residual variance - shared part; residuals winsorised at 3 SD.
   spatial_pull: one component, R = 5 km (fitted ell = 4 km, tw = 0 for every party; ActionSA
   ts = 0.024 vs nugget 0.197 on the full live case: a counted neighbour 1 km away gets weight
   ~0.1). spatial_nw: R = 3 km (ell 1.5) + R = 15 km (ell 8); ActionSA ts = 0.029 + 0.066,
   nugget 0.124, ward tw ~ 0.
4. Shift = simple kriging (BLUP / EB posterior mean) of the field at every uncounted VD, on the
   log-ratio scale, then back to counts.
5. Draws (S = 200): `spatial_pull` = hybrid_cf's draws re-centred on the pulled mean;
   `spatial_nw` = base draws without the independent VD residual + a joint draw of the kriging
   posterior of the field (conditional simulation with a cached Cholesky of each kernel over
   all VDs; ward N(0, tw)) + resampled VD nugget. `spatial_rake_half` uses spatial_nw's draws
   re-centred on the raked mean. (`spatial_pull_ff` = spatial_pull + the field posterior on
   top: the extra log-scale noise shifts aggregate shares (Jensen), live random group cov90 1.00 ->
   0.94; not used.)
6. Raking (`_raked`): VD pattern = w x alr(spatial_nw) + (1 - w) x alr(spatial_pull) (w = 0.5 in
   the best, 1 in `spatial_rake`), then IPF so that every municipality x category total of the
   uncounted VDs equals spatial_pull's and every VD sums to its electorate.

## Tuning sweep (live check, point projections, group_mae_pp mean by task kind; vd_ = VD MAE)

`full` = hybrid_cf base (F = 10 here), `nw` = ward-free base; R = taper cap (km), `2s` =
3 km + 15 km components, g = gain on the shift, `incf` = all pairs instead of cross-fold,
`adapt` = local-spacing (Paciorek) kernel, `nw0` = ward-free base without pull.

| method      |   group |   muni |   random |   time |   vd_group |   vd_random |   vd_time |
|:------------|--------:|-------:|---------:|-------:|-----------:|------------:|----------:|
| hcf         |   1.638 |  2.048 |    0.520 |  0.122 |      3.598 |       3.367 |     3.330 |
| full        |   1.616 |  2.035 |    0.490 |  0.116 |      3.596 |       3.352 |     3.325 |
| full_F5     |   1.615 |  2.035 |    0.482 |  0.115 |      3.595 |       3.348 |     3.317 |
| full_R3     |   1.625 |  2.034 |    0.487 |  0.114 |      3.601 |       3.357 |     3.326 |
| full_R8     |   1.607 |  2.036 |    0.504 |  0.124 |      3.589 |       3.350 |     3.302 |
| full_2s     |   1.596 |  2.034 |    0.492 |  0.131 |      3.582 |       3.352 |     3.276 |
| full_g15    |   1.611 |  2.035 |    0.489 |  0.118 |      3.594 |       3.364 |     3.326 |
| full_g2     |   1.606 |  2.035 |    0.498 |  0.118 |      3.595 |       3.389 |     3.341 |
| full_noward |   1.616 |  2.035 |    0.490 |  0.116 |      3.596 |       3.352 |     3.325 |
| nw0         |   1.680 |  2.023 |    0.538 |  0.209 |      3.676 |       3.440 |     3.476 |
| nw          |   1.664 |  2.024 |    0.474 |  0.161 |      3.631 |       3.344 |     3.312 |
| nw_2s       |   1.618 |  2.029 |    0.471 |  0.173 |      3.590 |       3.326 |     3.239 |
| nw_R3       |   1.668 |  2.022 |    0.490 |  0.158 |      3.633 |       3.359 |     3.319 |
| nw_R8       |   1.644 |  2.029 |    0.473 |  0.166 |      3.612 |       3.326 |     3.276 |
| nw_noward   |   1.665 |  2.024 |    0.455 |  0.174 |      3.645 |       3.353 |     3.348 |
| nw_adapt    |   1.663 |  2.025 |    0.476 |  0.156 |      3.633 |       3.357 |     3.309 |
| nw_incf     |   1.661 |  2.024 |    0.478 |  0.150 |      3.626 |       3.351 |     3.303 |
| nw_g05      |   1.671 |  2.023 |    0.491 |  0.193 |      3.646 |       3.333 |     3.368 |
| nw_g15      |   1.655 |  2.024 |    0.525 |  0.119 |      3.632 |       3.455 |     3.325 |

Reading: on the hybrid_cf base the pull helps everywhere but is small (the EB ward level
already does most of the local work; tw fits to 0). Removing the ward level costs 0.04 / 0.09
(group / time) and the kriging only partly gives it back in the aggregate, while it gives the
best VD-level accuracy (vd_time 3.24). Gains are flat in R (3-8 km) and gain 1-1.5.

## Live check (pinned 15:18 snapshot, all 29 tasks; `live_check.summarise`)

| method / kind |   group_mae |   vd_mae |   vd_leader_acc |   group_cov90 |   vd_cov90 |   s |
|:--|--:|--:|--:|--:|--:|--:|
| hybrid_cf group         | 1.638 | 3.598 | 0.845 | 0.892 | 0.873 | 0.6 |
| hybrid_cf muni          | 2.048 | 4.434 | 0.801 | 0.896 | 0.848 | 0.5 |
| hybrid_cf random        | 0.520 | 3.367 | 0.831 | 1.000 | 0.895 | 0.6 |
| hybrid_cf time          | 0.122 | 3.330 | 0.868 | 1.000 | 0.902 | 0.7 |
| **spatial_rake_half** group  | 1.605 | 3.584 | 0.855 | 0.922 | 0.885 | 5.6 |
| **spatial_rake_half** muni   | 2.035 | 4.442 | 0.800 | 0.917 | 0.872 | 5.1 |
| **spatial_rake_half** random | 0.471 | 3.310 | 0.831 | 0.944 | 0.912 | 5.5 |
| **spatial_rake_half** time   | 0.148 | 3.270 | 0.868 | 1.000 | 0.904 | 8.4 |
| spatial_rake group      | 1.613 | 3.585 | 0.859 | 0.931 | 0.883 | 5.0 |
| spatial_rake muni       | 2.041 | 4.441 | 0.801 | 0.917 | 0.872 | 4.8 |
| spatial_rake random     | 0.465 | 3.331 | 0.825 | 0.944 | 0.914 | 5.6 |
| spatial_rake time       | 0.182 | 3.262 | 0.856 | 1.000 | 0.910 | 8.6 |
| spatial_pull group      | 1.615 | 3.595 | 0.846 | 0.892 | 0.871 | 1.6 |
| spatial_pull muni       | 2.035 | 4.432 | 0.800 | 0.896 | 0.846 | 1.4 |
| spatial_pull random     | 0.482 | 3.348 | 0.827 | 1.000 | 0.898 | 1.2 |
| spatial_pull time       | 0.115 | 3.317 | 0.868 | 1.000 | 0.899 | 3.6 |
| spatial_pull_ff group   | 1.615 | 3.595 | 0.846 | 0.892 | 0.881 | 1.5 |
| spatial_pull_ff random  | 0.482 | 3.348 | 0.827 | 0.944 | 0.911 | 1.7 |
| spatial_2s group        | 1.598 | 3.581 | 0.852 | 0.892 | 0.872 | 0.9 |
| spatial_2s muni         | 2.037 | 4.433 | 0.800 | 0.896 | 0.848 | 0.9 |
| spatial_2s random       | 0.483 | 3.347 | 0.831 | 1.000 | 0.898 | 0.7 |
| spatial_2s time         | 0.120 | 3.277 | 0.868 | 1.000 | 0.904 | 0.6 |
| spatial_nw group        | 1.628 | 3.591 | 0.862 | 0.941 | 0.882 | 2.2 |
| spatial_nw muni         | 2.030 | 4.433 | 0.799 | 0.917 | 0.873 | 2.2 |
| spatial_nw random       | 0.465 | 3.325 | 0.820 | 0.944 | 0.912 | 2.5 |
| spatial_nw time         | 0.169 | 3.248 | 0.856 | 1.000 | 0.912 | 5.2 |

Mean |ActionSA error| of the held-out aggregate (pp), group / muni / random / time:
hybrid_cf 1.741 / 4.328 / 0.343 / 0.109; spatial_rake_half 1.643 / 4.343 / 0.303 / 0.196;
spatial_pull 1.717 / 4.335 / 0.382 / 0.112. spatial_pull beats hybrid_cf on 23 of 29 tasks
(worst: group:mixed:TSH +0.04). The `time` regression of the raked variants is ActionSA -0.20
and DA +0.26 pp on the 167 late VDs (raking fixes municipal totals over all ~2000 uncounted VDs,
not over this subset).

## Benchmark (`harness.summarise`, both cases, all orders, n_random 3)

| method            | prov_mae | prov_max | muni_mae | ward_acc | ward_mae | vd_mae | prov_cov90 | prov_crps | muni_cov90 | ward_brier |    s |
|:------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| **spatial_rake_half** | 0.871 | 1.952 | 1.294 | 0.952 | 2.325 | 3.555 | 0.969 | 0.854 | 0.894 | 0.087 | 8.5 |
| spatial_rake      | 0.871 | 1.952 | 1.294 | 0.952 | 2.311 | 3.541 | 0.969 | 0.854 | 0.894 | 0.086 | 8.3 |
| spatial_pull      | 0.871 | 1.952 | 1.294 | 0.951 | 2.375 | 3.611 | 0.941 | 0.878 | 0.847 | 0.099 | 7.0 |
| spatial_2s        | 0.873 | 1.957 | 1.290 | 0.951 | 2.361 | 3.597 | 0.938 | 0.885 | 0.848 | 0.099 | 6.1 |
| hybrid_cf         | 0.877 | 1.968 | 1.298 | 0.951 | 2.389 | 3.628 | 0.944 | 0.872 | 0.847 | 0.099 | 6.5 |
| spatial_nw        | 0.909 | 2.086 | 1.302 | 0.951 | 2.320 | 3.554 | 0.957 | 0.897 | 0.889 | 0.088 | 7.4 |

Random + real2026 orders only:

| method            | prov_mae | muni_mae | ward_mae | vd_mae | prov_cov90 | prov_crps | muni_cov90 | ward_brier |
|:------------------|------:|------:|------:|------:|------:|------:|------:|------:|
| **spatial_rake_half** | 0.242 | 0.738 | 1.693 | 2.991 | 1.000 | 0.342 | 0.910 | 0.057 |
| spatial_rake      | 0.242 | 0.738 | 1.676 | 2.972 | 1.000 | 0.342 | 0.910 | 0.056 |
| spatial_pull      | 0.242 | 0.738 | 1.749 | 3.052 | 0.967 | 0.344 | 0.852 | 0.063 |
| hybrid_cf         | 0.251 | 0.743 | 1.768 | 3.076 | 0.967 | 0.335 | 0.851 | 0.064 |
| spatial_nw        | 0.283 | 0.732 | 1.680 | 2.982 | 0.989 | 0.395 | 0.907 | 0.058 |

By case (prov / muni / ward MAE / CRPS / ward Brier): LGE2021 hybrid_cf 1.013 / 1.404 / 2.558 /
0.891 / 0.113, rake_half 1.010 / 1.407 / 2.487 / 0.822 / 0.097; PE2024 hybrid_cf 0.742 / 1.192 /
2.220 / 0.852 / 0.085, rake_half 0.732 / 1.181 / 2.164 / 0.885 / 0.076. Ward MAE improves in every
order (real2026 1.618 -> 1.544, small_first 2.486 -> 2.401, ANC-first 3.082 -> 3.043, DA-first
3.928 -> 3.903); ward Brier most at 5-10% counted (0.149 -> 0.122, 0.122 -> 0.105). Failure
cases are hybrid_cf's: DA / ANC strongholds first (prov 2.28 / 1.77), which a local pull cannot
fix (no counted neighbours of the other kind). spatial_nw's own province error (0.909) comes
mostly from the ward-free base (in the aggregate hybrid_cf's in-sample ward EB is better than
kriging); raking removes it.

## Flagged wards (full live case at 15:18; ActionSA % of valid votes)

Ward = its VDs in the case, counted at actual + uncounted projected, plus counted VDs outside
the VD set at actual (as live.py does). [90% interval from draws.]

| ward | VDs in case (counted) | counted VDs' share | hybrid_cf | spatial_rake_half | spatial_pull | spatial_nw |
|---|---|---|---|---|---|---|
| 79700053 | 8 (1) | 48.5 | 29.3 [20.2-40.2] | 29.3 [20.8-39.1] | 29.2 | 29.8 |
| 79700061 | 5 (0; 1 counted outside VD set) | 42.6 | 18.7 [12.2-26.5] | 17.9 [11.6-26.3] | 18.8 | 17.3 |
| 79900057 | 3 (0; 1 outside VD set) | 49.9 | 25.5 [17.6-37.0] | 25.0 [17.2-35.1] | 25.5 | 24.6 |
| 79700024 | 4 (1) | 33.6 | 16.2 [11.2-21.5] | 15.8 [11.0-21.9] | 16.1 | 15.7 |
| 74801006 | 2 (1) | 29.3 | 6.0 [5.0-7.1] | 6.0 [4.8-8.2] | 6.0 | 6.0 |
| 79700099 | 7 (2) | 23.7 | 24.9 [18.4-32.7] | 24.1 [16.8-32.3] | 24.7 | 23.8 |

Per VD (79700053): the counted 33000499 had LGE2021 37.6% (ward-mates 2.4-7.5% except 33000488
23.7%, projected 50%); 79700024's counted 33010591 had 19.3% (its look-alike 33010861, 19.3%,
is projected 33%; 33010614 has 0.03% in PE2024 -> 0.8%); 74801006's other VD (4278 voters) had 0
ActionSA votes in both PE2024 and LGE2021 (not stood) -> 1.4%; 79700099's counted 33020604 polled
12.6% vs 7.8% projected (residual +0.53) and 33020817 45.5% vs 47.0% (-0.15): they cancel.

## Run time

Live case (2797 VDs, 773 counted), machine at load 8-11: spatial_pull 1.2-3.6 s per projection
with 200 draws (hybrid_cf 0.5-0.7 s; 5 out-of-fold refits of hybrid_cf's point projection
~0.3 s each + kriging); spatial_rake_half 5-8 s (two out-of-fold sets, conditional simulation,
one Cholesky of a 2797 x 2797 kernel per range, cached per case). Benchmark 6-9 s per projection.

## Risks / caveats

* Gains are modest and measured on one snapshot (29 overlapping tasks) and two backtests; the
  live `time` task (the most realistic one) is worse for the raked variants (0.148 vs 0.122;
  spatial_pull 0.115). If that matters more than ward accuracy, use `spatial_pull`.
* w = 0.5 in `spatial_rake_half` was set after seeing `spatial_rake` (w = 1) on the live check
  (one extra look, not a grid search); benchmark difference w = 0.5 vs 1 is small.
* Province intervals of the raked variants are a little wide (prov cov90 0.969, random +
  real2026 1.00); municipal intervals are better calibrated than hybrid_cf's (0.894 vs 0.847).
* The covariance is fitted from counted pairs; at 5% counted there are few close pairs and the
  fit falls back towards no pull (by design). Wards whose only counted VD is outside the VD set
  get nothing from that VD (pipeline fix needed). Centroid distance ignores barriers (highways,
  mine dumps) between townships and suburbs; the cross-fold covariance shrinks the pull where
  neighbours disagree, but a group hold-out shows the pull can go the wrong way across area
  types (group:mixed:TSH +0.04).
* The user's expectation (ward-mates near the counted VD's level) is contradicted by the live
  data; the remaining ActionSA gap in those wards is a baseline / history question (see
  notes_v_asa.md), not a missing neighbour pull.
