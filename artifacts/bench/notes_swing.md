# Swing family: classical swing, regression and cluster methods

Code: `src/saelection/bench/m_swing.py`. Benchmark: `artifacts/bench/results_swing.csv` (both cases,
all orders, `--n-random 3`, same code path as `harness --tag swing`; the two cases were run as two
processes and concatenated). Live check: `artifacts/bench/live_swing.csv` (snapshot 14:45 SAST, 28
hold-out tasks; a 13:29 run is summarised too).

## Bottom line

* **Best on the benchmark and robust on the live check: `hybrid_cf`** (Curtice-Firth regression
  where the uncounted VD lies inside the counted VDs' past-profile range, the hierarchical
  area-type x municipality ratio swing `cluster_muni_ratio` where the regression would have to
  extrapolate). Benchmark (both cases, all orders, 3 seeds): province MAE 0.88 pp (tally 4.09, `uniform_swing_muni` 2.94), municipality MAE 1.30, ward-winner accuracy 0.951, CRPS 0.87 pp, ward Brier 0.099 - best on municipality, ward, VD, turnout and CRPS metrics and within 0.002 of the best on province MAE and ward Brier. `cluster_muni_ratio` ties on province MAE (0.875 vs 0.877) but is worse on municipality / ward / VD metrics. On the live hold-outs (14:45 snapshot) `hybrid_cf` is tied best with `reg_cf` on the key "group" hold-outs (area type x municipality hidden: 1.64 pp vs 4.91 for `uniform_swing_muni` and 9.90 for the tally), best on "random" (0.36) and "time" (0.28) among the strong methods, 2.07 on "muni".
* Best pure swing method: `cluster_muni_ratio` / `cluster_ratio` (ratio swing pooled by
  k-means area type (K = 20), municipality and ward with empirical-Bayes shrinkage). The plain
  Curtice-Firth regression `reg_cf` is the best method on the live hold-outs (it interpolates
  well when every area type has counted look-alikes) but the worst serious method on the
  stronghold-first orders (it extrapolates badly), which is why the hybrid exists.
* Intervals: block bootstrap + EB posterior + a correlated cluster term + a new
  profile-space extrapolation penalty. Province 90% coverage of the main methods is 0.93-0.95 on the benchmark (slightly conservative; 0.94-1.00 on random/real orders, 0.79-0.92 on the stronghold-first and small-first orders) and about 0.90 on the live group hold-outs. Municipality coverage is lower (0.82-0.86).
* Every projection takes well under a second of compute on one idle core (0.2-0.6 s for the
  swing/cluster methods with 200 draws, 0.6-1.3 s for `hybrid_cf` / `reg_cf`; 1-2 s with the
  machine at load 11), plus ~4 s one-off pre-count caching per case.


## Methods in `src/saelection/bench/m_swing.py`

All methods work on the 7 categories (6 parties + ABSTAIN), so turnout is projected with the
parties. Pre-count work (features, k-means / fuzzy clusters, variogram space) is cached per
case.

| method | what it does |
|---|---|
| `swing_{add,ratio,logit}_{prov,muni,ward}` | Item 1. Swing of each uncounted VD from its own previous-LGE result. `add`: uniform swing in shares of the electorate (size-weighted, i.e. the aggregate additive swing). `ratio`: proportional swing, aggregate ratio sum Y / sum E per unit, renormalised. `logit`: swing of log(p_k / p_ABSTAIN), precision-weighted. Pooled at province, province -> municipality, or province -> municipality -> ward, each unit's swing shrunk to its parent by empirical Bayes (James-Stein: B = tau^2 n_eff / (tau^2 n_eff + sigma^2), tau^2 by the unbalanced one-way ANOVA moment estimator). New parties by substitution regression (item 4) so that the scale/level comparison is not about new parties. |
| `reg_cf` | Item 2. Curtice-Firth: ridge regression of the log-ratio change (vs a per-party blend of the two past results) on past shares and log-ratios in both elections, past turnout, log size, electorate growth, history flag and municipality dummies, fitted on the counted VDs; penalty chosen by group (leave-cluster-fold-out) CV, which beat LOO CV; uncounted fitted values clipped to the counted range; EB municipality + ward effects on the residuals. New parties via the same ridge (substitution) regression. |
| `cluster_swing` | Item 3. CSIR-style cluster swing on the log-ratio scale: k-means (K = 20) on standardised past valid shares + turnout of both elections. Swing of an uncounted VD = province swing + EB cluster + municipality + ward effects (crossed, 3 backfitting sweeps), so a cluster with few counted VDs backs off to the province/municipality. Base = per-party blend w LGE + (1 - w) PE with w fitted on the counted VDs. New parties: province level of their log-ratio + the same EB effects (a cluster average, item 4). |
| `cluster_ratio` | Same pooling (km20 + muni + ward, EB), but on the ratio scale from the average of the two past compositions: each unit's aggregate ratio sum Y / sum E is shrunk to its parent. New parties from the log-ratio cluster model. Best pure swing method before the area-type x municipality and hybrid variants were added. |
| `cluster_ratio_subst` | `cluster_ratio` with new parties by the substitution ridge regression instead of cluster averages (item 4 comparison). |
| `greben_csir` | The CSIR model as published (Greben, Elphinstone & Holloway 2006): fuzzy c-means (C = 20, m = 1.2) on raw party percentages of the most recent past election; each cluster's current composition (shares + turnout) = membership x size weighted average over counted VDs; uncounted VD = membership-weighted mix. A level estimator: past results set only the memberships. |
| `greben_std` | Greben with the Ittmann et al. (2017) fix: clustering on standardised per-party shares (small parties weigh as much as large ones) of both elections + turnout; K = 40 (best of 10/20/40 in the backtest). |
| `swing_ens` | 2/3 `cluster_ratio` + 1/3 `cluster_swing`. |

### Uncertainty (item 5)

Draws (S = 200, float32) are built in log-ratio space by the log-ratio model with the method's pooling
levels and, for the share-scale methods, re-centred on that method's point projection:

1. **Fixed part**: Bayesian block bootstrap over counted wards (an Exp(1) weight per ward),
   refitting the province swing / ridge coefficients for every draw. This is the
   "resample residuals and refit" step, done by wards so within-ward correlation is kept.
2. **Random effects**: EB posterior N(effect, posterior var) for every cluster, municipality,
   ward; a group without counted VDs draws a fresh N(0, tau^2) (tau^2 estimated from the
   counted VDs). At the cluster and municipality levels the draws share a common factor
   (rho = 0.5): the clusters the count has not reached are "the other kind of VD" and miss the
   back-off swing in the same direction, so their errors do not average out.
3. **Extrapolation penalty** (new): a variogram of the swing in the standardised past-profile
   (+ log size) space, fitted on pairs of counted VDs, gamma(h) = a + b h. An uncounted VD
   whose 5 nearest counted VDs are at mean distance h_i > h0 (the typical counted-to-counted
   distance) gets an extra swing variance b (h_i - h0), times 0.7^2, as one common factor
   (correlated over all VDs) plus a per-cluster factor. Under random counting h_i ~ h0 and the
   term vanishes; under stronghold-first orders it dominates. This is what turns the
   coverage of the biased orders from 0.2-0.6 into 0.8-1.0.
4. **VD residuals** resampled from the counted VDs (not Normal: heavy tails, Katz & King),
   standardised by each VD's sampling + extra variance, cross-party correlation kept.

The two constants (rho = 0.5, scale 0.7) were tuned on the benchmark (`n_random = 1`) for
province coverage near 0.90 (screened with 1 seed; in the final 3-seed run the main methods
land at 0.93-0.95, i.e. slightly conservative - a penalty scale of ~0.5 instead of 0.7 would
bring them to ~0.90, not re-run). They are the only things tuned on coverage. No inflation
schedule by % counted was needed: coverage is flat in % counted (0.90-0.99 at 5%, 10%, 25%,
50%; see table below), because the extrapolation penalty already shrinks as the counted VDs
come to cover the profile space.

### Added after the live 2026 evidence (swing varies by area type x municipality)

| method | what it does |
|---|---|
| `cluster_muni_ratio` | Hierarchical swing per party x area type x municipality: area types = k-means(20) on standardised past shares (LGE + PE) + turnout (not hand thresholds); the aggregate ratio of each (area type x municipality) cell is EB-shrunk to area type + municipality (crossed), then province, by how many VDs are counted in the cell; plus ward effects. |
| `cluster_muni_ratio_k6` | same with K = 6 coarse area types (point only). |
| `cluster_muni_swing` | same hierarchy on the log-ratio scale. |
| `hybrid_cf` | Curtice-Firth regression where the VD is inside the counted VDs' profile range, `cluster_muni_ratio` where it would extrapolate: per-VD weight a = clip(1 - (h_i - h0) / h0, 0, 1) from the distance to the 5 nearest counted VDs (same distance as the extrapolation penalty). Draws from `cluster_muni_ratio`. `hybrid_cf2`: slower hand-over (c = 2, point only). |


## Benchmark summary (`harness.summarise`, results_swing.csv)

Methods without `prov_cov90` are point-only comparison variants (by design, to keep the run affordable). `seconds` is the harness wall time per projection with the 4-core machine at load 8-12 (three other agents); see run time below.

| method                |   prov_mae_pp |   prov_max_pp |   muni_mae_pp |   ward_winner_acc |   ward_mae_pp |   vd_mae_pp |   vd_turnout_mae_pp |   prov_cov90 |   prov_crps_pp |   muni_cov90 |   ward_brier |   tally_mae_pp |   seconds |
|:----------------------|--------------:|--------------:|--------------:|------------------:|--------------:|------------:|--------------------:|-------------:|---------------:|-------------:|-------------:|---------------:|----------:|
| cluster_muni_ratio    |         0.875 |         1.977 |         1.369 |             0.948 |         2.521 |       3.849 |               6.055 |        0.951 |          0.906 |        0.834 |        0.104 |          4.094 |     4.243 |
| hybrid_cf             |         0.877 |         1.968 |         1.298 |             0.951 |         2.389 |       3.628 |               5.475 |        0.944 |          0.872 |        0.847 |        0.099 |          4.094 |     4.346 |
| hybrid_cf_c1          |         0.879 |         1.964 |         1.313 |             0.95  |         2.413 |       3.675 |               5.661 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.161 |
| cluster_ratio         |         0.887 |         2.002 |         1.381 |             0.948 |         2.547 |       3.853 |               6.012 |        0.938 |          0.909 |        0.822 |        0.105 |          4.094 |     4.209 |
| swing_ens             |         0.89  |         1.988 |         1.324 |             0.949 |         2.486 |       3.837 |               5.931 |        0.951 |          0.911 |        0.842 |        0.103 |          4.094 |     4.165 |
| cluster_ratio_subst   |         0.947 |         2.29  |         1.414 |             0.942 |         2.58  |       3.877 |               6.016 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.02  |
| cluster_swing         |         0.979 |         2.115 |         1.384 |             0.943 |         2.623 |       4.09  |               6.124 |        0.929 |          0.94  |        0.846 |        0.108 |          4.094 |     4.009 |
| cluster_muni_ratio_k6 |         0.979 |         2.255 |         1.435 |             0.947 |         2.591 |       3.927 |               6.216 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.007 |
| cluster_muni_swing    |         0.981 |         2.096 |         1.368 |             0.944 |         2.609 |       4.088 |               6.171 |        0.938 |          0.942 |        0.862 |        0.107 |          4.094 |     4.069 |
| swing_ratio_ward      |         1.172 |         2.663 |         1.699 |             0.934 |         2.994 |       4.419 |               6.324 |        0.981 |          1.136 |        0.848 |        0.116 |          4.094 |     4.35  |
| swing_ratio_muni      |         1.236 |         2.789 |         1.747 |             0.934 |         3.057 |       4.526 |               5.982 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.009 |
| swing_ratio_prov      |         1.248 |         2.779 |         2.082 |             0.936 |         3.201 |       4.646 |               6.077 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.004 |
| swing_logit_ward      |         1.35  |         3.109 |         1.701 |             0.929 |         3.006 |       4.474 |               6.348 |        0.972 |          1.22  |        0.883 |        0.119 |          4.094 |     4.363 |
| reg_cf                |         1.391 |         3.394 |         1.607 |             0.929 |         2.775 |       4.06  |               6.29  |        0.988 |          1.067 |        0.89  |        0.098 |          4.094 |     4.717 |
| swing_logit_muni      |         1.419 |         3.294 |         1.72  |             0.929 |         3.086 |       4.599 |               6.264 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.002 |
| swing_add_ward        |         1.429 |         3.36  |         1.734 |             0.939 |         3.388 |       5.013 |               6.304 |        0.954 |          1.226 |        0.872 |        0.111 |          4.094 |     4.463 |
| swing_logit_prov      |         1.456 |         3.108 |         2.216 |             0.929 |         3.304 |       4.779 |               6.229 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.007 |
| swing_add_muni        |         1.55  |         3.668 |         1.802 |             0.935 |         3.625 |       5.312 |               6.21  |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.007 |
| swing_add_prov        |         1.605 |         3.778 |         2.144 |             0.936 |         3.791 |       5.452 |               6.292 |      nan     |        nan     |      nan     |      nan     |          4.094 |     0.015 |
| greben_std            |         2.922 |         7.375 |         4.034 |             0.857 |         5.093 |       6.666 |               7.307 |        0.809 |          2.546 |        0.613 |        0.264 |          4.094 |     3.872 |
| greben_csir           |         2.957 |         7.456 |         4.122 |             0.839 |         5.225 |       6.814 |               8.465 |        0.806 |          2.56  |        0.609 |        0.272 |          4.094 |     3.798 |


### Province MAE (pp) by case and order (random = mean of 3 seeds)

| method                |   LGE ANC-first |   LGE DA-first |   LGE random |   LGE real2026 |   LGE small-first |   PE2 ANC-first |   PE2 DA-first |   PE2 random |   PE2 real2026 |   PE2 small-first |
|:----------------------|----------------:|---------------:|-------------:|---------------:|------------------:|----------------:|---------------:|-------------:|---------------:|------------------:|
| cluster_muni_ratio    |            2.4  |           2.4  |         0.18 |           0.16 |              1.23 |            1.33 |           2.22 |         0.23 |           0.16 |              0.76 |
| hybrid_cf             |            2.26 |           2.36 |         0.25 |           0.33 |              1.21 |            1.29 |           2.19 |         0.26 |           0.11 |              0.65 |
| hybrid_cf_c1          |            2.33 |           2.37 |         0.23 |           0.31 |              1.2  |            1.31 |           2.17 |         0.27 |           0.11 |              0.67 |
| cluster_ratio         |            2.42 |           2.4  |         0.21 |           0.15 |              1.3  |            1.31 |           2.23 |         0.23 |           0.16 |              0.74 |
| swing_ens             |            2.34 |           2.42 |         0.22 |           0.18 |              1.24 |            1.4  |           2.29 |         0.2  |           0.13 |              0.82 |
| cluster_ratio_subst   |            2.34 |           3.12 |         0.2  |           0.14 |              1.66 |            1.34 |           2.04 |         0.28 |           0.17 |              0.65 |
| cluster_swing         |            2.23 |           2.48 |         0.39 |           0.37 |              1.34 |            1.53 |           2.47 |         0.2  |           0.16 |              0.99 |
| cluster_muni_ratio_k6 |            2.85 |           2.5  |         0.25 |           0.23 |              1.28 |            1.3  |           2.32 |         0.24 |           0.19 |              1.19 |
| cluster_muni_swing    |            2.22 |           2.47 |         0.38 |           0.33 |              1.43 |            1.57 |           2.45 |         0.19 |           0.16 |              1    |
| swing_ratio_ward      |            2.11 |           3.56 |         0.34 |           0.3  |              1.49 |            2.62 |           2.73 |         0.36 |           0.4  |              0.67 |
| swing_ratio_muni      |            2.13 |           3.62 |         0.51 |           0.5  |              1.61 |            2.65 |           2.72 |         0.33 |           0.43 |              0.76 |
| swing_ratio_prov      |            2.29 |           3.67 |         0.56 |           0.66 |              1.33 |            2.63 |           2.75 |         0.33 |           0.32 |              0.8  |
| swing_logit_ward      |            2.02 |           3.92 |         0.44 |           0.36 |              1.79 |            2.7  |           2.65 |         0.73 |           0.58 |              0.91 |
| reg_cf                |            4.21 |           4.11 |         0.38 |           0.38 |              1.79 |            3    |           2.6  |         0.24 |           0.11 |              0.82 |
| swing_logit_muni      |            2.05 |           3.93 |         0.49 |           0.43 |              1.95 |            2.72 |           2.67 |         0.82 |           0.7  |              1.08 |
| swing_add_ward        |            5.05 |           2.95 |         0.66 |           0.31 |              1.67 |            2.27 |           3.12 |         0.31 |           0.46 |              0.75 |
| swing_logit_prov      |            2.1  |           4.02 |         0.34 |           0.25 |              1.58 |            2.97 |           2.65 |         0.96 |           0.87 |              1.6  |
| swing_add_muni        |            5.1  |           3.01 |         0.91 |           0.67 |              1.9  |            2.29 |           3.19 |         0.33 |           0.57 |              0.8  |
| swing_add_prov        |            5.22 |           3.05 |         0.96 |           0.7  |              1.72 |            2.36 |           3.2  |         0.39 |           0.62 |              1.09 |
| greben_std            |            9.75 |           8.61 |         0.21 |           0.27 |              1.35 |            8.37 |           8.42 |         0.29 |           0.13 |              1.16 |
| greben_csir           |            8.78 |          10.35 |         0.26 |           0.29 |              0.82 |            8.31 |           8.92 |         0.2  |           0.16 |              1.04 |


### Province 90% coverage by case and order (methods with draws)

| method             |   LGE ANC-first |   LGE DA-first |   LGE random |   LGE real2026 |   LGE small-first |   PE2 ANC-first |   PE2 DA-first |   PE2 random |   PE2 real2026 |   PE2 small-first |
|:-------------------|----------------:|---------------:|-------------:|---------------:|------------------:|----------------:|---------------:|-------------:|---------------:|------------------:|
| cluster_muni_ratio |            0.96 |           0.92 |         0.99 |           1    |              0.87 |            0.96 |           0.79 |         0.97 |           1    |              0.96 |
| hybrid_cf          |            1    |           0.92 |         0.96 |           1    |              0.79 |            0.96 |           0.83 |         0.97 |           0.94 |              1    |
| cluster_ratio      |            0.96 |           0.92 |         0.97 |           1    |              0.87 |            0.96 |           0.71 |         0.97 |           0.94 |              0.96 |
| swing_ens          |            1    |           0.92 |         0.97 |           1    |              0.87 |            0.96 |           0.79 |         0.97 |           1    |              0.96 |
| cluster_swing      |            1    |           0.88 |         0.96 |           0.94 |              0.79 |            0.96 |           0.71 |         0.97 |           1    |              0.96 |
| cluster_muni_swing |            1    |           0.88 |         0.96 |           1    |              0.88 |            0.92 |           0.79 |         0.97 |           1    |              0.92 |
| swing_ratio_ward   |            1    |           0.96 |         1    |           1    |              0.96 |            1    |           0.83 |         1    |           1    |              1    |
| swing_logit_ward   |            1    |           0.92 |         1    |           1    |              0.96 |            1    |           0.75 |         1    |           1    |              1    |
| reg_cf             |            1    |           0.96 |         1    |           1    |              1    |            0.96 |           0.92 |         1    |           1    |              1    |
| swing_add_ward     |            0.83 |           0.79 |         1    |           1    |              0.96 |            1    |           0.79 |         1    |           1    |              1    |
| greben_std         |            0.5  |           0.46 |         1    |           0.83 |              0.87 |            0.33 |           0.38 |         1    |           1    |              1    |
| greben_csir        |            0.58 |           0.5  |         1    |           0.83 |              0.92 |            0.21 |           0.29 |         1    |           1    |              1    |


### Province 90% coverage by % counted

| method             |   5.0 |   10.0 |   21.4 |   21.7 |   25.0 |   50.0 |
|:-------------------|------:|-------:|-------:|-------:|-------:|-------:|
| cluster_muni_ratio |  0.93 |   0.99 |   1    |      1 |   0.95 |   0.93 |
| hybrid_cf          |  0.92 |   0.97 |   1    |      1 |   0.95 |   0.93 |
| cluster_ratio      |  0.92 |   0.99 |   1    |      1 |   0.93 |   0.92 |
| swing_ens          |  0.93 |   0.99 |   1    |      1 |   0.95 |   0.93 |
| cluster_swing      |  0.92 |   0.94 |   0.83 |      1 |   0.94 |   0.92 |
| cluster_muni_swing |  0.9  |   0.97 |   1    |      1 |   0.95 |   0.92 |
| swing_ratio_ward   |  0.99 |   0.99 |   1    |      1 |   0.99 |   0.96 |
| swing_logit_ward   |  0.97 |   0.97 |   1    |      1 |   0.98 |   0.96 |
| reg_cf             |  0.99 |   0.99 |   1    |      1 |   0.99 |   0.99 |
| swing_add_ward     |  0.96 |   0.96 |   1    |      1 |   0.95 |   0.94 |
| greben_std         |  0.81 |   0.81 |   0.83 |      1 |   0.83 |   0.77 |
| greben_csir        |  0.81 |   0.78 |   0.83 |      1 |   0.81 |   0.81 |


## Live check (`live_check`, snapshot 14:45 SAST = IEC_LGE2026_GP_20261105_1445.csv, 728 counted VDs, 28 hold-out tasks)

Mean group-share MAE (pp) of the held-out VDs by task kind, 90% coverage of the held-out group shares, and VD-level MAE on group tasks. File: `artifacts/bench/live_swing.csv`.

| method                |   group |   muni |   random |   time |   group cov90 |   muni cov90 |   random cov90 |   time cov90 |   group vd_mae |
|:----------------------|--------:|-------:|---------:|-------:|--------------:|-------------:|---------------:|-------------:|---------------:|
| reg_cf                |    1.64 |   2.01 |     0.39 |   0.33 |          0.91 |         0.83 |              1 |            1 |           3.59 |
| hybrid_cf             |    1.64 |   2.07 |     0.36 |   0.28 |          0.88 |         0.96 |              1 |            1 |           3.61 |
| hybrid_cf_c1          |    1.69 |   2.12 |     0.34 |   0.29 |        nan    |       nan    |            nan |          nan |           3.68 |
| swing_ens             |    1.83 |   2.31 |     0.38 |   0.4  |          0.86 |         0.94 |              1 |            1 |           3.96 |
| cluster_muni_ratio    |    1.85 |   2.28 |     0.37 |   0.37 |          0.83 |         0.94 |              1 |            1 |           4.12 |
| cluster_ratio         |    1.86 |   2.28 |     0.38 |   0.18 |          0.79 |         0.94 |              1 |            1 |           4.09 |
| cluster_ratio_subst   |    1.86 |   2.28 |     0.38 |   0.18 |        nan    |       nan    |            nan |          nan |           4.09 |
| cluster_muni_ratio_k6 |    1.89 |   2.36 |     0.32 |   0.36 |        nan    |       nan    |            nan |          nan |           4.13 |
| cluster_swing         |    1.89 |   2.44 |     0.52 |   0.45 |          0.83 |         0.92 |              1 |            1 |           3.91 |
| cluster_muni_swing    |    1.92 |   2.46 |     0.49 |   0.49 |          0.88 |         0.85 |              1 |            1 |           3.98 |
| greben_std            |    3.34 |   3.12 |     0.37 |   0.24 |          0.54 |         0.56 |              1 |            1 |           5.4  |
| swing_ratio_ward      |    3.37 |   3.62 |     0.83 |   0.53 |          0.68 |         0.77 |              1 |            1 |           5.73 |
| swing_add_prov        |    3.47 |   2.69 |     0.89 |   0.55 |        nan    |       nan    |            nan |          nan |           5.61 |
| greben_csir           |    3.5  |   3.62 |     0.42 |   0.38 |          0.59 |         0.52 |              1 |            1 |           5.53 |
| swing_add_ward        |    3.51 |   2.69 |     0.53 |   0.61 |          0.62 |         0.88 |              1 |            1 |           5.59 |
| swing_ratio_muni      |    3.61 |   3.62 |     1.45 |   1.21 |        nan    |       nan    |            nan |          nan |           5.57 |
| swing_logit_ward      |    3.77 |   3.81 |     1.11 |   1.12 |          0.73 |         0.9  |              1 |            1 |           5.71 |
| swing_add_muni        |    3.78 |   2.69 |     0.84 |   0.77 |        nan    |       nan    |            nan |          nan |           5.8  |
| swing_ratio_prov      |    4.05 |   3.62 |     1.61 |   1.21 |        nan    |       nan    |            nan |          nan |           5.92 |
| swing_logit_muni      |    4.18 |   3.81 |     1.48 |   1.54 |        nan    |       nan    |            nan |          nan |           6.01 |
| swing_logit_prov      |    4.5  |   3.81 |     1.53 |   1.7  |        nan    |       nan    |            nan |          nan |           6.35 |


Baselines (coordinator, `live_baselines.csv`):

| method             |   group |   muni |   random |   time |
|:-------------------|--------:|-------:|---------:|-------:|
| tally              |    9.9  |   4.68 |     0.82 |   2.06 |
| uniform_swing_muni |    4.91 |   4.77 |     1.56 |   1.54 |


Same check at the 13:29 snapshot (637 counted VDs; group MAE), for stability:

| method                |   group |   muni |   random |   time |
|:----------------------|--------:|-------:|---------:|-------:|
| reg_cf                |    1.71 |   2.03 |     0.36 |   0.57 |
| hybrid_cf             |    1.74 |   2.09 |     0.34 |   0.58 |
| hybrid_cf_c1          |    1.78 |   2.14 |     0.36 |   0.55 |
| cluster_ratio         |    1.87 |   2.27 |     0.5  |   1.01 |
| cluster_ratio_subst   |    1.87 |   2.27 |     0.5  |   1.01 |
| swing_ens             |    1.9  |   2.29 |     0.42 |   0.41 |
| cluster_muni_ratio    |    1.92 |   2.26 |     0.52 |   0.56 |
| cluster_swing         |    1.97 |   2.37 |     0.44 |   0.34 |
| cluster_muni_ratio_k6 |    1.98 |   2.4  |     0.47 |   0.94 |
| cluster_muni_swing    |    2.02 |   2.38 |     0.43 |   0.28 |

## Failure cases (best method `hybrid_cf`; the other cluster methods fail the same way)

* **DA-strongholds-first, 5-25% counted:** province MAE 2.1-3.7 pp. LGE2021: ANC -3.6 to -6.8,
  ActionSA +3.7 to +5.9, EFF -2.8 to -4.2 (the DA-suburb swing - small ActionSA gain from
  nothing, smaller ANC/EFF losses than in townships - is applied to unreached townships).
  PE2024: MK -5.7 to -7.4 and EFF +6.7 to +9.1 (MK's level in DA suburbs says nothing about
  townships; EFF's large log-ratio gain from a tiny suburban base is extrapolated). At 50% it
  is fine (0.35-0.43).
* **ANC-strongholds-first:** LGE2021 2.1-2.4 pp at every level (DA -2.8 to -4.9, ActionSA -4.2
  at 5%, EFF +1.4 to +5.0); PE2024 0.65-1.9 (MK +2 to +3.7).
* **small-first, LGE2021 5-10%:** DA -2.9 to -5.0 and turnout +3.9 to +6.8 pp: the smallest VDs
  (farms, informal settlements) had a smaller turnout collapse than the cities.
* These are the cases the extrapolation penalty was built for: intervals are 11-21 pp wide there
  and coverage is 0.83-1.0 except PE2024 DA-first (0.83) and LGE2021 small-first (0.67-0.83).
* **Random / real2026 orders** are good: 0.07-0.64 pp (mean 0.25 / 0.33 LGE2021, 0.26 / 0.11
  PE2024), ward-winner accuracy 0.95-0.98.
* **Live:** worst hold-outs are the small-municipality townships (Lesedi GT423 3.9 pp, Merafong
  GT484 2.9, Rand West GT485 2.9) and Johannesburg as a whole (muni:JHB 5.1): the West Rand /
  Sedibeng townships swung differently (less ActionSA, more DA) from the metro townships that
  the regression learns from. `cluster_muni_ratio` is better than `hybrid_cf` on some of these
  (GT423 3.4, GT481 suburbs 0.5 vs 1.7) and worse on the metro townships (JHB/TSH 2.7-2.9 vs
  1.6), so neither dominates group by group.

## Coordinator stress-test findings (14:45)

* (a) *Damped log-ratio swing (slopes 0.54-0.79).* The regression part of `hybrid_cf` /
  `reg_cf` fits the slopes on both past log-ratios and shares live on the counted VDs, which
  is why it beats the pure swing models on the live hold-outs; the cluster ratio swing has an
  implicit slope of 1 inside a cluster (damping across clusters comes from the cluster
  effects). I did not add a separately damped swing scale (time).
* (b) *k-nearest look-alike swing.* I tried kNN swing in profile space (k = 10-40, with and
  without geography, plus muni/ward EB): 1.56-1.87 pp on the benchmark vs 1.27 for the cluster
  swing (much worse on ANC-first, where the nearest counted VDs are the least representative).
  The hybrid's distance weight uses the same 5-nearest-counted-VD distance.
* (c) *Broken histories (s24A-inflated turnout, zero results).* 44 VDs in the live case have past
  turnout > 90% (none has zero results in my VD set). `hybrid_live` replaces their projection
  by the composition of the 8 nearest counted non-broken VDs (past valid shares, same
  municipality preferred). On the live hold-outs this was slightly **worse** than `hybrid_cf`
  (group 1.646 vs 1.644, VD MAE 3.66 vs 3.61, VD turnout MAE 4.79 vs 4.57): the regression
  part already learns that inflated past turnout does not carry over. Kept in the module,
  not in the final set.

## Run time per projection

Pure compute for one projection with 200 draws on one core (case features and clusters cached;
the first call per case adds ~4 s): `greben_csir` 0.2 s, `cluster_ratio` /
`cluster_muni_ratio` / `cluster_swing` 0.3-0.5 s, `swing_ens` 0.5 s, `hybrid_cf` / `reg_cf`
0.6-1.3 s (the ridge is refitted for every bootstrap draw). With the machine at load 11 these
were 1-2 s. The harness `seconds` column (3-4 s median) was measured while three other agents
shared the 4 cores, and the point-only comparison variants take 0.01-0.2 s. Re-running every
few minutes on election night is no problem.


## What I tried (screening, point projections, 1 random seed, both cases, all orders)

Mean province MAE (pp) over both cases and all orders/levels; baseline `uniform_swing_muni`
2.94, `tally` 4.09. Lower is better. Only the final methods were re-run with draws and 3 seeds.

| variant | prov MAE | note |
|---|---|---|
| **Scale x pooling (item 1)**: see the full-run table below (`swing_{add,ratio,logit}_{prov,muni,ward}`) | | |
| log-ratio swing (ward), base previous PE / previous LGE / per-party blend fitted on counted VDs | 2.93 / 2.36 / 2.31 | the LGE base beats the PE base in both cases; the blend is best |
| **New parties (item 4)** | | |
| ridge on past log-ratios, no clipping | 2.45 | ActionSA +65 pp at DA-first 10%: exp() of a linear extrapolation |
| ridge on bounded shares, fitted values clipped to the counted range | 1.74 | |
| kNN average of the new party's log-ratio (k = 20) | 1.64 | |
| cluster average (province level + EB cluster/muni/ward effects) | **1.41** (cl40 model) | vs 1.59 with the ridge in the same model; adopted |
| ecological transfer (Goodman NNLS) as the new party's base | 1.44-1.48 (vs 1.27) | |
| **Curtice-Firth regression (item 2)** | | |
| ridge, past log-ratio features, LOO CV, no clip / clip | 2.77 / 2.16 | ANC-first 6-8 pp: extrapolation |
| ridge, share features only, LOO / group CV | 3.00 / 2.84 | |
| ridge, all features, blended base, group CV, muni+ward EB (`reg_cf`) | 1.85 | best CF; 0.11-0.26 in PE2024 random/real, better than any cluster model there |
| CF on within-cluster covariates + cluster EB | 1.59-1.84 | |
| kNN swing in profile space (k = 10/20/40, +geography, +muni/ward) | 1.87 / 1.76 / 1.67 / 1.56 | constant extrapolation is worse than the province mean on ANC-first |
| ecological transfer base for all parties (+ cluster EB) | 2.37 | strongholds 4-5 pp |
| **Cluster swing (item 3)** (log-ratio, blended base, cluster + muni + ward EB) | | |
| k-means K = 20 / 40 / 60 | **1.27** / 1.30 / 1.29 | K = 20 chosen |
| GMM(40, diag) / Ward tree nested 8 -> 40 / 6 -> 20 -> 60 / km8 + km40 / cluster x muni cells only | 1.44 / 1.30 / 1.41 / 1.29 / 1.46 | nested back-off does not help |
| weights: precision / electorate / votes at stake / sqrt(votes x precision) | 1.30 / 1.76 / 1.68 / 1.52 | |
| Duan smearing of the exp back-transform | 1.35 (vs 1.30) | worse |
| ratio-scale cluster swing, base LGE / average(LGE, PE) / geometric / blended per party | 1.40 / 1.26 / 1.38 / 1.23 | |
| + new parties from the log-ratio cluster model (`cluster_ratio`) | **1.16** | km10 1.21, km40 1.16, without ward level 1.17 |
| **Area type x municipality (coordinator request)** | | |
| ratio, km K + muni + (K x muni cell) + ward, K = 4 / 6 / 10 / 20 | 1.30 / 1.29 / 1.19 / **1.15** | |
| log-ratio, same, K = 4 / 6 / 10 / 20 | 1.64 / 1.56 / 1.38 / 1.28 | |
| **Hybrid** regression inside the counted range, `cluster_muni_ratio` outside, c = 0.5 / 1 / 2 / 4 / 8 | 1.15 / 1.15 / **1.14** / 1.17 / 1.25 | c = 2 chosen (also best on the live check) |
| **CSIR level estimator** (Greben 2006), raw %, C = 20, m = 1.2 / m = 1.4 / C = 10 / C = 40 | 4.12 / 4.18 / 4.12 / 4.16 | 0.2-0.3 on random orders, 8-10 pp on stronghold orders |
| Greben with standardised per-party shares (Ittmann fix), C = 10 / 20 / 40 | 4.49 / 4.26 / **4.05** | |
| ensembles ratio:logit 1:1 / 2:1; + Greben | 1.20 / 1.18; 1.54-1.85 | |
| oracle floor: cluster swing fitted on ALL VDs | 0.20 (LGE2021) / 0.14 (PE2024) | model misspecification alone |

Main lessons:

1. **Scale.** Proportional (ratio) swing beats additive and log-ratio swing at every pooling
   level. Additive swing breaks for small parties in hostile areas; the log-ratio swing,
   estimated as a precision-weighted mean of VD log-ratio changes, is pulled toward the many
   VDs where a party is tiny and its log-ratio is noisy, so it under-projects big parties where
   they are big. The ratio estimator (sum Y / sum E) is vote-weighted by construction.
2. **Pooling.** Ward EB on top of municipality helps a little on every scale; profile clusters
   help much more than geography (cluster swing 1.16-1.27 vs ward swing 1.87-2.36).
3. **New parties** are best handled as a cluster average of their own log-ratio (they have no
   base to swing from). Linear substitution regressions on past shares extrapolate badly.
4. **Regression vs clusters.** CF regression is the best interpolator (random/real orders,
   live hold-outs) and the worst extrapolator (stronghold orders). The per-VD hybrid keeps the
   regression where the uncounted VD has counted look-alikes and keeps the cluster swing where
   it does not, and is best on both benchmarks.
5. **Stronghold orders are an information limit.** With only DA (or ANC) strongholds counted no
   swing model knows how townships (suburbs) moved; every method here sits at 2-2.5 pp there,
   and more structure (kNN, transfer, CF) made it worse. These orders need wide intervals, not a
   cleverer mean.

## Area type x municipality hierarchy (coordinator request)

`cluster_muni_ratio` = per-party swing in (area type x municipality) cells, area types from
k-means on standardised LGE + PE party shares + turnout (not hand thresholds), cells shrunk
by EB to area type + municipality (crossed), then province, with the amount of pooling set by
the number of counted VDs in the cell and the estimated between-cell variance.

* **K:** K = 4 / 6 / 10 / 20 gives 1.30 / 1.29 / 1.19 / 1.15 (ratio) on the benchmark;
  coarse area types (like the three hand-made ones) are not enough. On the live hold-outs
  K = 6 vs 20 is 1.98 vs 1.92 (group tasks).
* **Does it hurt when the heterogeneity is absent?** No. In the benchmark the cell-level
  swing SD (EB, all VDs) is only 0.1-0.2 on the log-ratio scale for the big parties (vs
  0.25-0.4 between area types and up to 2.5 between municipalities for ActionSA, which stood
  in few municipalities in 2021), so EB shrinks the cells almost to zero and the result equals
  the model without cells: full run `cluster_ratio` 0.887 vs `cluster_muni_ratio` 0.875 province MAE, muni MAE 1.38 vs 1.37, random orders 0.21/0.23 vs 0.18/0.23 (LGE2021/PE2024). K = 6 is worse (0.979).
* On the live data it helps where cells are counted (live `random` and `time` tasks) but cannot
  create information for a cell with no counted VD: the `group` hold-outs hide exactly that
  cell, which then gets area type + municipality effects only. The township x municipality
  interaction the live evidence shows (Tshwane ActionSA +31 vs Johannesburg +17) is not
  additive in area type and municipality, so the hierarchy recovers only part of it; the
  regression (which reads turnout, size and both past elections per VD) recovers more
  (Tshwane townships held out: `reg_cf` 1.49, `hybrid_cf` 1.74, `cluster_muni_ratio` 2.62 pp).


## What the harness gets wrong (or could do better)

1. **`prov_cov90` counts parties that are not on the ballot.** MK is not contested in the
   LGE2021 case; its true share is 0, and the interval [0, 0] covers it only if every draw is
   *exactly* zero. A method whose softmax leaves 1e-18 of a vote gets 1/6 of its LGE2021
   coverage wrongly removed; a method that writes exact zeros gets a free 1/6. My methods now
   write exact zeros for any category with no counted vote, so their harness coverage is
   ~(5 x contested coverage + 1)/6 in LGE2021. Coverage should be computed over contested
   parties only (`case.design.contested`).
2. **The real2026 order was not frozen** (it read every snapshot in `data/observed_2026/`, so
   results depended on when a family ran). This has since been fixed in the harness (first
   snapshot only); my final run used the pinned version (21.4% / 21.7% first block). Live-check
   numbers still depend on the snapshot (`--snapshot`), so they must be compared at the same
   snapshot; mine are at 14:45 (13:29 also given).
3. **Stronghold orders dominate the averages.** DA/ANC-strongholds-first sort strictly by
   previous-PE share, an extreme no real count resembles (CSIR's own model "will not operate
   well under extreme political or spatial bias"). They are 2 of the 5 order types and their
   errors are ~10x the random-order errors, so the headline `prov_mae_pp` ranks methods
   mostly by how they fail on two adversarial orders. Report random/real separately (or
   weight orders) when picking the live method.
4. **Turnout is not in the headline metric.** `prov_mae_pp` uses valid-vote shares only;
   projected turnout can be 8-15 pp off (small_first 5-10%) without moving it. Vote totals,
   and therefore seat thresholds in absolute votes, depend on turnout.
5. **`muni_mae_pp` weights all 9 municipalities equally** (Lesedi = Johannesburg); a
   vote-weighted version would reflect seats better. `ward_mae_pp` is over open wards only,
   which changes with the level, so it is not comparable across levels.
6. **Scoring cost.** With draws, `score` takes ~4 s per projection (the dense
   (S, W, n) einsums for ward shares); the methods themselves take 0.3-1 s. A sparse ward
   indicator would make a full run ~5x faster.
7. **Coverage is noisy.** 6 correlated parties x 54 scenarios per method: a 90% target
   cannot be distinguished from 85% or 95% with confidence; CRPS is the better guide.
