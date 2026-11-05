# ML, spatial smoothing and conformal methods (`--tag ml`, `src/saelection/bench/m_ml.py`)

## Bottom line

* **Recommended: `ml_mixed_gbm_shift`.** The swing model is a REML crossed-random-effects
  log-ratio swing: municipality + area type + municipality × area type + 8 k-means profile
  clusters + ward. LightGBM models what that leaves; its number of trees is chosen by grouped CV,
  and 0 trees is allowed. A GREG ratio calibration follows. The draws combine posterior effect
  draws, CV+/jackknife+ residuals and an ACI-style order-shift shock.
  * **Backtest** (both cases, all orders, 3 random seeds):
    * Province MAE is **0.81 pp**, against tally 4.09 and `uniform_swing_muni` 2.94.
    * Its municipality MAE (1.20 pp) and ward-winner accuracy (0.951) are the best of all methods.
    * VD MAE is 3.66 pp, CRPS 0.63 pp and ward Brier 0.080.
    * Province 90% coverage is **0.94**; municipality coverage is 0.84.
  * **live_check** (2026 count, pinned to the 14:45 snapshot, 728 counted VDs):
    * Area type × municipality hold-outs: **1.58 pp** with group_cov90 **0.91**, against uniform
      swing 4.9 and tally 9.9.
    * Municipality: 1.80 pp (cov 0.90). Random: 0.23 pp. Time: 0.25 pp.
* **Runner-up without ML: `ml_mixed_look_shift`.** It replaces the GBM with k = 8 look-alike
  smoothing. Backtest 0.80 pp, cov 0.93. Live group 1.70, municipality 1.98, random 0.26,
  time 0.17.
* **Backtest point accuracy is a near-tie at the top.** krig 0.794, mixed_look 0.796, knn 0.800,
  mixed_gbm 0.805, gbm 0.81, mixed 0.81, gbm_change 0.82, swing_muni 0.84. The differences come
  from the two stronghold-first orders. The recommendation rests on two things:
  1. **Calibrated draws.** Only the mixed + shift variants reach about 0.93 province coverage; the
     others reach 0.68–0.83.
  2. **The live group test**, where the area type × municipality structure matters and
     geographic smoothing fails (knn 2.62, swing_muni 2.88).
* **Boosting vs ridge, honestly.**
  * Backtest: `ml_gbm` 0.81 vs `ml_ridge` 1.12. They are equal on random orders (0.19). Ridge loses
    on stronghold orders through linear extrapolation, even with clamped features.
  * Live: `ml_gbm` 1.59 (group) and 1.70 (muni) vs `ml_ridge_int` 2.18 and 2.24. Boosting adds
    value once several hundred VDs are counted, but `ml_gbm` alone under-covers (group_cov90 0.62).
    Putting it on top of the mixed base keeps the accuracy and fixes the intervals.
  * At 5% counted the CV usually picks 0–25 trees, so the GBM falls back to the base.
* **Change vs level.** `ml_gbm_change` scores 0.82 pp and `ml_gbm_level` 2.78. On random orders
  they are equal; under stronghold orders the level model is 8–9 pp off. Predict the change.
* **Faithful WaPo-style (`ml_wapo`).** It scores 2.15 pp backtest and 1.95 live group. Adding
  municipality × area-type effects (`ml_wapo_int`) gives 1.97 backtest; live it is 2.01 on groups
  and 1.92 on municipalities.
  * It suffers from no GREG: the log-scale bias is about 0.5 pp even on random orders.
  * Linear CQR extrapolates under stronghold orders.
* **The two biggest wins were not ML.**
  1. **GREG ratio calibration.** It rescales projections so that the out-of-fold predictions of the
     counted VDs sum to their actual totals. Without it, the mean of VD log-ratios is a biased
     estimate of the change in totals. It cut random-order province MAE at 25–50% counted from
     about 0.5 pp to about 0.1–0.2 pp.
  2. **The area type × municipality structure** (mixed model), as the 2026 live evidence asked.
     Live group error fell from 2.88 to 1.77 pp, and group coverage rose from 0.46 to 0.82.

## What I tried (all in `m_ml.py`; one engine `_run`, variants differ in the swing model)

1. **Target space and regimes.**
   * Targets are the log valid-vote share per party and the turnout logit, mapped back with
     softmax/sigmoid.
   * Parties with local history (more than 0.2% of the municipality vote in a predictor election)
     are modelled as *change* versus a history baseline. The LGE/PE mix weight is chosen on the
     counted VDs.
   * Parties without history are modelled as *level*: ActionSA in LGE2021, MK in PE2024, and
     ActionSA outside the metros in PE2024.
   * Broken histories get a neighbour fallback (`_fix_broken`): zero valid votes, or turnout above
     95% from s24A voting. A first, broader version (neighbour-fill of shares too) broke the mixed
     model under stronghold orders (one PE2024 run went to a 6.7 pp error), so only turnout is
     replaced for s24A VDs.
2. **Base swing models.**
   * `ml_swing_muni`: empirical-Bayes municipality swing (DerSimonian–Laird, ward-clustered).
   * `ml_mixed`: REML crossed random effects with an analytic gradient, heteroscedastic noise
     φ + Poisson part.
   * `ml_mixed_slope`: adds a learned damped slope, prior N(1, 0.3²). The slopes came out at
     0.5–0.8, matching the stress test. It still lost on both the backtest (1.08 vs 0.80) and live
     (group 1.84 vs 1.77): the type, cell and cluster effects already absorb the damping, and the
     slope extrapolates badly under stronghold orders.
3. **Spatial smoothing of the residual swing.**
   * `ml_knn`: geographic k-NN with distance weights and shrinkage to the municipality swing.
   * `ml_gmrf`: contiguity-graph propagation, a sparse solve on `case.edges`.
   * `ml_krig`: GP / kriging, Matern-3/2 on lon/lat with ML hyper-parameters.
   * `ml_look`: k = 8 nearest *look-alike* counted VDs in standardised LGE+PE shares and turnout,
     with a penalty for another municipality.
   * All smoothing hyper-parameters are set by ward-grouped CV.
   * Geographic smoothing helps a little on top of the municipality swing: backtest knn 0.800 and
     krig 0.794 vs swing_muni 0.844. Live it barely helps (group 2.62 vs 2.88).
   * Look-alike smoothing helps slightly on top of mixed: live random 0.26 vs 0.30, group 1.70 vs
     1.77.
4. **ML / linear on features.** Features: previous shares and turnout of both elections, own
   log-ratios, log size, electorate growth, lon/lat, imputed flag, municipality, area type,
   cluster, and spatial lags (mean history of touching VDs).
   * `ml_ridge`, `ml_ridge_int`: ridge, the second with type × muni, cluster and shares × metro
     interactions.
   * `ml_enet`: elastic net.
   * `ml_gbm`, `ml_mixed_gbm`: LightGBM residual learners, 3-level trees with strong L2. The
     number of trees is chosen by grouped CV, and **0 trees is allowed**, so they fall back to the
     base at low counts.
   * `ml_gbm_change` vs `ml_gbm_level`: LightGBM directly on the change or the level, with no base.
   * Ridge and enet prediction features are **clamped to the counted range** to stop linear
     extrapolation.
5. **Hybrids.** `ml_gbm_krig` and `ml_ridge_krig` (base + features + kriging), and mixed + kNN /
   look-alike / GBM / ridge.
   * Screened on one seed only, with earlier code, and left out of the final CSV:
     * `ml_gmrf`: about the same as knn (1.04 vs 1.06 province MAE) and 6× slower.
     * `ml_mixed_knn` and `ml_mixed_ridge`: no gain over `ml_mixed` (1.10 / 1.15 vs 1.10).
   * Implemented but not benchmarked for lack of time: `ml_enet`, `ml_gbm_krig`, `ml_ridge_krig`,
     `ml_mixed_krig`, `ml_mixed_gmrf`, `ml_look`, `ml_mixed_slope_look`, `ml_mixed_nw`.
6. **WaPo-style (faithful as far as the setting allows): `ml_wapo`, `ml_wapo_int`.**
   * Each VD's change against its own history is ridge-regressed on covariates and municipality
     fixed effects, weighted by size.
   * Unit intervals come from CQR: linear quantile regressions at 5% and 95% on a ward-grouped
     half, conformalised on the other half with size-weighted conformal quantiles.
   * Aggregation uses an equicorrelated Gaussian model of the standardised scores (one-way ANOVA
     within municipality) plus a contest-level effect.
   * `_int` adds municipality × area-type fixed effects.
   * No GREG. That is the main reason it trails: the log-scale bias is about 0.5 pp.
7. **Uncertainty.** The first version (jackknife of fold fits + municipal + ward + VD +
   conformal inflation from ward aggregates) was replaced by `_draw2`:
   * Ward-grouped 10-fold CV gives out-of-fold (CV+/jackknife+) residuals.
   * VD noise: robust extra-Poisson φ plus the Poisson part. A ward-correlated share ρ is
     estimated from residuals of touching counted VDs. Standardised OOF residual *vectors* are
     resampled (heavy tails, cross-party correlation).
   * Correlated unit-level swing errors: posterior (seen) or prior (unseen) draws of the
     municipality / type / cell / cluster / ward effects. These are **propagated through the GREG
     ratio**, so a municipality-wide shift is absorbed by the count, while an unseen area type is
     not. This is what widens the intervals for area types with few counted look-alikes.
   * Estimation error: a Gaussian-multiplier ward-grouped jackknife of the GREG ratios.
   * **Order-shift shock (`shift_cal`, ACI-style)** (I named this the "pseudo-arrival" check):
     * A propensity model ranks counted VDs by how "counted-like" they are. The 30% least like the
       rest are projected from the others with the cheap swing+GREG model.
     * Their excess aggregate error over random splits, scaled by the real counted/uncounted gap
       along the propensity axis, becomes a province-wide shock.
     * The harness gives no arrival order, so true ACI (Gibbs & Candès) cannot run inside it. Live,
       the real arrival order should replace the propensity ranking: score each new batch against
       its pre-arrival projection.
     * The shock raised backtest province coverage from 0.81 to 0.94 for `ml_mixed_gbm` →
       `_shift`, and from 0.82 to 0.93 for `ml_mixed_look`. The mean projection is unchanged.
       Live group coverage rose from 0.84 to 0.89 for `ml_mixed_look`.
   * In the harness, ACI-style adaptivity helps coverage under stronghold orders but cannot fix
     them (see failures).

## Kriging and spatial smoothing

* `ml_krig` (0.794) and `ml_knn` (0.800) have the best backtest mean, mostly from the DA-first
  order (1.8 vs 2.1 pp for the mixed variants).
* They lose to the mixed model on municipality MAE, coverage (0.70 / 0.68) and the live group test
  (knn 2.62 pp).
* Every variant gets the same conformal/posterior draws, so kriging is judged here on its mean.
  WaPo dropped kriging partly because its Gaussian intervals covered poorly; that is not tested
  here.
* Kriging is also the slowest variant (median 40 s per projection under load).

## Summary table (`harness.summarise`, results_ml.csv: both cases, all orders, --n-random 3)

Run with the harness loop (my parallel runner calls `harness.load_case/orders/score` exactly as `harness.run`). The tally / uniform_swing_muni baselines at the same points: 4.09 / 2.94 pp.

| method              |   prov_mae_pp |   prov_max_pp |   muni_mae_pp |   ward_winner_acc |   ward_mae_pp |   vd_mae_pp |   vd_turnout_mae_pp |   prov_cov90 |   prov_crps_pp |   muni_cov90 |   ward_brier |   tally_mae_pp |   seconds |
|:--------------------|--------------:|--------------:|--------------:|------------------:|--------------:|------------:|--------------------:|-------------:|---------------:|-------------:|-------------:|---------------:|----------:|
| ml_krig             |         0.794 |         1.824 |         1.268 |             0.949 |         2.523 |       3.876 |               5.655 |        0.701 |          0.68  |        0.675 |        0.082 |          4.094 |    74.572 |
| ml_mixed_look       |         0.796 |         1.796 |         1.229 |             0.95  |         2.394 |       3.714 |               5.94  |        0.821 |          0.609 |        0.765 |        0.078 |          4.094 |    22.861 |
| ml_mixed_look_shift |         0.796 |         1.796 |         1.229 |             0.95  |         2.394 |       3.714 |               5.94  |        0.932 |          0.623 |        0.842 |        0.082 |          4.094 |    23.254 |
| ml_knn              |         0.8   |         1.805 |         1.271 |             0.949 |         2.501 |       3.846 |               5.582 |        0.682 |          0.688 |        0.679 |        0.081 |          4.094 |     2.454 |
| ml_mixed_gbm        |         0.805 |         1.794 |         1.202 |             0.951 |         2.365 |       3.663 |               5.713 |        0.806 |          0.614 |        0.771 |        0.076 |          4.094 |    27.263 |
| ml_mixed_gbm_shift  |         0.805 |         1.794 |         1.202 |             0.951 |         2.365 |       3.663 |               5.713 |        0.938 |          0.63  |        0.844 |        0.08  |          4.094 |    16.422 |
| ml_gbm              |         0.81  |         1.869 |         1.299 |             0.947 |         2.389 |       3.642 |               5.145 |        0.694 |          0.706 |        0.676 |        0.083 |          4.094 |    31.606 |
| ml_mixed            |         0.81  |         1.84  |         1.234 |             0.95  |         2.422 |       3.746 |               5.937 |        0.83  |          0.618 |        0.77  |        0.078 |          4.094 |     7.522 |
| ml_mixed_shift      |         0.81  |         1.84  |         1.234 |             0.95  |         2.422 |       3.746 |               5.937 |        0.935 |          0.632 |        0.843 |        0.082 |          4.094 |    11.304 |
| ml_gbm_change       |         0.822 |         1.797 |         1.308 |             0.948 |         2.37  |       3.627 |               5.117 |        0.685 |          0.721 |        0.636 |        0.082 |          4.094 |    40.654 |
| ml_swing_muni       |         0.844 |         2.005 |         1.314 |             0.946 |         2.732 |       4.113 |               5.54  |        0.719 |          0.716 |        0.68  |        0.084 |          4.094 |     1.15  |
| ml_mixed_slope      |         1.079 |         2.297 |         1.372 |             0.949 |         2.548 |       3.869 |               5.84  |        0.772 |          0.847 |        0.745 |        0.079 |          4.094 |    14.137 |
| ml_ridge_int        |         1.106 |         2.794 |         1.635 |             0.925 |         2.744 |       4.067 |               5.478 |        0.707 |          0.99  |        0.656 |        0.121 |          4.094 |     2.869 |
| ml_ridge            |         1.116 |         2.791 |         1.659 |             0.926 |         2.744 |       4.024 |               5.323 |        0.707 |          1.009 |        0.666 |        0.119 |          4.094 |     3.944 |
| ml_wapo_int         |         1.965 |         4.824 |         2.422 |             0.888 |         3.519 |       4.852 |               5.63  |        0.722 |          1.682 |        0.68  |        0.191 |          4.094 |     1.998 |
| ml_wapo             |         2.148 |         5.28  |         2.57  |             0.877 |         3.731 |       5.068 |               5.587 |        0.682 |          1.881 |        0.656 |        0.212 |          4.094 |     3.002 |
| ml_gbm_level        |         2.775 |         6.756 |         3.086 |             0.847 |         4.234 |       5.672 |               5.911 |        0.639 |          2.669 |        0.602 |        0.283 |          4.094 |    21.535 |

### prov_mae by order type

| method              |   ANC_strongholds_first |   DA_strongholds_first |   random |   real2026 |   small_first |
|:--------------------|------------------------:|-----------------------:|---------:|-----------:|--------------:|
| ml_gbm              |                    1.92 |                   2.31 |     0.19 |       0.16 |          0.56 |
| ml_gbm_change       |                    1.96 |                   2.34 |     0.21 |       0.18 |          0.48 |
| ml_gbm_level        |                    9.23 |                   8.07 |     0.22 |       0.19 |          0.63 |
| ml_knn              |                    1.92 |                   1.85 |     0.24 |       0.28 |          0.7  |
| ml_krig             |                    1.91 |                   1.79 |     0.25 |       0.24 |          0.74 |
| ml_mixed            |                    1.99 |                   2.14 |     0.22 |       0.22 |          0.52 |
| ml_mixed_gbm        |                    2.03 |                   2.08 |     0.21 |       0.18 |          0.54 |
| ml_mixed_gbm_shift  |                    2.03 |                   2.08 |     0.21 |       0.18 |          0.54 |
| ml_mixed_look       |                    1.97 |                   2.06 |     0.21 |       0.22 |          0.54 |
| ml_mixed_look_shift |                    1.97 |                   2.06 |     0.21 |       0.22 |          0.54 |
| ml_mixed_shift      |                    1.99 |                   2.14 |     0.22 |       0.22 |          0.52 |
| ml_mixed_slope      |                    3.58 |                   2.38 |     0.22 |       0.2  |          0.52 |
| ml_ridge            |                    2.81 |                   3.59 |     0.19 |       0.18 |          0.41 |
| ml_ridge_int        |                    2.64 |                   3.55 |     0.21 |       0.2  |          0.5  |
| ml_swing_muni       |                    1.98 |                   1.85 |     0.27 |       0.29 |          0.85 |
| ml_wapo             |                    4.98 |                   6.46 |     0.51 |       0.5  |          1.14 |
| ml_wapo_int         |                    4.83 |                   5.59 |     0.49 |       0.49 |          1.01 |

### prov_cov90 by order type

| method              |   ANC_strongholds_first |   DA_strongholds_first |   random |   real2026 |   small_first |
|:--------------------|------------------------:|-----------------------:|---------:|-----------:|--------------:|
| ml_gbm              |                    0.17 |                   0.27 |     0.93 |       0.92 |          0.77 |
| ml_gbm_change       |                    0.21 |                   0.21 |     0.94 |       0.83 |          0.77 |
| ml_gbm_level        |                    0.08 |                   0.17 |     0.92 |       0.81 |          0.71 |
| ml_knn              |                    0.27 |                   0.31 |     0.92 |       0.78 |          0.67 |
| ml_krig             |                    0.27 |                   0.37 |     0.92 |       0.83 |          0.69 |
| ml_mixed            |                    0.56 |                   0.67 |     0.97 |       0.89 |          0.79 |
| ml_mixed_gbm        |                    0.5  |                   0.6  |     0.94 |       0.97 |          0.79 |
| ml_mixed_gbm_shift  |                    0.85 |                   0.92 |     0.98 |       0.97 |          0.9  |
| ml_mixed_look       |                    0.56 |                   0.67 |     0.96 |       0.89 |          0.77 |
| ml_mixed_look_shift |                    0.87 |                   0.92 |     0.98 |       0.94 |          0.85 |
| ml_mixed_shift      |                    0.87 |                   0.9  |     0.99 |       0.92 |          0.88 |
| ml_mixed_slope      |                    0.33 |                   0.54 |     0.97 |       0.89 |          0.77 |
| ml_ridge            |                    0.21 |                   0.17 |     0.97 |       0.83 |          0.87 |
| ml_ridge_int        |                    0.31 |                   0.15 |     0.94 |       0.89 |          0.83 |
| ml_swing_muni       |                    0.29 |                   0.48 |     0.94 |       0.86 |          0.6  |
| ml_wapo             |                    0.23 |                   0.31 |     0.92 |       0.86 |          0.67 |
| ml_wapo_int         |                    0.29 |                   0.29 |     0.94 |       0.92 |          0.77 |

### prov_width90 by order type

| method              |   ANC_strongholds_first |   DA_strongholds_first |   random |   real2026 |   small_first |
|:--------------------|------------------------:|-----------------------:|---------:|-----------:|--------------:|
| ml_gbm              |                    1.2  |                   1.13 |     1.01 |       0.65 |          1.64 |
| ml_gbm_change       |                    1.2  |                   1.06 |     1.01 |       0.64 |          1.53 |
| ml_gbm_level        |                    0.94 |                   0.91 |     1.08 |       0.65 |          1.65 |
| ml_knn              |                    1.16 |                   1.1  |     1.1  |       0.74 |          1.73 |
| ml_krig             |                    1.19 |                   1.09 |     1.14 |       0.75 |          1.75 |
| ml_mixed            |                    3.86 |                   4.61 |     1.16 |       0.76 |          1.93 |
| ml_mixed_gbm        |                    3.79 |                   4.63 |     1.14 |       0.73 |          1.93 |
| ml_mixed_gbm_shift  |                    9.84 |                  10.7  |     1.47 |       1.08 |          3.65 |
| ml_mixed_look       |                    3.84 |                   4.61 |     1.15 |       0.74 |          1.93 |
| ml_mixed_look_shift |                    9.84 |                  10.62 |     1.47 |       1.09 |          3.66 |
| ml_mixed_shift      |                    9.87 |                  10.6  |     1.48 |       1.09 |          3.65 |
| ml_mixed_slope      |                    4.13 |                   4.58 |     1.17 |       0.77 |          1.85 |
| ml_ridge            |                    1.24 |                   1.23 |     1.04 |       0.68 |          1.7  |
| ml_ridge_int        |                    1.25 |                   1.22 |     1.05 |       0.7  |          1.68 |
| ml_swing_muni       |                    1.33 |                   1.36 |     1.24 |       0.87 |          1.82 |
| ml_wapo             |                    2.52 |                   1.94 |     2.61 |       1.59 |          3.12 |
| ml_wapo_int         |                    2.53 |                   2.03 |     2.7  |       1.49 |          2.96 |

### prov_mae by level

| method              |   5.0 |   10.0 |   21.4 |   21.7 |   25.0 |   50.0 |
|:--------------------|------:|-------:|-------:|-------:|-------:|-------:|
| ml_gbm              |  1.17 |   1.21 |   0.31 |   0.13 |   0.63 |   0.43 |
| ml_gbm_change       |  1.17 |   1.25 |   0.32 |   0.18 |   0.64 |   0.42 |
| ml_gbm_level        |  4.22 |   3.95 |   0.3  |   0.21 |   2.37 |   1.29 |
| ml_knn              |  1.22 |   1.05 |   0.44 |   0.32 |   0.66 |   0.42 |
| ml_krig             |  1.21 |   1.04 |   0.38 |   0.23 |   0.66 |   0.43 |
| ml_mixed            |  1.37 |   1.05 |   0.37 |   0.09 |   0.64 |   0.39 |
| ml_mixed_gbm        |  1.27 |   1.1  |   0.32 |   0.08 |   0.63 |   0.41 |
| ml_mixed_gbm_shift  |  1.27 |   1.1  |   0.32 |   0.08 |   0.63 |   0.41 |
| ml_mixed_look       |  1.36 |   1    |   0.36 |   0.12 |   0.63 |   0.39 |
| ml_mixed_look_shift |  1.36 |   1    |   0.36 |   0.12 |   0.63 |   0.39 |
| ml_mixed_shift      |  1.37 |   1.05 |   0.37 |   0.09 |   0.64 |   0.39 |
| ml_mixed_slope      |  1.43 |   1.41 |   0.28 |   0.14 |   0.99 |   0.71 |
| ml_ridge            |  1.49 |   1.79 |   0.38 |   0.15 |   0.93 |   0.53 |
| ml_ridge_int        |  1.49 |   1.72 |   0.36 |   0.2  |   0.94 |   0.54 |
| ml_swing_muni       |  1.24 |   1.08 |   0.41 |   0.4  |   0.71 |   0.51 |
| ml_wapo             |  3.4  |   3.44 |   0.7  |   0.58 |   1.56 |   0.77 |
| ml_wapo_int         |  3.24 |   3.03 |   0.75 |   0.58 |   1.41 |   0.7  |

### prov_cov90 by level

| method              |   5.0 |   10.0 |   21.4 |   21.7 |   25.0 |   50.0 |
|:--------------------|------:|-------:|-------:|-------:|-------:|-------:|
| ml_gbm              |  0.69 |   0.57 |   0.67 |   1    |   0.71 |   0.76 |
| ml_gbm_change       |  0.64 |   0.64 |   0.5  |   1    |   0.69 |   0.75 |
| ml_gbm_level        |  0.61 |   0.6  |   0.5  |   1    |   0.58 |   0.74 |
| ml_knn              |  0.67 |   0.6  |   0.5  |   0.83 |   0.73 |   0.73 |
| ml_krig             |  0.69 |   0.67 |   0.67 |   1    |   0.71 |   0.7  |
| ml_mixed            |  0.86 |   0.86 |   0.67 |   1    |   0.81 |   0.8  |
| ml_mixed_gbm        |  0.86 |   0.83 |   1    |   1    |   0.81 |   0.7  |
| ml_mixed_gbm_shift  |  0.93 |   0.96 |   1    |   1    |   0.95 |   0.9  |
| ml_mixed_look       |  0.85 |   0.85 |   0.83 |   1    |   0.8  |   0.79 |
| ml_mixed_look_shift |  0.9  |   0.96 |   1    |   1    |   0.94 |   0.92 |
| ml_mixed_shift      |  0.92 |   0.94 |   0.83 |   1    |   0.95 |   0.93 |
| ml_mixed_slope      |  0.82 |   0.81 |   0.83 |   1    |   0.75 |   0.7  |
| ml_ridge            |  0.69 |   0.69 |   0.67 |   0.83 |   0.7  |   0.73 |
| ml_ridge_int        |  0.68 |   0.69 |   0.67 |   0.83 |   0.7  |   0.74 |
| ml_swing_muni       |  0.72 |   0.68 |   0.83 |   0.67 |   0.76 |   0.7  |
| ml_wapo             |  0.67 |   0.57 |   0.5  |   0.83 |   0.71 |   0.76 |
| ml_wapo_int         |  0.74 |   0.65 |   0.67 |   1    |   0.73 |   0.75 |

### new parties |err| pp

| method              |   ActionSA LGE2021 (no history) |   MK PE2024 (no history) |   ActionSA PE2024 (metro-only history) |
|:--------------------|--------------------------------:|-------------------------:|---------------------------------------:|
| ml_gbm              |                            0.89 |                     1.46 |                                   0.28 |
| ml_gbm_change       |                            1    |                     1.3  |                                   0.26 |
| ml_gbm_level        |                            1    |                     1.38 |                                   0.28 |
| ml_knn              |                            0.74 |                     1.77 |                                   0.22 |
| ml_krig             |                            0.79 |                     1.72 |                                   0.2  |
| ml_mixed            |                            1.41 |                     1.47 |                                   0.23 |
| ml_mixed_gbm        |                            1.45 |                     1.5  |                                   0.28 |
| ml_mixed_gbm_shift  |                            1.45 |                     1.5  |                                   0.28 |
| ml_mixed_look       |                            1.41 |                     1.46 |                                   0.24 |
| ml_mixed_look_shift |                            1.41 |                     1.46 |                                   0.24 |
| ml_mixed_shift      |                            1.41 |                     1.47 |                                   0.23 |
| ml_mixed_slope      |                            1.32 |                     1.79 |                                   0.17 |
| ml_ridge            |                            1.07 |                     1.34 |                                   0.3  |
| ml_ridge_int        |                            1.11 |                     1.26 |                                   0.34 |
| ml_swing_muni       |                            0.69 |                     2.12 |                                   0.18 |
| ml_wapo             |                            3.08 |                     3.99 |                                   0.77 |
| ml_wapo_int         |                            2.54 |                     3.62 |                                   0.73 |

### by case

| method              |   ('prov_cov90', 'LGE2021') |   ('prov_cov90', 'PE2024') |   ('prov_mae_pp', 'LGE2021') |   ('prov_mae_pp', 'PE2024') |
|:--------------------|----------------------------:|---------------------------:|-----------------------------:|----------------------------:|
| ml_gbm              |                        0.72 |                       0.67 |                         0.89 |                        0.73 |
| ml_gbm_change       |                        0.71 |                       0.66 |                         0.91 |                        0.74 |
| ml_gbm_level        |                        0.65 |                       0.63 |                         3    |                        2.55 |
| ml_knn              |                        0.68 |                       0.69 |                         0.84 |                        0.76 |
| ml_krig             |                        0.69 |                       0.71 |                         0.83 |                        0.75 |
| ml_mixed            |                        0.82 |                       0.84 |                         0.91 |                        0.71 |
| ml_mixed_gbm        |                        0.81 |                       0.8  |                         0.89 |                        0.72 |
| ml_mixed_gbm_shift  |                        0.95 |                       0.93 |                         0.89 |                        0.72 |
| ml_mixed_look       |                        0.8  |                       0.84 |                         0.89 |                        0.7  |
| ml_mixed_look_shift |                        0.94 |                       0.92 |                         0.89 |                        0.7  |
| ml_mixed_shift      |                        0.94 |                       0.93 |                         0.91 |                        0.71 |
| ml_mixed_slope      |                        0.81 |                       0.73 |                         1.17 |                        0.99 |
| ml_ridge            |                        0.73 |                       0.69 |                         1.41 |                        0.82 |
| ml_ridge_int        |                        0.73 |                       0.69 |                         1.39 |                        0.82 |
| ml_swing_muni       |                        0.77 |                       0.67 |                         0.84 |                        0.85 |
| ml_wapo             |                        0.71 |                       0.65 |                         2.38 |                        1.91 |
| ml_wapo_int         |                        0.73 |                       0.71 |                         2.13 |                        1.8  |

### seconds per projection (median, max)

| method              |   median |   mean |   max |
|:--------------------|---------:|-------:|------:|
| ml_gbm              |     28.6 |   31.6 |  69.1 |
| ml_gbm_change       |     31.5 |   40.7 | 128.5 |
| ml_gbm_level        |     16.4 |   21.5 |  70.7 |
| ml_knn              |      2.4 |    2.5 |   5.4 |
| ml_krig             |     40.4 |   74.6 | 343.5 |
| ml_mixed            |      7.6 |    7.5 |  13.1 |
| ml_mixed_gbm        |     22.5 |   27.3 |  80.8 |
| ml_mixed_gbm_shift  |     16.8 |   16.4 |  34.8 |
| ml_mixed_look       |     21.9 |   22.9 |  55.2 |
| ml_mixed_look_shift |     22.8 |   23.3 |  54.4 |
| ml_mixed_shift      |      9   |   11.3 |  35.9 |
| ml_mixed_slope      |     12.8 |   14.1 |  38.6 |
| ml_ridge            |      3   |    3.9 |  10.2 |
| ml_ridge_int        |      2.7 |    2.9 |   6.1 |
| ml_swing_muni       |      1.1 |    1.2 |   2.5 |
| ml_wapo             |      3   |    3   |   7   |
| ml_wapo_int         |      1.7 |    2   |   7.2 |

### worst projections for ml_mixed_gbm_shift

| case    | order                 |   pct_counted |   prov_mae_pp |   prov_cov90 |   err_DA |   err_ANC |   err_ACTIONSA |   err_MK |   err_EFF |   err_OTHER |
|:--------|:----------------------|--------------:|--------------:|-------------:|---------:|----------:|---------------:|---------:|----------:|------------:|
| LGE2021 | DA_strongholds_first  |             5 |          3.66 |         0.83 |     6.73 |     -4.4  |          -4.25 |     0    |     -2.33 |        4.26 |
| PE2024  | DA_strongholds_first  |            10 |          3.52 |         0.67 |    -0.24 |     -2.17 |           0.67 |    -8.16 |      7.17 |        2.73 |
| PE2024  | DA_strongholds_first  |             5 |          3.26 |         0.83 |     0.58 |     -1.46 |           1.27 |    -8.32 |      6.31 |        1.63 |
| LGE2021 | ANC_strongholds_first |             5 |          3.08 |         0.83 |    -1.32 |     -1.85 |          -6.06 |     0    |      4.08 |        5.15 |
| LGE2021 | ANC_strongholds_first |            10 |          3.01 |         1    |    -3.33 |      0.92 |          -5.69 |     0    |      3.87 |        4.23 |
| LGE2021 | ANC_strongholds_first |            50 |          2.55 |         0.83 |    -3.11 |      2.65 |          -4.53 |     0    |      1.68 |        3.32 |
| PE2024  | DA_strongholds_first  |            25 |          2.48 |         1    |    -0.3  |     -0.75 |          -0.02 |    -6.39 |      4.75 |        2.7  |
| LGE2021 | DA_strongholds_first  |            10 |          2.36 |         1    |     6.43 |     -1.28 |          -5.01 |     0    |     -0.8  |        0.66 |

## live_check (artifacts/bench/live_ml.csv; all methods pinned to the 14:45 snapshot, 728 counted VDs)

Baselines (average group error, pp): group tally 9.9 / uniform_swing_muni 4.9; muni 4.7 / 4.8; random 0.8 / 1.6; time 2.1 / 1.5.

### error (pp) by task kind

| method              |   group |   muni |   random |   time |
|:--------------------|--------:|-------:|---------:|-------:|
| ml_gbm              |    1.59 |   1.7  |     0.28 |   0.21 |
| ml_knn              |    2.62 |   2.14 |     0.39 |   0.27 |
| ml_mixed            |    1.77 |   2    |     0.3  |   0.13 |
| ml_mixed_gbm_shift  |    1.58 |   1.8  |     0.23 |   0.25 |
| ml_mixed_look       |    1.7  |   1.98 |     0.26 |   0.17 |
| ml_mixed_look_shift |    1.7  |   1.98 |     0.26 |   0.17 |
| ml_mixed_slope      |    1.84 |   2.1  |     0.33 |   0.27 |
| ml_ridge_int        |    2.18 |   2.24 |     0.27 |   0.24 |
| ml_swing_muni       |    2.88 |   2.21 |     0.39 |   0.26 |
| ml_wapo             |    1.95 |   2.16 |     0.57 |   0.4  |
| ml_wapo_int         |    2.01 |   1.92 |     0.48 |   0.58 |

### group_cov90 by task kind

| method              |   group |   muni |   random |   time |
|:--------------------|--------:|-------:|---------:|-------:|
| ml_gbm              |    0.62 |   0.88 |     0.94 |      1 |
| ml_knn              |    0.46 |   0.79 |     1    |      1 |
| ml_mixed            |    0.82 |   0.85 |     0.94 |      1 |
| ml_mixed_gbm_shift  |    0.91 |   0.9  |     1    |      1 |
| ml_mixed_look       |    0.84 |   0.87 |     0.94 |      1 |
| ml_mixed_look_shift |    0.89 |   0.92 |     1    |      1 |
| ml_mixed_slope      |    0.77 |   0.79 |     1    |      1 |
| ml_ridge_int        |    0.52 |   0.75 |     0.94 |      1 |
| ml_swing_muni       |    0.46 |   0.81 |     0.83 |      1 |
| ml_wapo             |    0.65 |   0.58 |     0.94 |      1 |
| ml_wapo_int         |    0.58 |   0.6  |     0.94 |      1 |

## Failure cases (recommended method)

* **Stronghold-first orders.** The order is sorted on the predictor itself, so the data are missing
  not at random and nothing in the count describes the uncounted areas.
  * LGE2021 DA-first at 5–10%: DA is over-projected by 6–7 pp and ActionSA under-projected by
    4–5 pp. ActionSA has no history, and its level in DA suburbs says little about townships.
  * LGE2021 ANC-first: ActionSA is under-projected by 4.5–6 pp, and OTHER and EFF are
    over-projected by 4 pp, even at 50% counted.
  * PE2024 DA-first: MK is under-projected by 6–8 pp and EFF over-projected by 5–7 pp. MK has no
    history and is concentrated in the not-yet-counted townships.
  * The order-shift shock makes these intervals wide (about 10 pp), so coverage there is
    0.85–0.92. That is honest about the uncertainty, but the mean is still off.
* **New parties under biased orders** are the weak spot. Under random or real orders they are fine:
  the mean |error| of ActionSA 2021 and MK 2024 is 0.1–0.4 pp at every level from 5% counted.
* **The mixed base cuts both ways for no-history parties under biased orders.**
  * ActionSA 2021: the simple swing (0.69) and knn (0.74) beat mixed (1.41–1.45). Type and cell
    effects estimated on unrepresentative counted VDs hurt a pure level target.
  * MK 2024 goes the other way: mixed 1.47, swing 2.12.
  * For 2026, every party has PE2024 history, so this matters less.
* **live_check.** The worst groups are small townships outside the metros, e.g. Lesedi (GT423)
  townships: 3.6 pp, with MK +5.5 and EFF −4. With few counted look-alikes there, the shock and
  the cell prior widen the intervals, but not always enough: GT423, GT484 and EKU townships have
  group cov90 0.67–0.83.
* **Over-coverage on random/real orders.** Province cov90 is 0.97–0.98, with mean width 1.1–1.5 pp.
  The shift shock is never exactly zero, and the province coverage in LGE2021 counts MK (always
  covered).

## Run time per projection

The `seconds` column is wall-clock time measured while 3–4 other agents shared the 4 cores (load
8–14), so it is about 2–3 times the unloaded time.

| method | median seconds per projection (loaded) |
|---|---|
| `ml_swing_muni` | about 1 |
| `ml_knn` | 2.4 |
| `ml_ridge*`, `ml_wapo*` | 2–4 |
| `ml_mixed` | 7.6 |
| `ml_mixed_shift` | about 11 |
| `ml_mixed_slope` | 13 |
| **`ml_mixed_gbm_shift`** | **16.8** (max 35) |
| `ml_mixed_look*` | 22 |
| `ml_gbm` | 29 |
| `ml_gbm_change` | 32 |
| `ml_krig` | 40 (max 340) |

See the seconds table above for all methods.

* The cost is the 10-fold ward-grouped CV, which is needed for out-of-fold residuals, GREG and
  the jackknife.
* REML uses an analytic gradient and is fitted on at most 450 VDs.
* Early in the session, under light load, `ml_swing_muni` took 1–3 s and `ml_mixed` 3–18 s per
  projection.

## What the harness gets wrong (or makes hard to read)

1. **MK in LGE2021.**
   * MK did not stand, so `err_MK` = 0 and MK is always "covered". `prov_mae_pp` and
     `prov_cov90` average over 6 parties, so LGE2021 MAE is diluted by 1/6 and its coverage is
     inflated.
   * Coverage takes only the values k/6 per projection.
   * Report coverage per party-projection over contested parties only.
2. **The mean prov_mae is dominated by the two stronghold-first orders.** They are deterministic
   sorts on the previous PE share of the very party being scored: extreme MNAR that no
   swing/ML model can see through. Rankings on random/real orders differ: ridge and GBM win
   there. Report the mean by order type, and give the real order more weight.
3. **`real2026` reads every IEC file present when `orders()` runs.** The fetcher keeps adding
   snapshots, so two runs (or two workers of one run) can replay different first-reported sets.
   Here the set was stable at 637 VDs, but `live_check` moved from 728 to 794 counted VDs during
   my runs. I pinned the live results to the 14:45 snapshot. Pin snapshots in both.
4. **PE2024 `registered`.** It is `electorate = max(registered, ballots cast)`, so for 13 s24A
   VDs the target election's ballots leak into the "known" electorate. This is minor.
5. **No arrival order is passed to methods.** Adaptive conformal (ACI) and "score each new batch
   against its pre-arrival projection" cannot be tested in the harness, only emulated (my
   propensity-ranked pseudo-arrival check). Passing the order or arrival snapshot index for the
   counted VDs would allow real ACI.
6. **Intervals are scored only at province and municipality level, plus ward Brier.** A cell-level
   (area type × municipality) coverage like `live_check`'s `group_cov90` is the more relevant
   test for still-uncounted areas. It is not in the backtest.
7. **`seconds` depends on machine load**, so it is not comparable across families run at different
   times.
