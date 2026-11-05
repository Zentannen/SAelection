# v_ward: pulling uncounted VDs toward their counted ward-mates

Code: `src/saelection/bench/m_vward.py`. Best method: **`m_vward:ward_pull`**.
Live check: `artifacts/bench/live_v_ward.csv` (pinned snapshot `IEC_LGE2026_GP_20261105_1518.csv`, all 28
tasks; `ward_pull` and `ward_eb_rake` from `live_check --tag v_ward`, plus `hybrid_cf` from a
`live_check.run` at the same snapshot with the same code). Benchmark: `artifacts/bench/results_v_ward.csv` (both
cases, all orders, `--n-random 3`; `ward_pull` and `m_swing:hybrid_cf`, one process per case through
`harness.run`, then concatenated).

## Bottom line

* **The flagged wards are mostly a reporting-order effect, not a model failure.** In every flagged ward
  with a counted VD, that VD was already the ward's ActionSA stronghold before 2026. In 79700053 the counted VD
  (33000499) had 37.6% ActionSA in LGE2021 and now has 48.5%. Six of its seven uncounted ward-mates had 2-8%
  in LGE2021, and the seventh had 24%. In 79700024 two of the three uncounted VDs are DA strongholds (DA
  76-81%, ActionSA 0-6% in 2021/2024). So the counted VD's out-of-fold residual is small or negative (-0.66
  on the log-ratio scale for 33000499), and a correct pull hardly moves those wards. Two other flagged wards,
  79700061 and 79900057, have **no** counted VD in the model at all. The VDs that reported there
  (33000523, 32980620) are not in the model's 2026 VD set; 35 such counted VDs exist at 15:18. `live.py`
  adds their votes to the ward totals without modelling them, so no pull can come from them.
* **Projecting uncounted VDs from their counted ward-mates' shares, as the ward map invites, is about 1.7x
  worse.** Test: in the live hold-outs (time, random x3), take the held VDs that have counted ward-mates
  (115-127 VDs in 89-99 wards per task). Predicting each one as the counted ward-mates' share gives ActionSA
  ward MAE 6.37 pp. `hybrid_cf` gives 3.77 pp and the unraked spatial pull 3.85 pp. Over all parties the
  ward MAE is 6.31 (ward-mates) vs 3.17 (`hybrid_cf`) vs 3.14 (pull). Out of fold, `hybrid_cf`'s ActionSA
  is calibrated at every level: predicted 2.9 / 7.1 / 11.9 / 17.9 / 24.7 / 34.3 / 46.3% vs actual
  3.8 / 7.1 / 12.3 / 18.9 / 25.7 / 34.6 / 45.3% (vote-weighted, by predicted-share bin, 773 counted VDs).
* **A ward-mate pull still helps a little everywhere wards matter, and does no harm at municipality
  level.** `ward_pull` against `hybrid_cf` on the benchmark:
  * ward MAE 2.32 vs 2.39 pp
  * VD MAE 3.55 vs 3.63
  * ward Brier 0.090 vs 0.099 (-9%)
  * CRPS 0.84 vs 0.87
  * ward-winner accuracy 0.952 vs 0.951

  It is better on ward MAE and ward Brier in every order (random, real2026, stronghold-first, small-first).
  The cost is +0.014 pp on province MAE (0.891 vs 0.877) and on municipality MAE (1.313 vs 1.298). This
  comes from the base dropping hybrid_cf's own ward level, because `ward_pull` keeps the base's municipal
  totals. Live check (15:18), `ward_pull` vs `hybrid_cf`:
  * group: 1.61 vs 1.64 pp; VD leader accuracy 0.864 vs 0.845; group cov90 0.93 vs 0.89; VD cov90 0.91 vs 0.87
  * muni: 2.04 vs 2.05
  * random: 0.52 vs 0.52
  * time: 0.20 vs 0.12 (167 VDs, both are noise-level); time VD MAE 3.25 vs 3.33
* **Recommended cheap next step (not run, out of time):** rake the pulled VDs to *`hybrid_cf`'s*
  municipal totals instead of the no-ward base's. Province and municipality results would then equal
  `hybrid_cf`'s exactly, and the ward and VD gains would stay. It costs one extra `hybrid_cf` call per
  projection.

## Method (`ward_pull`)

1. **Base** = `hybrid_cf` without its ward level: `cluster_muni_ratio` on km20 + municipality + km20 x
   municipality cells, plus the Curtice-Firth ridge with municipality EB effects only, with the same
   distance hand-over (c = 2). The ward level is removed because the pull replaces it; keeping both would
   count the ward twice.
2. **Out-of-fold residuals** of the counted VDs. The counted wards are split into 5 folds, and all counted
   VDs of a ward go in one fold. The base is refitted without each fold and predicts it. The residual is
   r_i = alr(Y_i) - alr(base_oof_i), where alr is the log-ratio of each party vs ABSTAIN. Neither the VD's
   own result nor its ward-mates' enters the prediction it is compared with.
3. **Covariance of the residuals by moments**, per party. Residuals are centred at their
   precision-weighted mean and winsorised at 3 SD, so one broken-history VD cannot drag its ward. Products
   of residuals over counted pairs (same ward, or within 25 km) are regressed with NNLS on [same ward] and
   exp(-d/ell), where d is the centroid distance and ell is chosen from a grid of 0.5-8 km. The fit is
   Cov = tw [same ward] + ts exp(-d/ell), and Var = tw + ts + s2 + v_i, where v_i is the sampling variance
   of alr(Y_i). At 15:18 the fit gives:
   * ActionSA: tw 0.019, ts 0.097, ell 3 km, s2 0.10, against a raw residual variance of 0.36
   * MK: ts 0.20, ell 1 km

   The same-ward correlation is mostly a short-range spatial correlation. Two counted VDs under 1 km apart
   in *different* wards co-vary as much as ward-mates. That is why the spatial version beats the pure ward
   version.
4. **Simple kriging (BLUP)** of each uncounted VD's residual from all counted VDs. With the ward term alone
   (`ward_eb`) this is exactly the James-Stein / EB ward mean shrunk to 0 by n_counted and tw/s2, i.e.
   u_w = tw * sum_i r_i/(s2+v_i) / (1 + tw * sum_i 1/(s2+v_i)). For a ward with one counted VD, ActionSA
   is pulled by about 20-35% of that VD's residual. The variance components are moment estimates, not
   tuned. A `gain` multiplier of 0.5-3 was screened on the live check (table below) and 1 was kept.
5. The shift is added on the alr scale. Then, per municipality and category, the uncounted VDs are
   **raked** (IPF, each VD keeps its electorate) back to the base's total. A zero-mean shift on the alr
   scale is not zero-mean in votes. In the LGE2021 backtest, where ActionSA is a new party with large
   skewed residuals, the unraked pull added +0.7 pp ActionSA at province level (random 10%: province MAE
   0.31 -> 0.45). Raking removes that bias and keeps the ward/VD gain (LGE2021 random 25%: ward MAE 1.91 ->
   1.68, VD MAE 3.37 -> 3.09).
6. **Draws.** Each base draw is shifted by the kriged mean plus a joint draw from the kriging posterior.
   The posterior is a Cholesky factor of the posterior covariance of the residual field over all uncounted
   VDs, per party. It is wide for wards with no counted VD (it gives back the ward variance the base
   dropped), narrow next to counted ward-mates, and correlated within wards and space. Each draw is then
   raked to that base draw's municipal totals.

## Variants (all in `m_vward.py`)

| method | what |
|---|---|
| `base_nw` | base only (no pull) |
| `ward_eb` | ward-level EB pull, no raking |
| `ward_cell` | two-level: ward, plus municipality x area-type (km20) cell, no raking |
| `ward_sp` | ward + spatial kriging, no raking |
| `ward_sp_cell` | ward + cell + spatial, no raking |
| `ward_sp_fill` | `ward_sp` on a case whose LGE2021 ActionSA is filled where it did not stand (cause 2) |
| `ward_eb_rake` | `ward_eb` raked to municipal totals |
| **`ward_pull`** | `ward_sp` raked to municipal totals (best) |

Live-check screening of the point projections at 15:18. "No raking" here means the base draws were only
shifted, without the kriging posterior draws, so cov90 is not comparable.

| variant | group | muni | random | time | VD MAE group / random / time |
|---|---|---|---|---|---|
| hybrid_cf | 1.638 | 2.048 | 0.520 | 0.122 | 3.598 / 3.367 / 3.330 |
| base_nw | 1.691 | 2.035 | 0.545 | 0.222 | 3.670 / 3.440 / 3.474 |
| ward_eb (gain 1) | 1.632 | 2.035 | 0.536 | 0.166 | 3.600 / 3.370 / 3.326 |
| ward_eb (gain 2) | 1.612 | 2.035 | 0.547 | 0.135 | 3.611 / 3.419 / 3.301 |
| ward_cell | 1.628 | 2.035 | 0.549 | 0.155 | 3.603 / 3.372 / 3.321 |
| ward_sp gain 0.5 | 1.639 | 2.038 | 0.465 | 0.175 | 3.586 / 3.324 / 3.252 |
| **ward_sp gain 1** | 1.622 | 2.035 | 0.482 | 0.157 | 3.581 / 3.348 / 3.226 |
| ward_sp gain 2 | 1.609 | 2.028 | 0.500 | 0.154 | 3.595 / 3.390 / 3.223 |
| ward_sp gain 3 | 1.605 | 2.022 | 0.510 | 0.154 | 3.607 / 3.420 / 3.236 |
| ward_sp_cell | 1.624 | 2.035 | 0.480 | 0.157 | 3.582 / 3.356 / 3.226 |

The cell level adds nothing: its variance component is about 0, because the base already has km20 x
municipality cell effects. The ward-only version has no reach beyond the ward boundary, so the spatial term
is the useful part.

## Final live check (snapshot 15:18, `live_check.summarise`)

| method | kind | group_mae | vd_mae | vd_leader_acc | group_cov90 | vd_cov90 | s |
|---|---|---|---|---|---|---|---|
| hybrid_cf | group | 1.638 | 3.598 | 0.845 | 0.892 | 0.873 | 2.3 |
| hybrid_cf | muni | 2.048 | 4.434 | 0.801 | 0.896 | 0.848 | 3.0 |
| hybrid_cf | random | 0.520 | 3.367 | 0.831 | 1.000 | 0.895 | 3.2 |
| hybrid_cf | time | 0.122 | 3.330 | 0.868 | 1.000 | 0.902 | 7.2 |
| **ward_pull** | group | **1.607** | **3.575** | **0.864** | 0.931 | 0.907 | 12.6 |
| **ward_pull** | muni | **2.041** | **4.432** | 0.802 | 0.917 | 0.884 | 11.3 |
| **ward_pull** | random | 0.521 | 3.375 | 0.833 | 1.000 | 0.926 | 9.1 |
| **ward_pull** | time | 0.195 | **3.247** | 0.862 | 1.000 | 0.943 | 19.1 |
| ward_eb_rake | group | 1.640 | 3.605 | 0.849 | 0.912 | 0.894 | 10.2 |
| ward_eb_rake | muni | 2.035 | 4.429 | 0.801 | 0.917 | 0.865 | 8.7 |
| ward_eb_rake | random | 0.559 | 3.378 | 0.825 | 1.000 | 0.915 | 7.7 |
| ward_eb_rake | time | 0.163 | 3.340 | 0.868 | 1.000 | 0.933 | 12.3 |

Mean absolute ActionSA error of the held-out group (pp), group / muni / random / time:
* `hybrid_cf`: 1.74 / 4.33 / 0.34 / 0.11
* `ward_pull`: 1.61 / 4.30 / 0.24 / 0.31

The large ActionSA misses are municipal-level: JHB held out +15 pp, TSH -6, EKU -4. They come from the
ActionSA swing differing by municipality relative to its 2021/2024 base. A ward pull cannot fix that.

## Benchmark (`harness.summarise`, both cases, all orders, n_random 3)

| method | prov_mae | prov_max | muni_mae | ward_acc | ward_mae | vd_mae | vd_turnout | prov_cov90 | crps | muni_cov90 | ward_brier | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_cf | **0.877** | 1.968 | **1.298** | 0.951 | 2.389 | 3.628 | 5.475 | 0.944 | 0.872 | 0.847 | 0.099 | 16.5 |
| ward_pull | 0.891 | 2.010 | 1.313 | **0.952** | **2.320** | **3.547** | **5.393** | 0.978 | **0.839** | 0.858 | **0.090** | 30.0 |

Random + real2026 orders only:

| method | prov_mae | muni_mae | ward_acc | ward_mae | vd_mae | prov_cov90 | ward_brier |
|---|---|---|---|---|---|---|---|
| hybrid_cf | 0.251 | 0.743 | 0.963 | 1.768 | 3.076 | 0.967 | 0.064 |
| ward_pull | 0.274 | 0.754 | 0.965 | 1.683 | 2.977 | 1.000 | 0.058 |

Ward MAE and ward Brier by order, `hybrid_cf` -> `ward_pull`:

| order | ward MAE | ward Brier |
|---|---|---|
| random_s0 | 1.83 -> 1.76 | 0.065 -> 0.059 |
| random_s1 | 1.79 -> 1.71 | 0.067 -> 0.059 |
| random_s2 | 1.80 -> 1.70 | 0.067 -> 0.060 |
| real2026 | 1.62 -> 1.53 | 0.054 -> 0.050 |
| small_first | 2.49 -> 2.38 | 0.115 -> 0.095 |
| DA_strongholds_first | 3.93 -> 3.92 | 0.163 -> 0.146 |
| ANC_strongholds_first | 3.08 -> 3.05 | 0.152 -> 0.149 |

The pull helps less under the stronghold-first orders: residuals of counted strongholds do not carry over
to their non-stronghold neighbours. By case, ward MAE goes 2.56 -> 2.47 in LGE2021 and 2.22 -> 2.17 in
PE2024.

## The flagged wards

The live case at 15:18 counts 773 VDs. A ward is the sum of its VDs, with counted VDs at their actual
result and uncounted VDs projected; 90% intervals are from the draws. All shares are ActionSA.

| ward | VDs (counted) | hybrid_cf | ward_pull | counted VDs |
|---|---|---|---|---|
| 79700053 | 8 (1) | 29.3 [20.2, 40.2] | 29.3 [18.3, 42.4] | 48.5 |
| 79700061 | 5 (0) | 18.3 [11.7, 26.2] | 16.4 [8.2, 27.8] | none in model (42.6 in unmatched VD 33000523, 122 votes) |
| 79900057 | 3 (0) | 22.4 [13.6, 35.4] | 21.1 [11.6, 36.9] | none in model (49.9 in unmatched VD 32980620, 1136 votes) |
| 79700024 | 4 (1) | 16.2 [11.2, 21.5] | 15.4 [10.5, 24.0] | 33.6 |
| 74801006 | 3 (1) | 10.3 [6.1, 18.1] | 10.0 [5.7, 16.6] | 29.3 |
| 79700099 | 6 (2) | 25.1 [18.8, 33.3] | 23.9 [16.5, 34.4] | 23.7 |

`ward_sp_fill`, which fills LGE2021 ActionSA in Sedibeng and West Rand at 2.6 x its PE2024 share, moves
these wards by at most 1.7 pp against `ward_sp`, mostly *down* (79700053: 29.7 -> 28.0); it was not
benchmarked. Cause (2) in the brief is weak in the data. The
per-party LGE/PE blend fitted on the counted VDs puts weight 0-0.2 on LGE2021 for ActionSA. The VD-level
correlation of log 2026 ActionSA share with log LGE2021 vs log PE2024 share is 0.51 vs 0.57 in JHB, 0.47 vs
0.50 in EKU and 0.64 vs 0.49 in TSH.

## Risks / failure cases

* `ward_pull` keeps the base's municipal totals, which are slightly worse than `hybrid_cf`'s on the
  benchmark (province MAE +0.014). See the recommended anchor-to-`hybrid_cf` step above.
* Province intervals are conservative on the benchmark: prov cov90 0.98, against 0.94 for `hybrid_cf`.
  Live group cov90 is 0.93 and VD cov90 0.91. The kriging posterior adds correlated variance on top of the
  base's independent VD residual draws, which already carry the same variance.
* The unraked pull is biased for a party with large, skewed log-ratio residuals (a new party). Raking
  fixes this; keep it on.
* Under stronghold-first orders the spatial pull can carry a stronghold's residual into unlike
  neighbours. The effect is small on average (ward MAE 3.93 -> 3.92), but it is a risk if the real count
  order is extreme.
* Counted VDs missing from the model's VD set (35 at 15:18) cannot pull anything. A fix in `live.py` would
  give them a residual against the ward's projected composition.
* Run time: about 30 s per projection on the loaded shared machine (16 s for `hybrid_cf`); about 6-12 s
  when idle. The cost is 5 out-of-fold base fits plus 6 Cholesky factorisations of the uncounted x
  uncounted posterior covariance.
