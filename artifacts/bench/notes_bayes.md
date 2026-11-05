# Bayesian hierarchical and spatial models (`--tag bayes`)

* Methods: `src/saelection/bench/m_bayes.py`
* Benchmark results: `artifacts/bench/results_bayes.csv` (both cases, all orders, 3 random seeds)
* Live-check results: `artifacts/bench/live_bayes.csv` (live_check on the 2026 count; tasks group, time and random)

Every method returns 300 posterior-predictive draws per VD: parameter uncertainty, VD residual and
multinomial noise.

**Best method: `car_exact_cal`.** It is the repo's model with three changes:
* exact Gaussian inference;
* an area-type × municipality swing term, plus count-adaptive slopes;
* intervals widened by a % counted inflation schedule.

`car_exact_cmb` is the same model without the inflation. It has the best point accuracy, with 0.70
province coverage. `hier_tcm8`, the learned-from-count hierarchical model, is the best on live_check
accuracy. It breaks under strongly biased counting orders, so it suits a second ensemble member rather
than the primary model.

## 0. Hyper-parameter files (coordinator's point f)

I re-ran `python scripts/run_train.py` (4 min on 1 thread). It re-created `artifacts/hyper_T1.npz`,
`hyper_T2.npz`, `spatial_test.json` and `training_summary.json`. **The new files are the ones to use.**

The old files did not match the code, for two reasons:
* They were fitted with OTHER's CAR field off. The current `spatial_test.json` keeps it on.
* They came from the older model, in which a switched-off CAR field stayed in as an iid VD effect.
  - The current `model.py` and `live.py` drop switched-off fields entirely, and psi carries that variation instead.
  - So the old files had psi sd (tau) of 0.021 for ANC, 0.031 for EFF and 0.029 for OTHER. Their VD variation sat in the dropped field.
  - With the current `live.py`, ANC, EFF and OTHER therefore had almost no VD residual.
  - The new T1 values are tau = 0.435 (ANC), 0.551 (EFF) and 0.548 (OTHER).

The spatial hold-out test reproduced:
* DA, ActionSA, MK, OTHER and the turnout field keep the CAR term.
* ANC and EFF do not.

The 13:00 live output in `output/live/` (generated 11:08 UTC) used the old files.

## 1. What was tried

| method | what it is |
|---|---|
| `car_full` | (a) The repo model via `live.project`: Leroux CAR, municipal swing and LKJ residuals. Structural parameters are fixed at the training posterior mean. Mean-field SVI, 5000 steps. Leak-free hyper: PE2024 <- `merged_hyper(("T1",))`; LGE2021 <- **T2 only**. T2 is a later transition of the other election type, so it is weaker, but nothing from 2021 enters the fitted values. The 6 per-party CAR on/off switches come from the T1 hold-out test, which saw 2021 data; that leak is negligible. Naive prior, as in `backtest.py`: the previous same-type election's Gauteng result (LGE2016 for LGE2021, PE2019 for PE2024), A_sd 0.5. A party with no history gets a 2% share and A_sd 1.0. |
| `car_full_fast` | car_full with 1500 SVI steps |
| `car_off` | (b) No CAR for any party. Structural parameters are **re-trained without CAR** (cached in `artifacts/bench/cache_bayes/`), so psi carries the VD variation. Switching the fields off in CAR-trained hyper-parameters would understate it. |
| `car_exact` | (a') Same model and hyper-parameters, but the **exact posterior**. Given the hyper-parameters the live model is linear-Gaussian. At counted VDs, psi plus the sampling noise is integrated out. (A, delta, phi on every VD) is drawn with one sparse LU and 300 solves ("perturb-then-solve"). No SVI. |
| `car_exact_b` | car_exact plus slopes re-estimated from the count with prior sd 0.25 and no guard. Rejected. |
| `car_exact_cm` | car_exact plus an **area-type swing**: cluster × party (sd 0.3) and cluster × municipality × party (sd 0.2) effects. The clusters are 8 k-means clusters on standardised per-party shares and turnout from both predictor elections. A cell with no counted VD gets a prior draw, so it is wider where nothing comparable has been counted. The municipal-swing sd has a floor of 0.15 (T2's is 0.002 for EFF). Broken histories are cleaned. |
| `car_exact_cmb` | car_exact_cm plus slopes **and substitution coefficients** that adapt to the count. The trained values are the prior means, with an adjustment sd of 0.1. A party with no history gets slopes on all 12 past alr columns (sd 0.25). Extrapolation guard: each column's adjustment sd shrinks linearly to 0 as the standardised mean difference between counted and uncounted VDs on that column reaches 0.6. |
| **`car_exact_cal`** | car_exact_cmb with its latent draws widened around their mean by f(% counted). f is the 90% quantile of abs(error)/sd, divided by 1.645, fitted **leave-one-case-out**: LGE2021 uses the f fitted on PE2024 replays and vice versa. Live uses both cases: f = 1.32 / 1.60 / 1.79 / 2.32 at 5 / 10 / 25 / 50% counted. |
| `hier`, `hier_t` | (c) Simple hierarchical log-ratio swing model learned **from the count only**. It has its own slopes on the 12 past alr columns and partial pooling ward -> municipality -> province. Student-t likelihood, NUTS. The predictive VD residual is a bootstrap of counted VDs' residual vectors. `hier_t` has tight slope priors. |
| `hier_tcm8` | (d) hier_t plus cluster × party and cluster × municipality × party effects (8 clusters) with learned scales. |
| `hier_bym2` | (d) hier plus a BYM2 effect (scaled ICAR + iid), fitted with SVI. Tested on one replay only. |
| `hier_clus10/20/40`, `hier_cm6/10` | Cluster variants with loose slopes. Not benchmarked, because loose slopes failed (section 3). |

**Broken histories** (`_clean_X`). 140 PE2019 VDs and 12 PE2024 VDs have valid votes above 98% of the
roll (section-24A voting). Their past alrs reach 13. Their past alr is replaced by the average of their
touching neighbours. A 90% cutoff was worse (sum of 4 test replays: 11.4 vs 8.6 pp at 98% and 8.7 with
no cleaning), because it also hits genuinely high-turnout 2019 suburbs.

## 2. Summary (harness.summarise, full benchmark; baselines from results_baselines.csv)

| method             | prov_mae_pp | prov_max_pp | muni_mae_pp | ward_winner_acc | ward_mae_pp | vd_mae_pp | vd_turnout_mae_pp | prov_cov90 | prov_crps_pp | muni_cov90 | ward_brier | seconds |
|:-------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| car_exact_cmb      | 1.066 | 2.606 | 1.464 | 0.945 | 2.658 | 4.122 | 6.375 | 0.704 | 0.836 | 0.764 | 0.087 | 28.3 |
| **car_exact_cal**  | 1.146 | 2.770 | 1.498 | 0.945 | 2.705 | 4.174 | 6.501 | **0.793** | 0.848 | **0.861** | 0.091 | 34.9 |
| car_full           | 1.517 | 3.730 | 1.783 | 0.947 | 2.973 | 4.383 | 6.270 | 0.213 | 1.436 | 0.417 | 0.086 | 56.1 |
| uniform_swing_muni | 2.943 | 8.185 | 2.816 | 0.935 | 4.210 | 5.962 | 7.202 | – | – | – | – | 0 |
| tally              | 4.094 | 10.583 | 5.744 | 0.677 | 9.723 | 12.391 | 9.512 | – | – | – | – | 0 |

Province MAE (pp) by order:

| case, order | car_exact_cal | car_exact_cmb | car_full | tally | uniform_swing_muni |
|:--|--:|--:|--:|--:|--:|
| LGE2021 ANC_strongholds_first | 3.55 | 3.57 | **2.98** | 10.68 | 3.38 |
| LGE2021 DA_strongholds_first  | 2.98 | 3.04 | 3.09 | 11.77 | 3.89 |
| LGE2021 random (3 seeds)      | 0.38 | **0.34** | 1.20 | 0.47 | 2.72 |
| LGE2021 real2026              | 0.27 | **0.23** | 0.97 | 1.01 | 2.55 |
| LGE2021 small_first           | 0.87 | **0.82** | 1.73 | 2.90 | 3.26 |
| PE2024 ANC_strongholds_first  | 2.10 | **2.08** | 2.21 | 9.46 | 3.15 |
| PE2024 DA_strongholds_first   | 2.28 | **2.12** | 2.92 | 12.50 | 3.77 |
| PE2024 random (3 seeds)       | 0.38 | **0.25** | 0.55 | 0.62 | 2.52 |
| PE2024 real2026               | 0.44 | **0.20** | 0.40 | 1.07 | 2.35 |
| PE2024 small_first            | 0.87 | **0.68** | 1.25 | 3.13 | 2.88 |

Coverage and width by % counted. The 21.4 / 21.7 rows are real2026 at its first level.

| % counted | prov_cov90 cal / cmb / full | muni_cov90 cal / cmb / full | prov 90% width (pp) cal / cmb / full |
|--:|:--|:--|:--|
| 5  | 0.86 / 0.78 / 0.28 | 0.83 / 0.78 / 0.41 | 4.99 / 3.74 / 0.78 |
| 10 | 0.87 / 0.74 / 0.18 | 0.86 / 0.77 / 0.39 | 4.33 / 2.96 / 0.58 |
| 21.4 (LGE2021 real2026) | 0.83 / 0.83 / 0.33 | 0.87 / 0.72 / 0.37 | 1.32 / 0.85 / 0.43 |
| 21.7 (PE2024 real2026) | 1.00 / 1.00 / 0.50 | 0.98 / 0.93 / 0.54 | 1.81 / 0.97 / 0.46 |
| 25 | 0.80 / 0.68 / 0.17 | 0.87 / 0.75 / 0.40 | 3.31 / 1.83 / 0.42 |
| 50 | 0.64 / 0.61 / 0.20 | 0.87 / 0.76 / 0.46 | 1.89 / 0.92 / 0.33 |

Mean absolute province error per party (pp), averaged over all rows:

| party | DA | ANC | ActionSA | MK | EFF | OTHER |
|:--|--:|--:|--:|--:|--:|--:|
| car_exact_cal | 2.14 | 1.48 | 0.59 | 0.90 | 1.29 | 0.48 |
| car_exact_cmb | 2.01 | 1.35 | 0.51 | 0.78 | 1.28 | 0.47 |
| car_full | 3.15 | 1.77 | 0.84 | 1.11 | 1.72 | 0.51 |

Ablations were run with `--n-random 1` (both cases, 38 projections each; car_exact and car_exact_b
before the history cleaning):

| method | prov_mae | muni_mae | ward_acc | prov_cov90 | prov_crps | muni_cov90 | ward_brier | seconds |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| car_full_fast (1500 SVI steps) | 1.736 | 2.037 | 0.944 | 0.219 | 1.648 | 0.367 | 0.092 | 33.9 |
| car_full (5000) | 1.775 | 1.981 | 0.944 | 0.211 | 1.690 | 0.394 | 0.090 | 61.4 |
| car_exact (exact posterior, same model) | 1.785 | 1.949 | 0.945 | 0.298 | 1.652 | 0.473 | 0.090 | 14.6 |
| car_off (b, no CAR, re-trained) | 1.895 | 1.940 | 0.941 | 0.193 | 1.808 | 0.410 | 0.094 | 27.0 |
| car_exact_b (free slopes, sd 0.25, no guard) | 2.270 | 2.347 | 0.898 | 0.360 | 1.997 | 0.571 | 0.161 | 26.1 |

Tuning grid: 8 replays (both cases × ANC-first 10%, DA-first 10%, random 10%, small-first 25%):

| method | prov_mae | prov_max | muni_mae | prov_cov90 | width | sec |
|:--|--:|--:|--:|--:|--:|--:|
| car_exact_cm (8 clusters) | 2.01 | 4.84 | 2.51 | 0.63 | 3.19 | 10 |
| hier_t (c) | 2.70 | 6.55 | 2.83 | 0.52 | 3.52 | 56 |
| hier_tcm8 (d) | 2.58 | 6.19 | 2.80 | 0.56 | 4.35 | 56 |

hier_t and hier_tcm8 are close to the trained models in random order (LGE2021 0.33, PE2024 0.55 at
10%). They are far worse in ANC-first: 5.3-6.4 pp, against 1.6-2.3 for car_exact_cm.

## 3. Findings

### Why the repo model is overconfident

Its province coverage is 0.21 and its municipality coverage 0.42. There are three causes.

1. **Mean-field SVI.** For a Gaussian posterior, mean-field VI gets the means right but returns only the
   *conditional* variances.
   * A_k trades off against every psi_i and phi_i, so var(A) collapses.
   * The same model with the exact posterior (`car_exact`) has the same MAE (1.79 vs 1.78).
   * Its 90% width is 1.7× larger (0.95 vs 0.57 pp) and its province coverage 0.30 vs 0.21.
   * It is also 4× faster.
2. **Bias of the fixed structure.** This is the bigger cause.
   * Slopes, substitution, sig_d and the CAR parameters come from another transition and carry no
     uncertainty. Errors are ~2 pp while widths are ~1 pp.
   * Even in random order at 50% counted, `car_full` misses by 0.7 pp in LGE2021, where the tally misses
     by 0.19 pp.
   * With the T2 slopes (b1 + b2 = 0.84), LGE2021 DA is 4.4 pp too low in random order: its strongholds
     are shrunk.
   * The coordinator's stress test found the same thing live: the uncounted projection is about 0.98-0.999
     correlated with the fixed part.
3. **No area-type swing.** Only A and delta_muni respond to the count, but 2026 townships and suburbs
   swing very differently within one municipality.

Two other suspects are not the cause:
* The A_sd prior: A is dominated by the data after 140 VDs.
* Multinomial-only noise: psi is drawn for uncounted VDs.

### What each change does

* **Area-type × municipality swing with cleaned histories (`car_exact_cm`).** At 10% random order it cuts
  the province MAE from 1.46 to 0.45 (LGE2021) and from 0.51 to 0.25 (PE2024). Intervals widen where a
  cluster × municipality cell has no counted VD. On live_check, group MAE goes from 4.07 (car_full) to
  2.61, and coverage from 0.31 to 0.73.
* **Slopes and substitution with trained priors (`cmb` vs `cm`).**
  * Live (2026 count): group 2.28 vs 2.61, random 0.53 vs 0.71, time 0.44 vs 0.22.
  * Replays: better in random, small-first and real2026 orders.
  * Without the extrapolation guard the slopes are dangerous. car_exact_b and the first cmb put DA
    15-20 pp too low in stronghold-first orders, because slopes learned on ANC strongholds were applied
    to DA strongholds.
  * Even with the guard, cmb is worse than car_full in LGE2021 ANC-first at 25-50% (3.6 vs 3.0).
* **Inflation schedule (`cal`).**
  * Benchmark province coverage goes from 0.70 to 0.79 and municipality coverage from 0.76 to 0.86.
  * Live group coverage goes from 0.77 to 0.89, and VD coverage from 0.88 to 0.92.
  * The cost is about +0.08 pp MAE: widening eta moves the vote-space mean.
  * One fix to consider: report the uninflated mean with the inflated intervals.
  * Coverage still drops at 50% counted in the stronghold orders (0.64), where bias is constant and the
    width shrinks.

### What each variant adds over (c)

* **(c) itself.** Slopes learned from the count extrapolate wrongly when the count is biased. ANC-first at
  10%: DA -11.7 (PE2024) and -17.5 pp (LGE2021). In random order (c) is as good as the trained model.
  * The first (c) used Student-t predictive draws with nu ≈ 2.1. This gave a vote-space bias: big parties
    too low, small parties too high and turnout +1.6 pp (t(2) has no exponential moments).
  * Bootstrapping residual vectors from counted VDs fixed it.
* **(d) clusters.** `hier_tcm8` vs `hier_t`: −0.1 pp on the tuning grid, with wider intervals. On
  live_check it is the most accurate model: group 1.96, random 0.34, time 0.49, VD leader accuracy
  0.88 in the time task. That count is mixed, not adversarial. NUTS takes 25-70 s.
* **(d) BYM2.** One replay (PE2024, random, 10%): municipality MAE 1.02 vs 1.31, province no better, and SVI
  collapses the ward sd. Not pursued.
* **(b) CAR.** Without it (`car_off`) MAE rises from 1.78 to 1.90 and coverage does not improve. CAR
  helps the mean a little and does nothing for calibration. Keep it, as the hold-out test says, but do
  not count on it for interval width.

### Failure cases

* **LGE2021 ANC-strongholds-first.** DA is too low by 5-9 pp in every model, because DA's 2016->2021
  change in its strongholds cannot be seen from ANC areas.
* **PE2024 DA-strongholds-first.** MK and EFF are off by 4-7 pp, because MK's township strength cannot be
  seen from the suburbs.
* **New parties.** The new party (ActionSA in 2021, MK in 2024) is one of the biggest errors in biased
  orders.
* **Live groups.** The worst live groups for cal are DA suburbs in TSH and JHB (2.9-3.3 pp; cm alone gets
  1.3-1.6 there) and townships in GT484 and TSH (2.8-2.9).

## 4. The 13:00 live intervals (coordinator's question)

* **13:00 output** (old hyper files, SVI): DA 34.6-35.5 (±0.4 pp, ±1.5% of its votes), ANC ±0.2 pp at
  21% counted.
* **car_full in the replays** at the real 2026 order and the same % counted: bands of ±0.2-0.25 pp, with
  errors of 0.4-1.2 pp and coverage 0.33-0.50. So yes, it is overconfident by about 3-5×.
* **car_exact_cal on the 16:03 SAST snapshot** (841 VDs, 30%):

| party | estimate | 90% band |
|:--|--:|:--|
| DA | 33.1% | 32.0-34.3 |
| ANC | 21.9% | 21.2-22.6 |
| ActionSA | 17.2% | 16.6-17.8 |
| MK | 5.5% | 5.0-6.0 |
| EFF | 6.5% | 6.1-6.9 |
| OTHER | 15.8% | 15.1-16.7 |
| turnout | 51.7% | 50.8-52.6 |

The DA band now contains the stress test's look-alike estimate (33.4%). The live model's band (DA
35.3-36.1) did not. car_exact_cmb gives DA 33.2% (32.5-34.0).

## 5. live_check

Tasks: group, time and random. Snapshots up to 15:33 SAST (794 counted VDs); car_exact_cal was run on
16:03 (841). The baselines come from `live_baselines.csv`.

| method, kind | group_mae | group_max | vd_mae | vd_leader_acc | vd_turnout_mae | group_cov90 | vd_cov90 | sec |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| car_exact_cal, group | 2.24 | 5.57 | 4.32 | 0.824 | 4.56 | **0.889** | **0.919** | 26 |
| car_exact_cal, random | 0.44 | 0.95 | 3.80 | 0.808 | 4.12 | 1.00 | 0.922 | 15 |
| car_exact_cal, time | 0.47 | 0.96 | 3.64 | 0.860 | 4.19 | 1.00 | 0.926 | 28 |
| car_exact_cmb, group | 2.28 | 5.52 | 4.29 | 0.818 | 4.78 | 0.769 | 0.884 | 22 |
| car_exact_cmb, random | 0.53 | 1.27 | 3.73 | 0.810 | 4.19 | 0.889 | 0.909 | 11 |
| car_exact_cmb, time | 0.44 | 0.97 | 3.64 | 0.862 | 4.12 | 0.833 | 0.899 | 21 |
| car_exact_cm, group | 2.61 | 6.54 | 4.53 | 0.818 | 4.98 | 0.731 | 0.862 | 18 |
| car_exact_cm, random | 0.71 | 1.50 | 4.12 | 0.774 | 4.72 | 0.778 | 0.884 | 9 |
| car_exact_cm, time | 0.22 | 0.51 | 3.91 | 0.840 | 4.24 | 1.00 | 0.883 | 11 |
| hier_tcm8, group | **1.96** | 4.58 | **3.81** | 0.821 | 4.33 | 0.722 | 0.910 | 44 |
| hier_tcm8, random | **0.34** | 0.57 | 3.42 | 0.842 | 4.04 | 1.00 | 0.929 | 24 |
| hier_tcm8, time | 0.49 | 0.81 | 3.28 | 0.883 | 3.94 | 1.00 | 0.935 | 71 |
| car_full, group | 4.07 | 9.85 | 5.44 | 0.792 | 7.81 | 0.306 | 0.732 | 52 |
| car_full, random | 1.78 | 4.65 | 4.74 | 0.762 | 6.47 | 0.50 | 0.815 | 36 |
| car_full, time | 1.28 | 3.84 | 4.48 | 0.803 | 6.08 | 0.50 | 0.808 | 82 |
| uniform_swing_muni, group / random / time | 4.91 / 1.56 / 1.54 | | | | | | | |
| tally, group / random / time | 9.90 / 0.82 / 2.06 | | | | | | | |

## 6. Run time per projection

The box was shared (2 of 4 cores, other agents running), so these timings are pessimistic.

* `car_full` (SVI 5000): 55-60 s.
  * With 1500 steps (`car_full_fast`): 34 s, with the same accuracy (MAE 1.74 vs 1.78) and calibration.
  * So fewer SVI steps are fine, but the exact solver below is better and faster.
* `car_exact*`: 10-15 s when uncontended (28-35 s here). The time goes to:
  * `calibrate_A`: a Monte Carlo over CAR prior draws. It does not depend on the count, so it can be
    cached once per night.
  * The sparse LU of a ~14k-dimensional system: 1-3 s.
  * The 300 multinomial draws.
* `hier*` (NUTS, 300 warmup + 300 draws): 25-80 s, growing with the number of counted VDs.

All of these fit the few-minutes re-run cycle.

## 7. What the harness gets wrong (or makes hard)

* **`real2026` is not frozen.** It is built from whatever snapshots are in `data/observed_2026` at run time.
  Here it stayed at 21.4% / 21.7%, but it can change as files arrive during the night. Freeze it to a
  named snapshot.
* **MK in LGE2021.** MK did not stand in LGE2021, but it still counts as 1 of the 6 parties in that
  case's MAE, coverage and CRPS. Any method that projects MK = 0 gets a free 0 error and a free
  "covered". This lowers LGE2021 MAE by about 1/6 and raises coverage.
* **Province coverage is noisy.** It rests on 6 numbers per projection, so it moves in steps of 0.17.
  Pooling z-scores, as `infl` does here, is more informative.
* **The averages reward robustness to adversarial orders more than accuracy on realistic ones.**
  The stronghold and small-first orders are extreme, and they make up 3 of the 7 orders. Report the
  random and real2026 rows separately as well.
* **LGE2021 tests robustness more than accuracy.** Its leak-free hyper-parameters (T2) come from the
  other election type. For model-based methods, LGE2021 measures robustness to a wrong structure.
* **`live_check.load_live_case` crashes under pandas 3.** `vd_wards` returns a read-only array and
  `ward[i] = w` then fails. My driver used a writable copy; I did not edit the file.

## 8. Using it live

* `m_bayes.car_exact_cal(case, counted, rng)` works on the `LIVE2026` case from
  `live_check.load_live_case()`, with every reported VD counted.
* For that case it uses the T1 + T2 hyper-parameters, the poll prior in `config/live.json` and the
  both-case inflation schedule.
* Cache `calibrate_A` and the k-means clusters across re-runs.
* The broken-history rule (turnout > 98% or roll < 20) flags only 17 VDs in the LIVE2026 predictors
  (12 in PE2024, 5 in LGE2021). The stress test found 47; those with a zero LGE2021 result still need a
  flag.
