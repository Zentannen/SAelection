# v_asa: party baselines from the right election (ActionSA)

Code: `src/saelection/bench/m_vasa.py`. Live check: `artifacts/bench/live_v_asa.csv` (snapshot
`IEC_LGE2026_GP_20261105_1518.csv` pinned, all 29 tasks, every variant below). Benchmark:
`artifacts/bench/results_v_asa.csv` (both cases, all orders, `--n-random 3`, `hybrid_asa` and
`m_swing:hybrid_cf`). Note: `data/processed/vd_ward_iec_2026.csv` appeared at 13:38 (another
agent), so every run here uses the IEC ward of uncounted VDs.

## Bottom line

* **Best: `m_vasa:hybrid_asa`** = `hybrid_cf` on a re-based history + a district level:
  1. cross-election low-base floor 0.3 for ActionSA (a VD's ActionSA share in one predictor
     election may not be below 0.3 x what the other implies, via the province LGE2021/PE2024
     ratio 2.79). Where ActionSA did not stand in 2021 (Sedibeng, West Rand) this gives a
     *partial* LGE-equivalent base 0.3 x 2.79 x PE2024 share; it also repairs 73 metro VDs
     whose PE2024 ActionSA count is a spurious ~0 (LGE2021 2-27%; the counted ones polled 4-50%);
  2. a district level (metros / Sedibeng / West Rand) as an EB level in the ratio swing and
     dummies in the ridge regression.
* Live check (14:45 -> 15:18 snapshot, hybrid_cf -> hybrid_asa): group 1.638 -> **1.599**,
  muni 2.048 -> **1.904**, random 0.520 -> **0.478**, time 0.122 -> **0.098** pp; VD MAE better on
  every task kind; mean |ActionSA error| group 1.74 -> 1.59, muni 4.33 -> 4.02, random 0.34 -> 0.18.
  Coverage is a little lower: group cov90 0.892 -> 0.873, muni 0.896 -> 0.854.
* Benchmark (only the district level acts there; see "fallback"): province MAE 0.877 -> 0.875,
  muni 1.298 -> 1.254, ward MAE 2.389 -> 2.365, CRPS 0.872 -> 0.845, ward Brier 0.099 -> 0.093,
  cov90 0.944 -> 0.938. Random + real2026 orders: a tie (0.251 vs 0.252).
* **What the data says about the user's hypothesis.** A full LGE-equivalent baseline for
  ActionSA (LGE2021 where it stood, PE2024 x look-alike LGE/PE ratio where it did not) is
  clearly *worse*: muni hold-outs 2.05 -> 2.92 pp, ActionSA over-predicted by 6-10 pp in every
  Sedibeng/West Rand municipality hold-out. ActionSA's 2026 vote grew much less where it had no
  2021 local presence (2026/PE2024 ratio 1.9-3.8 there vs 2.6-6.8 in the metros); the zero in
  LGE2021 is information (it worked as a region indicator in hybrid_cf's clusters and ridge),
  not just missing data. Forcing LGE weighting is worse still (muni 3.01). The live data itself
  puts *less* weight on LGE2021 for ActionSA than for any other party: the log-ratio blend picks
  LGE weight 0.21 (DA 0.38, ANC 0.73, EFF 0.58, OTHER 0.56), the ratio-part blend picks 0.0 (PE
  only), and the ridge's biggest ActionSA coefficient is a negative one on its PE2024 log-ratio
  (2026 ActionSA is flatter than either past geography: a township surge, 28% in counted
  townships vs 12% LGE2021 and 4% PE2024).
* **The flagged wards are not a baseline problem** (table below): none of the re-basings moves
  them by more than ~1-2 pp. 79700061 and 79900057 have no counted VD *in the model's VD set*:
  their counted VDs (33000523, 219 registered; 32980620, 1344 registered) are new 2026 stations
  outside the 2026 VD boundary set, which a method cannot see through `case` (live.py adds their
  actual votes to the ward tally but nothing projects from them). In 79700053 and 79700024 the
  one counted VD is the ward's historical ActionSA stronghold (LGE2021 37.6% and 19.3% vs 2-8%
  for most ward-mates), so "the counted VD shows 48%" does not imply the ward does; its
  look-alike ward-mate 33000488 (LGE 23.7%) is projected 46-49%. In 74801006 (2 VDs) the counted
  VD is a 649-voter tent; the other VD (4278 voters) had 0 ActionSA votes in PE2024 and stands
  in a municipality where ActionSA did not stand in 2021.
* **Low-base floor.** Forecasts under 1% are mostly right: among 39 counted VDs with an
  ActionSA base ~0 in both elections, 2026 ActionSA has median ~0.7% (most 0-2.4%, a few 6-20%);
  in the hold-outs, VDs forecast < 1% polled 0.6-1.7% on average. A neighbour floor (share >=
  0.3 x touching VDs' share) lifts such VDs (74801006: 5.4% -> 11.6%) but does *not* improve the
  live check (group 1.599 -> 1.628, VD ActionSA MAE worse on group / muni hold-outs), so it is
  not in the best method. The real low-base issue is the 0.2%-PE bin, which hybrid_cf
  under-forecasts by ~2-3 pp (hold-out mean forecast 2.1-3.5% vs actual 3.7-6.7%); hybrid_asa
  narrows that (ActionSA VD MAE in that bin 4.9 -> 3.8 group, 4.6 -> 3.5 muni, 2.7 -> 2.2 random).

## Variants (all = hybrid_cf machinery on a re-based case; live check, pinned 15:18 snapshot)

`group/muni/random/time` = group_mae_pp by task kind; |ASA err| = mean absolute province-of-
held-out ActionSA share error (pp).

| method | what changes vs hybrid_cf |
|---|---|
| `vasa_imp` | full LGE-equivalent imputation: LGE2021 ActionSA in GT421-3/GT481/4/5 := PE2024 share x LGE/PE ratio of the 150 most similar VDs (PE profile) in the metros (median 2.82), other parties scaled down so turnout is unchanged |
| `vasa_imp_mix` | + ratio-part base blend learned live (`_mix_shares`) |
| `vasa_imp_force` | + forced LGE weighting for ActionSA (its PE column := LGE-equivalent / 2.79) |
| `vasa_imp_floor`, `vasa_imp_nb` | + cross-election floor 0.3 / + neighbour floor 0.3 |
| `vasa_g03` | imputation at strength gamma 0.3 (LGE* = 0.3 x r x PE) |
| `vasa_glive`, `_dist`, `_floor` | gamma learned live from counted VDs where it did / did not stand (0.70 at 15:18), shrunk to 0.5 |
| `vasa_floor` | cross-election floor 0.3 everywhere (= gamma 0.3 imputation via the province ratio + 73/73 metro VD floors) |
| `vasa_dist` | district level only, no re-basing |
| `vasa_floor_dist` = **`hybrid_asa`** | floor 0.3 + district level |
| `vasa_f02_dist`, `vasa_f05_dist` | floor 0.2 / 0.5 + district |
| `vasa_floor_dist2` | district level after the municipality (correlated draws on cluster + muni as in hybrid_cf) |
| `vasa_fnb`, `vasa_fnb_dist` | floor 0.3 + neighbour floor 0.3 (+ district) |
| `vasa_vdf*` | floors only where ActionSA stood (keeps the not-stood zero exactly); `_x` cross-election only, `_nb` neighbour only |

| method           |   group |   muni |   random |   time |   vd_mae group |   vd_mae random |   cov90 group |   cov90 muni |   abs ASA err group |   abs ASA err muni |   abs ASA err random |
|:-----------------|--------:|-------:|---------:|-------:|---------------:|----------------:|--------------:|-------------:|------------------:|-----------------:|-------------------:|
| **hybrid_asa**   |   1.599 |  1.904 |    0.478 |  0.098 |          3.573 |           3.326 |         0.873 |        0.854 |             1.588 |            4.024 |              0.176 |
| hybrid_cf        |   1.638 |  2.048 |    0.520 |  0.122 |          3.598 |           3.367 |         0.892 |        0.896 |             1.741 |            4.328 |              0.343 |
| vasa_floor_dist2 |   1.599 |  1.908 |    0.478 |  0.098 |          3.574 |           3.325 |         0.863 |        0.917 |             1.582 |            4.009 |              0.176 |
| vasa_f02_dist    |   1.622 |  1.918 |    0.480 |  0.103 |          3.598 |           3.298 |         0.863 |        0.896 |             1.612 |            3.943 |              0.177 |
| vasa_f05_dist    |   1.604 |  1.999 |    0.487 |  0.098 |          3.571 |           3.281 |         0.873 |        0.875 |             1.737 |            4.510 |              0.167 |
| vasa_fnb_dist    |   1.628 |  1.955 |    0.484 |  0.097 |          3.600 |           3.290 |         0.882 |        0.896 |             1.762 |            4.177 |              0.183 |
| vasa_vdf_dist    |   1.607 |  1.945 |    0.525 |  0.138 |          3.581 |           3.359 |         0.882 |        0.875 |             1.657 |            4.029 |              0.338 |
| vasa_dist        |   1.659 |  1.976 |    0.522 |  0.112 |          3.600 |           3.363 |         0.882 |        0.938 |             1.803 |            4.283 |              0.333 |
| vasa_floor       |   1.582 |  2.180 |    0.478 |  0.097 |          3.594 |           3.313 |         0.873 |        0.854 |             1.630 |            5.061 |              0.165 |
| vasa_glive_floor |   1.582 |  2.594 |    0.473 |  0.113 |          3.573 |           3.271 |         0.873 |        0.854 |             1.725 |            6.897 |              0.149 |
| vasa_vdf         |   1.590 |  2.052 |    0.514 |  0.145 |          3.584 |           3.364 |         0.882 |        0.896 |             1.611 |            4.217 |              0.324 |
| vasa_g03         |   1.620 |  2.294 |    0.480 |  0.153 |          3.578 |           3.299 |         0.863 |        0.875 |             1.863 |            5.525 |              0.205 |
| vasa_imp         |   1.605 |  2.922 |    0.479 |  0.184 |          3.578 |           3.282 |         0.853 |        0.833 |             1.767 |            8.122 |              0.182 |
| vasa_imp_floor   |   1.602 |  2.881 |    0.469 |  0.128 |          3.609 |           3.269 |         0.843 |        0.833 |             1.818 |            7.878 |              0.177 |
| vasa_imp_nb      |   1.624 |  2.795 |    0.494 |  0.104 |          3.578 |           3.293 |         0.863 |        0.812 |             1.668 |            7.811 |              0.232 |
| vasa_imp_mix     |   1.619 |  2.914 |    0.555 |  0.210 |          3.602 |           3.356 |         0.863 |        0.833 |             1.770 |            8.085 |              0.242 |
| vasa_imp_dist    |   1.659 |  2.109 |    0.480 |  0.179 |          3.580 |           3.276 |         0.873 |        0.896 |             1.842 |            4.932 |              0.181 |
| vasa_imp_force   |   1.673 |  3.009 |    0.487 |  0.197 |          3.607 |           3.351 |         0.873 |        0.792 |             1.941 |            8.461 |              0.320 |
| vasa_glive       |   1.678 |  2.671 |    0.493 |  0.158 |          3.609 |           3.303 |         0.833 |        0.833 |             1.856 |            7.364 |              0.182 |
| vasa_glive_dist  |   1.665 |  2.069 |    0.496 |  0.157 |          3.592 |           3.296 |         0.873 |        0.875 |             1.842 |            4.871 |              0.172 |
| vasa_fnb         |   1.628 |  2.197 |    0.489 |  0.103 |          3.623 |           3.295 |         0.853 |        0.854 |             1.835 |            4.912 |              0.208 |
| vasa_vdf_nb      |   1.619 |  2.124 |    0.529 |  0.138 |          3.598 |           3.338 |         0.863 |        0.875 |             1.560 |            4.412 |              0.354 |
| vasa_vdf_x       |   1.621 |  2.091 |    0.524 |  0.116 |          3.559 |           3.332 |         0.843 |        0.896 |             1.654 |            4.146 |              0.364 |

Reading: the partial imputation in the not-stood municipalities (gamma ~0.3, via the floor) buys
the random / group / time gains; on its own it hurts the Sedibeng / West Rand / EKU municipality
hold-outs (their swing now flows through clusters shared with the metros); the district level
gives that back and more (muni 2.18 -> 1.90). Full imputation (gamma 1), the live gamma (0.70)
and forced LGE weighting over-predict ActionSA where it did not stand. VD-level floors in the
metros alone (`vasa_vdf_x`) are neutral.

Per-task, hybrid_cf -> hybrid_asa (group_mae, ActionSA error of the held-out group):
muni:GT481 1.19 -> 0.42 (ASA -3.2 -> -0.2), GT485 2.20 -> 1.43 (-1.6 -> +0.3), GT484 1.54 -> 1.21,
TSH 2.19 -> 1.92, JHB 5.09 -> 4.88 (ASA +15.0 -> +14.6: JHB's 2026 ActionSA is flat on 2021, no
history predicts that), EKU 1.33 -> 1.56 (ASA -4.0 -> -4.5), GT422 0.42 -> 0.71, GT423 2.42 -> 3.10
(ASA +2.5 -> +4.1); group:township:GT481 1.79 -> 1.55, township:JHB 1.59 -> 1.32, other:JHB 1.35 ->
0.87, suburb:GT422 0.71 -> 0.51; worse: township:EKU 1.88 -> 2.07, suburb:EKU 0.78 -> 1.01,
township:GT423 3.78 -> 4.10.

## Flagged wards (full live case at 15:18: counted VDs at actual, uncounted projected; ActionSA %)

Ward share = (sum over the ward's VDs in the case + counted VDs outside the VD set) / valid.
"counted" = all counted VDs of the ward in the snapshot file.

| ward | VDs (case) | counted in case / snapshot | hybrid_cf | hybrid_asa | counted VDs | note |
|---|---|---|---|---|---|---|
| 79700053 | 8 | 1 / 1 | 29.3 | 28.1 | 48.5 | counted VD = ward's 2021 ActionSA stronghold (L21 37.6%; mates 2-8% except 33000488, 23.7% -> projected 46%) |
| 79700061 | 5 | 0 / 1 | 18.7 | 17.6 | 42.6 | counted VD 33000523 (122 valid) not in the 2026 VD set |
| 79900057 | 3 | 0 / 1 | 25.5 | 24.4 | 49.9 | counted VD 32980620 (1136 valid) not in the 2026 VD set |
| 79700024 | 4 | 1 / 1 | 16.2 | 16.2 | 33.6 | 33010614: PE24 ActionSA 0.03%, L21 4.8% -> 0.7% (hcf) / 1.8% (asa); 33010636 L21 5.8% |
| 74801006 | 2 | 1 / 1 | 6.0 | 5.4 | 29.3 | other VD 33250384 (4278 reg.): 0 ActionSA in PE24, not stood in 2021 -> ~1%; neighbour floor gives 11.6% |
| 79700099 | 7 | 2 / 2 | 24.9 | 23.4 | 23.7 | fine |

Ward 90% intervals (hybrid_asa): 79700053 20.8-38.3, 79700061 11.2-26.8, 79900057 13.0-32.7,
79700024 11.4-22.1, 74801006 4.6-6.4, 79700099 17.5-30.8. Within-ward pull from counted ward-mates
is a different fix (not this variant); hybrid_cf already has an EB ward effect, shrunk hard when a
ward has one counted VD.

## Benchmark (`harness.summarise`, both cases, all orders, n_random 3)

| method     | prov_mae_pp | prov_max_pp | muni_mae_pp | ward_winner_acc | ward_mae_pp | vd_mae_pp | vd_turnout_mae_pp | prov_cov90 | prov_crps_pp | muni_cov90 | ward_brier |
|:-----------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| hybrid_asa | 0.875 | 1.967 | 1.254 | 0.951 | 2.365 | 3.614 | 5.468 | 0.938 | 0.845 | 0.846 | 0.093 |
| hybrid_cf  | 0.877 | 1.968 | 1.298 | 0.951 | 2.389 | 3.628 | 5.475 | 0.944 | 0.872 | 0.847 | 0.099 |

Random + real2026 orders only: hybrid_cf prov 0.251 / muni 0.743 / ward MAE 1.768 / cov 0.967 /
CRPS 0.335; hybrid_asa 0.252 / 0.725 / 1.764 / 0.972 / 0.329. By case: LGE2021 prov 1.013 -> 1.010,
muni 1.404 -> 1.307; PE2024 prov 0.742 -> 0.739, muni 1.192 -> 1.201.

**Fallback.** In the LGE2021 case ActionSA has no prior LGE or PE, in the PE2024 case no prior PE
(PE2019); the floor needs the party in both predictor elections, so it is a no-op there and
hybrid_asa = hybrid_cf + district level (checked: `vasa_imp` reproduced hybrid_cf exactly on the 30
LGE2021 benchmark rows run before it was stopped; `vasa_ref`, no re-basing, reproduces hybrid_cf
on the full live case to 0.0 votes). The
benchmark therefore cannot test the ActionSA re-basing itself; only the live check does.

## Run time

Same as hybrid_cf plus a one-off re-basing / prep per case (~3-5 s): 1.4-1.5 s per live-check
projection with 200 draws (3.9 s for the first, "time", task which pays the prep) with the
machine at load 10-13 (hybrid_cf 1.2-1.5 s); 6-10 s per benchmark projection at that load.

## Risks / caveats

* The floor constant 0.3 and the district level were chosen on the live check at one snapshot
  (24 variants, 29 overlapping tasks): mild overfitting risk; 0.2 / 0.5 give similar results.
* The district level is Gauteng-specific (metros pooled, GT42x, GT48x) and doubles as "where
  ActionSA stood in 2021"; with 3 groups its EB variance is estimated from 3 units.
* Coverage drops a little (group 0.892 -> 0.873, muni 0.896 -> 0.854): the district level takes
  the correlated (rho) draw from the municipality level; `vasa_floor_dist2` keeps it there (muni
  cov 0.917, group 0.863, same means).
* GT421 (Emfuleni, 167 VDs, only 4 counted at 15:18) moves from 5.8% to 7.7% ActionSA: its
  forecast rests on the district swing and the 0.3 imputation, with little own data.
* The user's ward complaint is mostly not fixed by this variant (see table); the two wards whose
  only counted VD is outside the VD set need a pipeline fix (map new stations to a VD or use
  them in the ward effect), not a method change.
