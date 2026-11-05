# v_leak: leakage / overfitting audit of the judge's winner `m_vbest:asa_rake_half`

Code: `src/saelection/bench/m_vleak.py`. It has no new method, only sealed wrappers:
`strict_*` overwrites case.Y outside `counted` with random garbage; `strict2_*` also sets the
held-out VDs' electorate to the pre-count estimate. Live check pinned to
`IEC_LGE2026_GP_20261105_1518.csv`: `artifacts/bench/live_v_leak.csv`. Scratch scripts and logs:
`/tmp/claude-0/-home-user-SAelection/1147242a-1fcc-5810-b77a-713d267bbfe4/scratchpad/refute_leak/`.

## Verdict: not refuted

There is no leakage, and the data shows no overfitting relative to hybrid_cf. One cache bug
affects a rejected alternative, not the winner; it does not change the choice.

## 1. Leakage

| check | result |
|---|---|
| Poison / sealed test, all 29 pinned live tasks (`poison.py`): asa_rake_half on the live case vs on a copy with Y[~counted] = random counts (new case object, so every id-keyed cache is rebuilt) | max diff of uncounted-VD means = **0**, of draws = **0**, on every task. The sealed scores equal the judge's `live_v_judge.csv` exactly: group 1.568, muni 1.888, random 0.445, time 0.111, group cov90 0.902. |
| The same test on the replay benchmark (`poison_bench.py`): LGE2021 and PE2024 x {random 10%, DA strongholds 25%, real2026} | diff **0** for means and draws in all 6. These cases take the new-party code paths (ActionSA has no history in LGE2021, MK in PE2024). |
| Own-result cross-fitting with the winner's exact bases (`oof_own.py`): FULL = hybrid_asa and NW = hybrid_asa without ward level, inside `_asa_base()`; tasks random_s0 and group:township:TSH; 3 counted VDs each, ActionSA x3 + 500 votes | The VD's own out-of-fold prediction changes by **0**, and so do the other VDs in its fold. Only VDs in other folds change, as cross-fitting requires. |
| Per-VD inputs taken from held-out results by live_check | ward: the reported ward equals the official IEC list for all 773 counted VDs (0 differences). Electorate: held-out `registered` = 2026 count total (`Y.sum(1)`), which no method can avoid. strict2 replaces it with crosswalk x growth from counted VDs only (section 3). |

The judge's own leak test (`judge/leak.py`) covered only random_s0. It now covers every task
and both benchmark cases.

## 2. Cache bug: the asa_spatial_pull2 live numbers belong to asa_spatial_pull

`m_vspatial._OOF` is keyed by `(case.name, id(case), counted, base, F, seed)`. The key does not
record which base model `m_vbest._asa_base` has installed. In the judge's `live_b_sp` run,
asa_spatial_pull2 ran right after asa_spatial_pull on the same re-based case, so it reused that
method's out-of-fold residuals and base mean. In `live_v_judge.csv` the two methods are identical
on all 29 tasks.

A clean, separate-process re-run (`live_asp2_clean.csv`) gives group 1.577, muni 1.892 (cov90
0.917), random 0.445 and time 0.092. Per-task differences are at most 0.057 pp, with a mean of
0.009. The judge's trade-off does not change: asa_rake_half has better group MAE (1.568) and
group cov90 (0.902 vs 0.863).

The winner is not affected. It ran alone in `live_b_rh`, and my sealed run reproduces it to 4
decimals. Live deployment is safe as long as asa_rake_half is not run in the same process
after another `_asa_base(...)` configuration on the same counted mask.

## 3. Strict re-score on the pinned live check (`live_v_leak.csv`)

Each cell is group MAE / VD MAE / group cov90 / |ActionSA err|, in pp, averaged over the tasks
of that kind.
- **strict** (sealed Y): identical to the judge's numbers. asa_rake_half is bit-identical on all
  29 tasks (section 1). strict_hybrid_cf matches the judge's hybrid_cf on the 27 tasks it
  finished before I stopped that run (see below).
- **strict2** (sealed Y + pre-count electorate for the held-out VDs): re-run in full.

| kind | hybrid_cf (judge = strict) | asa_rake_half (judge = strict) | strict2_hybrid_cf | strict2_asa_rake_half |
|---|---|---|---|---|
| group (17) | 1.638 / 3.598 / .892 / 1.74 | 1.568 / 3.545 / .902 / 1.47 | 1.654 / 3.612 / .892 / 1.78 | **1.591 / 3.558 / .902 / 1.49** |
| muni (8) | 2.048 / 4.434 / .896 / 4.33 | 1.888 / 4.357 / .896 / 4.05 | 2.087 / 4.439 / .917 / 4.35 | **1.913 / 4.358 / .896 / 4.04** |
| random (3) | 0.520 / 3.367 / 1.00 / 0.34 | 0.445 / 3.272 / 1.00 / 0.17 | 0.536 / 3.371 / 1.00 / 0.30 | **0.454 / 3.269** / .944 / 0.16 |
| time (1) | 0.122 / 3.330 / 1.00 / 0.11 | 0.111 / 3.185 / 1.00 / 0.05 | **0.109** / 3.339 / 1.00 / 0.15 | 0.122 / **3.184** / 1.00 / 0.12 |

Under strict2 the winner's lead over hybrid_cf stays about the same:
- group -0.063 pp (8 wins / 9 losses);
- muni -0.174 (5 / 3);
- random -0.081 (3 / 0).

Two things change. The time task flips to +0.013 pp (VD MAE is still 0.155 better), and random
cov90 is 0.944 instead of 1.00.

The first strict run, with all four wrappers, was stopped before it wrote its file because its
tag collided with another agent's `live_v_refute.csv`. Its log is
`refute_leak/live_refute.log`. strict2 was then re-run under the tag `v_leak`.

## 4. Overfitting

The floor (0.3) and the district level were chosen from 24 v_asa variants on these same 29 tasks.
The judge then chose among ~11 combinations on the same tasks.

The pinned gain over hybrid_cf on group tasks is within noise: mean -0.070 pp, s.e. 0.060, 9
wins vs 8 losses over 17 tasks. Muni tasks: 5 wins / 3 losses, mean -0.16, s.e. 0.18. Random
tasks: 3/3. Time: 1/1. Just over half of the group gain comes from hybrid_asa (-0.039 of -0.070);
the spatial part adds -0.031.

**Forward test on data no tuning run saw** (`forward.py`; supplementary, the live check itself
stays pinned). counted = every VD reported by cut C. Scored = the VDs that reported after C,
up to 16:47. All of them reported after 15:18, so no agent tuned on them. Later cuts are nested
subsets of earlier ones.

| cut (n held) | hybrid_cf group / VD MAE | hybrid_asa | spatial_rake_half | **asa_rake_half** | asa_ward_pull | asa_spatial_pull |
|---|---|---|---|---|---|---|
| 15:18 (122) | 0.436 / 3.042 | 0.412 / 2.995 | 0.374 / 2.932 | **0.347 / 2.892** | 0.388 / 2.902 | 0.352 / 2.932 |
| 15:33 (101) | 0.453 / 3.074 | 0.464 / 3.027 | 0.411 / 2.968 | 0.432 / **2.906** | 0.429 / 2.926 | 0.430 / 2.948 |
| 16:03 (54)  | 0.566 / 3.243 | 0.438 / 3.207 | 0.577 / 3.152 | 0.494 / **3.092** | 0.459 / 3.122 | 0.432 / 3.117 |
| 16:20 (32)  | 0.347 / 3.298 | 0.223 / 3.270 | 0.283 / 3.195 | 0.251 / **3.169** | 0.237 / 3.187 | 0.236 / 3.189 |

At the 15:18 cut the winner is the best of the six methods on the fresh VDs:
- **Partly counted wards (83 wards).** Ward MAE of their held VDs is 2.832, against 2.982 for
  hybrid_cf. Mean |ActionSA error| is 3.40 against 3.76.
- **Calibration.** VD cov90 is 0.928.

**The floor on fresh low-base VDs** (`fwd_vd.py`). For the 9 fresh VDs with PE2024 ActionSA
< 0.5%, the mean ActionSA error is -2.97 pp, against -5.17 for hybrid_cf. One example is
32890349, a metro VD with a spurious PE zero: it actually polled 38.7%, the winner projects
31.5% and hybrid_cf 11.7%. In the other bins the two methods are within ±0.6 pp.

**Group / muni hold-outs scored only on fresh VDs** (`fresh_holdout.py`). counted = everything
reported by 16:47 except the group. Only the group's post-15:18 VDs are scored. This is 9 tasks
with 11-46 fresh VDs each.

| | asa_rake_half | hybrid_asa | hybrid_cf | spatial_rake_half |
|---|---|---|---|---|
| group MAE (5 tasks) | **1.707** | 1.723 | 1.789 | 1.787 |
| muni MAE (4 tasks) | **2.798** | 2.804 | 2.905 | 2.908 |
| |ActionSA err| group / muni | **2.27 / 6.70** | 2.34 / 6.72 | 2.64 / 7.78 | 2.63 / 7.74 |
| group cov90 / muni cov90 | 0.833 / 0.875 | 0.867 / 0.875 | 0.933 / 0.917 | 0.900 / 0.917 |

The winner wins 5 of 9 tasks. The pattern matches the pinned check: EKU is worse (+0.53 muni,
+0.28 township) and West Rand GT481 is better (-0.59 muni, -0.78 township).

## 5. Caveats (not refutations)

- **Coverage on fresh group hold-outs.** It drops to 0.833 (25 of 30 intervals), against 0.933
  for hybrid_cf. This is a small sample, but the same direction as hybrid_asa's lower coverage
  on the pinned check.
- **Double-zero VDs.** A VD with no ActionSA votes in both LGE2021 and PE2024 stays near 0;
  the floor needs a non-zero in one of the two elections. One example is the flagged ward
  74801006: its other VD, 33250384 (GT481), reported after 15:18 with 15.6% ActionSA. The winner
  projected 0.9% and hybrid_cf 1.5%. 33240406 (GT484) polled 7.3% against a projected 2.1%.
- **Emfuleni (GT421)** still cannot be tested: 4 counted VDs at 15:18 and 1 more by 16:47.
