# Live-projection bake-off: brief for method agents

## Goal
On election night (2026 Local Government Elections, Gauteng, PR ballot) voting districts (VDs)
report one at a time. From the VDs counted so far we must project the final result for every
uncounted VD, and from that the province, the 9 municipalities (council seats are allocated
there) and the 526 wards. The real reporting order is not random (it can favour some areas), so a
method must see through it. We are comparing families of methods on one shared backtest and will
pick the best, or an ensemble, for the live 2026 run.

## Repo and environment
* Worktree: `/home/user/sae-main` (code in `src/saelection/`, venv at `.venv/`; run
  `. .venv/bin/activate` and use `PYTHONPATH=src`). Raw + processed data are already present.
* Read `README.md` (the existing Bayesian spatial CAR model) and
  `src/saelection/bench/harness.py` (the shared benchmark) before starting.
* Categories everywhere: `CATS = [DA, ANC, ACTIONSA, MK, EFF, OTHER, ABSTAIN]`; ABSTAIN =
  registered − valid votes, so a VD's 7 counts sum to its electorate (covers turnout too).
* 4 CPU cores and 15 GB RAM are shared with three other agents. Set `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, use `n_jobs=1`, and for JAX
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`, unless your
  brief says otherwise. Compute time itself is not a constraint, so let long runs finish.
* `pip install` into the venv if you need a package (e.g. scikit-learn, lightgbm).

## The benchmark (do not edit harness.py)
`PYTHONPATH=src python -m saelection.bench.harness --methods <m1>,<m2> --tag <family>`
where a method is `module:function` under `saelection.bench`, e.g. `m_swing:ratio_swing_muni`.
A method has signature `fn(case, counted, rng) -> Projection(vd_mean, draws=None, info=None)`:
* `case.Y[counted]` may be used; `case.Y[~counted]` is the answer and must never be read.
  Everything else on `case` (previous-election counts `prev_lge`/`prev_pe`, `registered`,
  `muni`, `ward`, `lon`/`lat`, contiguity `edges`, `X1`/`X2` alr features, `design`) is known
  before counting.
* `vd_mean` (n, 7) projected final counts per VD; `draws` (S, n, 7) optional but strongly
  wanted: they give the 90% interval coverage, CRPS and ward-winner Brier score. Aim for
  calibrated intervals (coverage near 0.90), not just a good mean. S = 200–500 is fine.
* Cases: `LGE2021` (predictors LGE2016 + PE2019, same election type as 2026) and `PE2024`
  (predictors LGE2021 + PE2019). Orders: 3 random seeds, DA strongholds first, ANC strongholds
  first, small VDs first, and `real2026` (the VDs that actually reported first in the 2026 count).
  Levels 5, 10, 25, 50% counted.
* **New parties**: ActionSA has no 2016 history (LGE2021 case) and MK has no history at all in
  the PE2024 case, so `prev_*` is 0 for them. Your method must still project them, e.g. from
  counted VDs that are similar or nearby. (In the real 2026 run every party has 2024 history,
  but the backtests need this handled.)
* Baselines already scored (`artifacts/bench/results_baselines.csv`): `tally` (avg province
  MAE 4.09 pp) and `uniform_swing_muni` (2.94 pp; breaks on new parties, no intervals).
* For a quick iteration use `--n-random 1 --cases LGE2021`, then run the full set.

## Deliverables (write only these files; do not edit other files and do not git commit)
* `src/saelection/bench/m_<family>.py` with your methods (docstrings explaining each).
* `artifacts/bench/results_<family>.csv` from a full run of your final methods (both cases, all
  orders, `--n-random 3`).
* `artifacts/bench/notes_<family>.md`: what you tried, the summary table
  (`harness.summarise`), which variant is best and why, failure cases (which order/level/party),
  and run time per projection.
* Your final message: the summary table, the best method name, and anything you think the
  harness gets wrong.
