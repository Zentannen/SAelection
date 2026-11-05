# Handoff — state as of 2026-11-05 ~12:50 SAST

## Done (on `main`)
- **Data**: VD-level Gauteng history (2016/2021 LGE PR, 2019/2024 PE) in `data/processed/vd_history_gauteng.csv`;
  2026 VD/ward delimitation, touching-VD graphs, VD→ward crosswalk (`data/processed/`). Raw inputs: `scripts/00_fetch_data.sh`.
- **Model** (`src/saelection/model.py`, NumPyro, VI by default, `SAELECTION_METHOD=nuts` for HMC):
  past performance + ideological substitution (`config/ideology.json`) + Leroux CAR over touching VDs
  + LKJ-correlated party residuals + common turnout field.
- **Training** (`scripts/run_train.py`, ~8 min CPU) — artifacts in `artifacts/` are current.
  Spatial hold-out test (`artifacts/spatial_test.json`): CAR kept for DA (+15.7±3.6 elpd), ActionSA (+101.9±21.9),
  Other (+24.5±7.9), MK (untestable, default); dropped for ANC (+3.4±4.6) and EFF (+4.4±3.8).
- **Live projection** (`src/saelection/live.py`, `scripts/run_live.py`): conditions on VD-level IEC CSVs or on
  ward/municipal partial totals + % VDs counted (`data/observed_2026/progress_*.csv`, see README there).
  Writes `output/live/` (schema in README). Uncounted VDs use the exact CAR conditional.
  Smoke test passed (5% of 2024 VDs revealed → DA 25.0 vs 27.4 actual, ANC 36.4 vs 34.8, MK 8.1 vs 9.8).

## Not done yet
1. **Backtest** never completed (crashed on an initialiser bug, now fixed). Run:
   `PYTHONPATH=src python -m saelection.backtest` (~30–45 min CPU; 3 orders × 4 levels × VD/ward modes).
2. **2026 projection** not yet run with real counts → `output/live/` not generated yet.
3. **Frontend export**: the web app on branch `claude/eager-mayer-vzaen0` (repo default branch) reads
   `public/data/{wards.geojson,predictions.json,results.json}`:
   - `wards.geojson`: features with `properties.ward_id` (string, e.g. "79800001")
   - `predictions.json`: `{generated_at, wards: {ward_id: {shares: {party: pct 0-100}, confidence: 0-1}}}`
   - `results.json`: `{updated_at, source, wards: {ward_id: {status: not_started|counting|final, registered, votes: {party: n}}}}`
   - party codes there: `ANC, DA, EFF, MK, IFP, PA, ActionSA, FF+` (note `ActionSA`, not `ACTIONSA`).
   An exporter from `output/live/wards.json` → these files still needs writing (`confidence` ≈ winner's `p_first`).

## IEC 2026 live data (results.elections.org.za, now reachable)
- Municipal "Results Summary – All Ballots" (Excel, live, all parties, Ward/PR/DC40 columns):
  `https://results.elections.org.za/home/LGEPublicReports/1887/Detailed Results/GP/{JHB|TSH|EKU|GT421|...|DC42|DC48}.xls`
  (1887 = 2026 LGE event id). JHB at 12:40: PR valid 159,622, PR votes cast 161,262 — partial count, no VD count in the file.
- The download form (`/home/Downloads/ME-Results/`, ASP.NET postbacks; report 22, province 3) lists wards and VDs,
  but did **not** expose ward- or VD-level file links in my test — check the dashboard
  `/dashboards/lge/LGEResultsDashboard.aspx` network calls for a ward/VD JSON feed, or the IEC API
  (`api.elections.org.za`, needs an API key). Ward totals + % VDs counted are enough for `progress_wards.csv`.

## Kaggle
Env secrets are `KAGGLE_ACCOUNT` / `KAGGLE_API_TOKEN` (only visible to sessions started after they were added).
`scripts/kaggle_push.sh` maps them to the CLI's names and pushes `kaggle/saelection_gauteng_2026.ipynb` (Internet on).
