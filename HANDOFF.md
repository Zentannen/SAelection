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

## UPDATE 13:15 — live feed found and wired in
- Per-VD live data: `https://results.elections.org.za/dashboards/lge/MapsJason/2026/EEID1887Munic<id>.js`
  (ids in `src/saelection/iec_live.py`). VD figures are ward+PR combined → divided by ballots-per-voter (≈2.0).
- `scripts/update_live.sh` = fetch → project → export `output/frontend/` for the map app.
- First run (12:57 data, 22% of VDs): DA 35.7%, ANC 22.6%, Other 15.3%, ActionSA 14.9%, EFF 6.7%, MK 4.8%;
  turnout ~53% of registered (counted VDs: 52.7% vs 45.3% in the same VDs in 2021).
- Calibration (count-replay on 2024, `artifacts/backtest_summary.csv`, VD mode, 10%/25% counted):
  projection error vs running-tally error — random order 0.37 vs 0.52/0.33 pp (90% coverage 100%);
  ANC-strongholds-first 1.7–2.2 vs 9.2–9.9 pp (coverage 67%); DA-first 2.4–2.9 vs 12.9–14.8 pp (67%).
  Discrepancy sd = 0.06 + 0.08 x (counted-vs-uncounted gap in past party strength); slopes fixed (freeing
  them made stronghold-first worse). Ward-progress mode not backtested yet.
- 88 VDs exist only in the live feed (new 2026 VDs): their counted votes are added at ward level.

## IEC 2026 live data (older notes)
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
