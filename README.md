# Gauteng 2026 — Predictions by District and Ward

A web app for following the 2026 local government election forecast for **Gauteng** against the live count, at three levels: the **Gauteng prediction**, each **district** (the three metros, Sedibeng and West Rand), and each of the 526 **wards**.

## What it is

- **Gauteng prediction.** The headline panel shows:
  - the predicted largest party and its vote share (whether it has an outright majority is shown only for a district: councils are elected per municipality, so there is no Gauteng-wide majority)
  - predicted vote share against the votes counted so far, as two stacked bars split by party
  - wards reporting, how often the prediction is holding, upsets, and most predicted ward wins

  Pick a district (header menu, district list, or a click on the map) and the panel switches to that district's prediction.
- **Districts.** District borders are drawn heavier than ward borders, and each district is labelled on the map. The **Districts** tab lists every district with its predicted largest party, forecast vs counted vote share, wards reporting, how often the prediction is holding, and upsets. Hovering a district outlines it on the map; clicking one zooms to it and fades the others. Press **Esc** or click **All districts** to zoom back out.
- **Ward map.** Each ward is filled with its predicted winner's colour; darker shading means a more confident prediction. Scroll or pinch to zoom, drag to pan, or use the + / − / reset buttons. The **Current result** toggle recolours the map by who leads now.
- **Hover card.** Hovering a ward shows its count status, votes and turnout, whether the leader matches the prediction, and each party's current share with a tick at its predicted share.
- **Ward panel.** Clicking a ward opens its panel and zooms to its district. The panel has:
  - the current result against the prediction
  - **Vote share across elections:** past results (2016 local, 2019 provincial, 2021 local, 2024 provincial) as a solid line, continuing as a dashed line to the 2026 forecast (hollow dot), with the 2026 count so far as a filled dot
  - **Election night count:** each party's running vote share at each results update

  Lines are smooth curves that pass through every real data point.
- **Parties and Upsets tabs.** Predicted ward wins against wards each party leads now, and the wards where someone other than the predicted winner leads. Both follow the selected district.
- **Live updates.** `results.json` is reloaded every 60 seconds.

The layout is designed for laptop and desktop screens.

## Why it's being built

The ward is the basic unit of South Africa's local government elections, and wards roll up into municipalities and districts. On election night you want to know whether the forecast is holding at every level: overall, in each district, and in the individual wards that are swinging. Tables of thousands of wards don't show that at a glance. This app starts from the general prediction, lets you drill into a district, and puts each ward's trend and count one click away.

## Live site

**https://zentannen.github.io/SAelection/**, deployed by `.github/workflows/deploy-pages.yml` to GitHub Pages on every push to `main`, and every 10 minutes from the default branch.

Each deploy fetches a fresh IEC count with `npm run fetch:results` (`scripts/fetch-iec-results.mjs`): the PR ballot per ward from the IEC results dashboard feed, in the model's six groups, with each ward's election-night timeline extended from the previously deployed `results.json`. If the feed is down, the committed `results.json` is used. The forecast (`predictions.json`) only changes when a new model run is exported and committed (`scripts/build_gauteng_data.py`). With no forecast loaded, the map opens on **Current result**.

## Stack

[Vite](https://vite.dev) + React + TypeScript, styled with [Tailwind CSS](https://tailwindcss.com) and [shadcn/ui](https://ui.shadcn.com) components (Card, Badge, Button, Select, Tabs, Table, Toggle Group, Tooltip, Chart). Charts use shadcn's Recharts-based `Chart`.

The map is plain SVG: [d3-geo](https://d3js.org/d3-geo) projects the shapes, [d3-zoom](https://d3js.org/d3-zoom) handles pan and zoom, and [TopoJSON](https://github.com/topojson/topojson) works out the district borders from the ward boundaries. The past-elections line is drawn with [d3-shape](https://d3js.org/d3-shape) so it stays one smooth curve while switching from solid to dashed. Light and dark themes follow the OS setting.

## Running it

```bash
npm install
```

```bash
npm run dev
```

| Script | What it does |
|---|---|
| `npm run build` | Type-checks and builds to `dist/` |
| `npm run preview` | Serves the production build |
| `npm run generate:sample` | Overwrites `public/data/` with synthetic sample data (`-- --wards 300 --seed 3` to vary it) |

## Data

The app reads four static files from `public/data/`. They hold the real 2026 Gauteng data, built from the projection model on the `main` branch:

```bash
python3 scripts/build_gauteng_data.py <checkout-of-main> public/data
```

| File | From `main` |
|---|---|
| `wards.geojson` | `output/frontend/wards.geojson`, with each municipality's district added |
| `predictions.json` | `output/frontend/predictions.json` (ward forecasts), plus the model's own Gauteng and district figures from `output/live/{province,municipalities}.json` |
| `results.json` | `output/frontend/results.json` (IEC dashboard count, PR-equivalent votes) |
| `history.json` | `data/processed/vd_history_gauteng.csv`, moved onto the 2026 wards with `vd_ward_crosswalk_2026.csv` |

Everything is on the PR ballot and in the model's six groups: **DA, ANC, ActionSA, MK, EFF and Other**, so the forecast, the count and past results compare like for like. Past results are the 2016 and 2021 local elections (PR ballot) and the 2019 and 2024 **Gauteng provincial** ballots, the same four elections the model is trained on. Re-run the build after each new model run (`scripts/update_live.sh` on `main`).

`npm run generate:sample` overwrites these files with synthetic data (the app then shows a "Sample data" banner).

### `wards.geojson`: ward boundaries (and districts)

A GeoJSON `FeatureCollection` of `Polygon` or `MultiPolygon` features in WGS84 longitude/latitude. `ward_id` is required. `district` puts the ward in a district; wards without one appear only under "All districts". `name` and `municipality` are optional and appear in the hover card and ward panel.

```json
{ "type": "Feature",
  "properties": { "ward_id": "79800001", "name": "Ward 1", "municipality": "City of Johannesburg",
                  "district": "City of Johannesburg", "province": "Gauteng" },
  "geometry": { "type": "Polygon", "coordinates": [[[28.01, -26.20], ...]] } }
```

District borders are computed from the ward shapes: any edge shared by two wards in different districts is drawn as a district border, so there's no separate district file. In South Africa a metro is its own district, so give metro wards the metro's name as their `district`.

Official ward boundaries are published by the Municipal Demarcation Board as shapefiles. Convert and simplify them with [mapshaper](https://mapshaper.org); for example:

```bash
npx mapshaper wards.shp -proj wgs84 -simplify 10% keep-shapes -rename-fields ward_id=WardID -o format=geojson public/data/wards.geojson
```

Change `WardID` to whatever the shapefile calls its ward code, and add the `district` field from the municipality → district lookup. Either ring winding order works; the loader fixes it. Simplify before loading the whole country, because ~4,500 full-resolution wards make a very large file.

### `predictions.json`: the forecast

`shares` holds predicted vote share percentages (0–100). `confidence` (0–1) is optional and sets how strongly the ward is shaded.

```json
{ "generated_at": "2026-11-01T00:00:00Z",
  "wards": { "79800001": { "shares": { "ANC": 41.2, "DA": 30.5, "EFF": 12.0 }, "confidence": 0.72 } },
  "overall":   { "shares": { "ANC": 38.0, "DA": 27.1, "MK": 14.9 } },
  "districts": { "City of Johannesburg": { "shares": { "ANC": 35.2, "DA": 29.8 } } } }
```

`overall` and `districts` are optional. If your model produces its own area-wide or district figures, put them here and the app shows them as-is ("Model forecast"). Otherwise the app builds them from the ward forecasts: an average of each ward's predicted shares, weighted by registered voters (from `results.json`) when every ward has that figure, and counting each ward equally if not.

### `results.json`: the live count

`votes` holds cumulative vote counts so far. `status` is `not_started`, `counting` or `final`. `registered` is optional and used for turnout and weighting. `timeline` is optional: cumulative counts at each results update, oldest first. It drives the election-night chart, and its last entry should match `votes`.

```json
{ "updated_at": "2026-11-05T20:15:00Z", "source": "IEC",
  "wards": { "79800001": {
    "status": "counting", "registered": 12000,
    "votes": { "ANC": 1520, "DA": 1310, "EFF": 402 },
    "timeline": [
      { "at": "2026-11-05T19:00:00Z", "votes": { "ANC": 610, "DA": 580, "EFF": 150 } },
      { "at": "2026-11-05T20:15:00Z", "votes": { "ANC": 1520, "DA": 1310, "EFF": 402 } } ] } } }
```

Overwrite this file as results arrive, appending to each ward's `timeline`. The app picks up the new file within a minute. Times are shown in SAST.

### `history.json`: past elections (optional)

Vote share per ward in earlier elections, oldest first. Leave a party out of an election it didn't contest; the chart shows a gap rather than 0%. If the file is missing, the past-elections chart says so and everything else works.

```json
{ "elections": [ { "id": "2016", "label": "2016" }, { "id": "2021", "label": "2021" } ],
  "wards": { "79800001": { "2016": { "ANC": 52.3, "DA": 31.0, "EFF": 9.1 },
                           "2021": { "ANC": 44.8, "DA": 29.5, "EFF": 12.6 } } } }
```

Ward boundaries change between elections, so past results must be mapped onto **today's** wards before they go in this file. One way is to re-aggregate voting-district results by which current ward each voting station falls in.

## Coalition scenario plan

The app's **Coalition scenarios** view (header switch, or `/#scenarios`) shows the playbook: a Gauteng table from vote-share bands to likely governments, nine configurations with foreign cases drawn as vote-share bars, party archetypes, regularities, national pressure and the province. Its content lives in `src/scenarios.ts`; the long-form version is [`analysis/gauteng-coalition-game-plan.md`](analysis/gauteng-coalition-game-plan.md), a coalition scenario playbook built from comparative cases, mostly Central and Eastern Europe. It sets out what governments formed under different vote-share configurations and party values, how long they lasted and why, then maps those patterns onto Johannesburg, Tshwane, Ekurhuleni and the Gauteng provincial government.

## Party colours

Parties and colours are set in `src/parties.ts` and `src/index.css` (`--party-*`). The five parties the Gauteng model forecasts (DA, ANC, ActionSA, EFF, MK) get their own colour. Every other party is shown in a shared grey **Other** colour on the map, bars and charts, matching the model's "Other" group. Party names always appear next to the colour in hover cards, tooltips, legends and tables.

The DA, ANC, EFF and MK colours follow each party's usual colour and ActionSA is shown in teal; all were adjusted in lightness to stay distinguishable and were checked with a colour-blindness validator. The closest pair is MK and ANC for red-blind (protan) readers, which is acceptable only because names are always shown alongside. Re-check if you change the colours.

## Project layout

```
public/data/                 wards.geojson, predictions.json, results.json, history.json
scripts/build_gauteng_data.py     builds public/data/ from the model output on main
scripts/generate-sample-data.mjs  synthetic sample data
src/
  App.tsx                    view switch (ward map / coalition scenarios), kept in the URL hash
  map-app.tsx                ward map page: layout, district selection, colour-by toggle, side panel tabs
  scenarios.ts               coalition scenario content and foreign cases
  data.ts                    loading + ring-winding fix
  lib/model.ts               ward summaries, area/district roll-ups, 60s results polling
  lib/geo.ts                 district borders derived from ward topology
  parties.ts                 party names, colours, "Other" folding
  components/
    prediction-overview.tsx  general / district prediction panel
    district-list.tsx        district comparison list
    ward-map.tsx             SVG map: ward + district borders, labels, pan/zoom, focus
    scenarios-view.tsx       coalition scenarios page
    vote-bar.tsx             vote-share bar coloured by role in government
    ward-hover-card.tsx      hover card (current result vs prediction)
    ward-detail.tsx          ward panel
    ward-charts.tsx          vote share across elections + election-night count charts
    segmented-bar.tsx        stacked share bars with tooltips
    party-chart.tsx          wards-by-party bar chart
    upsets-table.tsx         upsets table
    ui/                      shadcn/ui components
```

## The projection model

The forecast comes from the Bayesian spatial model in `src/saelection/` (Python).

### How it works

For VD *i* and party *k* (log-ratio of the party's votes to non-voters, so one model covers both
party choice and turnout):

```
eta_ik = A_k                         provincial swing level             (from the count / polls)
       + delta_m(i),k                municipal swing                    (from the count)
       + b1_k X_LGE2021,ik           own past performance, last LGE     (learned 2016->2021)
       + b2_k X_PE2024,ik            own past performance, last PE
       + sum_j C_kj X_j              SUBSTITUTION: past strength of ideologically similar
                                     parties j (config/ideology.json: DA~ActionSA, ANC~MK~EFF,
                                     DA/ActionSA~Other)
       + phi_ik + g_i                SPATIAL: Leroux CAR over VDs whose polygons touch,
                                     one field per party + a common turnout field
       + psi_ik                      VD residual, correlated across parties (LKJ prior);
                                     negative correlations = substitution inside a VD
votes_i ~ Multinomial(registered_i, softmax(eta_i, 0))
```

**Spatial term.** Only contiguity matters: VDs are neighbours iff their boundaries touch.
For each party, λ_k measures how strongly touching VDs co-vary. A hold-out test (20% of VDs) compares
the model with and without the spatial term. The term is switched off for any party where it does not
improve out-of-sample prediction (`artifacts/spatial_test.json`).

**Live counting.** Counted VDs are fixed at their actual result. Each one is compared with *its
own* history, so the count tells the model about the swing (A, δ). Through the CAR term, it also
tells the model about the uncounted VDs that touch it. A party that leads early only because its
strongholds counted first is not projected to keep that lead. `artifacts/backtest_summary.csv`
replays the 2024 count in deliberately biased orders to check this.

### Running it

```bash
pip install -r requirements.txt
```
```bash
bash scripts/00_fetch_data.sh        # raw IEC results + boundaries (~1 GB, not committed)
```
```bash
python scripts/prepare_data.py       # data/processed/: VD history, contiguity graphs, VD->ward crosswalk
```
```bash
python scripts/run_train.py          # artifacts/: learned structure, spatial hold-out test (~10 min CPU)
```
```bash
PYTHONPATH=src python -m saelection.backtest   # artifacts/backtest_*.csv (count replay on 2024)
```
```bash
python scripts/run_live.py           # output/live/: projections for the frontend
```

**Live count (IEC dashboard feed).** `scripts/update_live.sh` runs one refresh cycle:
`python -m saelection.iec_live` pulls `results.elections.org.za/dashboards/lge/MapsJason/2026/EEID1887Munic<id>.js`
for the 9 Gauteng municipalities (per-VD votes, counting progress, 2026 registered voters). It converts the
per-VD all-ballot votes to PR-equivalent votes, re-runs the projection, and exports the web-app files to
`output/frontend/` (`predictions.json`, `results.json`, `wards.geojson`, in the contract of the map app on branch
`claude/eager-mayer-vzaen0`). About 3 minutes per cycle on 4 CPUs.

Alternatively, put the IEC 2026 per-VD result CSVs in `data/observed_2026/`
(see the README there) and re-run `scripts/run_live.py`. Before any VD is counted, the projection
uses the poll prior in `config/live.json`.

**Kaggle:** `kaggle/saelection_gauteng_2026.ipynb` runs the whole pipeline. Set your username in
`kaggle/kernel-metadata.json`, then run `kaggle kernels push -p kaggle`, or import the notebook
into Kaggle and turn Internet on.

### Output for the frontend (`output/live/`)

All files are regenerated on every run. JSON is the primary format and CSV is a flat copy of it.
Join to the maps by `id`.

| file | one record per | key fields |
|---|---|---|
| `meta.json` | run | `generated_at`, `n_vds`, `n_vds_counted`, `pct_vds_counted`, `registered_total`, `prior`, `model` (spatial λ per party, substitution matrices, residual correlations), `notes` |
| `province.json` | Gauteng | unit record (below) + `counted_so_far` (actual votes counted per category) |
| `municipalities.json/.csv` | municipality (9) | unit record + `name` |
| `wards.json/.csv` | ward (526) | unit record + `muni`, `ward_no`, `n_vd` |
| `vds.json/.csv` | VD (2,797) | unit record + `muni`, `ward` (main ward), `station`, `lon`, `lat`, `status` = `counted` or `projected` |
| `province_draws.csv` | posterior draw | final vote totals per category for each of the 500 draws (histograms / distributions) |
| `municipality_draws.csv` | draw × municipality | same, per municipality |
| `geo/vds_2026.geojson`, `geo/wards_2026.geojson` | polygon | `id` = VD number / ward id (simplified boundaries for mapping) |

**Unit record** (same shape at every level):

```json
{
  "id": 79800012,
  "pct_vds_counted": 37.5,
  "registered": 41234,
  "turnout_valid": {"mean": 0.41, "p05": 0.38, "p50": 0.41, "p95": 0.44},
  "parties": {
    "DA":  {"votes": {"mean": 6120, "p05": 5410, "p50": 6105, "p95": 6890},
            "share": {"mean": 0.36, "p05": 0.32, "p50": 0.36, "p95": 0.40},
            "p_first": 0.71},
    "ANC": {"...": "..."}, "ACTIONSA": {}, "MK": {}, "EFF": {}, "OTHER": {}
  },
  "projected_leader": "DA"
}
```

* `votes` = projected **final** PR votes, including votes already counted. For a counted VD,
  mean = p05 = p95 = the actual count.
* `share` = share of valid votes. `turnout_valid` = valid votes / registered voters.
* `p_first` = probability that the party finishes with the most votes in that unit.
* Remaining votes still to come = `votes` − `counted_so_far`, or per VD where `status` is `projected`.

### Data sources

* IEC VD-level results (results.elections.org.za), as republished in
  `xyclone-designs/database.iec`
* MDB/IEC voting-district and ward boundaries (2021, 2024, 2026), as mirrored in
  `garyvdm/demarcation-compare`
* Poll prior: Social Research Foundation (Q1 2026) and eNCA/MarkData Gauteng polls (see `config/live.json`)
