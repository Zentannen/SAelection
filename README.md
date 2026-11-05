# SA Election — Predictions by District and Ward

A web app for following a municipal election forecast against the live count, at three levels: the **general prediction** for the whole area, each **district**, and each **ward**.

## What it is

- **General prediction.** The headline panel shows:
  - the predicted largest party, its vote share, and whether it has an outright majority
  - predicted vote share against the votes counted so far, as two stacked bars split by party
  - wards reporting, how often the prediction is holding, upsets, and most predicted ward wins

  Pick a district (header menu, district list, or a click on the map) and the panel switches to that district's prediction.
- **Districts.** District borders are drawn heavier than ward borders, and each district is labelled on the map. The **Districts** tab lists every district with its predicted largest party, forecast vs counted vote share, wards reporting, how often the prediction is holding, and upsets. Hovering a district outlines it on the map; clicking one zooms to it and fades the others. Press **Esc** or click **All districts** to zoom back out.
- **Ward map.** Each ward is filled with its predicted winner's colour; darker shading means a more confident prediction. Scroll or pinch to zoom, drag to pan, or use the + / − / reset buttons. The **Current result** toggle recolours the map by who leads now.
- **Hover card.** Hovering a ward shows its count status, votes and turnout, whether the leader matches the prediction, and each party's current share with a tick at its predicted share.
- **Ward panel.** Clicking a ward opens its panel and zooms to its district. The panel has:
  - the current result against the prediction
  - **Vote share across elections:** past results (sample: 2016, 2021) as a solid line, continuing as a dashed line to the 2026 forecast (hollow dot), with the 2026 count so far as a filled dot
  - **Election night count:** each party's running vote share at each results update

  Lines are smooth curves that pass through every real data point.
- **Parties and Upsets tabs.** Predicted ward wins against wards each party leads now, and the wards where someone other than the predicted winner leads. Both follow the selected district.
- **Live updates.** `results.json` is reloaded every 60 seconds.

The layout is designed for laptop and desktop screens.

## Why it's being built

The ward is the basic unit of South Africa's local government elections, and wards roll up into municipalities and districts. On election night you want to know whether the forecast is holding at every level: overall, in each district, and in the individual wards that are swinging. Tables of thousands of wards don't show that at a glance. This app starts from the general prediction, lets you drill into a district, and puts each ward's trend and count one click away.

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
| `npm run generate:sample` | Regenerates the synthetic sample data (`-- --wards 300 --seed 3` to vary it) |

## Data

The app reads four static files from `public/data/`. **The files in the repo now are synthetic sample data.** The ward shapes, districts, predictions, past results and count are all made up, and the app shows a "Sample data" banner while they're in use.

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

## Party colours

Parties and colours are set in `src/parties.ts` and `src/index.css` (`--party-*`). Only the main ward-winning parties (ANC, DA, EFF, MK, IFP, ActionSA) get their own colour. Every other party is shown in a shared grey **Other** colour on the map, bars and charts, because eight colours can't all be told apart reliably, especially by colour-blind readers. Party names always appear next to the colour in hover cards, tooltips, legends and tables.

The ANC, DA, EFF, MK and IFP colours follow each party's usual colour and ActionSA is shown in teal; all were adjusted in lightness to stay distinguishable and were checked with a colour-blindness validator. The closest pair is MK and ANC for red-blind (protan) readers, which is acceptable only because names are always shown alongside. Re-check if you change the colours.

## Project layout

```
public/data/                 wards.geojson, predictions.json, results.json, history.json
scripts/generate-sample-data.mjs
src/
  App.tsx                    layout, district selection, colour-by toggle, side panel tabs
  data.ts                    loading + ring-winding fix
  lib/model.ts               ward summaries, area/district roll-ups, 60s results polling
  lib/geo.ts                 district borders derived from ward topology
  parties.ts                 party names, colours, "Other" folding
  components/
    prediction-overview.tsx  general / district prediction panel
    district-list.tsx        district comparison list
    ward-map.tsx             SVG map: ward + district borders, labels, pan/zoom, focus
    ward-hover-card.tsx      hover card (current result vs prediction)
    ward-detail.tsx          ward panel
    ward-charts.tsx          vote share across elections + election-night count charts
    segmented-bar.tsx        stacked share bars with tooltips
    party-chart.tsx          wards-by-party bar chart
    upsets-table.tsx         upsets table
    ui/                      shadcn/ui components
```
