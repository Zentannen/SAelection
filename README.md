# SA Election — Ward Predictions

A small web app that maps every ward in a municipal election. Each ward is coloured by its **predicted** winner, and hovering over a ward shows the **current result** next to the prediction.

## What it is

- **Ward map.** Every ward is drawn with its own boundary and filled with the colour of the party predicted to win it. Darker shading means a more confident prediction. Zoom by scrolling or pinching, pan by dragging, or use the + / − / reset buttons. Ward boundaries stay crisp at any zoom level.
- **Hover card.** Hover over a ward, or tab to it with the keyboard, to see:
  - the count status (no results yet, counting, or final)
  - votes counted and turnout
  - whether the current leader matches the prediction
  - each party's current share as a bar, with a tick marking its predicted share
- **"Current result" toggle.** Recolours the map by who is leading right now. Lighter shading means the ward is still counting; solid means final.
- **Summary cards.** Wards reporting, how often the prediction is holding, the number of upsets, and the predicted largest party.
- **Wards by party chart.** Predicted ward wins against wards each party leads now.
- **Upsets table.** Wards where someone other than the predicted winner leads. Hovering over a row highlights that ward on the map.
- **Live updates.** The app reloads `results.json` every 60 seconds, so the page keeps up as results come in on election night without a manual refresh.

## Why it's being built

The ward is the basic unit of South Africa's local government elections. A forecast is only useful on election night if you can quickly see where it is holding up and where it is going wrong. Tables of thousands of wards don't show that at a glance. A map coloured by the forecast, with the live count one hover away, does: you can scan a whole municipality, spot upsets as they happen, and drill into any ward without leaving the page.

## Stack

[Vite](https://vite.dev) + React + TypeScript, styled with [Tailwind CSS](https://tailwindcss.com) and [shadcn/ui](https://ui.shadcn.com) components (Card, Badge, Table, Toggle Group, Button, Chart). The charts use shadcn's Recharts-based `Chart` component. The map is plain SVG: [d3-geo](https://d3js.org/d3-geo) projects the shapes and [d3-zoom](https://d3js.org/d3-zoom) handles pan and zoom. Light and dark themes follow the OS setting.

## Running it

```bash
npm install
```

```bash
npm run dev
```

Other scripts:

| Script | What it does |
|---|---|
| `npm run build` | Type-checks and builds to `dist/` |
| `npm run preview` | Serves the production build |
| `npm run generate:sample` | Regenerates the synthetic sample data (`-- --wards 200 --seed 3` to vary it) |

## Data

The app reads three static files from `public/data/`. **The files in the repo now are synthetic sample data.** The ward shapes, names, predictions and results are all made up, and the app shows a "Sample data" banner while they are in use. Replace the files with real data and the banner disappears.

### `wards.geojson`: ward boundaries

A GeoJSON `FeatureCollection` of `Polygon` or `MultiPolygon` features in WGS84 longitude/latitude. Each feature needs a `ward_id`; the other properties are optional and appear in the hover card.

```json
{ "type": "Feature",
  "properties": { "ward_id": "79800001", "name": "Ward 1", "municipality": "City of Johannesburg", "province": "Gauteng" },
  "geometry": { "type": "Polygon", "coordinates": [[[28.01, -26.20], ...]] } }
```

Official ward boundaries are published by the Municipal Demarcation Board as shapefiles. Convert and simplify them with [mapshaper](https://mapshaper.org), for example:

```bash
npx mapshaper wards.shp -proj wgs84 -simplify 10% keep-shapes -rename-fields ward_id=WardID -o format=geojson public/data/wards.geojson
```

Change `WardID` to whatever the shapefile calls its ward code. Either ring winding order works; the loader normalises it. Simplify the shapes before loading the whole country, because ~4,500 full-resolution wards make a very large file.

### `predictions.json`: forecast per ward

`shares` holds predicted vote share percentages (0–100). `confidence` (0–1) is optional and sets how strongly the ward is shaded.

```json
{ "generated_at": "2026-11-01T00:00:00Z",
  "wards": { "79800001": { "shares": { "ANC": 41.2, "DA": 30.5, "EFF": 12.0 }, "confidence": 0.72 } } }
```

### `results.json`: current count per ward

`votes` holds raw vote counts so far. `status` is `not_started`, `counting` or `final`. `registered` is optional and used for turnout.

```json
{ "updated_at": "2026-11-05T20:15:00Z", "source": "IEC",
  "wards": { "79800001": { "status": "counting", "registered": 12000, "votes": { "ANC": 1520, "DA": 1310, "EFF": 402 } } } }
```

Overwrite this file as results arrive. The app picks up the new file within a minute.

## Coalition scenario plan

[`analysis/gauteng-coalition-game-plan.md`](analysis/gauteng-coalition-game-plan.md) sets out how different 2026 results could turn into governments in Johannesburg, Tshwane and Ekurhuleni, and what each outcome means for the Gauteng provincial government: the seat thresholds, what each party believes and needs, national-government pressure, metro-by-metro scenarios, and international precedents.

## Party colours

Parties and colours are set in `src/parties.ts` and `src/index.css` (`--party-*`). Only the main ward-winning parties (ANC, DA, EFF, MK, IFP) get their own colour; all other parties share a grey **Other** colour on the map and chart. Eight colours on one map can't all be told apart reliably, especially by colour-blind readers. Each party's name is always shown next to its colour in the hover card, legend and table.

The five party colours follow each party's usual colour but were adjusted in lightness so they stay distinguishable, and were checked with a colour-blindness validator. The closest pair is MK and ANC for red-blind (protan) readers, which sits in the acceptable range only because the party names are always shown. Re-check if you change the colours.

## Project layout

```
public/data/                 wards.geojson, predictions.json, results.json
scripts/generate-sample-data.mjs
src/
  App.tsx                    page layout, colour-by toggle, map legend
  data.ts                    loading + ring-winding fix
  lib/model.ts               per-ward summaries, 60s results polling
  parties.ts                 party names + colours
  components/
    ward-map.tsx             SVG map, boundaries, pan/zoom
    ward-hover-card.tsx      hover card (current result vs prediction)
    party-chart.tsx          wards-by-party bar chart
    stat-cards.tsx           summary cards
    upsets-table.tsx         upsets table
    ui/                      shadcn/ui components
```
