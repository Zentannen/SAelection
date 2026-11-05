// Generates SYNTHETIC sample data so the app runs before real data is wired in.
// The ward shapes, names, predictions and results are all made up.
//
// Outputs (overwrites):
//   public/data/wards.geojson     ward boundaries (Voronoi cells)
//   public/data/predictions.json  predicted vote share per ward
//   public/data/results.json      current (partial) count per ward
//
// Usage: npm run generate:sample [-- --wards 120 --seed 7]

import { Delaunay } from 'd3-delaunay';
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);
const WARD_COUNT = Number(args.wards ?? 120);
let seed = Number(args.seed ?? 7);

// Small deterministic PRNG (mulberry32) so the sample data is reproducible.
function rand() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-9)) * Math.cos(2 * Math.PI * rand());

// Rough lon/lat box (Gauteng-ish) purely so the map projects sensibly.
const BOX = { west: 27.75, east: 28.45, south: -26.45, north: -25.95 };
const PARTIES = ['ANC', 'DA', 'EFF', 'MK', 'IFP', 'PA', 'ActionSA', 'FF+'];

// Each party gets a few "strongholds"; support decays with distance from them.
const strongholds = Object.fromEntries(
  PARTIES.map((p, i) => [
    p,
    Array.from({ length: i < 4 ? 3 : 1 }, () => ({
      x: BOX.west + rand() * (BOX.east - BOX.west),
      y: BOX.south + rand() * (BOX.north - BOX.south),
      w: (i < 2 ? 1.6 : i < 4 ? 1.0 : 0.45) * (0.6 + rand() * 0.8),
    })),
  ]),
);

function sharesAt(x, y, noise) {
  const raw = {};
  for (const p of PARTIES) {
    let s = 0.05;
    for (const h of strongholds[p]) {
      const d2 = (x - h.x) ** 2 + (y - h.y) ** 2;
      s += h.w * Math.exp(-d2 / 0.02);
    }
    raw[p] = Math.max(0.01, s * (1 + noise * gauss()));
  }
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  return Object.fromEntries(PARTIES.map((p) => [p, raw[p] / total]));
}

const round1 = (n) => Math.round(n * 1000) / 10; // fraction -> percent, 1 dp

// --- Ward geometry -------------------------------------------------------
const points = Array.from({ length: WARD_COUNT }, () => [
  BOX.west + rand() * (BOX.east - BOX.west),
  BOX.south + rand() * (BOX.north - BOX.south),
]);
const voronoi = Delaunay.from(points).voronoi([BOX.west, BOX.south, BOX.east, BOX.north]);

const features = [];
const predictions = {};
const results = {};

points.forEach(([x, y], i) => {
  const ring = voronoi.cellPolygon(i);
  if (!ring) return;
  const wardId = `SAMPLE${String(i + 1).padStart(3, '0')}`;

  features.push({
    type: 'Feature',
    properties: {
      ward_id: wardId,
      name: `Sample Ward ${i + 1}`,
      municipality: 'Sample Metro',
      province: 'Sample Province',
    },
    geometry: { type: 'Polygon', coordinates: [ring.map(([a, b]) => [+a.toFixed(5), +b.toFixed(5)])] },
  });

  const predicted = sharesAt(x, y, 0.08);
  predictions[wardId] = {
    shares: Object.fromEntries(PARTIES.map((p) => [p, round1(predicted[p])])),
    confidence: +(0.55 + rand() * 0.4).toFixed(2),
  };

  // Current result: prediction plus extra noise, with a partially completed count.
  const r = rand();
  const registered = Math.round(6000 + rand() * 9000);
  if (r < 0.15) {
    results[wardId] = { status: 'not_started', registered, votes: {} };
    return;
  }
  const status = r < 0.55 ? 'counting' : 'final';
  const turnout = 0.35 + rand() * 0.25;
  const counted = status === 'final' ? 1 : 0.1 + rand() * 0.8;
  const actual = sharesAt(x, y, 0.3);
  const cast = Math.round(registered * turnout * counted);
  results[wardId] = {
    status,
    registered,
    votes: Object.fromEntries(PARTIES.map((p) => [p, Math.round(actual[p] * cast)])),
  };
});

const now = new Date().toISOString();
const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data');
mkdirSync(outDir, { recursive: true });

writeFileSync(join(outDir, 'wards.geojson'), JSON.stringify({ type: 'FeatureCollection', features }));
writeFileSync(
  join(outDir, 'predictions.json'),
  JSON.stringify({ sample: true, generated_at: now, wards: predictions }, null, 1),
);
writeFileSync(
  join(outDir, 'results.json'),
  JSON.stringify({ sample: true, updated_at: now, source: 'Synthetic sample data', wards: results }, null, 1),
);

console.log(`Wrote ${features.length} sample wards to ${outDir}`);
