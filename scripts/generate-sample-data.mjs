// Generates SYNTHETIC sample data so the app runs before real data is wired in.
// The ward shapes, names, predictions and results are all made up.
//
// Outputs (overwrites):
//   public/data/wards.geojson     ward boundaries (Voronoi cells)
//   public/data/predictions.json  predicted vote share per ward
//   public/data/results.json      current (partial) count per ward, with an
//                                 election-night timeline of cumulative counts
//   public/data/history.json      vote share per ward in past elections
//
// Usage: npm run generate:sample [-- --wards 200 --seed 7]

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
const WARD_COUNT = Number(args.wards ?? 200);
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
const at = (fx, fy) => [BOX.west + fx * (BOX.east - BOX.west), BOX.south + fy * (BOX.north - BOX.south)];

// Districts are placed by compass direction so their names match where they
// sit on the map. Each district is split into two local municipalities.
// Wards join the district / municipality whose seed point is nearest.
const DISTRICTS = [
  { name: 'North District', seed: at(0.5, 0.92) },
  { name: 'East District', seed: at(0.92, 0.5) },
  { name: 'South District', seed: at(0.5, 0.08) },
  { name: 'West District', seed: at(0.08, 0.5) },
  { name: 'Central District', seed: at(0.5, 0.5) },
];
const nearest = (list, [x, y]) =>
  list.reduce((best, d) => {
    // Measured in fractions of the box so districts come out similar in size.
    const dist = ((x - d.seed[0]) / (BOX.east - BOX.west)) ** 2 + ((y - d.seed[1]) / (BOX.north - BOX.south)) ** 2;
    return dist < best.dist ? { d, dist } : best;
  }, { d: list[0], dist: Infinity }).d;
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
const normalise = (shares) => {
  const total = Object.values(shares).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(shares).map(([p, s]) => [p, total ? s / total : 0]));
};
const jitter = (shares, amount) =>
  Object.fromEntries(Object.entries(shares).map(([p, s]) => [p, Math.max(0, s * (1 + amount * gauss()))]));
const toPercent = (shares) => Object.fromEntries(Object.entries(shares).map(([p, s]) => [p, round1(s)]));

// --- Ward geometry -------------------------------------------------------
const points = Array.from({ length: WARD_COUNT }, () => [
  BOX.west + rand() * (BOX.east - BOX.west),
  BOX.south + rand() * (BOX.north - BOX.south),
]);
const voronoi = Delaunay.from(points).voronoi([BOX.west, BOX.south, BOX.east, BOX.north]);

const districtOf = points.map((pt) => nearest(DISTRICTS, pt));
for (const d of DISTRICTS) {
  // Two municipality seeds per district: the member points furthest apart
  // along the district's longer axis.
  const members = points.filter((_, i) => districtOf[i] === d);
  const wide = Math.max(...members.map((p) => p[0])) - Math.min(...members.map((p) => p[0]));
  const tall = Math.max(...members.map((p) => p[1])) - Math.min(...members.map((p) => p[1]));
  const axis = wide >= tall ? 0 : 1;
  const sorted = [...members].sort((a, b) => a[axis] - b[axis]);
  const short = d.name.replace(' District', '');
  d.municipalities = [
    { name: `${short} A Local Municipality`, seed: sorted[Math.floor(sorted.length * 0.25)] },
    { name: `${short} B Local Municipality`, seed: sorted[Math.floor(sorted.length * 0.75)] },
  ];
}
const wardCounter = new Map();

const features = [];
const predictions = {};
const results = {};
const history = {};
const NOW = Math.floor(Date.now() / 60_000) * 60_000;

points.forEach(([x, y], i) => {
  const ring = voronoi.cellPolygon(i);
  if (!ring) return;
  const wardId = `SAMPLE${String(i + 1).padStart(3, '0')}`;
  const district = districtOf[i];
  const municipality = nearest(district.municipalities, [x, y]);
  const n = (wardCounter.get(municipality.name) ?? 0) + 1;
  wardCounter.set(municipality.name, n);

  features.push({
    type: 'Feature',
    properties: {
      ward_id: wardId,
      name: `${municipality.name.replace(' Local Municipality', '')} Ward ${n}`,
      municipality: municipality.name,
      district: district.name,
      province: 'Sample Province',
    },
    geometry: { type: 'Polygon', coordinates: [ring.map(([a, b]) => [+a.toFixed(5), +b.toFixed(5)])] },
  });

  const predicted = sharesAt(x, y, 0.08);
  predictions[wardId] = {
    shares: Object.fromEntries(PARTIES.map((p) => [p, round1(predicted[p])])),
    confidence: +(0.55 + rand() * 0.4).toFixed(2),
  };

  // Past elections. Parties founded recently get no earlier support: MK
  // (founded 2023) is folded into ANC/EFF for 2021, ActionSA (2020) into DA
  // for 2016. ~5% of wards are "new" and have no history, to exercise that state.
  if (rand() > 0.05) {
    const y2021 = { ...sharesAt(x, y, 0.15) };
    y2021.ANC += y2021.MK * 0.85;
    y2021.EFF += y2021.MK * 0.15;
    y2021.MK = 0;
    const y2016 = { ...normalise(jitter(y2021, 0.08)) };
    y2016.DA += y2016.ActionSA * 0.8;
    y2016.ANC += y2016.ActionSA * 0.2 + y2016.EFF * 0.25;
    y2016.EFF *= 0.75;
    y2016.ActionSA = 0;
    // Parties that didn't stand are left out rather than given 0%.
    const stood = (shares) => Object.fromEntries(Object.entries(toPercent(normalise(shares))).filter(([, v]) => v > 0));
    history[wardId] = { 2016: stood(y2016), 2021: stood(y2021) };
  }

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

  // Election-night timeline: votes arrive in batches (one per voting
  // district), each leaning a bit differently, so the running share moves.
  const batches = 2 + Math.floor(rand() * (status === 'final' ? 7 : 5));
  const sizes = Array.from({ length: batches }, () => 0.3 + rand());
  const sizeTotal = sizes.reduce((a, b) => a + b, 0);
  const lastAt = NOW - (status === 'final' ? 30 + rand() * 120 : rand() * 30) * 60_000;
  const gapMs = (25 + rand() * 35) * 60_000;
  const running = Object.fromEntries(PARTIES.map((p) => [p, 0]));
  const timeline = [];
  let assigned = 0;
  sizes.forEach((size, b) => {
    const batchVotes = b === batches - 1 ? cast - assigned : Math.round((cast * size) / sizeTotal);
    assigned += batchVotes;
    const lean = normalise(jitter(actual, 0.35));
    let left = batchVotes;
    PARTIES.forEach((p, j) => {
      const v = j === PARTIES.length - 1 ? left : Math.min(left, Math.round(lean[p] * batchVotes));
      running[p] += v;
      left -= v;
    });
    timeline.push({ at: new Date(lastAt - (batches - 1 - b) * gapMs).toISOString(), votes: { ...running } });
  });
  results[wardId] = { status, registered, votes: { ...running }, timeline };
});

const now = new Date(NOW).toISOString();
const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data');
mkdirSync(outDir, { recursive: true });

writeFileSync(join(outDir, 'wards.geojson'), JSON.stringify({ type: 'FeatureCollection', features }));
writeFileSync(
  join(outDir, 'predictions.json'),
  JSON.stringify({ sample: true, generated_at: now, wards: predictions }, null, 1),
);
writeFileSync(
  join(outDir, 'results.json'),
  JSON.stringify({ sample: true, updated_at: now, source: 'Synthetic sample data', wards: results }),
);
writeFileSync(
  join(outDir, 'history.json'),
  JSON.stringify({
    sample: true,
    elections: [
      { id: '2016', label: '2016' },
      { id: '2021', label: '2021' },
    ],
    wards: history,
  }),
);

console.log(`Wrote ${features.length} sample wards in ${DISTRICTS.length} districts to ${outDir}`);
for (const d of DISTRICTS) {
  const count = features.filter((f) => f.properties.district === d.name).length;
  console.log(`  ${d.name}: ${count} wards`);
}
