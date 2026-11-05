import type { PredictionsFile, ResultsFile, Shares, WardCollection, WardResult } from '@/types';

const BASE = `${import.meta.env.BASE_URL}data/`;

async function getJson<T>(file: string): Promise<T> {
  // no-store so polling picks up a freshly written results.json
  const res = await fetch(BASE + file, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`);
  return res.json() as Promise<T>;
}

export const loadWards = () => getJson<WardCollection>('wards.geojson').then(rewind);
export const loadPredictions = () => getJson<PredictionsFile>('predictions.json');
export const loadResults = () => getJson<ResultsFile>('results.json');

/** Parties sorted by share, highest first. */
export function ranked(shares: Shares): [string, number][] {
  return Object.entries(shares).sort((a, b) => b[1] - a[1]);
}

/** Convert raw vote counts to percentage shares. */
export function resultShares(result: WardResult): { shares: Shares; total: number } {
  const total = Object.values(result.votes).reduce((a, b) => a + b, 0);
  const shares: Shares = {};
  for (const [p, v] of Object.entries(result.votes)) shares[p] = total ? (v / total) * 100 : 0;
  return { shares, total };
}

/**
 * d3-geo treats polygons as spherical and expects exterior rings clockwise,
 * whereas most GeoJSON (RFC 7946, e.g. from ogr2ogr/mapshaper) is counter-
 * clockwise. Wrong winding makes a ward cover the whole globe, so normalise.
 */
function rewind(fc: WardCollection): WardCollection {
  const area = (ring: number[][]) => {
    let s = 0;
    for (let i = 0, n = ring.length, j = n - 1; i < n; j = i++) {
      s += (ring[i][0] - ring[j][0]) * (ring[i][1] + ring[j][1]);
    }
    return s; // > 0 means clockwise in lon/lat
  };
  const fixPolygon = (rings: number[][][]) =>
    rings.forEach((ring, i) => {
      const clockwise = area(ring) > 0;
      if ((i === 0) !== clockwise) ring.reverse();
    });
  for (const f of fc.features) {
    const g = f.geometry;
    if (g.type === 'Polygon') fixPolygon(g.coordinates);
    else if (g.type === 'MultiPolygon') g.coordinates.forEach(fixPolygon);
  }
  return fc;
}
