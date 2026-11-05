// Fetches the live 2026 LGE count for Gauteng from the IEC results dashboard feed
// and writes public/data/results.json in the app's format.
//
// Source: https://results.elections.org.za/dashboards/lge/ (the JSON files its map loads).
// Votes are the ward ballot (who wins the ward seat). A ward is "final" once every one of its
// voting districts has reported, "counting" once any has.
//
// If PREVIOUS_RESULTS_URL is set (the deployed results.json), each ward's election-night
// timeline is carried forward and extended whenever its count changes.
//
// Usage: node scripts/fetch-iec-results.mjs [out=public/data/results.json]

import { writeFileSync } from 'node:fs';

const OUT = process.argv[2] ?? 'public/data/results.json';
const BASE = 'https://results.elections.org.za/dashboards/lge/MapsJason/2026/';
const EVENT = 'EEID1887';
const PROVINCE = 'Gauteng';

const PARTY_CODES = {
  'AFRICAN NATIONAL CONGRESS': 'ANC',
  'DEMOCRATIC ALLIANCE': 'DA',
  'ECONOMIC FREEDOM FIGHTERS': 'EFF',
  'UMKHONTO WESIZWE PARTY': 'MK',
  'INKATHA FREEDOM PARTY': 'IFP',
  'PATRIOTIC ALLIANCE': 'PA',
  'ACTIONSA': 'ActionSA',
  'VRYHEIDSFRONT PLUS | FREEDOM FRONT PLUS': 'FF+',
};
const code = (name) => PARTY_CODES[name] ?? name;

async function getJson(url, { optional = false } = {}) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'cache-control': 'no-cache' } });
      if (!res.ok) throw new Error(`${res.status} ${url}`);
      return JSON.parse((await res.text()).replace(/^﻿/, ''));
    } catch (err) {
      if (optional) return null;
      if (attempt >= 4) throw err;
      await new Promise((r) => setTimeout(r, 2000 * 2 ** attempt));
    }
  }
}

const sum = (xs) => xs.reduce((a, b) => a + b, 0);

const national = await getJson(`${BASE}${EVENT}LiteMapsNational.js?rnd=${Date.now()}`);
const province = national.ProvinceResults.find((p) => p.Province === PROVINCE);
if (!province) throw new Error(`${PROVINCE} not in the national feed`);

const previous = process.env.PREVIOUS_RESULTS_URL
  ? await getJson(`${process.env.PREVIOUS_RESULTS_URL}?rnd=${Date.now()}`, { optional: true })
  : null;

const wards = {};
let latest = national.ReportDate;
for (const m of province.MunicipalityResults) {
  const munic = await getJson(`${BASE}${EVENT}Munic${m.MunicipalityID}.js?rnd=${Date.now()}`);
  if (munic.ReportDate > latest) latest = munic.ReportDate;
  for (const w of munic.WardResults) {
    const id = String(w.WardID);
    const votes = {};
    for (const p of w.PartyBallotResults) if (p.WardVotes > 0) votes[code(p.Name)] = p.WardVotes;
    const vds = w.VotingDistrictResults;
    const reported = vds.filter((v) => sum(v.PartyBallotResults.map((p) => p.TotalValidVotes)) > 0).length;
    const status = vds.length && reported === vds.length ? 'final' : reported > 0 ? 'counting' : 'not_started';

    const timeline = [...(previous?.wards?.[id]?.timeline ?? [])];
    const last = timeline.at(-1);
    if (sum(Object.values(votes)) > 0 && JSON.stringify(last?.votes) !== JSON.stringify(votes)) {
      timeline.push({ at: w.ReportDate, votes });
    }
    wards[id] = { status, votes, vds_reported: reported, vds_total: vds.length, timeline };
  }
}

const all = Object.values(wards);
writeFileSync(
  OUT,
  JSON.stringify({
    updated_at: latest,
    source: 'IEC (results.elections.org.za), 2026 Local Government Elections, Gauteng ward ballot',
    wards,
  }),
);
console.log(
  `${all.length} wards: ${all.filter((w) => w.status === 'final').length} final, ` +
    `${all.filter((w) => w.status === 'counting').length} counting; ` +
    `VDs ${sum(all.map((w) => w.vds_reported))}/${sum(all.map((w) => w.vds_total))}; IEC report ${latest}`,
);
