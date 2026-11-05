"""Fetch the live 2026 count for Gauteng from the IEC results dashboard.

The dashboard (results.elections.org.za/dashboards/lge) loads one JSON file per municipality:

    MapsJason/2026/EEID1887Munic<municipality id>.js
      .PartyBallotResults[]               municipal totals per party (WardVotes, PRVotes, ...)
      .VDExpected / .VDsComplete          counting progress, .RegPop registered voters
      .WardResults[].VotingDistrictResults[].PartyBallotResults[].TotalValidVotes
                                          per-VD votes, all ballots combined

Per-VD figures add the ward and PR ballots (and the DC 40% ballot in local municipalities), so
they are converted to PR-equivalent votes by dividing by the municipality's ballots-per-voter
ratio (all-ballot valid votes / PR valid votes, from the same file). A VD is "counted" once it
has any votes. Output: data/observed_2026/iec_live_vd.csv (+ iec_live_munis.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import PARTIES, PARTY_MAP, ROOT

BASE = "https://results.elections.org.za/dashboards/lge/MapsJason/2026/EEID1887Munic{}.js"
MUNI_IDS = {"EKU": 3002, "JHB": 3003, "TSH": 3004, "GT421": 3302, "GT422": 3303, "GT423": 3304,
            "GT481": 3040, "GT484": 3039, "GT485": 3332}
OBS = ROOT / "data" / "observed_2026"
CACHE = ROOT / "data" / "cache" / "iec_live"
NAME_MAP = dict(PARTY_MAP, **{"UMKHONTO WESIZWE PARTY": "MK"})


def _cat(name: str) -> str:
    return NAME_MAP.get(str(name).strip().upper(), "OTHER")


def fetch(muni: str, session=None) -> dict:
    s = session or requests.Session()
    r = s.get(BASE.format(MUNI_IDS[muni]), params={"rnd": int(time.time())}, timeout=120,
              headers={"User-Agent": "Mozilla/5.0 (SAelection research model)"})
    r.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{muni}.json").write_bytes(r.content)
    return json.loads(r.content.decode("utf-8-sig"))


def parse(muni: str, d: dict):
    mp = d["PartyBallotResults"]
    pr_valid = sum(p.get("PRVotes", 0) or 0 for p in mp)
    all_valid = sum(p.get("TotalValidVotes", 0) or 0 for p in mp)
    ratio = all_valid / pr_valid if pr_valid > 0 else (2.0 if muni in ("JHB", "TSH", "EKU") else 3.0)
    pr_by_cat = dict.fromkeys(PARTIES, 0)
    for p in mp:
        pr_by_cat[_cat(p["Name"])] += p.get("PRVotes", 0) or 0
    meta = {"muni": muni, "report_date": d.get("ReportDate"), "vd_expected": d.get("VDExpected"),
            "vds_complete": d.get("VDsComplete"), "pct_vds_complete": d.get("Perc_VDsComplete"),
            "registered_2026": d.get("RegPop"), "ballots_per_voter": round(ratio, 4),
            "pr_valid": pr_valid, "pr_by_party": pr_by_cat,
            "spoilt": mp[0].get("SpoiltVotes") if mp else None,
            "special_votes": mp[0].get("SpecialVotes") if mp else None}
    rows = []
    for w in d.get("WardResults", []):
        for v in w.get("VotingDistrictResults", []):
            c = dict.fromkeys(PARTIES, 0)
            for p in v.get("PartyBallotResults", []):
                c[_cat(p["Name"])] += p.get("TotalValidVotes", 0) or 0
            tot = sum(c.values())
            row = {"vd": int(v["VotingDistrictID"]), "ward": int(w["WardID"]), "muni": muni,
                   "counted": tot > 0, "report_date": v.get("ReportDate"), "all_ballot_valid": tot}
            row.update({k: int(round(c[k] / ratio)) for k in PARTIES})
            rows.append(row)
    return meta, rows


def update(munis=None) -> pd.DataFrame:
    s = requests.Session()
    metas, rows = [], []
    for m in munis or MUNI_IDS:
        meta, r = parse(m, fetch(m, s))
        metas.append(meta)
        rows += r
    df = pd.DataFrame(rows)
    OBS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OBS / "iec_live_vd.csv", index=False)
    (OBS / "iec_live_munis.json").write_text(json.dumps(metas, indent=2))
    return df, metas


if __name__ == "__main__":
    df, metas = update()
    for m in metas:
        print(f"{m['muni']:6s} {m['vds_complete']}/{m['vd_expected']} VDs ({m['pct_vds_complete']}%)  "
              f"ballots/voter {m['ballots_per_voter']}  report {m['report_date']}")
    c = df[df.counted]
    print(f"counted VDs: {len(c)} of {len(df)}")
    tot = c[PARTIES].sum()
    print((100 * tot / tot.sum()).round(1).to_string())
