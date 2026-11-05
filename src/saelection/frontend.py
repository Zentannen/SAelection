"""Export the projection in the data contract of the ward-map web app
(branch claude/eager-mayer-vzaen0: public/data/{wards.geojson,predictions.json,results.json}).

predictions.json : {generated_at, wards: {ward_id: {shares: {party: % of valid PR votes}, confidence}}}
                   shares = projected FINAL shares (counted + still to come); confidence = probability
                   the projected leader finishes first.
results.json     : {updated_at, source, wards: {ward_id: {status, registered, votes}}}
                   votes = PR-equivalent votes counted so far (IEC dashboard feed).
wards.geojson    : 2026 Gauteng wards, properties.ward_id/name/municipality/province.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .data import MUNI_NAMES, PARTIES, PROCESSED, ROOT

OUT = ROOT / "output" / "frontend"
CODE = {"DA": "DA", "ANC": "ANC", "ACTIONSA": "ActionSA", "MK": "MK", "EFF": "EFF", "OTHER": "Other"}


def export(live_dir: Path = ROOT / "output" / "live", out_dir: Path = OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    wards = json.loads((live_dir / "wards.json").read_text())
    meta = json.loads((live_dir / "meta.json").read_text())
    preds = {}
    for w in wards:
        lead = w["projected_leader"]
        preds[str(w["id"])] = {
            "shares": {CODE[p]: round(100 * x["share"]["mean"], 1) for p, x in w["parties"].items()},
            "confidence": round(w["parties"][lead]["p_first"], 3),
        }
    (out_dir / "predictions.json").write_text(json.dumps(
        {"generated_at": meta["generated_at"], "model": "SAelection Bayesian spatial VD model", "wards": preds}))

    res = {}
    live = ROOT / "data" / "observed_2026" / "iec_live_vd.csv"
    reg = {str(w["id"]): w["registered"] for w in wards}
    if live.exists():
        L = pd.read_csv(live)
        g = L.groupby("ward")
        for ward, t in g:
            n, c = len(t), int(t.counted.sum())
            status = "not_started" if c == 0 else ("final" if c == n else "counting")
            votes = {CODE[p]: int(t.loc[t.counted, p].sum()) for p in PARTIES}
            res[str(ward)] = {"status": status, "registered": int(reg.get(str(ward), 0)), "votes": votes,
                              "vds_counted": c, "vds_total": n}
    updated = max((m["report_date"] for m in (meta.get("iec_feed") or [])), default=meta["generated_at"])
    (out_dir / "results.json").write_text(json.dumps(
        {"updated_at": updated, "source": "IEC results dashboard (PR-equivalent: VD all-ballot votes / ballots per voter)",
         "wards": res}))

    gw = gpd.read_file(PROCESSED / "wards_2026_gauteng.gpkg")
    gw["ward_id"] = gw["ward"].astype(str)
    gw["name"] = "Ward " + gw["ward_no"].astype(int).astype(str)
    gw["municipality"] = gw["muni"].map(MUNI_NAMES).fillna(gw["muni"])
    gw["province"] = "Gauteng"
    gw[["ward_id", "name", "municipality", "province", "geometry"]].to_file(
        out_dir / "wards.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    return out_dir


if __name__ == "__main__":
    print(export())
