"""Export the live projection (output/live/) to the ward-map web app's data files.

Writes four files in the app's format (see the web app section of the README):
  wards.geojson     2026 ward boundaries with ward_id, name, municipality, district, province
  predictions.json  projected FINAL PR shares per ward (counted + projected remainder), for every
                    ward including the ones with nothing counted yet; plus the model's own
                    province ("overall") and district figures. Extra fields the app does not need
                    yet: per-party 90% range and P(first), % of VDs counted, projected leader.
  results.json      the actual PR count per ward so far, with a timeline over IEC snapshots
  history.json      2016 LGE, 2019 PE, 2021 LGE, 2024 PE PR shares mapped onto 2026 wards

Party codes follow the app (src/parties.ts): DA, ANC, ActionSA, MK, EFF and Other. The model
projects these six categories, so the count is reported in the same six for a like-for-like
comparison (FF+, PA, IFP and all other parties are in Other).

    python scripts/run_live.py && python scripts/export_webapp.py [--out public/data]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saelection.data import CATS, MUNI_NAMES, PARTIES, PROCESSED, load_history, load_results_file  # noqa: E402

LIVE = ROOT / "output" / "live"
OBS = ROOT / "data" / "observed_2026"
SAST = timezone(timedelta(hours=2))
CODE = {"DA": "DA", "ANC": "ANC", "ACTIONSA": "ActionSA", "MK": "MK", "EFF": "EFF", "OTHER": "Other"}
# metros are their own district; local municipalities belong to Sedibeng or West Rand
DISTRICT = {"JHB": "City of Johannesburg", "TSH": "City of Tshwane", "EKU": "Ekurhuleni",
            "GT421": "Sedibeng", "GT422": "Sedibeng", "GT423": "Sedibeng",
            "GT481": "West Rand", "GT484": "West Rand", "GT485": "West Rand"}


def pct(x):
    return round(100 * float(x), 2)


def shares_from_counts(row):
    tot = sum(row[p] for p in PARTIES)
    return {CODE[p]: (round(100 * row[p] / tot, 2) if tot else 0.0) for p in PARTIES}


def unit_prediction(rec):
    parties = rec["parties"]
    out = {"shares": {CODE[p]: pct(x["share"]["mean"]) for p, x in parties.items()},
           "confidence": round(max(x["p_first"] for x in parties.values()), 3),
           "leader": CODE[rec["projected_leader"]],
           "p_first": {CODE[p]: x["p_first"] for p, x in parties.items()},
           "range90": {CODE[p]: [pct(x["share"]["p05"]), pct(x["share"]["p95"])] for p, x in parties.items()},
           "pct_vds_counted": rec["pct_vds_counted"],
           "turnout": pct(rec["turnout_valid"]["mean"])}
    return out


def district_predictions():
    """District shares from the municipality draws (metros = their municipality)."""
    md = pd.read_csv(LIVE / "municipality_draws.csv")
    md["district"] = md["muni"].map(DISTRICT)
    g = md.groupby(["district", "draw"])[PARTIES].sum()
    sh = g.div(g.sum(axis=1), axis=0)
    out = {}
    for dist, s in sh.groupby(level=0):
        out[dist] = {"shares": {CODE[p]: pct(s[p].mean()) for p in PARTIES},
                     "range90": {CODE[p]: [pct(s[p].quantile(.05)), pct(s[p].quantile(.95))] for p in PARTIES}}
    return out


def ward_counts_by_snapshot():
    """{snapshot time: DataFrame(ward -> counts)} from every IEC snapshot (cumulative files)."""
    snaps = {}
    for f in sorted(OBS.glob("IEC_LGE2026_GP_*.csv")):
        stamp = pd.read_csv(f, encoding="utf-8-sig", usecols=["DateGenerated"], nrows=1)["DateGenerated"].iloc[0]
        t = datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=SAST)
        vd = load_results_file(f, "LGE", "LGE2026")
        snaps[t] = vd.groupby("ward")[CATS].sum()
    return dict(sorted(snaps.items()))


def history_by_ward(cw):
    """Past PR shares per 2026 ward: VD results allocated by the crosswalk's registered weights."""
    hist = load_history()
    elections = [("LGE2016", "2016 LGE"), ("PE2019", "2019 national"), ("LGE2021", "2021 LGE"), ("PE2024", "2024 national")]
    out = {}
    for key, _ in elections:
        h = hist[hist.election == key].set_index("vd")[PARTIES]
        stood = h.sum() > 0
        m = cw.merge(h, left_on="vd", right_index=True, how="inner")
        agg = m[PARTIES].mul(m["ward_weight"], axis=0).groupby(m["ward"]).sum()
        for w, row in agg.iterrows():
            tot = row.sum()
            if tot <= 0:
                continue
            out.setdefault(str(int(w)), {})[key] = {CODE[p]: round(100 * row[p] / tot, 2) for p in PARTIES if stood[p]}
    return [{"id": k, "label": lab} for k, lab in elections], out


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((LIVE / "meta.json").read_text())
    wards = json.loads((LIVE / "wards.json").read_text())
    prov = json.loads((LIVE / "province.json").read_text())
    cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")

    # wards.geojson
    geo = json.loads((LIVE / "geo" / "wards_2026.geojson").read_text())
    for f in geo["features"]:
        p = f["properties"]
        f["properties"] = {"ward_id": str(p["id"]), "name": f"Ward {p['ward_no']}",
                           "municipality": MUNI_NAMES.get(p["muni"], p["muni"]),
                           "district": DISTRICT.get(p["muni"]), "province": "Gauteng"}
    (out_dir / "wards.geojson").write_text(json.dumps(geo, separators=(",", ":")))

    # predictions.json
    preds = {"generated_at": meta["generated_at"], "model": "Bayesian spatial VD model (saelection.live)",
             "pct_vds_counted": meta["pct_vds_counted"],
             "wards": {str(w["id"]): unit_prediction(w) for w in wards},
             "overall": unit_prediction(prov), "districts": district_predictions()}
    (out_dir / "predictions.json").write_text(json.dumps(preds, separators=(",", ":")))

    # results.json
    snaps = ward_counts_by_snapshot()
    times = list(snaps)
    res = {"updated_at": times[-1].isoformat() if times else meta["generated_at"],
           "source": "IEC, results.elections.org.za (PR ballot)", "wards": {}}
    for w in wards:
        wid = int(w["id"])
        f = w["pct_vds_counted"]
        status = "not_started" if f <= 0 else ("final" if f >= 100 else "counting")
        tl = []
        for t in times:
            if wid in snaps[t].index:
                row = snaps[t].loc[wid]
                tl.append({"at": t.isoformat(), "votes": {CODE[p]: int(row[p]) for p in PARTIES}})
        votes = tl[-1]["votes"] if tl else {CODE[p]: 0 for p in PARTIES}
        res["wards"][str(wid)] = {"status": status, "registered": int(w["registered"]), "votes": votes,
                                  **({"timeline": tl} if tl else {})}
    (out_dir / "results.json").write_text(json.dumps(res, separators=(",", ":")))

    # history.json
    elections, hw = history_by_ward(cw)
    (out_dir / "history.json").write_text(json.dumps({"elections": elections, "wards": hw}, separators=(",", ":")))

    n_unc = sum(1 for w in wards if w["pct_vds_counted"] <= 0)
    print(f"wrote {out_dir}: {len(geo['features'])} wards ({n_unc} with nothing counted yet, all projected), "
          f"{len(times)} IEC snapshots in the timeline, history for {len(hw)} wards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "webapp"))
    main(Path(ap.parse_args().out))
