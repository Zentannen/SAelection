"""Official 2026 VD -> ward list for Gauteng from the IEC results site (every VD, counted or not).

The geometry-based crosswalk (data/processed/vd_ward_crosswalk_2026.csv) sometimes splits a VD
across wards or puts it in a different ward than the IEC, which reports every VD in exactly one
ward. This walks the IEC "Results Summary" form (2026 LGE -> Gauteng -> municipality -> ward)
and reads the voting-district list of each ward. Output: data/processed/vd_ward_iec_2026.csv
(vd, ward, muni). Takes about 5 minutes (one form post-back per ward).
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://results.elections.org.za/home/Downloads/ME-Results/"
P = "ctl00$MainContent$"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "vd_ward_iec_2026.csv"
MUNIS = {"3002~1": "EKU", "3302~2": "GT421", "3303~2": "GT422", "3304~2": "GT423", "3040~2": "GT481",
         "3039~2": "GT484", "3332~2": "GT485", "3003~1": "JHB", "3004~1": "TSH"}

s = requests.Session()
s.headers["User-Agent"] = "saelection research"


def hidden(t):
    return {m.group(1): html.unescape(m.group(2)) for m in
            re.finditer(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', t)}


def options(t, name):
    m = re.search(r'<select name="' + re.escape(name) + r'"[^>]*>(.*?)</select>', t, re.S)
    return [] if not m else re.findall(r'<option(?: selected="selected")? value="([^"]*)">', m.group(1))


def post(t, target, state):
    f = hidden(t)
    f.update({"__EVENTTARGET": target, "__EVENTARGUMENT": ""})
    f.update(state)
    for attempt in range(4):
        try:
            r = s.post(URL, data=f, timeout=120)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"post-back failed: {target}")


def main():
    rows = []
    for code, muni in MUNIS.items():
        state = {P + "ddlElections": "1887"}
        t = s.get(URL, timeout=60).text
        t = post(t, P + "ddlElections", state)
        state[P + "radReports"] = "22"
        t = post(t, P + "radReports$0", state)
        state[P + "ddlResultsSummaryProvinces"] = "3"
        t = post(t, P + "ddlResultsSummaryProvinces", state)
        state[P + "ddlResultsSummaryMunicipalities"] = code
        t = post(t, P + "ddlResultsSummaryMunicipalities", state)
        wards = [w for w in options(t, P + "ddlResultsSummaryWards") if w not in ("0", "-1", "-2")]
        for w in wards:
            state[P + "ddlResultsSummaryWards"] = w
            tw = post(t, P + "ddlResultsSummaryWards", state)
            for vd in options(tw, P + "ddlResultsSummaryVDs"):
                if vd not in ("0", "-1", "-2"):
                    rows.append({"vd": int(vd), "ward": int(w), "muni": muni})
        print(f"{muni}: {len(wards)} wards, {sum(r['muni'] == muni for r in rows)} VDs", flush=True)
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {df.vd.nunique()} VDs in {df.ward.nunique()} wards; VDs listed in >1 ward: {(df.groupby('vd').ward.nunique() > 1).sum()}")


if __name__ == "__main__":
    main()
