"""Download the IEC 2026 LGE Gauteng per-VD results and keep a timestamped snapshot.

Source: results.elections.org.za -> Downloads -> ME-Results -> "Downloadable detailed results",
2026 Local Government Elections, Gauteng, all municipalities (one CSV, every ballot type).
The file only lists VDs that have reported. A new snapshot is saved only when the IEC's
DateGenerated stamp changes, as data/observed_2026/IEC_LGE2026_GP_<YYYYMMDD_HHMM>.csv.

    python scripts/fetch_iec_2026.py            # once
    python scripts/fetch_iec_2026.py --every 600  # poll every 10 minutes
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

URL = "https://results.elections.org.za/home/LGEPublicReports/1887/Downloadable%20Party%20Results/GP.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "observed_2026"


def fetch_once() -> Path | None:
    r = requests.get(URL, timeout=300, headers={"User-Agent": "saelection research"})
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig", usecols=["VotingDistrict", "BallotType", "DateGenerated"])
    stamp = datetime.strptime(df["DateGenerated"].iloc[0], "%m/%d/%Y %I:%M:%S %p")
    f = OUT / f"IEC_LGE2026_GP_{stamp:%Y%m%d_%H%M}.csv"
    n_vd = df.loc[df.BallotType == "PR", "VotingDistrict"].nunique()
    if f.exists():
        print(f"{datetime.now():%H:%M:%S} no change (IEC file generated {stamp}, {n_vd} VDs)", flush=True)
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    print(f"{datetime.now():%H:%M:%S} NEW snapshot {f.name}: {n_vd} VDs reported", flush=True)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=0, help="poll interval in seconds (0 = once)")
    a = ap.parse_args()
    while True:
        try:
            fetch_once()
        except Exception as e:  # network hiccup: report and keep polling
            print(f"{datetime.now():%H:%M:%S} fetch failed: {e}", flush=True)
        if not a.every:
            break
        time.sleep(a.every)
