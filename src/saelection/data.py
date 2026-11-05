"""Load IEC voting-district (VD) results for Gauteng into a common tidy format.

Every election becomes one row per VD with counts for the modelled categories:

    DA, ANC, ACTIONSA, MK, EFF, OTHER  (valid PR / party votes)
    ABSTAIN                            (registered - valid; includes spoilt ballots)

so each row is a full multinomial over *registered* voters, which lets one model
cover both party choice and turnout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PARTIES = ["DA", "ANC", "ACTIONSA", "MK", "EFF", "OTHER"]
CATS = PARTIES + ["ABSTAIN"]

PARTY_MAP = {
    "DEMOCRATIC ALLIANCE": "DA",
    "AFRICAN NATIONAL CONGRESS": "ANC",
    "ACTIONSA": "ACTIONSA",
    "UMKHONTO WESIZWE": "MK",           # 2024 PE ballot name
    "UMKHONTO WESIZWE PARTY": "MK",     # 2026 LGE ballot name
    "ECONOMIC FREEDOM FIGHTERS": "EFF",
}

MUNI_NAMES = {
    "JHB": "City of Johannesburg",
    "TSH": "City of Tshwane",
    "EKU": "Ekurhuleni",
    "GT421": "Emfuleni",
    "GT422": "Midvaal",
    "GT423": "Lesedi",
    "GT481": "Mogale City",
    "GT484": "Merafong City",
    "GT485": "Rand West City",
}

ELECTIONS = {
    # key: (file, kind)
    "LGE2016": ("2016 LGE GP.csv", "LGE"),
    "PE2019": ("2019 PE.csv", "PE"),
    "LGE2021": ("2021 LGE GP.csv", "LGE"),
    "PE2024": ("2024 PE.csv", "PE"),
}


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def load_election(key: str) -> pd.DataFrame:
    fname, kind = ELECTIONS[key]
    return load_results_file(RAW / fname, kind, key)


def load_results_file(path, kind: str = "LGE", key: str = "LGE2026") -> pd.DataFrame:
    """Parse an IEC results CSV (results.elections.org.za download format) into one row per VD.

    kind="LGE": columns Province, Municipality, Ward, VotingDistrict, VotingStationName,
    RegisteredVoters, BallotType, SpoiltVotes, PartyName, TotalValidVotes (PR ballot rows used).
    kind="PE":  columns Province, Municipality, VD_Number, ..., Registered_Population,
    Spoilt_Votes, sPartyName, Party_Votes.
    """
    d = _read_csv(Path(path))
    d.columns = [c.strip().replace(" ", "_") for c in d.columns]
    d = d[d["Province"] == "Gauteng"].copy()
    if kind == "LGE":
        d = d[d["BallotType"] == "PR"]
        d = d.rename(columns={"VotingDistrict": "vd", "RegisteredVoters": "registered",
                              "SpoiltVotes": "spoilt", "PartyName": "party",
                              "TotalValidVotes": "votes", "Ward": "ward"})
        d["ward"] = d["ward"].str.extract(r"(\d+)")[0].astype(int)
    else:
        d = d.rename(columns={"VD_Number": "vd", "Registered_Population": "registered",
                              "Spoilt_Votes": "spoilt", "sPartyName": "party",
                              "Party_Votes": "votes"})
        d["ward"] = -1
    for c in ("vd", "registered", "spoilt", "votes"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "").str.strip(),
                             errors="coerce").fillna(0).astype(np.int64)
    d = d[d["vd"] > 0]
    d["muni"] = d["Municipality"].str.split(" - ").str[0].str.strip()
    d["cat"] = d["party"].str.strip().str.upper().map(PARTY_MAP).fillna("OTHER")

    meta = d.groupby("vd").agg(muni=("muni", "first"), ward=("ward", "first"),
                               registered=("registered", "first"), spoilt=("spoilt", "first"))
    counts = d.pivot_table(index="vd", columns="cat", values="votes", aggfunc="sum", fill_value=0)
    counts = counts.reindex(columns=PARTIES, fill_value=0)
    out = meta.join(counts)
    out["valid"] = out[PARTIES].sum(axis=1)
    # 2024 allowed out-of-district (s24A) voting, so a VD can record more ballots than
    # registered voters; treat the effective electorate as at least the ballots cast.
    out["electorate"] = np.maximum(out["registered"], out["valid"] + out["spoilt"])
    out["ABSTAIN"] = out["electorate"] - out["valid"]
    out["election"] = key
    return out.reset_index()



def build_history() -> pd.DataFrame:
    frames = [load_election(k) for k in ELECTIONS]
    hist = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hist.to_csv(PROCESSED / "vd_history_gauteng.csv", index=False)
    return hist


def load_history() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "vd_history_gauteng.csv")


if __name__ == "__main__":
    h = build_history()
    summ = h.groupby("election")[CATS + ["registered", "valid"]].sum()
    summ["turnout"] = (summ["valid"] + h.groupby("election")["spoilt"].sum()) / summ["registered"]
    print(h.groupby("election").vd.nunique())
    print((summ[PARTIES].div(summ["valid"], axis=0) * 100).round(1))
    print(summ[["registered", "valid", "turnout"]])
