"""Build the app's public/data/ files for the 2026 Gauteng local elections from the
projection model's output on the `main` branch.

    python3 scripts/build_gauteng_data.py <main-checkout> public/data

<main-checkout> is a checkout (or extract) of `main` holding:
    output/frontend/{wards.geojson,predictions.json,results.json}   (src/saelection/frontend.py)
    output/live/{province.json,municipalities.json}                  (scripts/run_live.py)
    data/processed/{vd_history_gauteng.csv,vd_ward_crosswalk_2026.csv}

Everything is on the model's ballot (the PR ballot) and in the model's six groups:
DA, ANC, ActionSA, MK, EFF and Other. Forecast, count and past results therefore compare
like for like.

Past results: 2016 and 2021 local (PR ballot), 2019 and 2024 Gauteng provincial ballot.
Voting-district (VD) results are moved onto the 2026 wards with the VD->ward crosswalk;
a VD split across wards is shared out by its registered-voter weight.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Municipality -> district. The three metros are their own district.
DISTRICT = {
    "City of Johannesburg": "City of Johannesburg",
    "City of Tshwane": "City of Tshwane",
    "Ekurhuleni": "Ekurhuleni",
    "Emfuleni": "Sedibeng",
    "Midvaal": "Sedibeng",
    "Lesedi": "Sedibeng",
    "Mogale City": "West Rand",
    "Merafong City": "West Rand",
    "Rand West City": "West Rand",
}
CODE = {"DA": "DA", "ANC": "ANC", "ACTIONSA": "ActionSA", "MK": "MK", "EFF": "EFF", "OTHER": "Other"}
ELECTIONS = [
    ("LGE2016", "2016 local"),
    ("PE2019", "2019 provincial"),
    ("LGE2021", "2021 local"),
    ("PE2024", "2024 provincial"),
]


def load(path: Path):
    return json.loads(path.read_text())


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, separators=(",", ":")))


def shares_from_votes(votes: dict) -> dict:
    total = sum(votes.values())
    return {CODE[p]: round(100 * v / total, 1) for p, v in votes.items()} if total else {}


def wards(src: Path, out: Path):
    g = load(src / "output/frontend/wards.geojson")
    g.pop("crs", None)
    for f in g["features"]:
        p = f["properties"]
        muni = p["municipality"]
        p["district"] = DISTRICT[muni]
        p["name"] = f"{muni} {p['name']}"  # "Ward 94" -> "City of Johannesburg Ward 94"
    dump(g, out / "wards.geojson")
    print(f"wards.geojson: {len(g['features'])} wards")


def predictions(src: Path, out: Path):
    p = load(src / "output/frontend/predictions.json")
    province = load(src / "output/live/province.json")
    munis = load(src / "output/live/municipalities.json")

    # The model's own Gauteng and district figures (expected final votes), so they
    # include its turnout projection rather than a plain average of ward shares.
    p["overall"] = {"shares": shares_from_votes({k: v["votes"]["mean"] for k, v in province["parties"].items()})}
    by_district = defaultdict(lambda: defaultdict(float))
    for m in munis:
        for k, v in m["parties"].items():
            by_district[DISTRICT[m["name"]]][k] += v["votes"]["mean"]
    p["districts"] = {d: {"shares": shares_from_votes(v)} for d, v in sorted(by_district.items())}
    p["pct_vds_counted"] = province.get("pct_vds_counted")
    dump(p, out / "predictions.json")
    print(f"predictions.json: {len(p['wards'])} wards, Gauteng {p['overall']['shares']}")


def results(src: Path, out: Path):
    r = load(src / "output/frontend/results.json")
    dump(r, out / "results.json")
    print(f"results.json: {len(r['wards'])} wards, updated {r.get('updated_at')}")


def history(src: Path, out: Path):
    weights = defaultdict(list)
    for r in csv.DictReader(open(src / "data/processed/vd_ward_crosswalk_2026.csv")):
        weights[r["vd"]].append((r["ward"], float(r["ward_weight"])))

    ids = dict(ELECTIONS)
    votes = defaultdict(lambda: defaultdict(float))  # (ward, election) -> party -> votes
    mapped, missed = defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(src / "data/processed/vd_history_gauteng.csv")):
        e = r["election"]
        if e not in ids:
            continue
        if r["vd"] not in weights:
            missed[e] += 1
            continue
        mapped[e] += 1
        for ward, w in weights[r["vd"]]:
            for col, party in CODE.items():
                votes[(ward, e)][party] += w * float(r[col] or 0)

    wards = defaultdict(dict)
    for (ward, e), v in votes.items():
        total = sum(v.values())
        if total > 0:
            # A party with no votes didn't stand (ActionSA before 2021, MK before 2024): leave it out.
            wards[ward][e] = {p: round(100 * n / total, 2) for p, n in v.items() if n > 0}

    dump({"elections": [{"id": e, "label": label} for e, label in ELECTIONS], "wards": wards}, out / "history.json")
    for e, label in ELECTIONS:
        print(f"history.json: {label}: {mapped[e]} VDs mapped, {missed[e]} VDs not in the 2026 crosswalk")
    print(f"history.json: {len(wards)} wards")


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    wards(src, out)
    predictions(src, out)
    results(src, out)
    history(src, out)
