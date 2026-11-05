"""Build public/data/history.json: past vote share per 2026 Gauteng ward.

Past voting-district (VD) results (2016/2021 local PR ballot, 2019/2024 Gauteng provincial ballot) are moved onto the 2026 wards with the VD->ward crosswalk
(a VD split across wards is shared by its registered-voter weight). VDs that no longer exist
in 2026 are left out, so coverage per ward is reported.

Usage: python3 scripts/build_gauteng_history.py <vd_history_gauteng.csv> <vd_ward_crosswalk_2026.csv> public/data/history.json
Both CSVs are on the `main` branch in data/processed/.
"""
import csv
import json
import sys
from collections import defaultdict

ELECTIONS = [("LGE2016", "2016 local"), ("PE2019", "2019 provincial"), ("LGE2021", "2021 local"), ("PE2024", "2024 provincial")]
PARTIES = {"DA": "DA", "ANC": "ANC", "ACTIONSA": "ActionSA", "MK": "MK", "EFF": "EFF", "OTHER": "Other"}


def main(hist_path, xwalk_path, out_path):
    weights = defaultdict(list)
    for r in csv.DictReader(open(xwalk_path)):
        weights[r["vd"]].append((r["ward"], float(r["ward_weight"])))

    votes = defaultdict(lambda: defaultdict(float))  # (ward, election) -> party -> votes
    seen = defaultdict(int)
    missed = defaultdict(int)
    for r in csv.DictReader(open(hist_path)):
        e = r["election"]
        if e not in dict(ELECTIONS):
            continue
        if r["vd"] not in weights:
            missed[e] += 1
            continue
        seen[e] += 1
        for ward, w in weights[r["vd"]]:
            for col, party in PARTIES.items():
                votes[(ward, e)][party] += w * float(r[col] or 0)

    wards = defaultdict(dict)
    for (ward, e), v in votes.items():
        total = sum(v.values())
        if total > 0:
            wards[ward][e] = {p: round(100 * n / total, 2) for p, n in v.items() if n > 0}

    out = {"elections": [{"id": e, "label": label} for e, label in ELECTIONS], "wards": wards}
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    for e, _ in ELECTIONS:
        print(f"{e}: {seen[e]} VDs mapped, {missed[e]} VDs not in the 2026 crosswalk")
    print(f"wrote {len(wards)} wards to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
