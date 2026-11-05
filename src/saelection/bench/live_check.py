"""Live-data check: score projection methods on the 2026 count itself.

The benchmark in harness.py replays past elections. This module instead uses the 2026 VDs
that have already reported: some of them are hidden, the method projects them from the rest,
and the projection is scored against their actual 2026 result. This is the closest test of how
well a method predicts the areas that are still uncounted tonight.

Hold-out tasks (only already-counted VDs are ever scored):
  time            counted = VDs in the first snapshot; predict the VDs that reported later
  group:<type>:<muni>  hide every counted VD of one area type in one municipality (e.g. all
                  counted Tshwane townships) and predict them from all other counted VDs. Tests
                  extrapolation to area types / places with no counted look-alikes.
  muni:<muni>     hide every counted VD of one municipality
  random_s<k>     hide a random 20% of counted VDs (baseline difficulty)

Area types for the grouping are fixed, simple and only used to define hold-out groups (methods
may cluster however they like): by PE2024 valid-vote shares, DA>=50% 'suburb', DA 25-50% 'mixed',
ANC+EFF+MK>=70% 'township', else 'other'.

The case is built like the live run: design LGE2026 <- LGE2021 + PE2024 on 2026 geometry, so
case.prev_lge = LGE2021 counts and case.prev_pe = PE2024 counts. case.name = 'LIVE2026'.
Methods use the same interface as harness.py; case.Y is the 2026 count for VDs that have
reported and 0 for VDs that have not (never read Y outside `counted`).

    PYTHONPATH=src python -m saelection.bench.live_check --methods tally,uniform_swing_muni --tag baselines
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..data import CATS, PARTIES, PROCESSED, ROOT, load_history, load_results_file
from ..features import K, build_design
from .harness import OUT, REAL2026, Case, _counts, _fill, _shares, resolve


def load_live_case(snapshot: str | None = None):
    """Case for 2026 plus per-VD arrival snapshot index (-1 = not reported yet)."""
    d = build_design(None, "LGE2021", "PE2024", 2026, name="LIVE2026")
    hist = load_history()
    files = sorted(REAL2026.glob("IEC_LGE2026_GP_*.csv"))
    if snapshot:
        files = [f for f in files if f.name <= snapshot]
    pos = {v: i for i, v in enumerate(d.vd)}
    iec_ward = {}
    Y = np.zeros((d.n, len(CATS)))
    reg_obs = np.full(d.n, np.nan)
    arrival = np.full(d.n, -1)
    for s, f in enumerate(files):
        t = load_results_file(f, "LGE", "LGE2026")
        t = t[t.valid > 0]
        for r in t.itertuples(index=False):
            i = pos.get(int(r.vd))
            if i is None:
                continue
            Y[i] = [getattr(r, c) for c in CATS]
            reg_obs[i] = r.registered
            iec_ward[i] = int(r.ward)
            if arrival[i] < 0:
                arrival[i] = s
    cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")
    base = cw.groupby("vd")["vd_regpop"].first().reindex(d.vd).to_numpy(float)
    rep = arrival >= 0
    growth = np.nansum(reg_obs[rep]) / base[rep].sum() if rep.any() else 1.0
    registered = np.where(rep, np.maximum(reg_obs, Y.sum(1)), base * growth)
    ward = vd_wards(d.vd)
    for i, w in iec_ward.items():          # a counted VD's ward as the IEC reports it
        ward[i] = w
    nodes = pd.read_csv(PROCESSED / "vd_nodes_2026.csv").set_index("vd").reindex(d.vd)
    lge, p1 = _counts(hist, "LGE2021", d.vd)
    pe, p2 = _counts(hist, "PE2024", d.vd)
    case = Case(name="LIVE2026", vd=d.vd, muni=d.muni, muni_codes=d.muni_codes, ward=ward,
                lon=nodes["lon"].to_numpy(), lat=nodes["lat"].to_numpy(), edges=d.edges, Y=Y,
                registered=registered, prev_lge=_fill(lge, p1, d.edges), prev_pe=_fill(pe, p2, d.edges),
                hist_present=p1 & p2, X1=d.X1, X2=d.X2, design=d)
    return case, arrival, [f.name for f in files]


def vd_wards(vds: np.ndarray) -> np.ndarray:
    """One ward per 2026 VD: the IEC's official list (scripts/fetch_iec_vd_wards.py) where
    available, else the geometry crosswalk's main ward. The IEC reports each VD in one ward."""
    cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")
    w = cw.sort_values("ward_weight").groupby("vd")["ward"].last()
    f = PROCESSED / "vd_ward_iec_2026.csv"
    if f.exists():
        iec = pd.read_csv(f).drop_duplicates("vd").set_index("vd")["ward"]
        w = iec.combine_first(w)
    return w.reindex(vds).fillna(-1).astype(np.int64).to_numpy().copy()


def area_type(case: Case) -> np.ndarray:
    s = case.prev_pe[:, :K] / np.maximum(case.prev_pe[:, :K].sum(1, keepdims=True), 1)
    da, town = s[:, 0], s[:, 1] + s[:, 3] + s[:, 4]
    return np.where(da >= .5, "suburb", np.where(da >= .25, "mixed", np.where(town >= .7, "township", "other")))


def tasks(case: Case, arrival: np.ndarray, min_group=12, n_random=3):
    rep = arrival >= 0
    out = {}
    if arrival.max() > 0:
        out["time"] = (arrival == 0, rep & (arrival > 0))
    at = area_type(case)
    for t in np.unique(at):
        for m, code in enumerate(case.muni_codes):
            g = rep & (at == t) & (case.muni == m)
            if g.sum() >= min_group and (rep & ~g).sum() >= 50:
                out[f"group:{t}:{code}"] = (rep & ~g, g)
    for m, code in enumerate(case.muni_codes):
        g = rep & (case.muni == m)
        if g.sum() >= min_group:
            out[f"muni:{code}"] = (rep & ~g, g)
    idx = np.where(rep)[0]
    for s in range(n_random):
        hide = np.random.default_rng(500 + s).choice(idx, size=int(0.2 * len(idx)), replace=False)
        g = np.zeros(case.n, bool)
        g[hide] = True
        out[f"random_s{s}"] = (rep & ~g, g)
    return out


def score_holdout(case: Case, counted: np.ndarray, held: np.ndarray, proj) -> dict:
    Y, V = case.Y[held], proj.vd_mean[held].astype(float)
    ta, pa = _shares(Y.sum(0)), _shares(V.sum(0))
    r = {"n_held": int(held.sum()), "n_counted": int(counted.sum()),
         "group_mae_pp": 100 * np.abs(pa - ta).mean(), "group_max_pp": 100 * np.abs(pa - ta).max(),
         "vd_mae_pp": 100 * np.abs(_shares(V) - _shares(Y)).mean(),
         "vd_leader_acc": float((_shares(V).argmax(1) == _shares(Y).argmax(1)).mean()),
         "vd_turnout_mae_pp": 100 * np.abs(V[:, :K].sum(1) / np.maximum(V.sum(1), 1) - Y[:, :K].sum(1) / np.maximum(Y.sum(1), 1)).mean()}
    for k, p in enumerate(PARTIES):
        r[f"err_{p}"] = 100 * (pa[k] - ta[k])
    if proj.draws is not None:
        sp = _shares(proj.draws[:, held].astype(float).sum(1))
        lo, hi = np.percentile(sp, [5, 95], axis=0)
        r["group_cov90"] = float(((lo <= ta) & (ta <= hi)).mean())
        r["group_width90_pp"] = float(100 * (hi - lo).mean())
        vs = _shares(proj.draws[:, held].astype(float))                     # (S, h, K)
        lo, hi = np.percentile(vs, [5, 95], axis=0)
        ty = _shares(Y)
        r["vd_cov90"] = float(((lo <= ty) & (ty <= hi)).mean())
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


def run(methods, tag=None, snapshot=None, only=None):
    case, arrival, files = load_live_case(snapshot)
    print(f"LIVE2026 from {files[-1]} ({len(files)} snapshots): {int((arrival >= 0).sum())} counted VDs in the model's VD set", flush=True)
    rows = []
    for tname, (counted, held) in tasks(case, arrival).items():
        if only and not any(tname.startswith(o) for o in only):
            continue
        for m in methods:
            label, fn = resolve(m)
            t = time.time()
            proj = fn(case, counted, np.random.default_rng(0))
            r = score_holdout(case, counted, held, proj)
            r.update(task=tname, method=label, seconds=round(time.time() - t, 1), snapshot=files[-1])
            rows.append(r)
            print(f"{tname:26s} {label:28s} held={r['n_held']:4d} group_mae={r['group_mae_pp']:5.2f} "
                  f"vd_mae={r['vd_mae_pp']:5.2f} lead_acc={r['vd_leader_acc']:.3f} "
                  f"cov90={r.get('group_cov90', float('nan')):.2f}", flush=True)
    df = pd.DataFrame(rows)
    if tag:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"live_{tag}.csv", index=False)
    return df


def summarise(df):
    df = df.copy()
    df["kind"] = df.task.str.split(":").str[0].str.replace(r"_s\d+$", "", regex=True)
    cols = [c for c in ["group_mae_pp", "group_max_pp", "vd_mae_pp", "vd_leader_acc", "vd_turnout_mae_pp",
                        "group_cov90", "vd_cov90", "seconds"] if c in df]
    return df.groupby(["method", "kind"])[cols].mean().round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="tally,uniform_swing_muni")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--snapshot", default=None, help="use snapshots up to this file name")
    ap.add_argument("--only", default="", help="comma list of task prefixes, e.g. group,time")
    a = ap.parse_args()
    df = run(a.methods.split(","), a.tag, a.snapshot, [o for o in a.only.split(",") if o])
    print(summarise(df).to_string())
