"""Shared count-replay benchmark for live-projection methods.

Every method answers the same question: given the voting districts (VDs) counted so far,
what are the final PR votes in every VD that is still uncounted? The harness replays a past
Gauteng election in several counting orders, calls each method, and scores the projection
against the actual final result. No method sees the target election's uncounted VDs.

Cases (target <- predictors). Neither needs a model trained on the target election:
  LGE2021 <- LGE2016 + PE2019   same election type as 2026 (geometry 2021)
  PE2024  <- LGE2021 + PE2019   most recent election (geometry 2024)

Counting orders:
  random_s{0..}           uniformly random reporting order (several seeds)
  DA_strongholds_first    VDs sorted by DA share in the previous PE (worst case for DA bias)
  ANC_strongholds_first   VDs sorted by ANC share in the previous PE
  small_first             VDs with the fewest registered voters first (small VDs finish first)
  real2026                the VDs that had reported in the 2026 count by the snapshot time,
                          then the rest in random order. The only order observed in reality.

Method interface
----------------
    def my_method(case: Case, counted: np.ndarray[bool], rng) -> Projection

`case.Y[counted]` may be used; `case.Y[~counted]` must not. Return a Projection with
  vd_mean : (n, 7) projected final counts per VD in CATS order (DA, ANC, ACTIONSA, MK, EFF,
            OTHER, ABSTAIN). Rows of counted VDs are overwritten with the actual result.
  draws   : optional (S, n, 7) draws of the same. If given, they are used for intervals and
            p(win); otherwise intervals are reported as missing.

Run:  PYTHONPATH=src python -m saelection.bench.harness --methods tally,uniform_swing_muni
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import CATS, PARTIES, ROOT, load_history
from ..features import K, build_design

OUT = ROOT / "artifacts" / "bench"
REAL2026 = ROOT / "data" / "observed_2026"
LEVELS = (0.05, 0.10, 0.25, 0.50)
CASES = {
    "LGE2021": ("LGE2021", "LGE2016", "PE2019", 2021),
    "PE2024": ("PE2024", "LGE2021", "PE2019", 2024),
}


@dataclass
class Case:
    name: str
    vd: np.ndarray            # (n,) VD numbers
    muni: np.ndarray          # (n,) municipality index
    muni_codes: list
    ward: np.ndarray          # (n,) ward id (int64); -1 if unknown
    lon: np.ndarray
    lat: np.ndarray
    edges: np.ndarray         # (m, 2) contiguity edges (VDs whose polygons touch)
    Y: np.ndarray             # (n, 7) actual final counts of the target election (CATS order)
    registered: np.ndarray    # (n,) electorate of the target election (known before counting)
    prev_lge: np.ndarray      # (n, 7) previous-LGE counts (neighbour-filled where VD is new)
    prev_pe: np.ndarray       # (n, 7) previous-PE counts
    hist_present: np.ndarray  # (n,) VD existed in both predictor elections (not imputed)
    X1: np.ndarray            # (n, 6) previous-LGE alr vs abstain (centred), from features.py
    X2: np.ndarray            # (n, 6) previous-PE alr vs abstain (centred)
    design: object            # the features.Design (for model-based methods)

    @property
    def n(self):
        return len(self.vd)


@dataclass
class Projection:
    vd_mean: np.ndarray
    draws: np.ndarray | None = None
    info: dict | None = None


def _counts(hist, key, vds):
    h = hist[hist.election == key].set_index("vd").reindex(vds)
    return h[CATS].to_numpy(float), h.index.isin(hist.loc[hist.election == key, "vd"])


def _fill(counts, present, edges):
    """Neighbour-average the *shares* of VDs that did not exist, scaled to their own size."""
    from ..features import _neighbour_fill
    tot = np.nan_to_num(counts).sum(1, keepdims=True)
    sh = np.where(present[:, None], np.nan_to_num(counts) / np.maximum(tot, 1), 0.0)
    sh = _neighbour_fill(sh, present, edges)
    size = np.where(present, tot[:, 0], np.nan)
    size = np.where(np.isnan(size), np.nanmedian(size), size)
    return sh * size[:, None]


def load_case(name: str) -> Case:
    target, pl, pp, gy = CASES[name]
    d = build_design(target, pl, pp, gy, name=f"bench {name}")
    hist = load_history()
    nodes = pd.read_csv(ROOT / "data" / "processed" / f"vd_nodes_{gy}.csv").set_index("vd").reindex(d.vd)
    lge, p1 = _counts(hist, pl, d.vd)
    pe, p2 = _counts(hist, pp, d.vd)
    if target.startswith("LGE"):
        ward = hist[hist.election == target].groupby("vd")["ward"].first().reindex(d.vd)
    else:
        import geopandas as gpd
        from ..geo import GEO_FILES
        w = gpd.read_file(GEO_FILES[2024], where="Province='Gauteng'", read_geometry=False)
        ward = w.drop_duplicates("VDNumber").set_index(w.drop_duplicates("VDNumber")["VDNumber"].astype(np.int64))["WardId"].reindex(d.vd)
    ward = pd.to_numeric(ward, errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    return Case(name=name, vd=d.vd, muni=d.muni, muni_codes=d.muni_codes, ward=ward,
                lon=nodes["lon"].to_numpy(), lat=nodes["lat"].to_numpy(), edges=d.edges,
                Y=d.Y, registered=d.registered, prev_lge=_fill(lge, p1, d.edges),
                prev_pe=_fill(pe, p2, d.edges), hist_present=p1 & p2, X1=d.X1, X2=d.X2, design=d)


def real2026_vds() -> np.ndarray:
    """VD numbers counted in the 2026 snapshot(s) in data/observed_2026 (PR ballot, >0 votes)."""
    vds = set()
    for f in sorted(REAL2026.glob("*.csv")):
        if f.name.startswith("progress_"):
            continue
        df = pd.read_csv(f, encoding="utf-8-sig", usecols=["VotingDistrict", "BallotType", "TotalValidVotes"])
        df = df[df.BallotType == "PR"].groupby("VotingDistrict").TotalValidVotes.sum()
        vds |= set(df[df > 0].index.astype(np.int64))
    return np.array(sorted(vds), dtype=np.int64)


def orders(case: Case, n_random=3):
    out = {}
    for s in range(n_random):
        out[f"random_s{s}"] = (np.random.default_rng(100 + s).permutation(case.n), LEVELS)
    pv = case.prev_pe[:, :K].sum(1)
    da = case.prev_pe[:, 0] / np.maximum(pv, 1)
    anc = case.prev_pe[:, 1] / np.maximum(pv, 1)
    out["DA_strongholds_first"] = (np.argsort(-da, kind="stable"), LEVELS)
    out["ANC_strongholds_first"] = (np.argsort(-anc, kind="stable"), LEVELS)
    out["small_first"] = (np.argsort(case.registered, kind="stable"), LEVELS)
    real = real2026_vds()
    if len(real):
        first = np.where(np.isin(case.vd, real))[0]
        rest = np.setdiff1d(np.arange(case.n), first)
        rest = np.random.default_rng(7).permutation(rest)
        frac = len(first) / case.n
        out["real2026"] = (np.concatenate([first, rest]), (round(frac, 4),) + tuple(l for l in LEVELS if l > frac + 0.02))
    return out


# ----------------------------------------------------------------------------- scoring
def _shares(v):
    """valid-vote shares over the K parties, last axis."""
    p = v[..., :K]
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-9)


def _group(M, v):
    """sum VD rows into groups: M (g, n) indicator, v (..., n, c) -> (..., g, c)."""
    return np.einsum("gn,...nc->...gc", M, v)


def score(case: Case, counted: np.ndarray, proj: Projection) -> dict:
    Y = case.Y
    V = proj.vd_mean.astype(float).copy()
    V[counted] = Y[counted]
    true_p = _shares(Y.sum(0))
    pred_p = _shares(V.sum(0))
    tally = _shares(Y[counted].sum(0))
    unc = ~counted
    r = {"prov_mae_pp": 100 * np.abs(pred_p - true_p).mean(),
         "prov_max_pp": 100 * np.abs(pred_p - true_p).max(),
         "tally_mae_pp": 100 * np.abs(tally - true_p).mean(),
         "turnout_err_pp": 100 * (V[:, :K].sum() / V.sum() - Y[:, :K].sum() / Y.sum())}
    for k, p in enumerate(PARTIES):
        r[f"err_{p}"] = 100 * (pred_p[k] - true_p[k])
    # municipalities (seat allocation happens here)
    Mm = (case.muni[None, :] == np.arange(len(case.muni_codes))[:, None]).astype(float)
    r["muni_mae_pp"] = 100 * np.abs(_shares(Mm @ V) - _shares(Mm @ Y)).mean()
    # wards: winner accuracy and share MAE for wards with at least one uncounted VD
    wards = np.unique(case.ward[case.ward >= 0])
    Mw = (case.ward[None, :] == wards[:, None]).astype(float)
    open_w = (Mw @ unc.astype(float)) > 0
    pw, tw = _shares(Mw @ V)[open_w], _shares(Mw @ Y)[open_w]
    r["ward_winner_acc"] = float((pw.argmax(1) == tw.argmax(1)).mean()) if open_w.any() else np.nan
    r["ward_mae_pp"] = 100 * np.abs(pw - tw).mean() if open_w.any() else np.nan
    # uncounted VDs: share MAE and turnout MAE
    if unc.any():
        r["vd_mae_pp"] = 100 * np.abs(_shares(V[unc]) - _shares(Y[unc])).mean()
        to_p = V[unc, :K].sum(1) / np.maximum(V[unc].sum(1), 1)
        to_t = Y[unc, :K].sum(1) / np.maximum(Y[unc].sum(1), 1)
        r["vd_turnout_mae_pp"] = 100 * np.abs(to_p - to_t).mean()
    # intervals from draws
    if proj.draws is not None:
        D = proj.draws.astype(float).copy()
        D[:, counted] = Y[counted][None]
        sp = _shares(D.sum(1))                                   # (S, K)
        lo, hi = np.percentile(sp, [5, 95], axis=0)
        r["prov_cov90"] = float(((lo <= true_p) & (true_p <= hi)).mean())
        r["prov_width90_pp"] = float(100 * (hi - lo).mean())
        # CRPS (energy form) averaged over parties, in pp
        S = sp.shape[0]
        a = np.abs(sp - true_p[None]).mean(0)
        b = np.abs(sp[:, None, :] - sp[None, :, :]).mean((0, 1)) if S <= 1000 else np.nan
        r["prov_crps_pp"] = float(100 * (a - 0.5 * b).mean())
        sm = _shares(np.einsum("gn,snc->sgc", Mm, D))
        lo, hi = np.percentile(sm, [5, 95], axis=0)
        tm = _shares(Mm @ Y)
        r["muni_cov90"] = float(((lo <= tm) & (tm <= hi)).mean())
        # Brier score of ward winner over open wards
        sw = _shares(np.einsum("gn,snc->sgc", Mw, D))[:, open_w]
        pwin = np.stack([(sw.argmax(-1) == k).mean(0) for k in range(K)], -1)
        onehot = np.eye(K)[tw.argmax(1)]
        r["ward_brier"] = float(((pwin - onehot) ** 2).sum(1).mean())
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


# ----------------------------------------------------------------------------- built-in baselines
def tally(case, counted, rng):
    """Naive: every uncounted VD votes like the average counted VD (what the dashboard shows)."""
    p = case.Y[counted].sum(0) / case.Y[counted].sum()
    return Projection(case.registered[:, None] * p[None, :])


def uniform_swing_muni(case, counted, rng):
    """Reference: per-municipality ratio swing vs the previous LGE on each category share of
    the electorate, applied to every uncounted VD's previous-LGE result (no intervals)."""
    prev = case.prev_lge / np.maximum(case.prev_lge.sum(1, keepdims=True), 1)
    V = np.zeros_like(case.Y)
    for m in range(len(case.muni_codes)):
        idx = case.muni == m
        c = idx & counted
        if c.sum() < 3:
            c = counted
        ratio = (case.Y[c].sum(0) / case.Y[c].sum()) / np.maximum(prev[c].mean(0), 1e-6)
        q = prev[idx] * ratio[None]
        V[idx] = case.registered[idx, None] * q / q.sum(1, keepdims=True)
    return Projection(V)


BUILTIN = {"tally": tally, "uniform_swing_muni": uniform_swing_muni}


def resolve(name: str):
    """'tally' (built-in) or 'module:function' under saelection.bench."""
    if name in BUILTIN:
        return name, BUILTIN[name]
    mod, fn = name.split(":")
    return fn, getattr(importlib.import_module(f"saelection.bench.{mod}"), fn)


def run(methods, cases=("LGE2021", "PE2024"), n_random=3, tag=None, order_filter=None):
    rows = []
    for cname in cases:
        case = load_case(cname)
        for oname, (order, levels) in orders(case, n_random).items():
            if order_filter and oname not in order_filter:
                continue
            for lev in levels:
                counted = np.zeros(case.n, bool)
                counted[order[: int(round(lev * case.n))]] = True
                for m in methods:
                    label, fn = resolve(m)
                    t = time.time()
                    proj = fn(case, counted, np.random.default_rng(0))
                    r = score(case, counted, proj)
                    r.update(case=cname, order=oname, pct_counted=round(100 * lev, 1), method=label,
                             seconds=round(time.time() - t, 1))
                    rows.append(r)
                    print(f"{cname:8s} {oname:22s} {100*lev:5.1f}% {label:28s} prov_mae={r['prov_mae_pp']:.2f} "
                          f"muni={r['muni_mae_pp']:.2f} ward_acc={r['ward_winner_acc']:.3f} "
                          f"cov90={r.get('prov_cov90', float('nan')):.2f} ({r['seconds']}s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    if tag:
        df.to_csv(OUT / f"results_{tag}.csv", index=False)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["prov_mae_pp", "prov_max_pp", "muni_mae_pp", "ward_winner_acc", "ward_mae_pp",
                        "vd_mae_pp", "vd_turnout_mae_pp", "prov_cov90", "prov_crps_pp", "muni_cov90",
                        "ward_brier", "tally_mae_pp", "seconds"] if c in df]
    return df.groupby(["method"])[cols].mean().round(3).sort_values("prov_mae_pp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="tally,uniform_swing_muni")
    ap.add_argument("--cases", default="LGE2021,PE2024")
    ap.add_argument("--orders", default="")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    df = run(a.methods.split(","), a.cases.split(","), a.n_random, a.tag,
             set(a.orders.split(",")) if a.orders else None)
    print(summarise(df).to_string())
