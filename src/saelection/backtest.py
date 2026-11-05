"""Count-replay backtest: does the live projection see through the order VDs report in?

Target: the 2024 Gauteng provincial ballot (PE2024), predictors LGE2021 + PE2019. Structural
parameters come from T1 only (LGE2021 <- LGE2016 + PE2019), so nothing from 2024 leaks in.
The prior is deliberately naive (roughly the 2019 result), so any accuracy comes from the count.

VDs are revealed in different orders - random, DA strongholds first, ANC strongholds first -
and at 5/10/25/50% counted we compare (a) the running tally and (b) the projected final against
the actual final, in two input modes:
  vd    : VD-level results for the counted VDs
  ward  : only running totals + % of VDs counted per ward (what the IEC dashboard shows)
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .data import PARTIES, ROOT, load_history
from .features import K, build_design
from .geo import GEO_FILES
from .live import merged_hyper, project

ART = ROOT / "artifacts"


def ward_matrix(d):
    w = gpd.read_file(GEO_FILES[2024], where="Province='Gauteng'", read_geometry=False)
    w = w.drop_duplicates("VDNumber").set_index(w.drop_duplicates("VDNumber")["VDNumber"].astype(np.int64))["WardId"]
    ward_of = w.reindex(d.vd).to_numpy()
    wards = np.unique(ward_of)
    W = (ward_of[None, :] == wards[:, None]).astype(float)
    return W


def main(levels=(0.05, 0.10, 0.25, 0.50), modes=("vd", "ward"), n_draws=300, vi_steps=5000):
    d = build_design("PE2024", "LGE2021", "PE2019", 2024, name="backtest PE2024")
    h = merged_hyper(sources=("T1",))
    R = d.registered
    Y = d.Y
    final = Y[:, :K].sum(0) / Y[:, :K].sum()
    prior = {"shares": [0.26, 0.47, 0.03, 0.05, 0.14, 0.05], "turnout": 0.65, "A_sd": 0.5}
    Wm = ward_matrix(d)

    hist = load_history()
    p19 = hist[hist.election == "PE2019"].set_index("vd").reindex(d.vd)
    da19 = (p19["DA"] / p19["valid"]).fillna(0).to_numpy()
    anc19 = (p19["ANC"] / p19["valid"]).fillna(0).to_numpy()
    rng = np.random.default_rng(42)
    orders = {"random": rng.permutation(d.n), "DA_strongholds_first": np.argsort(-da19),
              "ANC_strongholds_first": np.argsort(-anc19)}

    rows = []
    for oname, order in orders.items():
        for lev in levels:
            idx = order[: int(round(lev * d.n))]
            cmask = np.zeros(d.n, bool)
            cmask[idx] = True
            tally = Y[idx, :K].sum(0) / Y[idx, :K].sum()
            for mode in modes:
                if mode == "vd":
                    observed = {int(d.vd[i]): Y[i] for i in idx}
                    votes, counted, _ = project(d, h, R, observed, prior, n_draws=n_draws, seed=1, vi_steps=vi_steps)
                    tot = votes[..., :K].sum(1)
                    unc = ~counted
                    vs_pred = (votes[:, unc, :K] / np.maximum(votes[:, unc, :K].sum(-1, keepdims=True), 1)).mean(0)
                    vs_true = Y[unc, :K] / np.maximum(Y[unc, :K].sum(-1, keepdims=True), 1)
                    vd_mae = 100 * np.abs(vs_pred - vs_true).mean(0)
                else:
                    nvd = Wm.sum(1)
                    ncnt = Wm @ cmask
                    C = Wm @ (Y * cmask[:, None])
                    regc = Wm @ (R * cmask)
                    aggs = [{"W": Wm[j], "f": ncnt[j] / nvd[j], "n_counted": int(ncnt[j]), "counts": C[j, :K],
                             "abstain": max(regc[j] - C[j, :K].sum(), 1.0)} for j in range(len(Wm)) if ncnt[j] > 0]
                    fw = ncnt / nvd
                    votes, _, _ = project(d, h, R, {}, prior, n_draws=n_draws, seed=1, vi_steps=vi_steps, aggs=aggs)
                    Vw = np.einsum("wn,snc->swc", Wm, votes)
                    tot = (C[None, :, :K] + (1 - fw)[None, :, None] * Vw[..., :K]).sum(1)
                    vd_mae = np.full(K, np.nan)
                sh = tot / tot.sum(-1, keepdims=True)
                for k, p in enumerate(PARTIES):
                    lo, hi = np.percentile(sh[:, k], [5, 95])
                    rows.append({"mode": mode, "order": oname, "pct_counted": int(lev * 100), "party": p,
                                 "actual_final_pct": round(100 * final[k], 2),
                                 "running_tally_pct": round(100 * tally[k], 2),
                                 "projected_final_pct": round(100 * sh[:, k].mean(), 2),
                                 "proj_90_lo": round(100 * lo, 2), "proj_90_hi": round(100 * hi, 2),
                                 "covered": bool(lo <= final[k] <= hi),
                                 "uncounted_vd_share_mae_pp": round(float(vd_mae[k]), 2)})
                print(mode, oname, lev, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "backtest_count_replay.csv", index=False)
    df["tally_abs_err"] = (df.running_tally_pct - df.actual_final_pct).abs()
    df["proj_abs_err"] = (df.projected_final_pct - df.actual_final_pct).abs()
    summ = df.groupby(["mode", "order", "pct_counted"]).agg(
        tally_mae_pp=("tally_abs_err", "mean"), tally_max_pp=("tally_abs_err", "max"),
        projection_mae_pp=("proj_abs_err", "mean"), projection_max_pp=("proj_abs_err", "max"),
        coverage_90=("covered", "mean"), uncounted_vd_mae_pp=("uncounted_vd_share_mae_pp", "mean")).round(2)
    summ.to_csv(ART / "backtest_summary.csv")
    print(summ.to_string())
    return df, summ


if __name__ == "__main__":
    main()
