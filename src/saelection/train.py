"""Stage 1 - learn the model's structure from past Gauteng elections.

Transitions (target <- previous LGE + previous national/provincial ballot):
  T1: LGE2021 <- LGE2016 + PE2019   (same shape as 2026: an LGE after an LGE and a PE)
  T2: PE2024  <- LGE2021 + PE2019   (only transition where ActionSA has history; MK debuts)

Also runs the spatial check the model is built around: for every party, is there covariance
between *touching* VDs that improves out-of-sample prediction? 20% of VDs are held out; the
model is fitted with and without the CAR term and the held-out log predictive density is
compared. Parties whose spatial term does not help (delta-elpd < 2 SE) get it switched off.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .data import PARTIES, ROOT
from .features import K, build_design
from .model import fit, heldout_lpd_exact, omega, similarity, substitution_corr

ART = ROOT / "artifacts"
TRANSITIONS = {
    "T1": ("LGE2021", "LGE2016", "PE2019", 2021),
    "T2": ("PE2024", "LGE2021", "PE2019", 2024),
}
HYPER_KEYS = ["A", "b1", "b2", "C1", "C2", "sig_d", "sigma", "lam", "tau", "L_omega", "d_raw"]
STEPS = 10000


def save_hyper(name, s, d):
    ART.mkdir(exist_ok=True)
    np.savez_compressed(ART / f"hyper_{name}.npz", **{k: s[k] for k in HYPER_KEYS},
                        has_history=d.has_history, contested=d.contested, spatial=s["_spatial"],
                        muni_codes=np.array(d.muni_codes))


def summarise(name, s, d):
    Sm = 0.5 * similarity()
    Om = omega(s).mean(0)
    sub = substitution_corr(Om, s["tau"].mean(0))
    out = {"transition": d.name, "elbo": s["_elbo_final"], "parties": PARTIES,
           "contested": d.contested.tolist(), "has_history": d.has_history.tolist()}
    for k in ["b1", "b2", "sig_d", "tau"]:
        out[k] = {"mean": s[k].mean(0).round(3).tolist(), "sd": s[k].std(0).round(3).tolist()}
    for k in ["sigma", "lam"]:
        out[k] = {"mean": s[k].mean(0).round(3).tolist(), "sd": s[k].std(0).round(3).tolist(),
                  "index": PARTIES + ["TURNOUT_COMMON"]}
    out["C1_masked"] = (s["C1"] * Sm).mean(0).round(3).tolist()
    out["C2_masked"] = (s["C2"] * Sm).mean(0).round(3).tolist()
    out["omega"] = Om.round(3).tolist()
    out["substitution_corr"] = sub.round(3).tolist()
    return out


def spatial_test(seed=1, frac=0.2):
    target, pl, pp, gy = TRANSITIONS["T1"]
    d = build_design(target, pl, pp, gy, name="T1")
    rng = np.random.default_rng(seed)
    held = rng.random(d.n) < frac
    res = {}
    for label, sp in (("spatial", np.ones(K + 1, bool)), ("no_spatial", np.zeros(K + 1, bool))):
        t = time.time()
        s = fit(d, spatial=sp, obs_vd_mask=~held, vi_steps=STEPS, seed=seed)
        res[label] = heldout_lpd_exact(s, d, held)        # (n_held, K)
        print(f"  holdout fit {label}: {time.time() - t:.0f}s", flush=True)
    diff = res["spatial"] - res["no_spatial"]
    out = {"n_heldout": int(held.sum()), "parties": {}}
    for k, p in enumerate(PARTIES):
        if not d.contested[k]:
            out["parties"][p] = {"tested": False, "keep_spatial": True,
                                 "note": "did not contest 2021; kept (default)"}
            continue
        dk = diff[:, k]
        delta, se = float(dk.sum()), float(dk.std() * np.sqrt(len(dk)))
        out["parties"][p] = {"tested": True, "delta_elpd": round(delta, 1), "se": round(se, 1),
                             "elpd_spatial": round(float(res["spatial"][:, k].sum()), 1),
                             "elpd_no_spatial": round(float(res["no_spatial"][:, k].sum()), 1),
                             "keep_spatial": bool(delta > 2 * se)}
    return out


def main():
    ART.mkdir(exist_ok=True)
    print("spatial (touching-VD covariance) hold-out test on T1 ...", flush=True)
    test = spatial_test()
    keep = np.array([test["parties"][p]["keep_spatial"] for p in PARTIES] + [True])
    test["turnout_common_factor"] = {"keep_spatial": True}
    (ART / "spatial_test.json").write_text(json.dumps(test, indent=2))
    print(json.dumps(test["parties"], indent=1), flush=True)

    summary = {"spatial_test": test}
    for name, (target, pl, pp, gy) in TRANSITIONS.items():
        t = time.time()
        d = build_design(target, pl, pp, gy, name=name)
        s = fit(d, spatial=keep, vi_steps=STEPS, seed=7)
        save_hyper(name, s, d)
        summary[name] = summarise(name, s, d)
        print(f"{name} {d.name}: {time.time() - t:.0f}s  elbo={s['_elbo_final']:.0f}", flush=True)
    (ART / "training_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
