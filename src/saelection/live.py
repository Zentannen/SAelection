"""Stage 2 - live projection of the 2026 Gauteng count.

Given the VDs counted so far (IEC per-VD results), project the votes still to come in every
uncounted VD and the final distribution in every VD, ward, municipality and the province.

Why this is robust to the order in which VDs report: the model never extrapolates the running
tally. Each counted VD is compared with *its own* history (2021 LGE, 2024 PE), so it tells us
about the swing (provincial A_k, municipal delta_mk) and - through the touching-VD CAR term -
about its uncounted neighbours. A party that leads early because its strongholds counted first
does not get that lead projected onto VDs where it has always been weak.

Structural parameters (past-performance slopes b, substitution coefficients C, municipal-swing
spread, spatial sd / lambda, cross-party residual correlation Omega) are learned in stage 1
(train.py) and held fixed here; the swing and the VD fields are re-estimated from the count.
Before any VD is counted, the swing comes from the prior in config/live.json (polls).
"""
from __future__ import annotations

import json
import os
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from .data import CATS, MUNI_NAMES, PARTIES, PROCESSED, ROOT, load_results_file
from .features import K, Design, alr, alr_var, build_design
from .model import CarConditional, graph_eigen, sample_car, similarity, substitution_corr

ART = ROOT / "artifacts"
OUT = ROOT / "output" / "live"
OBSERVED = ROOT / "data" / "observed_2026"


# ----------------------------------------------------------------------------- hyper-parameters
def _nearest_corr(M):
    M = (M + M.T) / 2
    w, V = np.linalg.eigh(M)
    M = (V * np.clip(w, 1e-4, None)) @ V.T
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def merged_hyper(sources=("T1", "T2")):
    """Posterior-mean structural parameters for 2026, party by party.

    A party's parameters come from the first training transition in which it had a previous
    result (so its past-performance slopes mean the same thing as in 2026). Parties with no such
    transition (MK: first contested 2024) get the average of the established parties and no
    substitution row (their own 2024 result already carries their geography)."""
    H = {k: np.load(ART / f"hyper_{k}.npz") for k in sources}
    Sm = 0.5 * similarity()
    out = {k: np.zeros(K) for k in ["b1", "b2", "sig_d", "tau"]}
    out.update(sigma=np.zeros(K + 1), lam=np.zeros(K + 1), C1=np.zeros((K, K)), C2=np.zeros((K, K)))
    src = []
    for k in range(K):
        s = next((n for n in sources if H[n]["has_history"][k] and H[n]["contested"][k]), None)
        src.append(s or "pooled")
    base = sources[0]
    est = [k for k in range(K) if src[k] == base]
    for k in range(K):
        h = H[src[k]] if src[k] != "pooled" else None
        for name in ["b1", "b2", "sig_d", "tau"]:
            out[name][k] = h[name][:, k].mean() if h is not None else H[base][name][:, est].mean()
        for name in ["sigma", "lam"]:
            out[name][k] = h[name][:, k].mean() if h is not None else H[base][name][:, est].mean()
        if h is not None:
            out["C1"][k] = (h["C1"] * Sm).mean(0)[k]
            out["C2"][k] = (h["C2"] * Sm).mean(0)[k]
    out["sigma"][K] = H[base]["sigma"][:, K].mean()
    out["lam"][K] = H[base]["lam"][:, K].mean()
    out["spatial"] = H[base]["spatial"]
    # cross-party residual correlation: base transition, rows of parties that did not contest
    # there taken from the next transition in which they did
    Om = np.einsum("sij,skj->sik", H[base]["L_omega"], H[base]["L_omega"]).mean(0)
    for k in range(K):
        if not H[base]["contested"][k]:
            for n in sources[1:]:
                if H[n]["contested"][k]:
                    O2 = np.einsum("sij,skj->sik", H[n]["L_omega"], H[n]["L_omega"]).mean(0)
                    Om[k, :], Om[:, k] = O2[k, :], O2[:, k]
                    break
    np.fill_diagonal(Om, 1.0)
    out["omega"] = _nearest_corr(Om)
    out["Lcov"] = np.linalg.cholesky(out["omega"] * np.outer(out["tau"], out["tau"]))
    out["sources"] = dict(zip(PARTIES, src))
    return out


def fixed_effects(d: Design, h) -> np.ndarray:
    return h["b1"] * d.X1 + h["b2"] * d.X2 + d.X1 @ h["C1"].T + d.X2 @ h["C2"].T


# ----------------------------------------------------------------------------- live model
def _softmax_abstain_j(eta):
    e = jnp.concatenate([eta, jnp.zeros(eta.shape[:-1] + (1,), eta.dtype)], axis=-1)
    return jax.nn.softmax(e, axis=-1)


def live_model(F, muni, n_muni, edges, sig_d, sigma, lam, on, Lcov, A_loc, A_sd, R,
               z=None, v=None, obs_mask=None,
               agg_W=None, agg_z=None, agg_v=None, agg_num=None, agg_ref=None, agg_mask=None):
    """VD-level counts enter through z (alr of the VD result); ward / municipal partial counts
    enter through agg_* : the alr of the counted totals of unit u is compared with the alr of the
    model's expected totals over the unit's VDs, sum_i W_ui R_i p_i, with extra variance for
    which VDs happened to be counted first (see aggregate_observations)."""
    n = F.shape[0]
    J = K + 1
    A = numpyro.sample("A", dist.Normal(A_loc, A_sd).to_event(1))
    d_raw = numpyro.sample("d_raw", dist.Normal(0.0, 1.0).expand([n_muni, K]).to_event(2))
    phi = numpyro.sample("phi", dist.ImproperUniform(dist.constraints.real_vector, (), (n, J)))
    diff2 = jnp.sum((phi[edges[:, 0]] - phi[edges[:, 1]]) ** 2, axis=0)
    quad = ((1.0 - lam) * jnp.sum(phi ** 2, axis=0) + lam * diff2) / sigma ** 2
    numpyro.factor("car", -0.5 * jnp.sum(quad))
    psi = numpyro.sample("psi", dist.MultivariateNormal(jnp.zeros(K), scale_tril=Lcov).expand([n]).to_event(1))
    phi = phi * on
    eta = A + d_raw[muni] * sig_d + F + phi[:, :K] + phi[:, K:] + psi
    if z is not None:
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("z", dist.Normal(eta, jnp.sqrt(v)), obs=z)
    if agg_W is not None:
        E = agg_W @ (R[:, None] * _softmax_abstain_j(eta))            # (U, 7) expected totals
        logE = jnp.log(E + 1e-6)
        num = jnp.take_along_axis(logE, agg_num, axis=1)                # (U, K)
        ref = jnp.take_along_axis(logE, agg_ref[:, None], axis=1)       # (U, 1)
        with numpyro.handlers.mask(mask=agg_mask):
            numpyro.sample("agg_z", dist.Normal(num - ref, jnp.sqrt(agg_v)), obs=agg_z)


def _softmax_abstain(eta):
    e = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,), eta.dtype)], axis=-1)
    e = e - e.max(-1, keepdims=True)
    p = np.exp(e)
    return p / p.sum(-1, keepdims=True)


def calibrate_A(F, R, h, d, shares, turnout, rng, n_mc=60, iters=40):
    """Provincial level A such that expected Gauteng totals match the prior's valid-vote shares
    and turnout (registered-weighted, averaging over VD-level noise)."""
    target = np.append(turnout * np.asarray(shares), 1 - turnout)
    S = n_mc
    sig = np.tile(h["sigma"], (S, 1))
    lam = np.tile(np.where(h["spatial"], h["lam"], 0.0), (S, 1))
    phi = sample_car(sig, lam, d.edges, d.n, rng)
    psi = rng.standard_normal((S, d.n, K)) @ h["Lcov"].T
    delta = rng.standard_normal((S, len(d.muni_codes), K)) * h["sig_d"]
    phi = phi * h["spatial"].astype(float)
    noise = phi[..., :K] + phi[..., K:] + psi + delta[:, d.muni, :]
    A = np.log(target[:K] / target[K])
    for _ in range(iters):
        p = _softmax_abstain(A + F + noise)
        tot = np.einsum("snk,n->k", p, R) / (S * R.sum())
        A = A + np.log(target[:K] / tot[:K]) - np.log(target[K] / tot[K])
    return A


def _live_init(site=None, A_loc=None):
    if site is None:
        return partial(_live_init, A_loc=A_loc)
    if site["name"] == "A":
        return jnp.asarray(A_loc, jnp.float32)
    if site["name"] in ("phi", "psi", "d_raw"):
        return jnp.zeros(site["fn"].shape(), jnp.float32)
    return numpyro.infer.init_to_median(site)


def aggregate_observations(aggs, h):
    """aggs: list of dicts {W: (n,) VD weights of the unit, f: fraction of the unit's VDs counted,
    n_counted: VDs counted, counts: (6,) counted valid votes, abstain: counted registered - valid or None}.

    With the counted registered voters known, the observation is the alr against ABSTAIN (party
    choice + turnout); otherwise the alr against OTHER (party choice only). Sampling variance plus
    a selection term: the counted VDs are a subset of the unit, so their composition deviates
    from the whole unit's by ~ tau^2 (1 - f) / n_counted on the log-ratio scale."""
    U = len(aggs)
    W = np.stack([a["W"] for a in aggs])
    z = np.zeros((U, K)); v = np.ones((U, K)); num = np.tile(np.arange(K), (U, 1))
    ref = np.zeros(U, int); mask = np.zeros((U, K), bool)
    tau2 = h["tau"] ** 2
    for u, a in enumerate(aggs):
        c = np.asarray(a["counts"], float) + 0.5
        sel = (1 - a["f"]) / max(a["n_counted"], 1)
        if a.get("abstain") is not None:
            cr = a["abstain"] + 0.5
            z[u] = np.log(c) - np.log(cr)
            v[u] = 1 / c + 1 / cr + tau2 * sel
            ref[u] = K
            mask[u] = True
        else:
            o = PARTIES.index("OTHER")
            z[u] = np.log(c) - np.log(c[o])
            v[u] = 1 / c + 1 / c[o] + (tau2 + tau2[o]) * sel
            ref[u] = o
            mask[u] = np.arange(K) != o
    return dict(agg_W=W, agg_z=z, agg_v=v, agg_num=num, agg_ref=ref, agg_mask=mask)


def project(d: Design, h, R, observed: dict, prior: dict, n_draws=500, seed=0, vi_steps=6000, aggs=None):
    """Vote draws (S, n, 7) for every VD.

    observed: {vd: counts(7)} VD-level results -> those VDs fixed at their actual result, the
              uncounted VDs drawn from the exact CAR conditional given the counted ones.
    aggs:     ward/municipal partial counts (see aggregate_observations) -> VD draws are the
              posterior projection for the whole VD; the caller combines them with the counted part.
    """
    rng = np.random.default_rng(seed)
    F = fixed_effects(d, h)
    A_loc = calibrate_A(F, R, h, d, prior["shares"], prior["turnout"], rng)
    A_sd = np.full(K, prior.get("A_sd", 0.35))
    counted = np.zeros(d.n, bool)
    Yobs = np.zeros((d.n, len(CATS)))
    pos = {v: i for i, v in enumerate(d.vd)}
    for vd, row in observed.items():
        if vd in pos:
            counted[pos[vd]] = True
            Yobs[pos[vd]] = row
    S = n_draws
    lam_eff = np.where(h["spatial"], h["lam"], 0.0)
    J = K + 1
    if counted.sum() == 0 and not aggs:
        A = A_loc + rng.standard_normal((S, K)) * A_sd
        delta = rng.standard_normal((S, len(d.muni_codes), K)) * h["sig_d"]
        phi = sample_car(np.tile(h["sigma"], (S, 1)), np.tile(lam_eff, (S, 1)), d.edges, d.n, rng)
        info = {"method": "prior simulation (nothing counted yet)"}
    else:
        z, v = alr(Yobs), alr_var(Yobs)
        kw = dict(F=jnp.asarray(F, jnp.float32), muni=jnp.asarray(d.muni), n_muni=len(d.muni_codes),
                  edges=jnp.asarray(d.edges),
                  sig_d=jnp.asarray(h["sig_d"], jnp.float32), sigma=jnp.asarray(h["sigma"], jnp.float32),
                  lam=jnp.asarray(lam_eff, jnp.float32), on=jnp.asarray(h["spatial"], jnp.float32), Lcov=jnp.asarray(h["Lcov"], jnp.float32),
                  A_loc=jnp.asarray(A_loc, jnp.float32), A_sd=jnp.asarray(A_sd, jnp.float32),
                  R=jnp.asarray(R, jnp.float32))
        if counted.any():
            kw.update(z=jnp.asarray(z, jnp.float32), v=jnp.asarray(v, jnp.float32),
                      obs_mask=jnp.asarray(np.repeat(counted[:, None], K, 1)))
        if aggs:
            ao = aggregate_observations(aggs, h)
            kw.update({k: jnp.asarray(x, jnp.float32 if x.dtype == float else x.dtype) for k, x in ao.items()})

        guide = AutoNormal(live_model, init_loc_fn=partial(_live_init, A_loc=A_loc), init_scale=0.02)
        opt = optax.adam(optax.cosine_decay_schedule(0.02, vi_steps, alpha=0.02))
        res = SVI(live_model, guide, opt, Trace_ELBO()).run(jax.random.PRNGKey(seed), vi_steps,
                                                            progress_bar=False, **kw)
        post = guide.sample_posterior(jax.random.PRNGKey(seed + 1), res.params, sample_shape=(S,))
        A = np.asarray(post["A"])
        delta = np.asarray(post["d_raw"]) * h["sig_d"]
        phi = np.asarray(post["phi"]).copy()
        if counted.any() and not aggs:
            # uncounted VDs: exact CAR conditional given the counted VDs' fields
            for j in np.where(h["spatial"])[0]:
                cc = CarConditional(d.edges, d.n, float(h["sigma"][j]), float(lam_eff[j]), ~counted)
                phi[:, ~counted, j] = cc.sample(phi[:, counted, j], rng)
        info = {"method": "variational posterior given the count so far", "vi_steps": vi_steps,
                "n_vd_observations": int(counted.sum()), "n_aggregate_observations": len(aggs or []),
                "elbo": float(-np.asarray(res.losses)[-300:].mean())}
    phi = phi * h["spatial"].astype(float)              # fields switched off contribute nothing
    # VD-specific party residuals of uncounted VDs are not informed by any data: exact prior
    psi = rng.standard_normal((S, d.n, K)) @ h["Lcov"].T
    eta = A[:, None, :] + delta[:, d.muni, :] + F[None] + phi[..., :K] + phi[..., K:] + psi
    p = _softmax_abstain(eta)
    votes = np.empty((S, d.n, len(CATS)), np.int64)
    Rn = np.round(R).astype(np.int64)
    for s in range(S):
        votes[s] = rng.multinomial(Rn, p[s])
    votes[:, counted, :] = Yobs[counted].astype(np.int64)[None]
    info.update(A_prior_mean=A_loc.round(3).tolist(), A_post_mean=A.mean(0).round(3).tolist())
    return votes, counted, info


# ----------------------------------------------------------------------------- observed results
def load_observed(folder: Path = OBSERVED):
    """VD-level IEC result CSVs (any *.csv not named progress_*)."""
    files = sorted(f for f in folder.glob("*.csv") if not f.name.startswith(("progress_", "iec_live")))
    obs, reg = {}, {}
    for f in files:
        t = load_results_file(f, "LGE", "LGE2026")
        for r in t.itertuples(index=False):
            obs[int(r.vd)] = np.array([getattr(r, c) for c in CATS], float)
            reg[int(r.vd)] = float(r.registered)
    return obs, reg, [f.name for f in files]


def load_progress(folder: Path = OBSERVED):
    """Partial counts per ward / municipality (progress_*.csv), e.g. from the IEC dashboard.

    Columns: unit_id (ward id such as 79800012, or municipality code JHB/TSH/EKU/GT421/...),
    pct_vds_counted (0-100), counted valid votes per party (DA, ANC, ACTIONSA, MK, EFF, OTHER -
    full IEC party names also accepted, anything else is added to OTHER), optional
    registered_counted (registered voters in the counted VDs) and spoilt."""
    from .data import PARTY_MAP
    rows, files = [], sorted(folder.glob("progress_*.csv"))
    for f in files:
        t = pd.read_csv(f)
        t.columns = [c.strip() for c in t.columns]
        for _, r in t.iterrows():
            c = dict.fromkeys(PARTIES, 0.0)
            for col, val in r.items():
                key = col.upper()
                if col in ("unit_id", "pct_vds_counted", "registered_counted", "spoilt") or pd.isna(val):
                    continue
                cat = key if key in PARTIES else PARTY_MAP.get(key, "OTHER")
                c[cat] += float(val)
            rows.append({"unit_id": str(r["unit_id"]).strip(), "f": float(r["pct_vds_counted"]) / 100,
                         "counts": np.array([c[p] for p in PARTIES]),
                         "registered_counted": float(r["registered_counted"]) if "registered_counted" in r and pd.notna(r["registered_counted"]) else None,
                         "spoilt": float(r.get("spoilt", 0) or 0)})
    return rows, [f.name for f in files]


# ----------------------------------------------------------------------------- summaries
def _q(x, axis=0):
    return {"mean": np.mean(x, axis), "p05": np.percentile(x, 5, axis), "p50": np.percentile(x, 50, axis),
            "p95": np.percentile(x, 95, axis)}


def summarise_units(V, counted_frac, names, extra=None, counted=None):
    """V: (S, U, 7) final-vote draws for U units; counted: (S, U, 7) votes already counted.
    Returns list of dicts (frontend records)."""
    valid = V[..., :K].sum(-1)                                   # (S, U)
    share = V[..., :K] / np.maximum(valid, 1)[..., None]
    turnout_ballots = valid                                      # valid votes / registered
    reg = V.sum(-1)
    lead = share.argmax(-1)                                      # (S, U)
    recs = []
    for u, nm in enumerate(names):
        r = {"id": nm}
        if extra is not None:
            r.update(extra[u])
        r["pct_vds_counted"] = round(float(counted_frac[u]) * 100, 1)
        r["registered"] = int(np.round(reg[:, u].mean()))
        tq = _q(turnout_ballots[:, u] / np.maximum(reg[:, u], 1))
        r["turnout_valid"] = {k: round(float(v), 4) for k, v in tq.items()}
        r["parties"] = {}
        for k, p in enumerate(PARTIES):
            vq, sq = _q(V[:, u, k]), _q(share[:, u, k])
            r["parties"][p] = {
                "votes": {kk: int(round(float(vv))) for kk, vv in vq.items()},
                "share": {kk: round(float(vv), 4) for kk, vv in sq.items()},
                "p_first": round(float((lead[:, u] == k).mean()), 4),
            }
        r["projected_leader"] = PARTIES[int(np.bincount(lead[:, u], minlength=K).argmax())]
        if counted is not None:
            cu = counted[:, u, :K]
            r["counted_votes"] = {p: int(round(float(cu[:, k].mean()))) for k, p in enumerate(PARTIES)}
            rem = V[:, u, :K] - cu
            r["remaining_votes"] = {p: {kk: int(round(float(vv))) for kk, vv in _q(rem[:, k]).items()}
                                    for k, p in enumerate(PARTIES)}
        recs.append(r)
    return recs


def flat_table(recs):
    rows = []
    for r in recs:
        row = {k: v for k, v in r.items() if k not in ("parties", "turnout_valid", "counted_votes", "remaining_votes")}
        row["turnout_valid_mean"] = r["turnout_valid"]["mean"]
        for p, x in r["parties"].items():
            for q in ("mean", "p05", "p95"):
                row[f"{p}_votes_{q}"] = x["votes"][q]
            row[f"{p}_share_mean"] = x["share"]["mean"]
            row[f"{p}_p_first"] = x["p_first"]
            if "counted_votes" in r:
                row[f"{p}_counted"] = r["counted_votes"][p]
                row[f"{p}_remaining_mean"] = r["remaining_votes"][p]["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main entry
def run(n_draws=500, seed=0, out_dir: Path = OUT):
    cfg = json.loads((ROOT / "config" / "live.json").read_text())
    prior = cfg["prior"]
    prior["shares"] = [prior["valid_vote_shares"][p] for p in PARTIES]
    s_ = sum(prior["shares"])
    prior["shares"] = [x / s_ for x in prior["shares"]]

    d = build_design(None, "LGE2021", "PE2024", 2026, name="LGE2026<-LGE2021+PE2024")
    h = merged_hyper()
    observed, reg_obs, files = load_observed()
    progress, pfiles = load_progress()
    warnings = []
    if observed and progress:
        warnings.append("VD-level results present: progress_*.csv files ignored")
        progress = []
    cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")
    base = cw.groupby("vd")["vd_regpop"].first().reindex(d.vd).to_numpy(float)
    vdset = set(d.vd)
    counted_vds = [v for v in observed if v in vdset]

    # ward and municipality membership matrices (split VDs allocated by registered share)
    wards = sorted(cw.ward.unique())
    widx = {w: j for j, w in enumerate(wards)}
    vidx = {v: i for i, v in enumerate(d.vd)}
    Wm = np.zeros((len(wards), d.n))
    for r in cw.itertuples(index=False):
        Wm[widx[r.ward], vidx[r.vd]] += r.ward_weight
    wmeta = cw.groupby("ward").agg(muni=("muni", "first"), ward_no=("ward_no", "first"), n_vd=("vd", "nunique"))
    munis = d.muni_codes
    Mm = np.zeros((len(munis), d.n))
    Mm[d.muni, np.arange(d.n)] = 1
    ward_muni = np.array([munis.index(wmeta.muni[w]) for w in wards])

    live_csv = OBSERVED / "iec_live_vd.csv"
    live_extra = {}            # ward index -> PR-equivalent votes of counted VDs not in the 2026 layer
    live_meta = None
    if live_csv.exists():
        # IEC dashboard feed (iec_live.py): per-VD PR-equivalent votes + per-municipality RegPop
        L = pd.read_csv(live_csv)
        live_meta = json.loads((OBSERVED / "iec_live_munis.json").read_text())
        reg26 = {m["muni"]: m["registered_2026"] for m in live_meta}
        g_m = {m: reg26[m] / base[d.muni == munis.index(m)].sum() for m in munis if m in reg26}
        growth_vec = np.array([g_m.get(munis[k], 1.0) for k in d.muni])
        R = base * growth_vec
        growth = float(R.sum() / base.sum())
        growth_src = "per municipality: IEC 2026 RegPop / 2024 certified roll"
        files.append(live_csv.name)
        for r in L[L.counted].itertuples(index=False):
            cnt = np.array([getattr(r, p) for p in PARTIES], float)
            if r.vd in vidx:
                i = vidx[r.vd]
                R[i] = max(R[i], cnt.sum() * 1.02)
                observed[int(r.vd)] = np.append(cnt, R[i] - cnt.sum())
            elif r.ward in widx:
                live_extra.setdefault(widx[r.ward], np.zeros(K))
                live_extra[widx[r.ward]] += cnt
        counted_vds = [v for v in observed if v in vdset]
        live_vds = L.groupby("ward").agg(n=("vd", "size"), c=("counted", "sum"))
    elif counted_vds:
        b_ = pd.Series(base, index=d.vd)
        growth = sum(reg_obs[v] for v in counted_vds) / b_.loc[counted_vds].sum()
        growth_src = "estimated from counted VDs (2026 registered / 2024 certified)"
        R = base * growth
    else:
        growth = cfg.get("registered_growth_vs_2024", 1.0)
        growth_src = "config/live.json registered_growth_vs_2024"
        R = base * growth
    for v in counted_vds:
        if v in reg_obs:
            R[vidx[v]] = reg_obs[v]

    # partial counts per ward / municipality -> aggregate observations
    aggs, agg_units = [], []
    for pr in progress:
        uid = pr["unit_id"]
        if uid.isdigit() and int(uid) in widx:
            lvl, j = "ward", widx[int(uid)]
            W, n_vd = Wm[j], int(wmeta.n_vd[int(uid)])
        elif uid.upper() in munis:
            lvl, j = "muni", munis.index(uid.upper())
            W, n_vd = Mm[j], int(Mm[j].sum())
        else:
            warnings.append(f"progress unit {uid} not found; skipped")
            continue
        if pr["f"] <= 0:
            continue
        abst = None
        if pr["registered_counted"]:
            abst = max(pr["registered_counted"] - pr["counts"].sum(), 1.0)
        aggs.append({"W": W, "f": min(pr["f"], 1.0), "n_counted": max(1, round(pr["f"] * n_vd)),
                     "counts": pr["counts"], "abstain": abst})
        agg_units.append((lvl, j))
    if any(l == "ward" for l, _ in agg_units) and any(l == "muni" for l, _ in agg_units):
        warnings.append("both ward and municipality progress given: municipality rows used for fitting only")

    votes, counted, info = project(d, h, R, observed, prior, n_draws=n_draws, seed=seed, aggs=aggs)
    S = votes.shape[0]
    Yc = np.where(counted[:, None], votes[0], 0)                       # counted VD results (n, 7)

    # ---- wards: projected draws, then counted part + projected remainder where only partial
    V_w = np.einsum("wn,snc->swc", Wm, votes)
    C_w = np.broadcast_to(Wm @ Yc, V_w.shape).astype(float).copy()
    f_w = (Wm @ counted.astype(float)) / Wm.sum(1)
    # counted VDs that exist only in the live feed (new 2026 VDs): add their actual votes to the ward
    # and take the same number of valid votes out of the ward's still-projected VDs (those voters
    # were inside the parent VD's registered roll)
    unc = ~counted
    for j, ex in live_extra.items():
        U = np.einsum("n,snc->sc", Wm[j] * unc, votes)                    # projected, uncounted part
        uv = U[:, :K].sum(-1, keepdims=True)
        scale = np.clip(1 - ex.sum() / np.maximum(uv, 1), 0, 1)
        V_w[:, j, :K] = V_w[:, j, :K] - U[:, :K] * (1 - scale) + ex
        C_w[:, j, :K] += ex
    if live_meta is not None:
        f_w = np.array([live_vds.c.get(w, 0) / live_vds.n.get(w, 1) if w in live_vds.index else f_w[j]
                        for j, w in enumerate(wards)])
    ward_agg = {j: a for (l, j), a in zip(agg_units, aggs) if l == "ward"}
    for j, a in ward_agg.items():
        f = a["f"]
        C_w[:, j, :K] = a["counts"]
        C_w[:, j, K] = a["abstain"] if a["abstain"] is not None else f * V_w[:, j, K]
        V_w[:, j, :] = C_w[:, j, :] + (1 - f) * V_w[:, j, :]
        f_w[j] = f
    # ---- municipalities: sum of wards (keeps everything additive); municipal-only progress
    #      rows (no ward detail) are combined the same way
    V_m = np.stack([V_w[:, ward_muni == m, :].sum(1) for m in range(len(munis))], 1)
    C_m = np.stack([C_w[:, ward_muni == m, :].sum(1) for m in range(len(munis))], 1)
    nvd_w = np.array([wmeta.n_vd[w] for w in wards], float)
    f_m = np.array([(f_w[ward_muni == m] * nvd_w[ward_muni == m]).sum() / nvd_w[ward_muni == m].sum()
                    for m in range(len(munis))])
    if not ward_agg:
        for (l, j), a in zip(agg_units, aggs):
            if l == "muni":
                V_proj = np.einsum("n,snc->sc", Mm[j], votes)
                C_m[:, j, :K] = a["counts"]
                C_m[:, j, K] = a["abstain"] if a["abstain"] is not None else a["f"] * V_proj[:, K]
                V_m[:, j, :] = C_m[:, j, :] + (1 - a["f"]) * V_proj
                f_m[j] = a["f"]
    V_p = V_m.sum(1, keepdims=True)
    C_p = C_m.sum(1, keepdims=True)
    f_p = np.array([(f_m * Mm.sum(1)).sum() / d.n])
    V_w, V_m, V_p = (np.round(x).astype(np.int64) for x in (V_w, V_m, V_p))

    # ---- records
    nodes = pd.read_csv(PROCESSED / "vd_nodes_2026.csv").set_index("vd").reindex(d.vd)
    prim = cw.sort_values("ward_weight").groupby("vd").last().reindex(d.vd)
    vd_extra = [{"muni": prim.muni.iloc[i], "ward": int(prim.ward.iloc[i]), "station": prim.station.iloc[i],
                 "lon": float(nodes.lon.iloc[i]), "lat": float(nodes.lat.iloc[i]),
                 "status": "counted" if counted[i] else "projected",
                 "ward_pct_counted": round(100 * float(f_w[widx[int(prim.ward.iloc[i])]]), 1)} for i in range(d.n)]
    vd_recs = summarise_units(votes, counted.astype(float), [int(v) for v in d.vd], vd_extra)
    w_extra = [{"muni": wmeta.muni[w], "ward_no": int(wmeta.ward_no[w]), "n_vd": int(wmeta.n_vd[w])} for w in wards]
    ward_recs = summarise_units(V_w, f_w, [int(w) for w in wards], w_extra, counted=C_w)
    muni_recs = summarise_units(V_m, f_m, munis, [{"name": MUNI_NAMES.get(m, m)} for m in munis], counted=C_m)
    prov_rec = summarise_units(V_p, f_p, ["Gauteng"], counted=C_p)[0]
    prov_rec["counted_so_far"] = {c: int(round(float(C_p[:, 0, i].mean()))) for i, c in enumerate(CATS[:K])}

    # ---- write
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "generated_at": now, "election": "2026 Local Government Elections - Gauteng (PR ballot)",
        "parties": PARTIES, "categories": CATS,
        "n_vds": int(d.n), "n_vds_counted_exact": int(counted.sum()), "pct_vds_counted": round(100 * float(f_p[0]), 2),
        "observed_files": files, "progress_files": pfiles, "n_progress_units": len(aggs),
        "warnings": warnings, "n_draws": S,
        "iec_feed": live_meta,
        "registered_total": int(R.sum()), "registered_growth": round(float(growth), 4), "registered_growth_source": growth_src,
        "prior": {k: prior[k] for k in ("name", "valid_vote_shares", "turnout", "A_sd", "sources") if k in prior},
        "model": {
            "inference": info, "structural_parameter_source": h["sources"],
            "spatial_term_kept": dict(zip(PARTIES + ["TURNOUT_COMMON"], map(bool, h["spatial"]))),
            "spatial_lambda": dict(zip(PARTIES + ["TURNOUT_COMMON"], np.round(h["lam"], 3).tolist())),
            "substitution_C1_prev_LGE": np.round(h["C1"], 3).tolist(),
            "substitution_C2_prev_PE": np.round(h["C2"], 3).tolist(),
            "residual_corr_omega": np.round(h["omega"], 3).tolist(),
            "substitution_partial_corr": np.round(substitution_corr(h["omega"]), 3).tolist(),
        },
        "notes": [
            "Shares are of valid PR votes; turnout_valid = valid votes / registered voters.",
            "votes = projected FINAL votes (counted + still to come); counted_votes / remaining_votes split them.",
            "Counted VDs carry their actual result (p05 = p50 = p95). Projected VDs are posterior predictive draws.",
            "With ward-level progress only, a partially counted ward = its counted votes + (1 - pct counted) x its projection; VD records stay projections for the whole VD.",
            "Ward totals allocate a split VD's votes to wards by each piece's share of registered voters.",
            "p_first = probability the party has the most PR votes in that unit.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "province.json").write_text(json.dumps(prov_rec, indent=2))
    for nm, recs in (("municipalities", muni_recs), ("wards", ward_recs), ("vds", vd_recs)):
        (out_dir / f"{nm}.json").write_text(json.dumps(recs))
        flat_table(recs).to_csv(out_dir / f"{nm}.csv", index=False)
    # map layers (simplified polygons), joined to the records by id
    import geopandas as gpd
    (out_dir / "geo").mkdir(exist_ok=True)
    gv = gpd.read_file(PROCESSED / "vd_2026_gauteng.gpkg").rename(columns={"vd": "id"})
    gv.to_file(out_dir / "geo" / "vds_2026.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    if (PROCESSED / "wards_2026_gauteng.gpkg").exists():
        gw = gpd.read_file(PROCESSED / "wards_2026_gauteng.gpkg").rename(columns={"ward": "id"})
        gw.to_file(out_dir / "geo" / "wards_2026.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    # draws for distribution plots: province and municipalities, every draw
    pd.DataFrame(V_p[:, 0, :], columns=CATS).rename_axis("draw").to_csv(out_dir / "province_draws.csv")
    md = pd.DataFrame(V_m.reshape(-1, len(CATS)), columns=CATS)
    md.insert(0, "muni", np.tile(munis, S))
    md.insert(0, "draw", np.repeat(np.arange(S), len(munis)))
    md.to_csv(out_dir / "municipality_draws.csv", index=False)
    return meta, prov_rec
