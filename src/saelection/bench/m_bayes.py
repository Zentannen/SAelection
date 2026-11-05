"""Bayesian hierarchical and spatial methods for the live-projection bake-off (family: bayes).

All methods project the uncounted VDs on the additive-log-ratio scale against ABSTAIN
(eta_ik = log(p_ik / p_i,abstain), p = shares of the registered electorate), so one model covers
party choice and turnout, and every method returns posterior-predictive vote draws (S, n, 7):
parameter uncertainty + VD residual + multinomial noise.

Methods. Full benchmark: car_full, car_exact_cmb, car_exact_cal (FINAL). Live check: also
car_exact_cm, hier_tcm8. The rest were tried with --n-random 1 or on a few replays
(see artifacts/bench/notes_bayes.md).
-------
car_full        (a) The repo's model as it stands (live.project): Leroux CAR over touching VDs +
                municipal swing + LKJ-correlated party residuals, structural parameters (slopes
                b1 b2, substitution C1 C2, sig_d, CAR sigma/lambda, residual cov) fixed at the
                stage-1 training posterior means; mean-field SVI (5000 steps). Leak-free hyper:
                  PE2024  case -> T1 only (LGE2021 <- LGE2016 + PE2019)
                  LGE2021 case -> T2 only (PE2024 <- LGE2021 + PE2019): a later transition of a
                  different election type, weaker but nothing from 2021 enters the fitted values.
                  (The 0/1 per-party spatial switches come from the hold-out test on T1, i.e. on
                  2021 data; 6 booleans, a negligible leak.)
                  LIVE2026 (live_check) -> T1 + T2 and the poll prior, as in live.run().
                Naive prior: the previous same-type election's result (see naive_prior).
car_full_fast   car_full with 1500 SVI steps instead of 5000 (run-time check).
car_off         (b) CAR fields off for every party AND the structural parameters re-trained
                without them (psi then carries all VD-level variation; switching the fields off
                in hyper-parameters trained with them would understate the VD variance).
                Trained on first use, cached in artifacts/bench/cache_bayes/. SVI.
car_exact       (a') Same model, same fixed hyper-parameters, but the EXACT Gaussian posterior
                (sparse LU, no SVI). Given the hyper-parameters the live model is linear-Gaussian,
                so this is exact; mean-field SVI gets the mean right but collapses var(A).
                (Also uses the broken-history cleaning below; the pure SVI-vs-exact comparison
                is in the notes, from a run before the cleaning was added.)
car_off_exact   (b') car_off with the exact posterior.
car_exact_cm    (a'') car_exact + area-type swing: cluster x party and cluster x municipality x
                party effects (8 k-means clusters of standardised past shares + turnout), sds
                0.3 / 0.2, municipal-swing sd floored at 0.15; broken histories cleaned.
car_exact_cmb   (a'') car_exact_cm + slopes and substitution adapt to the count with the
                trained values as prior means (sd 0.1), guarded against extrapolation.
car_exact_cal   FINAL: car_exact_cmb with the latent draws widened by an inflation schedule by
                % counted (INFL; leave-one-case-out on the replays, both cases for live).
hier            (c) Hierarchical log-ratio swing model fitted to the count only (nothing
                pre-trained): eta = A + B X + delta_muni + omega_ward, Student-t likelihood,
                X = previous-LGE and previous-PE alr of all six parties (12 columns), partial
                pooling ward -> muni -> province, NUTS (1 chain). Predictive VD residual =
                bootstrap of counted VDs' residual vectors.
hier_t          (c) hier with tight slope priors (benchmarked version of (c)).
hier_svi        (c) fitted with SVI instead of NUTS.
hier_tcm8       (d) hier_t + cluster x party and cluster x municipality x party swing (8 clusters).
hier_cm6/10, hier_clus10/20/40   (d) cluster variants with loose slopes (not benchmarked).
hier_bym2       (d) hier + a BYM2-style spatial VD effect (scaled ICAR + iid mix) per party, SVI.

Broken histories (previous turnout > 98% of the roll, e.g. section-24A voting, or roll < 20)
have their past alr replaced by the touching-neighbour average in the car_exact_cm*, hier*
methods (_clean_X; pre-count information only).

A party with zero votes in every counted VD is taken to be off the ballot and projected at 0
(MK in the LGE2021 case); applied to every method.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache, partial

import numpy as np

from ..data import CATS, PARTIES, ROOT
from ..features import K, alr, alr_var
from .harness import Projection

CACHE = ROOT / "artifacts" / "bench" / "cache_bayes"
S_DRAWS = 300
BROKEN_T = 0.98          # previous-election turnout above this = broken history (s24A etc.)


# ============================================================================ shared helpers
def _broken(case):
    """VDs whose previous results are not usable as a baseline: turnout above 90% of the roll in
    either predictor election (e.g. section-24A voters counted at a VD in 2024) or a roll under
    20. Pre-count information only."""
    bad = np.zeros(case.n, bool)
    for prev in (case.prev_lge, case.prev_pe):
        tot = prev.sum(1)
        bad |= (prev[:, :K].sum(1) / np.maximum(tot, 1) > BROKEN_T) | (tot < 20)
    return bad


_XC: dict = {}


def _clean_X(case):
    """(X1, X2) with the rows of broken-history VDs replaced by the average of their touching
    neighbours (those would otherwise project e.g. 97% turnout or ActionSA 78%)."""
    key = (case.name, case.n)
    if key not in _XC:
        from ..features import _neighbour_fill
        bad = _broken(case)
        X = _neighbour_fill(np.concatenate([case.X1, case.X2], 1), ~bad, case.edges)
        _XC[key] = (X[:, :K], X[:, K:], bad)
    return _XC[key]


def naive_prior(case) -> dict:
    """Deliberately naive prior: the previous election *of the same type* (Gauteng totals of
    case.prev_lge for an LGE target, case.prev_pe for a PE target); for the LIVE2026 check the
    poll prior of config/live.json. A party with no previous
    result gets a 2% share floor and a wider prior on its level (A_sd 1.0 vs 0.5)."""
    if case.name == "LIVE2026":
        # live-data check: the live run's poll prior (config/live.json), as in live.run()
        import json
        cfg = json.loads((ROOT / "config" / "live.json").read_text())["prior"]
        sh = np.array([cfg["valid_vote_shares"][p] for p in PARTIES])
        return {"shares": (sh / sh.sum()).tolist(), "turnout": float(cfg["turnout"]),
                "A_sd": np.full(K, float(cfg.get("A_sd", 0.35)))}
    prev = case.prev_lge if case.name.startswith("LGE") else case.prev_pe
    tot = prev.sum(0)
    sh = tot[:K] / tot[:K].sum()
    new = sh < 1e-3
    sh = np.where(new, 0.02, sh)
    sh = sh / sh.sum()
    turnout = tot[:K].sum() / tot.sum()
    return {"shares": sh.tolist(), "turnout": float(turnout), "A_sd": np.where(new, 1.0, 0.5)}


def _off_ballot(case, counted):
    """Parties with zero votes in all counted VDs (not on the ballot)."""
    return case.Y[counted][:, :K].sum(0) == 0


def _finish(case, counted, votes, info, t0):
    """votes (S, n, 7) int -> Projection; zero parties that are off the ballot (their votes move
    to ABSTAIN so the VD total stays at its electorate), counted VDs at their actual result."""
    off = _off_ballot(case, counted)
    if off.any():
        votes = votes.copy()
        votes[..., K] += votes[..., :K][..., off].sum(-1)
        votes[..., np.where(off)[0]] = 0
    votes[:, counted, :] = case.Y[counted].astype(votes.dtype)[None]
    info = dict(info or {})
    info["seconds"] = round(time.time() - t0, 1)
    return Projection(votes.mean(0), votes, info)


def _softmax_abstain(eta):
    e = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,), eta.dtype)], axis=-1)
    e = e - e.max(-1, keepdims=True)
    p = np.exp(e)
    return p / p.sum(-1, keepdims=True)


def _multinomial(R, p, rng):
    """R (n,), p (S, n, 7) -> (S, n, 7) int64 vote draws."""
    Rn = np.round(R).astype(np.int64)
    out = np.empty(p.shape, np.int64)
    for s in range(p.shape[0]):
        out[s] = rng.multinomial(Rn, p[s])
    return out


# ============================================================================ (a), (b): live.project
def _merged_from(src_dir, sources):
    """live.merged_hyper reading the hyper_*.npz files of src_dir."""
    from .. import live
    old = live.ART
    try:
        live.ART = src_dir
        return live.merged_hyper(sources=sources)
    finally:
        live.ART = old


def _train_nocar(name):
    """Stage-1 fit of transition `name` with every CAR field off (psi carries all VD variation)."""
    from ..model import fit
    from ..train import TRANSITIONS, save_hyper, HYPER_KEYS
    target, pl, pp, gy = TRANSITIONS[name]
    from ..features import build_design
    d = build_design(target, pl, pp, gy, name=name)
    s = fit(d, spatial=np.zeros(K + 1, bool), vi_steps=10000, seed=7)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"hyper_{name}.npz", **{k: s[k] for k in HYPER_KEYS},
                        has_history=d.has_history, contested=d.contested, spatial=s["_spatial"],
                        muni_codes=np.array(d.muni_codes))


@lru_cache(maxsize=None)
def _hyper(case_name: str, variant: str = "full"):
    # leak-free: PE2024 <- T1 only; LGE2021 <- T2 only; the live 2026 run may use both
    src = {"PE2024": ("T1",), "LGE2021": ("T2",)}.get(case_name, ("T1", "T2"))
    if variant == "full":
        return _merged_from(ROOT / "artifacts", src)
    if variant == "nocar":
        for t in src:
            if not (CACHE / f"hyper_{t}.npz").exists():
                _train_nocar(t)
        return _merged_from(CACHE, src)
    raise ValueError(variant)


def _live(case, counted, rng, variant="full", vi_steps=5000, prior_mod=None, hyper_mod=None):
    from ..live import project
    t0 = time.time()
    h = dict(_hyper(case.name, variant))
    if hyper_mod:
        h = hyper_mod(h)
    prior = naive_prior(case)
    if prior_mod:
        prior = prior_mod(prior)
    observed = {int(case.vd[i]): case.Y[i] for i in np.where(counted)[0]}
    seed = int(rng.integers(1 << 30))
    votes, cm, info = project(case.design, h, case.registered, observed, prior, n_draws=S_DRAWS,
                              seed=seed, vi_steps=vi_steps)
    assert (cm == counted).all()
    info.update(hyper_sources=h["sources"], variant=variant)
    return _finish(case, counted, votes, info, t0)


def car_full(case, counted, rng):
    """(a) The repo's full model via live.project (5000 SVI steps, 300 draws)."""
    return _live(case, counted, rng, "full", 5000)


def car_full_fast(case, counted, rng):
    """(a) as car_full with 1500 SVI steps (run-time check)."""
    return _live(case, counted, rng, "full", 1500)


def car_off(case, counted, rng):
    """(b) No CAR term: every spatial field off, structural parameters re-trained without them."""
    return _live(case, counted, rng, "nocar", 5000)


# ============================================================================ (a') exact Gaussian posterior
def _incidence(edges, n):
    from scipy.sparse import coo_matrix
    m = len(edges)
    r = np.repeat(np.arange(m), 2)
    c = edges.reshape(-1)
    v = np.tile([1.0, -1.0], m)
    return coo_matrix((v, (r, c)), shape=(m, n)).tocsr()


def _exact_live(case, counted, rng, variant="full", learn_b=False, sig_d_floor=0.0, b_sd=0.25,
                cross_sd=0.5, S=None, freeze=(), prior=None, clusters=0, sig_c=0.3, sig_cm=0.2, infl=None,
                subst_sd=0.0, smd_max=0.6):
    """The live model of live.project with the same fixed structural parameters, but with the
    *exact* posterior instead of mean-field SVI.

    Given the hyper-parameters the live model is linear-Gaussian in all its latents:
        z_i = A + sig_d*d_m(i) + F_i [+ db X_i] + G phi_i + psi_i + eps_i,   eps_i ~ N(0, diag v_i)
    psi_i + eps_i is integrated out at each counted VD (covariance Lcov Lcov' + diag v_i), and
    the joint posterior of (A, d, [db], phi on every VD) is N(Q^-1 b, Q^-1) with a sparse Q.
    Draws come from one sparse LU of Q and S solves ("perturb-then-solve":
    x = Q^-1 (b + R' e1 + X' C e2) with R'R = prior precision and C C' = observation precision).
    Mean-field SVI recovers the posterior mean of a Gaussian but only the *conditional*
    variances, so A (traded off against every psi_i) came out far too narrow.

    learn_b: also re-estimate from the count an adjustment db to the past-performance slopes
    (own previous-LGE and previous-PE alr, prior sd b_sd) and, for a party with no history,
    slopes on all 12 past alr columns (prior sd cross_sd) - linear, so still exact.
    sig_d_floor: lower bound on the municipal-swing sd (T2's are ~0 for some parties).
    clusters: add an area-type swing, gamma_{c(i)k} (sd sig_c) + gamma2_{c(i)m(i)k} (sd sig_cm),
    c = k-means cluster of past voting profile (fixed sds: still exact Gaussian).
    freeze: diagnostics only - components ("A", "d", "b", "phi", "psi", "multi") replaced by their
    posterior mean / no noise, to see where the predictive variance comes from."""
    from scipy.sparse import coo_matrix, diags, vstack, block_diag
    from scipy.sparse.linalg import splu
    from ..live import calibrate_A, fixed_effects
    t0 = time.time()
    S = S or S_DRAWS
    seed = int(rng.integers(1 << 30))
    nrng = np.random.default_rng(seed)
    d = case.design
    h = _hyper(case.name, variant)
    n, M = case.n, len(case.muni_codes)
    prior = prior or naive_prior(case)
    X1c, X2c, _ = _clean_X(case)
    F = h["b1"] * X1c + h["b2"] * X2c + X1c @ h["C1"].T + X2c @ h["C2"].T      # = fixed_effects
    A_loc = calibrate_A(F, case.registered, h, d, prior["shares"], prior["turnout"], nrng)
    A_sd = np.full(K, 1.0) * prior["A_sd"]
    sig_d = np.maximum(h["sig_d"], sig_d_floor)
    on = np.where(h["spatial"])[0]                      # fields kept (0..K-1 parties, K = turnout)
    J = len(on)
    X = np.concatenate([X1c, X2c], 1)
    # adjustable slope columns: (party k, column j, prior sd); prior mean = the trained value
    # (the adjustment db has mean 0), so slopes and substitution adapt to the count
    bcols = []
    if learn_b:
        from ..model import similarity
        Sm = similarity()
        for k in range(K):
            if d.has_history[k]:
                bcols += [(k, k, b_sd), (k, K + k, b_sd)]
                if subst_sd:
                    bcols += [(k, jj + K * e, subst_sd) for jj in range(K) for e in (0, 1)
                              if Sm[k, jj] > 0 and d.has_history[jj]]
            else:
                bcols += [(k, j, cross_sd) for j in range(2 * K) if X[:, j].std() > 0]
        # extrapolation guard: a slope learned from counted VDs that do not span the uncounted
        # ones (standardised mean difference of that column) is not trusted - its prior sd
        # shrinks linearly to 0 at SMD = smd_max (e.g. DA-strongholds-first: DA SMD ~1.6)
        sx = np.where(X.std(0) > 0, X.std(0), 1.0)
        smd = np.abs(X[counted].mean(0) - X[~counted].mean(0)) / sx if (~counted).any() else np.zeros(2 * K)
        bcols = [(k, j, sd * (1 - smd[j] / smd_max)) for k, j, sd in bcols if smd[j] < smd_max]
    nb = len(bcols)
    if clusters:
        cl = _clusters(case, clusters)
        C_ = cl.max() + 1
        cm = cl * M + case.muni
    else:
        C_ = 0
    nG = C_ * K + C_ * M * K
    oA, oD, oB, oG = 0, K, K + M * K, K + M * K + nb
    oP = oG + nG
    N = oP + J * n
    # ---- prior square-root rows R (R'R = prior precision) and prior mean
    Bi = _incidence(case.edges, n)
    blocks = [diags(1.0 / A_sd), diags(np.ones(M * K))]
    if nb:
        blocks.append(diags(np.array([c[2] for c in bcols]) ** -1.0))
    if nG:
        blocks.append(diags(np.concatenate([np.full(C_ * K, 1 / sig_c), np.full(C_ * M * K, 1 / sig_cm)])))
    for j in on:
        lam, sg = float(min(h["lam"][j], 0.999)), float(h["sigma"][j])
        blocks.append(vstack([diags(np.full(n, np.sqrt(1 - lam) / sg)), Bi * (np.sqrt(lam) / sg)]))
    Rp = block_diag(blocks, format="csr")
    mu0 = np.zeros(N)
    mu0[oA:oA + K] = A_loc
    # ---- observations at counted VDs (psi + sampling noise integrated out)
    ci = np.where(counted)[0]
    nc = len(ci)
    z, v = alr(case.Y[ci]), alr_var(case.Y[ci])
    r = z - F[ci]
    Sig_psi = h["Lcov"] @ h["Lcov"].T
    P = np.linalg.inv(Sig_psi[None] + v[:, :, None] * np.eye(K)[None])           # (nc, K, K)
    C = np.linalg.cholesky(P)                                                     # P = C C'
    rows, cols, vals = [], [], []
    for k in range(K):
        rr = np.arange(nc) * K + k
        rows += [rr, rr]
        cols += [np.full(nc, oA + k), oD + case.muni[ci] * K + k]
        vals += [np.ones(nc), np.full(nc, sig_d[k])]
        for jj, j in enumerate(on):
            if j == k or j == K:
                rows.append(rr); cols.append(oP + jj * n + ci); vals.append(np.ones(nc))
    for b, (k, j, _) in enumerate(bcols):
        rows.append(np.arange(nc) * K + k); cols.append(np.full(nc, oB + b)); vals.append(X[ci, j])
    if nG:
        for k in range(K):
            rr = np.arange(nc) * K + k
            rows += [rr, rr]
            cols += [oG + cl[ci] * K + k, oG + C_ * K + cm[ci] * K + k]
            vals += [np.ones(nc), np.ones(nc)]
    Xo = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(nc * K, N)).tocsr()
    Pb = block_diag(list(P), format="csr")
    Cb = block_diag(list(C), format="csr")
    Q = (Rp.T @ Rp + Xo.T @ Pb @ Xo).tocsc()
    bvec = Rp.T @ (Rp @ mu0) + Xo.T @ (Pb @ r.reshape(-1))
    lu = splu(Q, permc_spec="COLAMD")
    mean = lu.solve(bvec)
    E = Rp.T @ nrng.standard_normal((Rp.shape[0], S)) + Xo.T @ (Cb @ nrng.standard_normal((nc * K, S)))
    draws = mean[:, None] + lu.solve(np.asarray(E))                              # (N, S)
    if infl is not None:
        # calibration: widen every latent (swing, slopes, area-type, CAR field) around its posterior
        # mean by a factor set by % counted (fitted on the replays); VD noise is left alone
        f = infl(counted.mean()) if callable(infl) else float(infl)
        draws = mean[:, None] + f * (draws - mean[:, None])
    for comp, sl in (("A", slice(oA, oD)), ("d", slice(oD, oB)), ("b", slice(oB, oG)), ("g", slice(oG, oP)),
                     ("phi", slice(oP, N))):
        if comp in freeze:
            draws[sl] = mean[sl, None]
    A = draws[oA:oA + K].T                                                       # (S, K)
    dm = draws[oD:oB].T.reshape(S, M, K) * sig_d
    phi = draws[oP:].T.reshape(S, J, n)
    eta = A[:, None, :] + dm[:, case.muni] + F[None]
    for b, (k, j, _) in enumerate(bcols):
        eta[:, :, k] += draws[oB + b][:, None] * X[None, :, j]
    if nG:
        g1 = draws[oG:oG + C_ * K].T.reshape(S, C_, K)
        g2 = draws[oG + C_ * K:oP].T.reshape(S, C_ * M, K)
        eta += g1[:, cl] + g2[:, cm]
    for jj, j in enumerate(on):
        if j < K:
            eta[:, :, j] += phi[:, jj]
        else:
            eta += phi[:, jj][..., None]
    if "psi" not in freeze:
        eta += nrng.standard_normal((S, n, K)) @ h["Lcov"].T                     # psi (prior)
    if "multi" in freeze:
        votes = case.registered[None, :, None] * _softmax_abstain(eta)
    else:
        votes = _multinomial(case.registered, _softmax_abstain(eta), nrng)
    info = {"method": "exact Gaussian posterior (sparse LU)", "variant": variant, "learn_b": learn_b,
            "inflation": None if infl is None else round(float(f), 3),
            "A_post_mean": A.mean(0).round(3).tolist(), "A_post_sd": A.std(0).round(3).tolist(),
            "A_prior_mean": A_loc.round(3).tolist()}
    if nb:
        info["db_mean"] = {f"{PARTIES[k]}:{j}": round(float(draws[oB + b].mean()), 3)
                           for b, (k, j, _) in enumerate(bcols)}
    return _finish(case, counted, votes, info, t0)


def car_exact(case, counted, rng):
    """(a') The full model with the exact Gaussian posterior (no SVI): fixes the overconfidence
    caused by mean-field VI; same hyper-parameters, prior and predictive as car_full."""
    return _exact_live(case, counted, rng)


def car_exact_b(case, counted, rng):
    """(a') car_exact + past-performance slopes re-estimated from the count (prior sd 0.25 on the
    adjustment; for a party with no history, slopes on all past alr columns) + municipal-swing
    sd floored at 0.15, so the model is less tied to the training transition."""
    return _exact_live(case, counted, rng, learn_b=True, sig_d_floor=0.15)


def car_exact_cm(case, counted, rng):
    """(a') car_exact + area-type swing: cluster x party (sd 0.3) and cluster x municipality x
    party (sd 0.2) effects, 8 k-means clusters of past voting profile, municipal-swing sd floored
    at 0.15; slopes stay at their trained values; still exact."""
    return _exact_live(case, counted, rng, sig_d_floor=0.15, clusters=8)


def car_off_exact(case, counted, rng):
    """(b') car_off (no CAR, re-trained) with the exact Gaussian posterior."""
    return _exact_live(case, counted, rng, variant="nocar")


# ============================================================================ (c), (d): hierarchical swing
def _hier_model(Xc, mc, wc, cc, n_m, n_w, n_c, A0, own, z=None, v=None, sp=None, cmc=None, n_cm=0,
                own_mu=0.5, own_sd=0.5, off_scale=0.5):
    """Hierarchical log-ratio model for the counted VDs (all arrays restricted to them).

    eta_ik = A_k + sum_j B_kj X_ij + delta_{m(i)k} + omega_{w(i)k} [+ kappa_{c(i)k}]
             [+ gamma_{c(i)m(i)k}] [+ spatial]
    z_ik   ~ StudentT(nu, eta_ik, sqrt(sig_e_k^2 + v_ik))    (v = multinomial sampling variance)
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    P = Xc.shape[1]
    A = numpyro.sample("A", dist.Normal(A0, 1.0).to_event(1))
    b_own = numpyro.sample("b_own", dist.Normal(own_mu, own_sd).expand([K, 2]).to_event(2))
    s_off = numpyro.sample("s_off", dist.HalfNormal(1.0).expand([K]).to_event(1)) * off_scale
    B_raw = numpyro.sample("B_raw", dist.Normal(0.0, 1.0).expand([K, P]).to_event(2))
    B = B_raw * s_off[:, None] * (1.0 - own)
    B = B.at[jnp.arange(K), jnp.arange(K)].add(b_own[:, 0]).at[jnp.arange(K), K + jnp.arange(K)].add(b_own[:, 1])
    sig_m = numpyro.sample("sig_m", dist.HalfNormal(0.3).expand([K]).to_event(1))
    dm = numpyro.sample("dm_raw", dist.Normal(0.0, 1.0).expand([n_m, K]).to_event(2)) * sig_m
    sig_w = numpyro.sample("sig_w", dist.HalfNormal(0.3).expand([K]).to_event(1))
    dw = numpyro.sample("dw_raw", dist.Normal(0.0, 1.0).expand([n_w, K]).to_event(2)) * sig_w
    eta = A + Xc @ B.T + dm[mc] + dw[wc]
    if n_c:
        sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.3).expand([K]).to_event(1))
        dc = numpyro.sample("dc_raw", dist.Normal(0.0, 1.0).expand([n_c, K]).to_event(2)) * sig_c
        eta = eta + dc[cc]
    if n_cm:
        # area type x municipality swing (only cells with counted VDs are latent)
        sig_cm = numpyro.sample("sig_cm", dist.HalfNormal(0.3).expand([K]).to_event(1))
        dcm = numpyro.sample("dcm_raw", dist.Normal(0.0, 1.0).expand([n_cm, K]).to_event(2)) * sig_cm
        eta = eta + dcm[cmc]
    sig_e = numpyro.sample("sig_e", dist.HalfNormal(0.5).expand([K]).to_event(1))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    scale2 = sig_e ** 2
    if sp is not None:
        # BYM2-style split of the VD residual: rho_k of its variance is a scaled ICAR field over
        # the touching-VD graph (latent on all VDs), the rest stays iid Student-t.
        rho = numpyro.sample("rho", dist.Beta(1.0, 1.0).expand([K]).to_event(1))
        n_all, edges, idx = sp["n"], sp["edges"], sp["idx"]
        # vague proper base (sd 100) so the site can be sampled; the ICAR factor below does the work
        u = numpyro.sample("u", dist.Normal(0.0, 100.0).expand([n_all, K]).to_event(2))
        d2 = jnp.sum((u[edges[:, 0]] - u[edges[:, 1]]) ** 2, axis=0)
        numpyro.factor("icar", -0.5 * jnp.sum(d2) - 0.5 * jnp.sum((u.sum(0) / (0.001 * n_all)) ** 2))
        eta = eta + sig_e * jnp.sqrt(rho / sp["scale"]) * u[idx]
        scale2 = sig_e ** 2 * (1.0 - rho)
    numpyro.sample("z", dist.StudentT(nu, eta, jnp.sqrt(scale2 + v)), obs=z)


_CL: dict = {}


def _clusters(case, n_clusters: int = 20):
    """k-means clusters of past voting profile (pre-count information only): per-party shares of
    valid votes and turnout in both predictor elections, each standardised so small parties
    (EFF, ActionSA, MK, OTHER) weigh as much as the big two (CSIR / Ittmann lesson)."""
    from sklearn.cluster import KMeans
    key = (case.name, n_clusters)
    if key in _CL:
        return _CL[key]
    from ..features import _neighbour_fill
    c = case
    feats = []
    for prev in (c.prev_lge, c.prev_pe):
        valid = prev[:, :K].sum(1, keepdims=True)
        feats += [prev[:, :K] / np.maximum(valid, 1), valid / np.maximum(prev.sum(1, keepdims=True), 1)]
    X = _neighbour_fill(np.concatenate(feats, 1), ~_broken(c), c.edges)
    X = X[:, X.std(0) > 1e-9]
    X = (X - X.mean(0)) / X.std(0)
    _CL[key] = KMeans(n_clusters, random_state=0, n_init=10).fit_predict(X)
    return _CL[key]


@lru_cache(maxsize=None)
def _icar_scale(case_name: str):
    """BYM2 scaling factor: geometric mean of the ICAR marginal variances (per connected
    component, from the Laplacian pseudo-inverse)."""
    from ..model import graph_eigen
    from .harness import load_case
    c = load_case(case_name)
    mu, U = graph_eigen(c.edges, c.n)
    pos = mu > 1e-8
    var = (U[:, pos] ** 2 / mu[pos]).sum(1)
    return float(np.exp(np.mean(np.log(var[var > 0]))))


def _icar_conditional(case, rng, u_obs, counted, ridge=1e-4):
    """Draws of the ICAR field at uncounted VDs given its value at counted ones:
    u_H | u_O ~ N(-Q_HH^-1 Q_HO u_O, Q_HH^-1), Q = Laplacian (+ tiny ridge so uncounted
    components with no counted neighbour stay proper). u_obs: (S, n_counted, K)."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    from ..model import laplacian
    Q = laplacian(case.edges, case.n) + ridge * np.eye(case.n)
    H, O = ~counted, counted
    Qhh = Q[np.ix_(H, H)]
    L = np.linalg.cholesky(Qhh)
    B = -cho_solve((L, True), Q[np.ix_(H, O)])                    # (|H|, |O|)
    S, _, Kk = u_obs.shape
    mean = np.einsum("ho,sok->shk", B, u_obs)
    eps = rng.standard_normal((H.sum(), S * Kk))
    noise = solve_triangular(L, eps, lower=True, trans="T").T.reshape(S, Kk, H.sum()).transpose(0, 2, 1)
    return mean + noise


def _init_zero_u(site=None):
    import jax.numpy as jnp
    from numpyro.infer import init_to_median
    if site is None:
        return partial(_init_zero_u)
    if site["name"] == "u":
        return jnp.zeros(site["fn"].shape(), jnp.float32)
    return init_to_median(site)


def _hier(case, counted, rng, clusters=0, spatial=False, method="nuts", warmup=300, samples=None, infl=None,
          clus_muni=False, own_mu=0.5, own_sd=0.5, off_hist=0.5, off_new=0.5, resid="boot"):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
    t0 = time.time()
    S = samples or S_DRAWS
    seed = int(rng.integers(1 << 30))
    nrng = np.random.default_rng(seed)
    X1c, X2c, _ = _clean_X(case)
    X = np.concatenate([X1c, X2c], 1)                           # (n, 12) centred past alr
    own = np.zeros((K, 2 * K))
    own[np.arange(K), np.arange(K)] = own[np.arange(K), K + np.arange(K)] = 1
    ci = np.where(counted)[0]
    z, v = alr(case.Y[ci]), alr_var(case.Y[ci])
    prior = naive_prior(case)
    t = prior["turnout"]
    A0 = np.log(np.asarray(prior["shares"]) * t) - np.log(1 - t)
    # wards: only those with counted VDs are latent; the rest are drawn from the prior later
    wards_all, w_idx = np.unique(case.ward, return_inverse=True)
    w_seen = np.unique(w_idx[ci])
    w_map = -np.ones(len(wards_all), int)
    w_map[w_seen] = np.arange(len(w_seen))
    n_c, cl, c_seen, c_map = 0, None, None, None
    if clusters:
        cl = _clusters(case, clusters)
        c_seen = np.unique(cl[ci])
        c_map = -np.ones(cl.max() + 1, int)
        c_map[c_seen] = np.arange(len(c_seen))
        n_c = len(c_seen)
    n_cm, cm_all, cm_seen, cm_map = 0, None, None, None
    if clusters and clus_muni:
        M_ = len(case.muni_codes)
        cm_all = cl * M_ + case.muni                         # area type x municipality cell
        cm_seen = np.unique(cm_all[ci])
        cm_map = -np.ones((cl.max() + 1) * M_, int)
        cm_map[cm_seen] = np.arange(len(cm_seen))
        n_cm = len(cm_seen)
    sp = None
    if spatial:
        sp = dict(n=case.n, edges=jnp.asarray(case.edges), idx=jnp.asarray(ci), scale=_icar_scale(case.name))
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    kw = dict(Xc=f32(X[ci]), mc=jnp.asarray(case.muni[ci]), wc=jnp.asarray(w_map[w_idx[ci]]),
              cc=jnp.asarray(c_map[cl[ci]]) if clusters else None,
              n_m=len(case.muni_codes), n_w=len(w_seen), n_c=n_c, A0=f32(A0), own=f32(own),
              z=f32(z), v=f32(v), sp=sp,
              cmc=jnp.asarray(cm_map[cm_all[ci]]) if n_cm else None, n_cm=n_cm,
              own_mu=own_mu, own_sd=own_sd,
              off_scale=f32(np.where(case.design.has_history, off_hist, off_new)))
    key = jax.random.PRNGKey(seed)
    if method == "nuts":
        mcmc = MCMC(NUTS(_hier_model, target_accept_prob=0.85, max_tree_depth=8,
                         init_strategy=init_to_median), num_warmup=warmup, num_samples=S,
                    num_chains=1, progress_bar=False)
        mcmc.run(key, **kw)
        post = {k: np.asarray(x) for k, x in mcmc.get_samples().items()}
        div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        info = {"method": "NUTS", "divergences": div}
    else:
        import optax
        from numpyro.infer.autoguide import AutoGuideList, AutoMultivariateNormal, AutoNormal
        glob = ["A", "b_own", "s_off", "B_raw", "sig_m", "sig_w", "sig_c", "sig_cm", "sig_e", "nu", "rho"]
        guide = AutoGuideList(_hier_model)
        guide.append(AutoMultivariateNormal(numpyro.handlers.block(numpyro.handlers.seed(_hier_model, 0), expose=glob),
                                            init_loc_fn=init_to_median, init_scale=0.05))
        guide.append(AutoNormal(numpyro.handlers.block(numpyro.handlers.seed(_hier_model, 0), hide=glob),
                                init_loc_fn=partial(_init_zero_u), init_scale=0.05))
        steps = method if isinstance(method, int) else 3000
        opt = optax.adam(optax.cosine_decay_schedule(0.02, steps, alpha=0.05))
        res = SVI(_hier_model, guide, opt, Trace_ELBO()).run(key, steps, progress_bar=False, **kw)
        post = {k: np.asarray(x) for k, x in guide.sample_posterior(jax.random.PRNGKey(seed + 1), res.params,
                                                                     sample_shape=(S,), **kw).items() if k != "z"}
        info = {"method": f"SVI {steps} steps", "elbo": float(-np.asarray(res.losses)[-200:].mean())}
    # ---- posterior predictive for every VD
    B = post["B_raw"] * (post["s_off"] * np.where(case.design.has_history, off_hist, off_new))[:, :, None] * (1 - own)[None]
    B[:, np.arange(K), np.arange(K)] += post["b_own"][:, :, 0]
    B[:, np.arange(K), K + np.arange(K)] += post["b_own"][:, :, 1]
    dm = post["dm_raw"] * post["sig_m"][:, None, :]
    dw_all = nrng.standard_normal((S, len(wards_all), K))
    dw_all[:, w_seen] = post["dw_raw"]
    dw_all *= post["sig_w"][:, None, :]
    eta = post["A"][:, None, :] + np.einsum("np,skp->snk", X, B) + dm[:, case.muni] + dw_all[:, w_idx]
    if clusters:
        dc_all = nrng.standard_normal((S, cl.max() + 1, K))
        dc_all[:, c_seen] = post["dc_raw"]
        eta += (dc_all * post["sig_c"][:, None, :])[:, cl]
    if n_cm:
        # cells with no counted VD: drawn from N(0, sig_cm) - wider where the count says nothing
        dcm_all = nrng.standard_normal((S, len(cm_map), K))
        dcm_all[:, cm_seen] = post["dcm_raw"]
        eta += (dcm_all * post["sig_cm"][:, None, :])[:, cm_all]
        info["sig_cm"] = post["sig_cm"].mean(0).round(3).tolist()
    if infl is not None:
        # calibration: widen the shared (province / slope / muni / ward / cluster) part of the
        # draws around its posterior mean by a factor set by % counted (fitted on the replays)
        f = infl(counted.mean()) if callable(infl) else float(infl)
        m = eta.mean(0, keepdims=True)
        eta = m + f * (eta - m)
        info["inflation"] = round(f, 3)
    nu = post["nu"][:, None, None]
    e = nrng.standard_t(np.broadcast_to(nu, (S, case.n, K)))
    sig = post["sig_e"][:, None, :]
    if spatial:
        rho = post["rho"][:, None, :]
        u = np.zeros((S, case.n, K))
        u[:, counted] = post["u"][:, ci] if post["u"].shape[1] == case.n else post["u"]
        u[:, ~counted] = _icar_conditional(case, nrng, post["u"][:, ci], counted)
        eta += sig * np.sqrt(rho / sp["scale"]) * u
        sig = sig * np.sqrt(1 - rho)
    if resid == "boot":
        # VD residual = a whole 6-party residual vector of a random counted VD (same draw):
        # keeps the real tails and cross-party correlation. Student-t draws with nu ~ 2 have no
        # exp-moments, which biased the vote-space mean (big parties down, small up, turnout up).
        r = z[None] - eta[:, ci]                                        # (S, nc, K)
        j = nrng.integers(len(ci), size=(S, case.n))
        eta += np.take_along_axis(r, j[..., None], axis=1)
    else:
        eta += sig * e
    votes = _multinomial(case.registered, _softmax_abstain(eta), nrng)
    info.update(nu=float(post["nu"].mean()), sig_e=post["sig_e"].mean(0).round(3).tolist(),
                sig_m=post["sig_m"].mean(0).round(3).tolist(), sig_w=post["sig_w"].mean(0).round(3).tolist())
    return _finish(case, counted, votes, info, t0)


def hier(case, counted, rng):
    """(c) Hierarchical log-ratio swing model (ward -> muni -> province, Student-t residual), NUTS."""
    return _hier(case, counted, rng)


def hier_svi(case, counted, rng):
    """(c) as hier, fitted with SVI (3000 steps; MVN guide on the globals, mean-field on the rest)."""
    return _hier(case, counted, rng, method=3000)


def hier_clus10(case, counted, rng):
    """(d) hier + cluster x party swing effect, 10 k-means clusters of past voting profile, NUTS."""
    return _hier(case, counted, rng, clusters=10)


def hier_clus20(case, counted, rng):
    """(d) hier + cluster x party swing effect, 20 clusters, NUTS."""
    return _hier(case, counted, rng, clusters=20)


def hier_clus40(case, counted, rng):
    """(d) hier + cluster x party swing effect, 40 clusters, NUTS."""
    return _hier(case, counted, rng, clusters=40)


TIGHT = dict(own_mu=0.5, own_sd=0.15, off_hist=0.1, off_new=0.5)


def hier_t(case, counted, rng):
    """(c) hier with tight slope priors (own previous-LGE/PE slopes ~ N(0.5, 0.15), cross-party
    slopes scale 0.1 for parties with history, 0.5 for a party without): slopes learned from a
    biased sample (e.g. only ANC strongholds) must not be extrapolated freely."""
    return _hier(case, counted, rng, **TIGHT)


def hier_tcm8(case, counted, rng):
    """(d) hier_t + area-type swing: cluster x party and cluster x municipality x party effects
    (8 k-means clusters of standardised past shares + turnout), NUTS."""
    return _hier(case, counted, rng, clusters=8, clus_muni=True, **TIGHT)


def hier_cm6(case, counted, rng):
    """(d) hier + area-type swing: cluster x party AND cluster x municipality x party effects
    (6 k-means clusters of past voting profile), NUTS. Cells with no counted VD get prior draws."""
    return _hier(case, counted, rng, clusters=6, clus_muni=True)


def hier_cm10(case, counted, rng):
    """(d) as hier_cm6 with 10 clusters."""
    return _hier(case, counted, rng, clusters=10, clus_muni=True)


def hier_bym2(case, counted, rng):
    """(d) hier + BYM2-style spatial VD effect (scaled ICAR on touching VDs), SVI."""
    return _hier(case, counted, rng, spatial=True, method=4000)


def car_exact_cm4(case, counted, rng):
    """car_exact_cm with 4 clusters."""
    return _exact_live(case, counted, rng, sig_d_floor=0.15, clusters=4)


def car_exact_cm12(case, counted, rng):
    """car_exact_cm with 12 clusters."""
    return _exact_live(case, counted, rng, sig_d_floor=0.15, clusters=12)


def car_exact_cmb(case, counted, rng):
    """(a'') car_exact_cm + the past-performance slopes AND substitution coefficients adapt to the
    count: the trained values are prior means (adjustment sd 0.1 own slopes, 0.1 substitution;
    a party with no history gets slopes on all past columns, sd 0.25), each sd shrunk to 0 when
    the counted VDs do not span the uncounted ones on that column (extrapolation guard)."""
    return _exact_live(case, counted, rng, sig_d_floor=0.15, clusters=8, learn_b=True, b_sd=0.1,
                       subst_sd=0.1, cross_sd=0.25)


# inflation of the latent spread by fraction counted, fitted on the replays of car_exact_cmb
# (90% quantile of |error| / sd at the province level, divided by 1.645). For a benchmark case the
# schedule is fitted on the OTHER case only (leave-one-case-out); the live run uses both.
INFL_LEVELS = [0.05, 0.10, 0.25, 0.50]
INFL = {"LGE2021": [1.14, 1.53, 1.67, 1.94],    # fitted on the PE2024 replays
        "PE2024": [1.57, 1.47, 2.10, 2.60],     # fitted on the LGE2021 replays
        "both": [1.32, 1.60, 1.79, 2.32]}       # live run (and LIVE2026 check)


def _infl_schedule(case):
    vals = INFL.get(case.name, INFL["both"])
    return lambda frac: float(np.interp(frac, INFL_LEVELS, vals))


def car_exact_cal(case, counted, rng):
    """car_exact_cmb with its latent draws widened by the % counted inflation schedule INFL
    (leave-one-case-out on the benchmark; both cases for live)."""
    return _exact_live(case, counted, rng, sig_d_floor=0.15, clusters=8, learn_b=True, b_sd=0.1,
                       subst_sd=0.1, cross_sd=0.25, infl=_infl_schedule(case))
