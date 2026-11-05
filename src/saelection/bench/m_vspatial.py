"""Spatial residual pull on top of hybrid_cf (family: v_spatial). Best: ``spatial_rake_half``.

Problem (election night 2026): in a partly counted ward the uncounted VDs seem to ignore what
their counted ward-mates and neighbours show (reported for ActionSA). hybrid_cf projects every
uncounted VD from its own history + an area-type x municipality swing + a Curtice-Firth
regression; its only local term is an EB ward random effect (shrunk hard when a ward has one
counted VD), and nothing crosses a ward boundary.

Methods: ``spatial_pull`` (steps 1-5 below on hybrid_cf as it is: the safe variant, better
than hybrid_cf on every live-check task kind and every benchmark accuracy metric, small ward
gain); ``spatial_nw`` (steps 1-5 on hybrid_cf without its ward level: best ward / VD accuracy,
worse province); ``spatial_rake_half`` (BEST overall, see ``_raked``): the VD pattern = log-ratio
average of spatial_nw and spatial_pull, raked so every municipality's uncounted total equals
spatial_pull's (province / municipalities as spatial_pull, ward MAE and ward Brier as
spatial_nw); ``spatial_rake`` = the same with spatial_nw's pattern alone.

Spatial pull (``_spatial``):
 1. Base = hybrid_cf's point projection. ``base='full'`` (``spatial_pull``) keeps
    hybrid_cf exactly as it is (incl. its EB ward level) and pulls on top of it.
    ``base='nw'`` (``spatial_nw``) removes the ward level from both parts of hybrid_cf
    (cluster_muni_ratio on km20 + muni + km20 x muni cells, ridge with municipality effects,
    same hand-over c = 2) so that the ward becomes a special case of the kriging below.
 2. Out-of-fold residuals: the counted VDs are split at random into F = 5 folds; the base is
    refitted without each fold and predicts it; r_i = alr(Y_i) - alr(base_oof_i) per party
    (log-ratio vs ABSTAIN, so turnout is in it). A VD's own result never enters the
    prediction it is compared with; a residual is "what the base gets wrong for a VD whose
    neighbours are counted", i.e. the situation of an uncounted VD tonight.
 3. Residual covariance per party, by moments from products r_i r_j of CROSS-FOLD counted
    pairs only (r_i was predicted by a base that had seen r_j's VD, exactly as an uncounted
    VD's base has seen its counted neighbours; if the base's own ward / cell / municipality
    effects already carry a neighbour's signal, the fitted covariance, and with it the pull,
    comes out smaller, so nothing is counted twice):
        Cov(r_i, r_j) = tw [same ward] + sum_c ts_c k_c(d_ij)            (i != j)
        Var(r_i)      = tw + sum_c ts_c + s2 + v_i
    k_c = tapered exponential in centroid distance d (km): exp(-d / ell_c) times the Wendland
    taper (1 - d/R_c)^4 (1 + 4 d/R_c), i.e. a hard distance cap R_c that keeps the kernel
    positive definite. One component (R = 5 km) by default, two (3 km + 15 km) in
    ``spatial_2s`` / ``spatial_nw``. Ranges (pooled over parties) by grid search; tw, ts >= 0
    by weighted non-negative least squares (pair weight = product of precisions);
    s2 = nugget = residual variance - shared part; v_i = sampling variance of alr(Y_i).
    Residuals are centred (precision-weighted) and winsorised at 3 SD.
 4. Pull = simple kriging, i.e. the BLUP / empirical-Bayes posterior mean of the shared
    residual field at every uncounted VD given all counted residuals:
        shift_u = c_u' (C_cc + diag(s2 + v))^-1 r_c,  c_u = tw [same ward] + sum ts_c k_c(d_uc)
    The weight on a counted neighbour shrinks with distance, with the number of counted
    neighbours (they share the field) and with their sampling noise; a ward with one counted
    VD gets a partial pull toward it, a ward with none still gets a pull from counted VDs just
    across its boundary. Applied on the log-ratio scale, then back to counts.
 5. Draws (S = 200). base='full': hybrid_cf's own draws re-centred on the pulled mean
    (``full_field=True`` also adds the kriging posterior of the field: wider, and its
    log-scale noise biases province shares, so not the default). base='nw': the base's
    log-ratio predictive distribution WITHOUT its independent VD residual (bootstrap of the
    swing, EB cluster / municipality posteriors, extrapolation penalty), re-centred on the
    pulled mean, + a joint draw of the kriging posterior of the field by conditional
    simulation (unconditional field = ward N(0, tw) + Cholesky of each tapered kernel over
    all VDs x sqrt(ts_c); simulated counted residuals = field + noise are kriged and
    subtracted) + the VD nugget sqrt(s2 + v_u) x a resampled standardised out-of-fold
    residual (heavy tails, cross-party correlation kept).

Tuning and results: artifacts/bench/notes_v_spatial.md (live check pinned to the 15:18
snapshot; benchmark both cases, n_random 3).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..features import K
from . import m_swing as ms
from .harness import Projection

NCAT = K + 1
NW_LEVELS = ("km20", "muni", "cm20")          # cluster_muni_ratio levels without the ward
ELLS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)   # km, grid for the exponential range
_OOF: dict = {}
_LCH: dict = {}
BASE_CACHE: dict | None = None                 # set to {} to cache base draws (tuning only)


# ----------------------------------------------------------------------------- geometry
def _geo(case):
    """VD centroids in km (equirectangular) and the local VD spacing (distance to the 4th
    nearest VD centroid, relative to its median; known before counting)."""
    p = ms._prep(case)
    if "vsp_geo" not in p.clusters:
        lon, lat = case.lon.astype(float).copy(), case.lat.astype(float).copy()
        for arr in (lon, lat):
            bad = ~np.isfinite(arr)
            if bad.any():
                for m in np.unique(p.muni[bad]):
                    ok = (p.muni == m) & ~bad
                    arr[bad & (p.muni == m)] = arr[ok].mean() if ok.any() else np.nanmean(arr)
        lat0 = lat.mean()
        xy = np.column_stack([(lon - lon.mean()) * 111.32 * np.cos(np.deg2rad(lat0)),
                              (lat - lat0) * 110.57])
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(xy).query(xy, k=5)
        sp = dd[:, 4] / np.median(dd[:, 4])
        p.clusters["vsp_geo"] = (xy, np.clip(sp, 0.5, 4.0))
    return p.clusters["vsp_geo"]


def _dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))


def _kern(d, ell, R, li=None, lj=None):
    """Tapered exponential correlation (Wendland taper, support R km). With local scales li, lj
    the Paciorek-Schervish non-stationary exponential (range ell x local VD spacing); tried
    on the live check (pull reaches further between large rural VDs), no gain, unused."""
    if li is None:
        k = np.exp(-d / ell)
    else:
        L2 = 0.5 * (li[:, None] ** 2 + lj[None, :] ** 2)
        k = (li[:, None] * lj[None, :] / L2) * np.exp(-d / (ell * np.sqrt(L2)))
    t = np.clip(1.0 - d / R, 0.0, None)
    return k * t ** 4 * (1.0 + 4.0 * d / R)


# ----------------------------------------------------------------------------- base model
def _base_mean(case, counted, base="nw", c=2.0):
    """hybrid_cf point projection; base='nw' drops its ward level (both parts)."""
    p = ms._prep(case)
    unc = ~counted
    levels, rg = (NW_LEVELS, ("muni",)) if base == "nw" else (ms.CM_LEVELS, ("muni", "ward"))
    rp = ms._share_projection(case, counted, None, scale="ratio", base="avg", levels=levels,
                              nd=0, **ms.UNC)
    gp = ms._logit_projection(case, counted, None, nd=0, base="mix", fixed="ridge", feats="all",
                              cv="group", groups=rg, new_party="reg")
    if counted.sum() > 10:
        hu, h0 = ms._extrap_sd(p, counted, np.zeros((counted.sum(), 1)),
                               np.ones((counted.sum(), 1)), return_dist=True)
        a = np.clip(1 - (hu - h0) / (c * h0), 0, 1)
    else:
        a = np.zeros(unc.sum())
    mean = rp.vd_mean.copy()
    mean[unc] = a[:, None] * gp.vd_mean[unc] + (1 - a[:, None]) * rp.vd_mean[unc]
    return mean


def _base_draws(case, counted, rng, base, nd):
    """Base predictive distribution as (vd_mean, draws). 'nw': the log-ratio cluster x muni
    model without ward level and WITHOUT the VD residual term (the spatial field and nugget
    are drawn separately). 'full': hybrid_cf's own draws (cluster_muni_ratio, re-centred)."""
    key = None
    if BASE_CACHE is not None:
        key = (id(case), counted.tobytes(), base, nd)
        if key in BASE_CACHE:
            return BASE_CACHE[key]
    if base == "nw":
        lp = ms._logit_projection(case, counted, rng, nd=nd, base="mix", new_party="const",
                                  cv="loo", resid=False, **ms.UNC, **ms._logit_kw_for(NW_LEVELS))
    else:
        lp = ms._share_projection(case, counted, rng, scale="ratio", base="avg",
                                  levels=ms.CM_LEVELS, nd=nd, **ms.UNC)
    out = (lp.vd_mean, lp.draws)
    if key is not None:
        BASE_CACHE[key] = out
    return out


def oof_residuals(case, counted, base="full", F=5, seed=0):
    """Out-of-fold base predictions for the counted VDs (random VD folds). Returns
    (r (nc, K) alr residuals, v (nc, K) sampling variance of alr(Y), fold (nc,), base mean)."""
    key = (case.name, id(case), counted.tobytes(), base, F, seed)
    if key in _OOF:
        return _OOF[key]
    idx = np.where(counted)[0]
    fold = np.random.default_rng(seed).permutation(len(idx)) % F
    P = np.zeros((len(idx), NCAT))
    for f in range(F):
        tr = counted.copy()
        tr[idx[fold == f]] = False
        P[fold == f] = _base_mean(case, tr, base)[idx[fold == f]]
    Yc = case.Y[counted].astype(float)
    out = (ms._alr(Yc) - ms._alr(P), ms._alr_var(Yc), fold, _base_mean(case, counted, base))
    if len(_OOF) > 40:
        _OOF.clear()
    _OOF[key] = out
    return out


# ----------------------------------------------------------------------------- covariance
def _nnls_small(G, b):
    """min th'G th - 2 b'th, th >= 0, by enumerating active sets (2-3 parameters)."""
    m = len(b)
    best, bth = np.inf, np.zeros(m)
    for mask in range(1, 2 ** m):
        s = np.array([(mask >> j) & 1 for j in range(m)], bool)
        try:
            th = np.linalg.solve(G[np.ix_(s, s)], b[s])
        except np.linalg.LinAlgError:
            continue
        if (th < 0).any():
            continue
        full = np.zeros(m)
        full[s] = th
        obj = full @ G @ full - 2 * b @ full
        if obj < best:
            best, bth = obj, full
    return bth, (best if np.isfinite(best) else 0.0)


def fit_cov(r, v, xy, ward, fold, comps=((5.0, ELLS),), use_ward=True, robust=3.0,
            cross_fold=True):
    """Moment fit of the residual covariance (module docstring, step 3), per party k:
        Cov(r_i, r_j) = tw_k [same ward] + sum_c ts_ck k_c(d_ij)       (i != j)
    with one tapered-exponential kernel per component c = (support R_c, grid of ranges);
    the ranges (pooled over parties) by grid search, tw, ts >= 0 by weighted NNLS on the
    products of centred, winsorised residuals of cross-fold counted pairs.
    Returns dict(mu, rc, tw (K,), ts (K, C), s2 (K,), ell (C,))."""
    nc, Kp = r.shape
    C = len(comps)
    w0 = 1.0 / (v + np.maximum(np.var(r, 0) - v.mean(0), 0.02))
    mu = (w0 * r).sum(0) / w0.sum(0)
    rc = r - mu
    sd = np.sqrt(np.average(rc ** 2, 0, w0))
    rc = np.clip(rc, -robust * sd, robust * sd)
    iu, ju = np.triu_indices(nc, 1)
    if cross_fold:
        keep = fold[iu] != fold[ju]
        iu, ju = iu[keep], ju[keep]
    d = np.sqrt(((xy[iu] - xy[ju]) ** 2).sum(1))
    sw = ((ward[iu] == ward[ju]) & (ward[iu] >= 0)).astype(float)
    Rmax = max(R for R, _ in comps)
    near = (d < Rmax) | (sw > 0)
    # far pairs carry no kernel and no ward column: they add the same constant to every
    # candidate's loss, so they are dropped
    iu, ju, d, sw = iu[near], ju[near], d[near], sw[near]
    Y = rc[iu] * rc[ju]                                   # (P, K)
    W = w0[iu] * w0[ju]
    W = W / np.maximum(W.mean(0), 1e-12)
    out = {"mu": mu, "rc": rc, "tw": np.zeros(Kp), "ts": np.zeros((Kp, C)), "s2": np.zeros(Kp),
           "ell": tuple(e[0] for _, e in comps), "n_pairs": len(d)}
    var = np.average(rc ** 2 - v, 0, w0)
    if len(d) < 10:
        out["s2"] = np.maximum(var, 0.01)
        return out
    kcols = [[_kern(d, ell, R) for ell in ells] for R, ells in comps]
    yy = (W * Y ** 2).sum(0)
    best = None
    for combo in np.ndindex(*[len(e) for _, e in comps]):
        cols = ([sw] if use_ward else []) + [kcols[c][combo[c]] for c in range(C)]
        A = np.column_stack(cols)
        tot, ths = 0.0, []
        for k in range(Kp):
            Aw = A * W[:, k:k + 1]
            th, obj = _nnls_small(A.T @ Aw, Aw.T @ Y[:, k])
            tot += (yy[k] + obj) / max(yy[k], 1e-12)          # relative SSE
            ths.append(th)
        if best is None or tot < best[0]:
            best = (tot, combo, ths)
    _, combo, ths = best
    out["ell"] = tuple(comps[c][1][combo[c]] for c in range(C))
    for k in range(Kp):
        th = ths[k]
        if use_ward:
            out["tw"][k], out["ts"][k] = th[0], th[1:]
        else:
            out["ts"][k] = th
        sh = out["tw"][k] + out["ts"][k].sum()
        cap = 0.9 * max(var[k], 0.02)
        if sh > cap:                                       # shared part cannot exceed the total
            out["tw"][k] *= cap / sh
            out["ts"][k] *= cap / sh
            sh = cap
        out["s2"][k] = max(var[k] - sh, 0.01)
    return out


# ----------------------------------------------------------------------------- the method
def _chol_all(case, ell, R):
    """Cholesky factor of the tapered kernel over all VDs (cached per case and range)."""
    key = (case.name, id(case), ell, R)
    if key not in _LCH:
        if len(_LCH) > 8:
            _LCH.clear()
        xy, _ = _geo(case)
        Kall = _kern(_dist(xy, xy), ell, R)
        Kall[np.diag_indices_from(Kall)] += 1e-6
        _LCH[key] = np.linalg.cholesky(Kall).astype(np.float32)
    return _LCH[key]


TWO_SCALE = ((3.0, (0.5, 1.0, 1.5)), (15.0, (3.0, 5.0, 8.0)))


def _spatial(case, counted, rng, *, base="full", F=5, comps=((5.0, ELLS),), gain=1.0,
             use_ward=True, nd=ms.S_DRAWS, vscale=1.0, cross_fold=True, parties=None,
             full_field=False):
    """Spatial residual pull (module docstring). gain scales the kriged shift (1 = BLUP);
    vscale scales the field + nugget draws; parties = indices of the parties to pull (None =
    all). Draws: base='nw' -> base draws without VD residual + kriging posterior of the field
    (conditional simulation) + nugget; base='full' -> hybrid_cf's draws re-centred on the
    pulled mean, plus (full_field=True) the kriging posterior of the field."""
    p = ms._prep(case)
    unc = ~counted
    nc, nu = int(counted.sum()), int(unc.sum())
    zero = case.Y[counted, :K].sum(0) == 0
    if nc < 12 or nu == 0:
        mean = _base_mean(case, counted, base)
        if nd <= 0:
            return Projection(mean)
        bm, bd = _base_draws(case, counted, rng, base, nd)
        return Projection(mean, ms._recentre(mean, Projection(bm, bd), counted))
    r, v, fold, bmean = oof_residuals(case, counted, base, F)
    xy, _ = _geo(case)
    ward = case.ward
    cv = fit_cov(r, v, xy[counted], ward[counted], fold, comps=comps, use_ward=use_ward,
                 cross_fold=cross_fold)
    if parties is not None:
        off = np.ones(K, bool)
        off[list(parties)] = False
        cv["tw"][off] = 0.0
        cv["ts"][off] = 0.0
    xc, xu = xy[counted], xy[unc]
    Dcc, Duc = _dist(xc, xc), _dist(xu, xc)
    Kcc = [_kern(Dcc, ell, R) for ell, (R, _) in zip(cv["ell"], comps)]
    Kuc = [_kern(Duc, ell, R) for ell, (R, _) in zip(cv["ell"], comps)]
    wc, wu = ward[counted], ward[unc]
    SWcc = ((wc[:, None] == wc[None]) & (wc[:, None] >= 0)).astype(float)
    SWuc = ((wu[:, None] == wc[None]) & (wu[:, None] >= 0)).astype(float)
    shift = np.zeros((nu, K))
    facs = []
    for k in range(K):
        tw, ts, s2 = cv["tw"][k], cv["ts"][k], cv["s2"][k]
        if zero[k] or (tw + ts.sum()) <= 0:
            facs.append(None)
            continue
        Cm = tw * SWcc + sum(t * Kk for t, Kk in zip(ts, Kcc))
        Cm[np.diag_indices_from(Cm)] += s2 + v[:, k]
        cf = cho_factor(Cm, lower=True)
        Cu = tw * SWuc + sum(t * Kk for t, Kk in zip(ts, Kuc))
        shift[:, k] = Cu @ cho_solve(cf, cv["rc"][:, k])
        facs.append((cf, Cu))
    eta = ms._alr(bmean[unc]) + gain * shift
    eta[:, zero] = -40.0
    mean = bmean.copy()
    mean[unc] = ms._softmax_counts(eta, p.reg[unc])
    mean[np.ix_(np.where(unc)[0], np.where(zero)[0])] = 0.0
    info = {"ell": cv["ell"], "tw": np.round(cv["tw"], 4).tolist(),
            "ts": np.round(cv["ts"], 4).tolist(), "s2": np.round(cv["s2"], 4).tolist(),
            "mean_abs_shift": np.abs(shift).mean(0).round(4).tolist()}
    if nd <= 0:
        return Projection(mean, None, info)

    bm, bd = _base_draws(case, counted, rng, base, nd)
    f32 = np.float32
    if base != "nw" and not full_field:
        return Projection(mean, ms._recentre(mean, Projection(bm, bd), counted), info)
    eps = f32(1e-3)
    lb = np.log(bd[:, unc].astype(f32) + eps)
    dev = lb[..., :K] - lb[..., [K]]
    l0 = np.log(bm[unc].astype(f32) + eps)
    dev -= (l0[:, :K] - l0[:, [K]])[None]                     # base deviation (S, nu, K)
    etas = eta.astype(f32)[None] + dev
    # kriging posterior of the shared field: conditional simulation
    Ls = [_chol_all(case, ell, R) for ell, (R, _) in zip(cv["ell"], comps)]
    widx = np.unique(ward, return_inverse=True)[1]
    Wn = int(widx.max()) + 1
    cidx, uidx = np.where(counted)[0], np.where(unc)[0]
    for k in range(K):
        if facs[k] is None:
            continue
        cf, Cu = facs[k]
        tw, ts, s2 = cv["tw"][k], cv["ts"][k], cv["s2"][k]
        f = np.zeros((case.n, nd), f32)
        for t, L in zip(ts, Ls):
            if t > 0:
                f += f32(np.sqrt(t)) * (L @ rng.standard_normal((case.n, nd), dtype=f32))
        if tw > 0:
            zw = rng.standard_normal((Wn, nd), dtype=f32)
            zw[np.unique(widx[ward < 0])] = 0.0
            f += f32(np.sqrt(tw)) * zw[widx]
        sim_c = f[cidx] + np.sqrt(s2 + v[:, k]).astype(f32)[:, None] * rng.standard_normal((nc, nd), dtype=f32)
        krig = Cu @ cho_solve(cf, sim_c.astype(float))
        etas[:, :, k] += (vscale * gain * (f[uidx] - krig)).T.astype(f32)
    if base == "nw":
        # VD nugget: resampled standardised out-of-fold residuals x sqrt(s2 + v_u)
        tot = cv["s2"] + cv["tw"] + cv["ts"].sum(1)
        zs = cv["rc"] / np.sqrt(np.maximum(tot[None] + v, 1e-6))
        zs = zs / np.maximum(zs.std(0), 1e-6)
        j = rng.integers(0, nc, size=(nd, nu))
        vu = ms._alr_var(mean[unc])
        etas += vscale * zs.astype(f32)[j] * np.sqrt(cv["s2"][None] + vu).astype(f32)[None]
    etas[..., zero] = -40.0
    draws = np.repeat(mean[None].astype(f32), nd, 0)
    draws[:, unc] = ms._softmax_counts(etas, p.reg[unc].astype(f32))
    draws[..., :K][..., zero] = 0.0
    return Projection(mean, draws, info)


def _make(doc, **kw):
    def method(case, counted, rng):
        return _spatial(case, counted, rng, **kw)
    method.__doc__ = doc
    return method


spatial_pull = _make(
    "hybrid_cf + spatial residual pull: simple kriging of hybrid_cf's out-of-fold (5 random "
    "folds) log-ratio residuals with a same-ward + tapered-exponential covariance (support "
    "R = 5 km, range by grid search) fitted on cross-fold counted pairs; draws = hybrid_cf's "
    "draws re-centred on the pulled mean.", base="full", F=5)

spatial_pull_ff = _make(
    "spatial_pull + a joint draw of the kriging posterior of the residual field (conditional "
    "simulation) added to hybrid_cf's re-centred draws (wider intervals).",
    base="full", F=5, full_field=True)

spatial_2s = _make(
    "spatial_pull with a two-scale kernel: short (R = 3 km, range 0.5-1.5) + long (R = 15 km, "
    "range 3-8) components.", base="full", F=5, comps=TWO_SCALE)

spatial_nw = _make(
    "Ward level replaced by the spatial term: hybrid_cf WITHOUT its ward level as base, kriging "
    "of its out-of-fold residuals with same-ward + two-scale spatial covariance (ward-mates as a "
    "special case); draws = base draws without VD residual + kriging posterior of the field "
    "(conditional simulation) + resampled VD nugget.", base="nw", F=5, comps=TWO_SCALE)

FINAL = ["spatial_rake_half", "spatial_rake", "spatial_pull", "spatial_pull_ff", "spatial_2s",
         "spatial_nw"]


def _raked(case, counted, rng, nd=ms.S_DRAWS, iters=10, w_nw=1.0):
    """spatial_nw's VD pattern raked to spatial_pull's municipal totals: for every
    municipality and category the uncounted VDs' projected total is set to spatial_pull's
    (iterative proportional fitting against each VD's electorate), so the province and
    municipalities keep spatial_pull's accuracy and the within-municipality (ward / VD)
    pattern comes from the ward-free base + ward / spatial kriging (w_nw < 1: the pattern is
    the log-ratio blend w_nw x spatial_nw + (1 - w_nw) x spatial_pull). Draws: spatial_nw's
    predictive distribution re-centred on the raked mean."""
    p = ms._prep(case)
    unc = ~counted
    A = _spatial(case, counted, rng, base="full", F=5, nd=0).vd_mean
    pb = _spatial(case, counted, rng, base="nw", F=5, comps=TWO_SCALE, nd=nd)
    B = pb.vd_mean.astype(float).copy()
    if w_nw < 1:
        eta = w_nw * ms._alr(B[unc]) + (1 - w_nw) * ms._alr(A[unc])
        Bb = ms._softmax_counts(eta, p.reg[unc])
        Bb[:, :K][:, B[unc][:, :K].sum(0) == 0] = 0.0
        B[unc] = Bb
    mu = p.muni[unc]
    Bu, Au, reg = B[unc], A[unc], p.reg[unc]
    T = ms._gsum(mu, Au, p.M)
    for _ in range(iters):
        S = ms._gsum(mu, Bu, p.M)
        Bu = Bu * np.where(S > 0, T / np.where(S > 0, S, 1), 1.0)[mu]
        Bu = Bu * (reg / np.maximum(Bu.sum(1), 1e-9))[:, None]
    B[unc] = Bu
    if nd <= 0:
        return Projection(B, None, pb.info)
    return Projection(B, ms._recentre(B, pb, counted), pb.info)


def spatial_rake(case, counted, rng):
    """spatial_nw's ward / VD pattern raked to spatial_pull's municipal totals (see _raked)."""
    return _raked(case, counted, rng)


def spatial_rake_half(case, counted, rng):
    """BEST. Ward / VD pattern = log-ratio average of spatial_nw (ward-free hybrid_cf + ward /
    two-scale spatial kriging of out-of-fold residuals) and spatial_pull (hybrid_cf + spatial
    kriging), raked to spatial_pull's municipal totals per category; draws = spatial_nw's
    predictive distribution (kriging posterior of the field included) re-centred."""
    return _raked(case, counted, rng, w_nw=0.5)
