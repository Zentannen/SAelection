"""Ward-mate pull on top of hybrid_cf (family: v_ward). Best method: ``ward_sp``.

Problem reported on election night 2026: in a partly counted ward the uncounted VDs seem to
ignore what their counted ward-mates show (ActionSA). hybrid_cf predicts each uncounted VD
from its own history + area-type x municipal swing; its only local term is an EB ward random
effect estimated in-sample after the cell effects, which is small.

Method (``_pull``; variants below):
 1. Base = hybrid_cf *without* its ward level (cluster_muni_ratio on km20 + muni + km20 x
    muni cells, Curtice-Firth ridge with municipality effects; same distance hand-over c = 2).
 2. Out-of-fold residuals of the counted VDs: counted wards are split into F = 5 folds (all
    counted VDs of a ward in one fold), the base is refitted without each fold and predicts
    it; r_i = alr(Y_i) - alr(base_oof_i) on the log-ratio (party vs ABSTAIN) scale. Neither a
    VD's own result nor its ward-mates' enters the prediction it is compared with.
 3. Covariance of r by moments (per party, residuals centred at their precision-weighted
    mean and winsorised at 3 SD): products of residuals over counted pairs are regressed
    (NNLS) on [same ward], optionally [same municipality x area-type cell], and
    exp(-d / ell) (d = centroid distance in km, ell from a grid 0.5-8 km):
        Cov(r_i, r_j) = tw [same ward] + tc [same cell] + ts exp(-d_ij / ell),
        Var(r_i)      = tw + tc + ts + s2 + v_i  (v_i = sampling variance of alr(Y_i)).
 4. Every uncounted VD gets the BLUP (simple kriging) of its residual from all counted
    VDs, times `gain` on the shared components. With the ward level only (``ward_eb``) this is
    exactly the James-Stein / EB ward mean
        u_w = tw sum_i r_i / (s2 + v_i) / (1 + tw sum_i 1 / (s2 + v_i)),
    i.e. shrunk to 0 by the number (and size) of counted ward-mates and tw / s2. The
    spatial term (``ward_sp``) also lets counted VDs just across a ward boundary pull, and
    gives wards with no counted VD a pull from nearby wards.
 5. Draws: base draws (cluster_muni log-ratio predictive distribution re-centred on the base
    mean) + the kriged shift + a joint draw of the kriging posterior (Cholesky of the
    posterior covariance of the residual field over all uncounted VDs, per party), so the
    ward-level variance dropped from the base comes back, correlated within wards / space,
    and shrinks where ward-mates are counted.

Finding (see notes_v_ward.md): the flagged wards' counted VDs were already ActionSA
strongholds in LGE2021 (e.g. 33000499: 37.6% then, 48.5% now, ward-mates 2-8% then), so
their residuals are small or negative and the pull hardly moves those wards; predicting
uncounted VDs from counted ward-mates' shares (what the ward map invites) is ~1.7x worse.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.linalg import solve_triangular

from ..features import K
from . import m_swing as ms
from .harness import Projection

NW_LEVELS = ("km20", "muni", "cm20")     # cluster_muni_ratio levels without the ward
_OOF_CACHE: dict = {}
BASE_CACHE: dict | None = None           # set to {} to cache base projections (tuning only)


# ----------------------------------------------------------------------------- base model
def _hybrid_base(case, counted, rng, c=2.0, nd=ms.S_DRAWS, levels=NW_LEVELS, reg_groups=("muni",)):
    """hybrid_cf with configurable pooling levels (default: no ward level anywhere)."""
    p = ms._prep(case)
    unc = ~counted
    rp = ms._share_projection(case, counted, rng, scale="ratio", base="avg", levels=levels,
                              nd=nd, **ms.UNC)
    gp = ms._logit_projection(case, counted, rng, nd=0, base="mix", fixed="ridge", feats="all",
                              cv="group", groups=reg_groups, new_party="reg")
    if counted.sum() > 10:
        hu, h0 = ms._extrap_sd(p, counted, np.zeros((counted.sum(), 1)),
                               np.ones((counted.sum(), 1)), return_dist=True)
        a = np.clip(1 - (hu - h0) / (c * h0), 0, 1)
    else:
        a = np.zeros(unc.sum())
    mean = rp.vd_mean.copy()
    mean[unc] = a[:, None] * gp.vd_mean[unc] + (1 - a[:, None]) * rp.vd_mean[unc]
    if nd <= 0:
        return Projection(mean)
    return Projection(mean, ms._recentre(mean, rp, counted))


def _folds(case, counted, F, seed=0):
    """fold label per counted VD; all counted VDs of one ward share a fold."""
    w = case.ward[counted]
    wu, inv = np.unique(w, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(wu))
    fl = (perm % F)[inv]
    # unknown ward (-1): spread individually
    unk = w < 0
    if unk.any():
        fl[unk] = np.random.default_rng(seed + 1).integers(0, F, unk.sum())
    return fl


def oof_residuals(case, counted, F=5, seed=0, base_kw=None):
    """Cross-fitted residuals of the counted VDs against the base: alr(Y) - alr(base_oof).
    Returns (r (nc, K), v (nc, K) sampling variance of alr(Y))."""
    base_kw = base_kw or {}
    key = (case.name, id(case), counted.tobytes(), F, seed, tuple(sorted(base_kw.items())))
    if key in _OOF_CACHE:
        return _OOF_CACHE[key]
    idx = np.where(counted)[0]
    fl = _folds(case, counted, F, seed)
    oof = np.zeros((len(idx), K + 1))
    rng = np.random.default_rng(0)
    for f in range(F):
        held = idx[fl == f]
        tr = counted.copy()
        tr[held] = False
        pr = _hybrid_base(case, tr, rng, nd=0, **base_kw)
        oof[fl == f] = pr.vd_mean[held]
    Yc = case.Y[counted].astype(float)
    r = ms._alr(Yc) - ms._alr(oof)
    v = ms._alr_var(Yc)
    if len(_OOF_CACHE) > 64:
        _OOF_CACHE.clear()
    _OOF_CACHE[key] = (r, v)
    return r, v


# ----------------------------------------------------------------------------- geometry
def _xy(case):
    """VD centroids in km (equirectangular around the province centre)."""
    lat0 = np.nanmean(case.lat)
    x = (case.lon - np.nanmean(case.lon)) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (case.lat - lat0) * 110.57
    return np.column_stack([x, y])


def _dist(A, B):
    return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0))


def _cell_labels(case, kind="cm20"):
    """municipality x area-type cell per VD: 'cm20' = k-means(20) past-profile cluster x
    municipality (the base model's cells); 'at' = live_check's hand area types x municipality."""
    p = ms._prep(case)
    if kind == "at":
        from ..features import K as _K
        s = case.prev_pe[:, :_K] / np.maximum(case.prev_pe[:, :_K].sum(1, keepdims=True), 1)
        da, town = s[:, 0], s[:, 1] + s[:, 3] + s[:, 4]
        at = np.where(da >= .5, 0, np.where(da >= .25, 1, np.where(town >= .7, 2, 3)))
        return at * p.M + p.muni
    return ms._clusters(p, 20, "kmeans") * p.M + p.muni


# ----------------------------------------------------------------------------- variance components
def _fit_cov(r, v, xy, ward, cell=None, use_spatial=True, ells=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
             dmax=25.0, robust=3.0):
    """Moment fit of the residual covariance, per party k:
        Cov(r_i, r_j) = tw_k [same ward] + tc_k [same cell] + ts_k exp(-d_ij / ell_k)   (i != j)
        Var(r_i)      = tw_k + tc_k + ts_k + s2_k + v_ik
    from products of centred residuals over counted pairs (same ward, same cell, or within
    dmax km), least squares with non-negative components, ell by grid search.
    Residuals are first winsorised at `robust` SD (one broken-history VD must not drag its
    ward). Returns dict with mu (K,), tw, tc, ts, ell, s2 (K,) and the clipped residuals."""
    nc, Kp = r.shape
    w0 = 1.0 / (v + np.maximum(np.var(r, 0) - v.mean(0), 0.02))
    mu = (w0 * r).sum(0) / w0.sum(0)
    rc = r - mu
    sd = np.sqrt(np.average(rc ** 2, 0, w0))
    rc = np.clip(rc, -robust * sd, robust * sd)
    iu, ju = np.triu_indices(nc, 1)
    d = np.sqrt(((xy[iu] - xy[ju]) ** 2).sum(1))
    sw = (ward[iu] == ward[ju]) & (ward[iu] >= 0)
    sc = (cell[iu] == cell[ju]) if cell is not None else np.zeros_like(sw)
    keep = sw | sc | (d < dmax)
    iu, ju, d, sw, sc = iu[keep], ju[keep], d[keep], sw[keep], sc[keep]
    from scipy.optimize import nnls
    out = {k: np.zeros(Kp) for k in ("tw", "tc", "ts", "s2")}
    out["ell"] = np.ones(Kp)
    out["mu"] = mu
    for k in range(Kp):
        y = rc[iu, k] * rc[ju, k]
        pw = np.sqrt(w0[iu, k] * w0[ju, k])        # precision weights of the pair
        pw = pw / pw.mean()
        sq = np.sqrt(pw)
        best = None
        for ell in (ells if use_spatial else (None,)):
            cols = [sw.astype(float)]
            if cell is not None:
                cols.append(sc.astype(float))
            if ell is not None:
                cols.append(np.exp(-d / ell))
            A = np.column_stack(cols)
            coef, res = nnls(A * sq[:, None], y * sq)
            if best is None or res < best[0]:
                best = (res, coef, ell)
        _, coef, ell = best
        j = 0
        out["tw"][k] = coef[j]; j += 1
        if cell is not None:
            out["tc"][k] = coef[j]; j += 1
        if ell is not None:
            out["ts"][k] = coef[j]
            out["ell"][k] = ell
        tot = np.average(rc[:, k] ** 2 - v[:, k], weights=w0[:, k])
        out["s2"][k] = max(tot - out["tw"][k] - out["tc"][k] - out["ts"][k], 0.02)
    out["rc"] = rc
    return out


def _krige(cv, v, xy_c, xy_u, ward_c, ward_u, cell_c=None, cell_u=None, gain=1.0, nd=0,
           rng=None, draw_scale=1.0):
    """BLUP of the residual field at uncounted VDs (per party) and optional joint draws of
    its posterior (nd, nu, K). gain scales the shared components (shrinkage tuning)."""
    rc = cv["rc"]
    nu, Kp = len(xy_u), rc.shape[1]
    Dcc, Duc = _dist(xy_c, xy_c), _dist(xy_u, xy_c)
    Scc = ((ward_c[:, None] == ward_c[None]) & (ward_c[:, None] >= 0)).astype(float)
    Suc = ((ward_u[:, None] == ward_c[None]) & (ward_u[:, None] >= 0)).astype(float)
    if cell_c is not None:
        Ccc = (cell_c[:, None] == cell_c[None]).astype(float)
        Cuc = (cell_u[:, None] == cell_c[None]).astype(float)
    m = np.zeros((nu, Kp))
    draws = np.zeros((nd, nu, Kp), np.float32) if nd > 0 else None
    pvar = np.zeros((nu, Kp))
    if nd > 0:
        Duu = _dist(xy_u, xy_u)
        Suu = ((ward_u[:, None] == ward_u[None]) & (ward_u[:, None] >= 0)).astype(float)
        np.fill_diagonal(Suu, 1.0)
        if cell_c is not None:
            Cuu = (cell_u[:, None] == cell_u[None]).astype(float)
    for k in range(Kp):
        tw, tc, ts, ell = gain * cv["tw"][k], gain * cv["tc"][k], gain * cv["ts"][k], cv["ell"][k]
        if tw + tc + ts <= 0:
            continue
        Kcc = tw * Scc + ts * np.exp(-Dcc / ell)
        Kuc = tw * Suc + ts * np.exp(-Duc / ell)
        if cell_c is not None:
            Kcc = Kcc + tc * Ccc
            Kuc = Kuc + tc * Cuc
        np.fill_diagonal(Kcc, tw + tc + ts)          # same ward/cell with itself
        A = Kcc + np.diag(cv["s2"][k] + v[:, k])
        L = np.linalg.cholesky(A)
        alpha = solve_triangular(L.T, solve_triangular(L, rc[:, k], lower=True), lower=False)
        m[:, k] = Kuc @ alpha
        W = solve_triangular(L, Kuc.T, lower=True)        # (nc, nu)
        pvar[:, k] = np.maximum(tw + tc + ts - (W ** 2).sum(0), 0)
        if nd > 0:
            Kuu = tw * Suu + ts * np.exp(-Duu / ell)
            if cell_c is not None:
                Kuu = Kuu + tc * Cuu
            P = Kuu - W.T @ W
            P = 0.5 * (P + P.T)
            jit = 1e-6 * max(tw + tc + ts, 1e-6)
            for _ in range(6):
                try:
                    R = np.linalg.cholesky(P + jit * np.eye(nu))
                    break
                except np.linalg.LinAlgError:
                    jit *= 10
            else:
                ev, U = np.linalg.eigh(P)
                R = U * np.sqrt(np.maximum(ev, 0))[None]
            z = rng.standard_normal((nd, nu)).astype(np.float32)
            draws[:, :, k] = draw_scale * (z @ R.T.astype(np.float32))
    return m, pvar, draws


# ----------------------------------------------------------------------------- history fill
_FILL_CACHE: dict = {}


def fill_lge_case(case):
    """Copy of `case` whose previous-LGE counts are filled for parties that stood in some
    municipalities but not in others (ActionSA in LGE2021 stood in JHB/TSH/EKU only): in a
    municipality where party k got no previous-LGE votes but did get previous-PE votes, its
    previous-LGE valid share is set to rho_k x its previous-PE valid share, rho_k = ratio of
    its LGE to PE valid shares where it stood in both (ActionSA 2021/2024: about 2.6), and the
    other parties' LGE counts are scaled down so the VD's valid total is unchanged. A party
    absent from the previous LGE everywhere (MK) is left to the base model's own fallback.
    Pre-count data only."""
    key = id(case)
    if key in _FILL_CACHE:
        return _FILL_CACHE[key][1]
    import copy
    c2 = copy.copy(case)
    L = case.prev_lge.astype(float).copy()
    P = case.prev_pe.astype(float)
    Lv, Pv = L[:, :K].sum(1), P[:, :K].sum(1)
    M = int(case.muni.max()) + 1
    for k in range(K):
        stood = np.array([L[case.muni == m, k].sum() > 0 for m in range(M)])
        if stood.all() or not stood.any():
            continue
        sm = stood[case.muni]
        rho = (L[sm, k].sum() / Lv[sm].sum()) / max(P[sm, k].sum() / Pv[sm].sum(), 1e-9)
        tgt = ~sm & (P[:, k] > 0)
        sh = np.clip(rho * P[tgt, k] / np.maximum(Pv[tgt], 1), 0, 0.9)
        others = [j for j in range(K) if j != k]
        L[np.ix_(tgt, others)] *= (1 - sh)[:, None]
        L[tgt, k] = sh * Lv[tgt]
    c2.prev_lge = L
    c2.name = case.name + "_fill"
    _FILL_CACHE[key] = (case, c2)          # keep the original alive so its id is not reused
    return c2


# ----------------------------------------------------------------------------- method
def _shift(V, rows, m):
    """add m (len(rows), K) to the alr (party vs ABSTAIN) of count rows V[..., rows, :],
    keeping each VD's total; exact zeros (party not on the ballot) stay zero."""
    eps = 1e-3
    X = V[..., rows, :].astype(np.float64)
    tot = X.sum(-1, keepdims=True)
    lg = np.log(X + eps)
    lg[..., :K] += m
    lg -= lg.max(-1, keepdims=True)
    e = np.exp(lg) * (X > 0)
    out = V.copy()
    out[..., rows, :] = (e / np.maximum(e.sum(-1, keepdims=True), 1e-30) * tot).astype(V.dtype)
    return out


def _pull(case, counted, rng, *, F=5, gain=1.0, spatial=True, cell=None, nd=ms.S_DRAWS,
          base_kw=None, draw_scale=1.0, robust=3.0, min_counted=20, krige_draws=True,
          fill_lge=False, rake=None):
    """Base (hybrid_cf without ward level) + kriged out-of-fold residuals (see module doc)."""
    if fill_lge:
        case = fill_lge_case(case)
    base_kw = dict(base_kw or {})
    bkey = (id(case), counted.tobytes(), nd, tuple(sorted(base_kw.items())))
    if BASE_CACHE is not None and bkey in BASE_CACHE:
        base = BASE_CACHE[bkey]
    else:
        base = _hybrid_base(case, counted, rng, nd=nd, **base_kw)
        if BASE_CACHE is not None:
            BASE_CACHE[bkey] = base
    unc = ~counted
    if counted.sum() < min_counted or not unc.any() or gain <= 0:
        return base
    r, v = oof_residuals(case, counted, F, base_kw=base_kw)
    xy = _xy(case)
    cl = _cell_labels(case, cell) if cell else None
    cv = _fit_cov(r, v, xy[counted], case.ward[counted], None if cl is None else cl[counted],
                  use_spatial=spatial, robust=robust)
    m, pvar, dz = _krige(cv, v, xy[counted], xy[unc], case.ward[counted], case.ward[unc],
                         None if cl is None else cl[counted], None if cl is None else cl[unc],
                         gain=gain, nd=nd if (base.draws is not None and krige_draws) else 0, rng=rng,
                         draw_scale=draw_scale)
    rows = np.where(unc)[0]
    mean = _shift(base.vd_mean.astype(float), rows, m)
    if rake:
        lab = case.muni.astype(int) if rake == "muni" else np.zeros(case.n, int)
        mean = _rake(mean, base.vd_mean.astype(float), rows, lab)
    info = {k: np.round(cv[k], 4).tolist() for k in ("tw", "tc", "ts", "ell", "s2", "mu")}
    if base.draws is None:
        return Projection(mean, None, info)
    shift = m[None] if dz is None else m[None] + dz
    draws = _shift(base.draws, rows, shift.astype(np.float32))
    if rake:
        # each draw keeps the base draw's group totals plus the posterior spread of the pulled
        # field's group totals is dropped; the ward-level redistribution stays in the draw
        draws = _rake(draws, base.draws, rows, lab)
    return Projection(mean, draws, info)


def _rake(V, base, rows, lab, iters=8):
    """Raking (IPF) of the pulled counts V[rows] so that, per group `lab` (e.g. municipality)
    and category, the uncounted VDs' total equals the base's total, while each VD keeps its
    electorate. The pull then only redistributes votes inside the group; the group level stays
    with the base model (whose aggregate ratio estimators are calibrated in count space - an
    alr-scale shift is not: exp() of a zero-mean shift adds votes, badly so for a new party).
    Works on (n, 7) or (S, n, 7) arrays."""
    X = V[..., rows, :].astype(np.float64)
    B = base[..., rows, :].astype(np.float64)
    tot = X.sum(-1, keepdims=True)
    G = int(lab.max()) + 1
    oh = np.eye(G)[lab[rows]]                                  # (nu, G)
    tgt = np.einsum("ug,...uc->...gc", oh, B)
    for _ in range(iters):
        cur = np.einsum("ug,...uc->...gc", oh, X)
        f = np.where(cur > 0, tgt / np.maximum(cur, 1e-12), 1.0)
        X = X * np.einsum("ug,...gc->...uc", oh, f)
        X = X / np.maximum(X.sum(-1, keepdims=True), 1e-12) * tot
    out = V.copy()
    out[..., rows, :] = X.astype(V.dtype)
    return out


def _make(doc, **kw):
    def method(case, counted, rng):
        return _pull(case, counted, rng, **kw)
    method.__doc__ = doc
    return method


base_nw = _make("Base only: hybrid_cf without its ward level (no pull). Comparison.", gain=0.0)
ward_eb = _make("Ward-mate pull, ward level only: EB (James-Stein) ward mean of out-of-fold "
                "residuals added to the uncounted ward-mates.", spatial=False)
ward_cell = _make("Two-level pull: ward, else municipality x area-type cell (k-means 20 x muni).",
                  spatial=False, cell="cm20")
ward_sp = _make("Ward + spatial (exponential covariance on centroid distance) kriging of "
                "out-of-fold residuals.", spatial=True)
ward_sp_cell = _make("Ward + cell + spatial kriging of out-of-fold residuals.", spatial=True,
                     cell="cm20")
ward_sp_fill = _make("ward_sp on a case whose LGE2021 ActionSA history is filled in the "
                     "municipalities where it did not stand (fill_lge_case). Exploratory.",
                     spatial=True, fill_lge=True)

ward_pull = _make("BEST. Ward + spatial kriging of out-of-fold residuals (ward_sp), raked so each "
                  "municipality's uncounted total per category stays at the base model's: the "
                  "pull only redistributes votes between the municipality's uncounted VDs.",
                  spatial=True, rake="muni")
ward_eb_rake = _make("ward_eb (ward level only, James-Stein ward mean) raked to municipality "
                     "totals like ward_pull.", spatial=False, rake="muni")
