"""Machine-learning, spatial-smoothing and conformal projection methods (bench family ``ml``).

All methods share one engine (`_run`) and differ only in the *model of the swing*.
Recommended: ``ml_mixed_gbm_shift``; runner-up without ML: ``ml_mixed_look_shift`` (see
artifacts/bench/notes_ml.md).

Target space
------------
Every VD's 7 counts are mapped to 7 unbounded targets:
    c_k = log(votes_k + .5) - log(valid + 3)          k = DA..OTHER  (log valid-vote share)
    t   = log(valid + .5) - log(abstain + .5)                         (turnout logit)
Projections are mapped back with valid = R * sigmoid(t) and shares = softmax(c) over the
parties that are active (a party with no history and no counted votes gets exactly 0).

History regimes and broken histories
------------------------------------
The history baseline B_ik is the same transform of the previous LGE and/or previous PE result,
mixed with a weight chosen on the counted VDs (minimum within-municipality swing variance).
A party has *local history* in VD i if it won >0.2% of the valid vote in VD i's municipality in a
predictor election. Then
    swing regime  (local history):   y = c_ik - B_ik    (the change)
    level regime  (no history, e.g. ActionSA 2021, MK 2024, ActionSA 2024 outside the metros):
                                     y = c_ik           (the level)
New parties are therefore projected from their counted level, the municipality / area-type /
cluster effects of the base, and the spatial or look-alike smoothing of counted VDs.
VDs with unusable history (no valid votes, or turnout >95% from s24A voting) get a
neighbour-average history / neighbour turnout (`_fix_broken`).

Base model of y
---------------
* muni   : empirical-Bayes municipality swing (DerSimonian-Laird random effects, ward-clustered
           variance), shrunk to the provincial swing.
* mixed  : crossed random effects fitted by REML (analytic gradient): municipality + area type
           (suburb / mixed / township / other from previous-PE shares) + municipality x area type
           + k-means profile cluster (8) + ward, heteroscedastic noise phi + Poisson part.
           Captures swing that varies by area type x municipality; cells / types / wards with few
           counted VDs are shrunk and get their posterior (or prior) variance in the draws.
* mixed_slope : + a learned slope on the baseline, c = mu + (1+gamma) B + ..., gamma ~ N(0,.3^2)
           (damped multiplicative swing), baseline clamped to the counted range.

Residual models on top of the base
----------------------------------
knn (geographic Nadaraya-Watson with shrinkage), look (k=8 nearest *look-alike* counted VDs in
standardised history space, same municipality preferred), gmrf (contiguity-graph propagation),
krig (Matern-3/2 GP on lon/lat, ML hyper-parameters), ridge / enet (features, optional
area-type / cluster / type x muni / shares x metro interactions, prediction features clamped to
the counted range), gbm (LightGBM, #trees by grouped CV including zero), and chains of these.
ml_gbm_change / ml_gbm_level fit LightGBM directly on the change or the level (no base).
ml_wapo / ml_wapo_int: a Washington-Post-style ridge-on-change + CQR + equicorrelated
aggregation analogue (no GREG, no clamping), with optional municipality x area-type effects.

GREG calibration
----------------
Log-scale swing models are biased for totals (the mean of log-ratios is not the log of the ratio
of totals). The out-of-fold predicted counts of the counted VDs are compared with their actual
totals per category and municipality (shrunk to the province with 20 pseudo-VDs) and the ratio
multiplies the uncounted projections - a design-consistent (GREG / ratio-estimator) correction.

Uncertainty (``draws``)
-----------------------
Ward-grouped 10-fold CV on the counted VDs gives out-of-fold (CV+ / jackknife+) residuals.
Each draw is built in target space and then in count space:
  base effects : posterior draws of the base effects (seen cells) or prior draws (unseen cells,
                 e.g. an area type with no counted look-alikes), propagated through the GREG
                 ratio so a municipality-wide shift is absorbed by the count;
  ward + VD    : extra-Poisson variance phi (robust, from OOF residuals) + the Poisson part,
                 a ward-correlated share rho (from touching counted VDs) and resampled
                 standardised OOF residual vectors (heavy tails, cross-party correlation);
  estimation   : Gaussian-multiplier ward-grouped jackknife of the GREG ratios;
  order shift  : (``shift_cal``) ACI-style pseudo-arrival check - the counted VDs least like
                 the counting so far (propensity model) are projected from the rest; their excess
                 aggregate error over random splits, scaled to the real counted/uncounted gap,
                 becomes a province-wide shock. Live, the real arrival order replaces the
                 propensity ranking.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .harness import Projection

K = 6          # parties
J = 7          # targets: 6 log valid shares + turnout logit
S_DRAWS = 300
G_FOLDS = 10
MIN_TRAIN = 15
HIST_MIN_SHARE = 0.002

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- transforms


def _to_targets(C):
    """(n, 7) counts in CATS order -> (n, 7) [log valid shares (6), turnout logit]."""
    C = np.asarray(C, float)
    valid = C[:, :K].sum(1)
    c = np.log(C[:, :K] + 0.5) - np.log(valid + 3.0)[:, None]
    t = np.log(valid + 0.5) - np.log(C[:, K] + 0.5)
    return np.column_stack([c, t])


def _to_counts(T, R, active):
    """(..., n, 7) targets -> counts (..., n, 7). Inactive parties get 0."""
    c = np.where(active, T[..., :K], -np.inf)
    c = c - c.max(-1, keepdims=True)
    e = np.exp(c)
    sh = e / e.sum(-1, keepdims=True)
    to = 1.0 / (1.0 + np.exp(-T[..., K]))
    valid = R * to
    out = np.empty(T.shape[:-1] + (7,))
    out[..., :K] = sh * valid[..., None]
    out[..., K] = R - valid
    return out


# ----------------------------------------------------------------------------- per-case prep
_CACHE: dict = {}


class Prep:
    pass


def _shares_turnout(C):
    C = np.asarray(C, float)
    v = C[:, :K].sum(1)
    sh = C[:, :K] / np.maximum(v, 1)[:, None]
    to = v / np.maximum(C.sum(1), 1)
    return sh, to


def _fix_broken(case):
    """VDs whose history is unusable get a fallback: no valid votes in a predictor election ->
    the size-scaled neighbour-average history of touching VDs (shares and turnout); turnout above
    95% (out-of-district s24A voting inflates the ballots) -> keep the party shares but replace
    the turnout by the touching VDs' mean turnout."""
    from ..features import _neighbour_fill
    out = []
    flag = np.zeros(case.n, bool)
    for C in (case.prev_lge, case.prev_pe):
        C = np.asarray(C, float).copy()
        v = C[:, :K].sum(1)
        tot = C.sum(1)
        to = v / np.maximum(tot, 1)
        zero = v < 10
        hi = (to > 0.95) & ~zero
        good = ~zero & ~hi
        if (zero | hi).any() and good.any():
            sh = _neighbour_fill(C / np.maximum(tot, 1)[:, None], ~zero, case.edges)
            C[zero] = sh[zero] * np.maximum(case.registered[zero], 1)[:, None]
            tn = _neighbour_fill(to[:, None], good, case.edges)[:, 0]
            tn = np.clip(tn, 0.05, 0.95)
            C[hi, K] = v[hi] * (1 - tn[hi]) / tn[hi]
        flag |= zero | hi
        out.append(C)
    return out[0], out[1], flag


def _prep(case):
    key = (case.name, case.n)
    if key in _CACHE:
        return _CACHE[key]
    P = Prep()
    n = case.n
    P.n = n
    P.muni = case.muni.astype(int)
    P.M = len(case.muni_codes)
    ward = case.ward.copy()
    # unknown wards: unique pseudo groups
    unk = ward < 0
    ward[unk] = -(np.arange(unk.sum()) + 1)
    P.ward = ward
    _, P.ward_idx = np.unique(ward, return_inverse=True)
    P.R = case.registered.astype(float)
    P.size = np.maximum(P.R, 50.0)
    lat0 = np.mean(case.lat)
    P.xy = np.column_stack([(case.lon - case.lon.mean()) * 111.32 * np.cos(np.deg2rad(lat0)),
                            (case.lat - lat0) * 110.57])
    prev_lge, prev_pe, P.broken = _fix_broken(case)
    H1 = _to_targets(prev_lge)
    H2 = _to_targets(prev_pe)
    s1 = prev_lge[:, :K].sum(0) > 0
    s2 = prev_pe[:, :K].sum(0) > 0
    # local history: party share in the VD's municipality in a predictor election
    has = np.zeros((n, J), bool)
    has[:, K] = True
    has1 = np.zeros((n, K), bool)
    has2 = np.zeros((n, K), bool)
    for m in range(P.M):
        i = P.muni == m
        for C, h, st in ((prev_lge, has1, s1), (prev_pe, has2, s2)):
            tot = C[i, :K].sum(0)
            sh = tot / max(tot.sum(), 1)
            h[i] = (sh > HIST_MIN_SHARE) & st
    has[:, :K] = has1 | has2
    P.has1 = np.column_stack([has1, np.ones(n, bool)])
    P.has2 = np.column_stack([has2, np.ones(n, bool)])
    P.has = has
    P.H1, P.H2 = H1, H2
    P.b1, P.b2 = _bvar(prev_lge), _bvar(prev_pe)
    # features
    sh1, to1 = _shares_turnout(prev_lge)
    sh2, to2 = _shares_turnout(prev_pe)
    e = case.edges
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    A.data[:] = 1.0
    P.A = A
    deg = np.asarray(A.sum(1)).ravel()
    Wn = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    base_hist = np.column_stack([sh1, sh2, to1, to2])
    lag = Wn @ base_hist
    lag[deg == 0] = base_hist[deg == 0]
    growth = np.log(P.size / np.maximum(prev_lge.sum(1), 50))
    onehot = (P.muni[:, None] == np.arange(P.M)[None]).astype(float)
    # log-ratio history (clipped) for linear models
    lr1 = np.where(P.has1, H1, 0.0)
    lr2 = np.where(P.has2, H2, 0.0)
    P.F_tree = np.column_stack([sh1, sh2, to1, to2, np.log(P.size), growth, P.xy,
                                case.hist_present.astype(float), P.muni, lag])
    P.F_lin = np.column_stack([sh1, sh2, to1, to2, lr1, lr2, np.log(P.size), growth, P.xy,
                               P.xy ** 2, P.xy[:, :1] * P.xy[:, 1:],
                               case.hist_present.astype(float), onehot, lag])
    P.onehot = onehot
    # area type from previous-PE valid shares (suburb DA>=50%, mixed DA 25-50%,
    # township ANC+EFF+MK>=70%, other) and k-means profile clusters on both elections' shares
    da, town = sh2[:, 0], sh2[:, 1] + sh2[:, 3] + sh2[:, 4]
    P.atype = np.where(da >= .5, 0, np.where(da >= .25, 1, np.where(town >= .7, 2, 3)))
    from sklearn.cluster import KMeans
    Hc = np.column_stack([sh1, sh2, to1, to2])
    Hc = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
    P.clus = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(Hc)
    P.cell = P.muni * 4 + P.atype
    tyoh = (P.atype[:, None] == np.arange(4)[None]).astype(float)
    celloh = (P.cell[:, None] == np.unique(P.cell)[None]).astype(float)
    cloh = (P.clus[:, None] == np.arange(8)[None]).astype(float)
    P.F_lin_int = np.column_stack([P.F_lin, tyoh, cloh, celloh,
                                   sh2[:, [0, 1, 4]] * onehot[:, [7]], sh2[:, [0, 1, 4]] * onehot[:, [8]],
                                   sh2[:, [0, 1, 4]] * onehot[:, [0]]])
    P.F_tree = np.column_stack([P.F_tree, P.atype, P.clus])
    P.tree = cKDTree(P.xy)
    # Laplacian for GMRF
    P.L = sp.diags(deg) - A
    _CACHE[key] = P
    return P


# ----------------------------------------------------------------------------- helpers


def _wmean(y, w):
    return float(np.sum(w * y) / max(np.sum(w), 1e-12))


def _folds(P, idx, G, rng):
    """ward-grouped fold labels for the training VDs idx."""
    groups = P.ward_idx[idx]
    ug = np.unique(groups)
    G = int(min(G, len(ug)))
    perm = rng.permutation(len(ug))
    lab = np.empty(len(ug), int)
    lab[perm] = np.arange(len(ug)) % G
    gmap = dict(zip(ug, lab))
    return np.array([gmap[g] for g in groups]), G


def _baseline_weight(P, idx, y_c, j, w):
    """mix weight of LGE vs PE history for target j, minimising within-muni swing variance."""
    both = P.has1[idx, j] & P.has2[idx, j]
    if both.sum() < 5:
        return 1.0 if P.has1[idx, j].mean() >= P.has2[idx, j].mean() else 0.0
    ii = idx[both]
    yy, ww = y_c[both], w[both]
    best, bw = np.inf, 1.0
    for a in np.linspace(0, 1, 11):
        d = yy - (a * P.H1[ii, j] + (1 - a) * P.H2[ii, j])
        r = d.copy()
        for m in np.unique(P.muni[ii]):
            s = P.muni[ii] == m
            r[s] -= _wmean(d[s], ww[s])
        v = _wmean(r ** 2, ww)
        if v < best - 1e-9:
            best, bw = v, a
    return bw


def _baseline(P, j, a):
    h1, h2 = P.has1[:, j], P.has2[:, j]
    B = np.where(h1 & h2, a * P.H1[:, j] + (1 - a) * P.H2[:, j],
                 np.where(h1, P.H1[:, j], P.H2[:, j]))
    return B


# ----------------------------------------------------------------------------- base model


class EBMuni:
    """Empirical-Bayes municipality mean of y (random-effects meta-analysis, ward-clustered)."""

    def fit(self, P, idx, y, w):
        M = P.M
        mu_all = _wmean(y, w)
        r_all = y - mu_all
        # pooled within-muni variance and design effect
        ybar = np.full(M, np.nan)
        v = np.full(M, np.inf)
        neff = np.zeros(M)
        res = np.zeros_like(y)
        for m in range(M):
            s = P.muni[idx] == m
            if s.sum() == 0:
                continue
            ybar[m] = _wmean(y[s], w[s])
            res[s] = y[s] - ybar[m]
            neff[m] = w[s].sum() ** 2 / np.sum(w[s] ** 2)
        has = ~np.isnan(ybar)
        sig2 = _wmean(res ** 2, w) * len(y) / max(len(y) - has.sum(), 1)
        # design effect from ward clustering (pooled): cluster-robust / naive
        g = P.ward_idx[idx]
        cl = np.bincount(g, weights=w * res)
        naive = np.sum((w * res) ** 2)
        deff = float(np.clip(np.sum(cl ** 2) / max(naive, 1e-12), 1.0, 5.0)) if naive > 0 else 1.0
        for m in np.where(has)[0]:
            v[m] = sig2 * deff / max(neff[m], 1e-9)
        vm = v[has]
        yb = ybar[has]
        wf = 1 / vm
        mu_fe = np.sum(wf * yb) / np.sum(wf)
        if has.sum() >= 2:
            Q = np.sum(wf * (yb - mu_fe) ** 2)
            den = np.sum(wf) - np.sum(wf ** 2) / np.sum(wf)
            tau2 = max((Q - (has.sum() - 1)) / max(den, 1e-12), 0.0)
        else:
            tau2 = 0.0
        tau2 = max(tau2, 0.05 ** 2)
        wr = 1 / (vm + tau2)
        mu = np.sum(wr * yb) / np.sum(wr) if has.any() else mu_all
        var_mu = 1 / np.sum(wr) if has.any() else sig2
        s_m = np.zeros(M)
        s_m[has] = tau2 / (tau2 + vm)
        eff = np.zeros(M)
        eff[has] = s_m[has] * (yb - mu)
        self.mu, self.var_mu, self.tau2, self.s_m, self.eff = mu, var_mu, tau2, s_m, eff
        self.sig2, self.deff = sig2, deff
        return self

    def predict(self, P, idx):
        return self.mu + self.eff[P.muni[idx]]


def _ebmuni_draw_dev(self, P, idx, S, rng, skip=()):
    if "muni" in skip:
        return np.zeros((S, len(idx)))
    sd = np.sqrt(self.tau2) * (1 - self.s_m)
    return (rng.standard_normal((S, P.M)) * sd[None])[:, P.muni[idx]]


EBMuni.draw_dev = _ebmuni_draw_dev
EBMuni.has_ward = False


GROUPS = {
    "muni": lambda P: P.muni,
    "atype": lambda P: P.atype,
    "cell": lambda P: P.cell,
    "clus": lambda P: P.clus,
    "mclus": lambda P: P.muni * 8 + P.clus,
    "ward": lambda P: P.ward_idx,
}


class EBMixed:
    """Crossed random-effects swing model fitted by REML:
        y_i = mu + sum_g u_g[label_g(i)] + e_i,   u_g ~ N(0, tau_g^2),  e_i ~ N(0, phi + b_i)
    with groups municipality, area type, municipality x area type (cell), profile cluster and
    ward. b_i is the Poisson part of the sampling variance (known from the counts), phi the
    extra-Poisson VD variance. Cells / types / wards with few counted VDs are shrunk to the
    higher levels, and their posterior variance is what the draws use: the intervals widen for
    area types with few counted look-alikes."""
    has_ward = True
    MAX_REML = 450

    def __init__(self, groups=("muni", "atype", "cell", "clus", "ward"), slope_x=None,
                 slope_sd=0.3):
        self.gnames = groups
        # optional learned slope on the history baseline (damped / amplified swing): a
        # continuous column x = B - mean(B) with coefficient gamma ~ N(0, slope_sd^2), i.e.
        # the swing model c = mu + (1 + gamma) B + effects. Prediction x is clamped to the
        # counted range (no linear extrapolation beyond the counted VDs).
        self.slope_x = slope_x
        self.slope_tau2 = slope_sd ** 2

    def _design(self, P, idx):
        labs = [GROUPS[g](P)[idx] for g in self.gnames]
        self.cols = []
        rows, cols, vals = [], [], []
        off = 0
        for L in labs:
            u, inv = np.unique(L, return_inverse=True)
            self.cols.append((u, off))
            rows.append(np.arange(len(idx)))
            cols.append(off + inv)
            vals.append(np.ones(len(idx)))
            off += len(u)
        gid = [np.full(len(u), gi) for gi, (u, _) in enumerate(self.cols)]
        if self.slope_x is not None:
            x = self.slope_x[idx]
            self.xrange = (x.min(), x.max())
            rows.append(np.arange(len(idx)))
            cols.append(np.full(len(idx), off))
            vals.append(x)
            self.slope_col = off
            off += 1
            gid.append(np.array([len(labs)]))
        Z = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(idx), off))
        return Z, np.concatenate(gid)

    def _tau_full(self, tau2):
        return np.r_[tau2, self.slope_tau2] if self.slope_x is not None else tau2

    @staticmethod
    def _solve(Z, gid, y, b, phi, tau2):
        W = 1.0 / (phi + b)
        ZtW = (Z.T.multiply(W)).tocsr()
        C = (ZtW @ Z).toarray() + np.diag(1.0 / tau2[gid])
        cf = cho_factor(C, lower=True)

        def sinv(v):
            return W * v - W * (Z @ cho_solve(cf, ZtW @ v))
        one = np.ones(len(y))
        s1 = sinv(one)
        mu = float(s1 @ y / s1.sum())
        r = y - mu
        sr = sinv(r)
        return W, ZtW, cf, s1, mu, r, sr

    def _reml(self, th, Z, gid, y, b):
        phi, tau2 = np.exp(th[0]), self._tau_full(np.exp(th[1:]))
        try:
            W, ZtW, cf, s1, mu, r, sr = self._solve(Z, gid, y, b, phi, tau2)
        except np.linalg.LinAlgError:
            return 1e10
        ld = np.sum(np.log(phi + b)) + np.sum(np.log(tau2[gid])) + 2 * np.sum(np.log(np.diag(cf[0])))
        return 0.5 * (ld + np.log(s1.sum()) + r @ sr)

    def _reml_grad(self, th, Z, gid, y, b):
        """REML objective (negative log restricted likelihood) and its analytic gradient in
        log-parameters."""
        phi, tau2 = np.exp(th[0]), self._tau_full(np.exp(th[1:]))
        try:
            W = 1.0 / (phi + b)
            ZtW = (Z.T.multiply(W)).tocsr()
            A = (ZtW @ Z).toarray()
            C = A + np.diag(1.0 / tau2[gid])
            cf = cho_factor(C, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        Ci = cho_solve(cf, np.eye(len(C)))

        def sinv(v):
            return W * v - W * (Z @ (Ci @ (ZtW @ v)))
        one = np.ones(len(y))
        s1 = sinv(one)
        ssum = s1.sum()
        mu = float(s1 @ y / ssum)
        Py = sinv(y - mu)
        ld = np.sum(np.log(phi + b)) + np.sum(np.log(tau2[gid])) + 2 * np.sum(np.log(np.diag(cf[0])))
        f = 0.5 * (ld + np.log(ssum) + (y - mu) @ Py)
        # gradient
        ZS1 = Z.T @ s1                                  # Z' Sigma^-1 1
        dZPZ = np.diag(A) - np.einsum("ck,kc->c", A, Ci @ A) - ZS1 ** 2 / ssum
        ZPy = Z.T @ Py
        g = np.zeros_like(th)
        for gi in range(len(th) - 1):
            m = gid == gi
            g[1 + gi] = 0.5 * (dZPZ[m].sum() - np.sum(ZPy[m] ** 2)) * tau2[gi]
        ZtW2Z = (ZtW.multiply(W[None, :]) @ Z).toarray()
        trS = W.sum() - np.sum(Ci * ZtW2Z)
        trP = trS - (s1 @ s1) / ssum
        g[0] = 0.5 * (trP - Py @ Py) * phi
        return f, g

    def fit(self, P, idx, y, w, b=None, hp=None):
        if b is None:
            b = np.full(len(y), 0.01)
        if hp is None:
            # REML for the variance components (on at most MAX_REML VDs for speed)
            sub = np.arange(len(idx))
            if len(sub) > self.MAX_REML:
                sub = np.sort(np.random.default_rng(3).choice(sub, self.MAX_REML, replace=False))
            Zs, gs = self._design(P, idx[sub])
            v = max(np.var(y), 1e-4)
            th0 = np.log(np.r_[0.5 * v, np.full(len(self.gnames), 0.1 * v)])
            bnd = [(np.log(1e-4 * v + 1e-8), np.log(5 * v + 1e-6))] + \
                  [(np.log(1e-5 * v + 1e-10), np.log(5 * v + 1e-6))] * len(self.gnames)
            o = minimize(self._reml_grad, th0, args=(Zs, gs, y[sub], b[sub]), method="L-BFGS-B",
                         jac=True, bounds=bnd, options={"maxiter": 60})
            hp = np.exp(o.x)
        Z, gid = self._design(P, idx)
        self.hp = hp
        phi, tau2 = hp[0], self._tau_full(hp[1:])
        W, ZtW, cf, s1, mu, r, sr = self._solve(Z, gid, y, b, phi, tau2)
        self.mu, self.var_mu = mu, 1.0 / s1.sum()
        self.u = cho_solve(cf, ZtW @ r)
        self.cf, self.gid, self.tau2 = cf, gid, tau2
        # for compatibility with EBMuni users
        self.s_m = np.ones(P.M)
        return self

    def _effects(self, P, idx):
        out = np.zeros(len(idx))
        for gi, g in enumerate(self.gnames):
            u, off = self.cols[gi]
            lab = GROUPS[g](P)[idx]
            pos = np.searchsorted(u, lab)
            pos = np.clip(pos, 0, len(u) - 1)
            ok = u[pos] == lab
            out[ok] += self.u[off + pos[ok]]
        if self.slope_x is not None:
            out += self.u[self.slope_col] * np.clip(self.slope_x[idx], *self.xrange)
        return out

    def predict(self, P, idx):
        return self.mu + self._effects(P, idx)

    def draw_dev(self, P, idx, S, rng, skip=()):
        """posterior (seen labels) / prior (unseen labels) deviations of the summed effects."""
        q = len(self.gid)
        L = self.cf[0]
        # C = L L^T ;  u ~ N(u_hat, C^-1)  ->  dev = L^-T z
        from scipy.linalg import solve_triangular
        z = rng.standard_normal((q, S))
        dev_u = solve_triangular(L, z, lower=True, trans="T").T          # (S, q)
        out = np.zeros((S, len(idx)))
        for gi, g in enumerate(self.gnames):
            if g in skip:
                continue
            u, off = self.cols[gi]
            lab = GROUPS[g](P)[idx]
            pos = np.clip(np.searchsorted(u, lab), 0, len(u) - 1)
            ok = u[pos] == lab
            out[:, ok] += dev_u[:, off + pos[ok]]
            if (~ok).any():
                ul, inv = np.unique(lab[~ok], return_inverse=True)
                out[:, ~ok] += (rng.standard_normal((S, len(ul))) * np.sqrt(self.tau2[gi]))[:, inv]
        if self.slope_x is not None and "slope" not in skip:
            out += dev_u[:, [self.slope_col]] * np.clip(self.slope_x[idx], *self.xrange)[None]
        if "mu" not in skip:
            out += rng.standard_normal((S, 1)) * np.sqrt(self.var_mu)
        return out


# ----------------------------------------------------------------------------- residual models
# each: fit_predict(P, tr, r, w, pred_sets, hp) -> list of predictions (one per pred set)
# tune(P, tr, r, w, folds, G) -> hp


class NoRes:
    name = "none"

    def tune(self, P, tr, r, w, fold, G, j):
        return None

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        return [np.zeros(len(p)) for p in preds]


class KNNRes:
    """Nadaraya-Watson on the k nearest counted VDs with shrinkage to 0."""
    name = "knn"
    grid = [(ell, lam) for ell in (1.0, 2.5, 6.0) for lam in (0.5, 2.0, 8.0)]
    k = 12

    def _pred(self, P, tr, r, w, p, hp):
        ell, lam = hp
        kk = min(self.k, len(tr))
        tree = cKDTree(P.xy[tr])
        d, nb = tree.query(P.xy[p], k=kk)
        if kk == 1:
            d, nb = d[:, None], nb[:, None]
        wn = w[nb] / np.mean(w)
        ker = np.exp(-d / ell) * wn
        return (ker * r[nb]).sum(1) / (ker.sum(1) + lam)

    def tune(self, P, tr, r, w, fold, G, j):
        best, bh = np.inf, self.grid[0]
        for hp in self.grid:
            err = 0.0
            for g in range(G):
                te = fold == g
                if te.all() or not te.any():
                    continue
                pr = self._pred(P, tr[~te], r[~te], w[~te], tr[te], hp)
                err += np.sum(w[te] * (r[te] - pr) ** 2)
            if err < best:
                best, bh = err, hp
        return bh

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        return [self._pred(P, tr, r, w, p, hp) for p in preds]


class LookalikeRes(KNNRes):
    """k nearest *look-alike* counted VDs: distance in standardised history space (valid shares
    and turnout of both predictor elections) plus a penalty for another municipality, so the
    same municipality is preferred. Nadaraya-Watson with shrinkage to 0, (l, lambda) by CV."""
    name = "look"
    grid = [(ell, lam) for ell in (0.5, 1.0, 2.0) for lam in (0.5, 2.0, 8.0)]
    k = 8
    muni_pen = 1.5

    def _coords(self, P):
        if not hasattr(P, "look_xy"):
            H = P.F_tree[:, :2 * K + 2]
            H = (H - H.mean(0)) / (H.std(0) + 1e-9) / np.sqrt(H.shape[1] / 4)
            P.look_xy = np.column_stack([H, self.muni_pen * P.onehot / np.sqrt(2)])
        return P.look_xy

    def _pred(self, P, tr, r, w, p, hp):
        ell, lam = hp
        X = self._coords(P)
        kk = min(self.k, len(tr))
        tree = cKDTree(X[tr])
        d, nb = tree.query(X[p], k=kk)
        if kk == 1:
            d, nb = d[:, None], nb[:, None]
        ker = np.exp(-d / ell) * (w[nb] / np.mean(w))
        return (ker * r[nb]).sum(1) / (ker.sum(1) + lam)


class GMRFRes:
    """Contiguity-graph propagation (Gaussian Markov random field interpolation)."""
    name = "gmrf"
    grid = [(a, b) for a in (0.3, 1.0, 3.0, 10.0) for b in (0.05, 0.3, 1.5)]

    def _solve(self, P, tr, r, w, hp):
        a, b = hp
        n = P.n
        wn = np.zeros(n)
        yy = np.zeros(n)
        wn[tr] = w / np.mean(w)
        yy[tr] = r
        Amat = (sp.diags(wn + b) + a * P.L).tocsc()
        return spsolve(Amat, wn * yy)

    def tune(self, P, tr, r, w, fold, G, j):
        best, bh = np.inf, self.grid[0]
        for hp in self.grid:
            err = 0.0
            for g in range(G):
                te = fold == g
                if te.all() or not te.any():
                    continue
                f = self._solve(P, tr[~te], r[~te], w[~te], hp)
                err += np.sum(w[te] * (r[te] - f[tr[te]]) ** 2)
            if err < best:
                best, bh = err, hp
        return bh

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        f = self._solve(P, tr, r, w, hp)
        return [f[p] for p in preds]


def _matern32(d, ell):
    a = np.sqrt(3.0) * d / ell
    return (1 + a) * np.exp(-a)


class KrigRes:
    """GP regression of the residual on location (Matern 3/2 + heteroscedastic nugget)."""
    name = "krig"
    max_hp_n = 700

    def _nll(self, th, D, r, nz):
        sf2, ell, sn2 = np.exp(th)
        Kx = sf2 * _matern32(D, ell) + np.diag(sn2 * nz + 1e-6)
        try:
            c, low = cho_factor(Kx, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        al = cho_solve((c, low), r)
        return 0.5 * r @ al + np.sum(np.log(np.diag(c)))

    def tune(self, P, tr, r, w, fold, G, j):
        rng = np.random.default_rng(1)
        sel = np.arange(len(tr))
        if len(sel) > self.max_hp_n:
            sel = rng.choice(sel, self.max_hp_n, replace=False)
        xy = P.xy[tr[sel]]
        D = cdist(xy, xy)
        nz = np.mean(w) / w[sel]
        rr = r[sel] - _wmean(r[sel], w[sel])
        v = max(np.var(rr), 1e-4)
        best = None
        for ell0 in (1.0, 5.0):
            th0 = np.log([0.3 * v, ell0, 0.7 * v])
            o = minimize(self._nll, th0, args=(D, rr, nz), method="L-BFGS-B",
                         bounds=[(np.log(1e-4 * v), np.log(5 * v)), (np.log(0.2), np.log(60.0)),
                                 (np.log(1e-3 * v), np.log(5 * v))])
            if best is None or o.fun < best.fun:
                best = o
        return np.exp(best.x)

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        sf2, ell, sn2 = hp
        xy = P.xy[tr]
        Kx = sf2 * _matern32(cdist(xy, xy), ell) + np.diag(sn2 * np.mean(w) / w + 1e-6)
        c = cho_factor(Kx, lower=True)
        al = cho_solve(c, r)
        return [sf2 * _matern32(cdist(P.xy[p], xy), ell) @ al for p in preds]


class LinRes:
    """Ridge (l1_ratio=0) or elastic net on standardised features, alpha by grouped CV."""

    def __init__(self, l1_ratio=0.0, feats="lin", clamp=True):
        self.clamp = clamp
        self.l1 = l1_ratio
        self.feats = feats
        self.name = ("ridge" if l1_ratio == 0 else "enet") + ("_int" if feats == "int" else "")
        self.grid = (np.logspace(0, 4, 9) if l1_ratio == 0 else np.logspace(-3.5, -0.5, 7))

    def _fitpred(self, X, y, w, Xp_list, alpha):
        # clamp prediction features to the training range: no linear extrapolation beyond the
        # counted VDs (strongholds-first orders otherwise blow up)
        if self.clamp:
            lo, hi = X.min(0), X.max(0)
            Xp_list = [np.clip(Xp, lo, hi) for Xp in Xp_list]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        ym = _wmean(y, w)
        if self.l1 == 0:
            ws = w / w.mean()
            xm = _wmean_cols(Xs, ws)
            Xc = Xs - xm
            A = Xc.T @ (Xc * ws[:, None]) + alpha * np.eye(X.shape[1])
            beta = np.linalg.solve(A, Xc.T @ (ws * (y - ym)))
            inter = ym - xm @ beta
        else:
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=alpha, l1_ratio=self.l1, max_iter=5000, tol=1e-3)
            m.fit(Xs, y, sample_weight=w / w.mean())
            beta, inter = m.coef_, m.intercept_
        return [((Xp - mu) / sd) @ beta + inter for Xp in Xp_list]

    def _F(self, P):
        return P.F_lin_int if self.feats == "int" else P.F_lin

    def tune(self, P, tr, r, w, fold, G, j):
        X = self._F(P)[tr]
        best, ba = np.inf, self.grid[-1]
        for a in self.grid:
            err = 0.0
            for g in range(G):
                te = fold == g
                if te.all() or not te.any():
                    continue
                pr = self._fitpred(X[~te], r[~te], w[~te], [X[te]], a)[0]
                err += np.sum(w[te] * (r[te] - pr) ** 2)
            if err < best:
                best, ba = err, a
        return ba

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        F = self._F(P)
        return self._fitpred(F[tr], r, w, [F[p] for p in preds], hp)


def _wmean_cols(X, w):
    return (X * w[:, None]).sum(0) / w.sum()


class GBMRes:
    """LightGBM, shallow + strongly regularised; #trees by grouped CV (0 allowed)."""
    name = "gbm"
    iters = (0, 10, 25, 50, 100, 200, 400)

    def __init__(self, features="tree"):
        self.features = features

    def _X(self, P, idx):
        return P.F_tree[idx]

    def _params(self, n):
        return dict(objective="regression", learning_rate=0.03, num_leaves=6, max_depth=3,
                    min_data_in_leaf=int(max(10, n // 25)), lambda_l2=10.0, feature_fraction=0.7,
                    bagging_fraction=0.8, bagging_freq=1, min_data_in_bin=3, max_bin=63,
                    verbose=-1, num_threads=1, seed=1, deterministic=True,
                    linear_tree=False)

    def _train(self, X, y, w, n_iter):
        import lightgbm as lgb
        ds = lgb.Dataset(X, y, weight=w / w.mean(), free_raw_data=False,
                         params={"min_data_in_bin": 3, "max_bin": 63, "verbose": -1})
        return lgb.train(self._params(len(y)), ds, num_boost_round=n_iter)

    def _pred(self, model, X, it, ym):
        if it == 0 or model is None:
            return np.full(len(X), ym)
        return model.predict(X, num_iteration=it)

    def tune(self, P, tr, r, w, fold, G, j):
        X = self._X(P, tr)
        errs = np.zeros(len(self.iters))
        for g in range(G):
            te = fold == g
            if te.all() or not te.any():
                continue
            ym = _wmean(r[~te], w[~te])
            mdl = self._train(X[~te], r[~te], w[~te], max(self.iters))
            for q, it in enumerate(self.iters):
                pr = self._pred(mdl, X[te], it, ym)
                errs[q] += np.sum(w[te] * (r[te] - pr) ** 2)
        # one-standard-error-ish rule: prefer fewer trees unless clearly better
        best = int(np.argmin(errs))
        tol = errs[best] * 1.005
        for q in range(len(self.iters)):
            if errs[q] <= tol:
                return self.iters[q]
        return self.iters[best]

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        ym = _wmean(r, w)
        if hp == 0:
            return [np.full(len(p), ym) for p in preds]
        mdl = self._train(self._X(P, tr), r, w, hp)
        return [mdl.predict(self._X(P, p)) for p in preds]


class Chain:
    """Residual models applied in sequence (e.g. GBM then kriging of what is left)."""

    def __init__(self, *models):
        self.models = models
        self.name = "+".join(m.name for m in models)

    def tune(self, P, tr, r, w, fold, G, j):
        hps = []
        rr = r.copy()
        for m in self.models:
            hp = m.tune(P, tr, rr, w, fold, G, j)
            hps.append(hp)
            # in-sample fit for the next stage (OOF would be cleaner; this is the usual stacking)
            oof = np.zeros_like(rr)
            for g in range(G):
                te = fold == g
                if te.all() or not te.any():
                    continue
                oof[te] = m.fit_predict(P, tr[~te], rr[~te], w[~te], [tr[te]], hp, j)[0]
            rr = rr - oof
        return hps

    def fit_predict(self, P, tr, r, w, preds, hp, j):
        out = [np.zeros(len(p)) for p in preds]
        rr = r.copy()
        for m, h in zip(self.models, hp):
            pr = m.fit_predict(P, tr, rr, w, [tr] + list(preds), h, j)
            rr = rr - pr[0]
            for q in range(len(preds)):
                out[q] = out[q] + pr[q + 1]
        return out


# ----------------------------------------------------------------------------- engine


BASES = {
    "mixed": ("muni", "atype", "cell", "clus", "ward"),
    "mixed_nw": ("muni", "atype", "cell", "clus"),
    "mixed_mc": ("muni", "atype", "cell", "clus", "mclus", "ward"),
}


def _fit_base(bkind, P, tr, y, w, b, bhp=None, x=None):
    if bkind == "muni":
        return EBMuni().fit(P, tr, y, w)
    slope = bkind.endswith("_slope") and x is not None
    groups = BASES[bkind.replace("_slope", "")]
    return EBMixed(groups, slope_x=x if slope else None).fit(P, tr, y, w, b=b, hp=bhp)


def _fit_predict(P, bkind, tr, y, w, b, bhp, resmodel, rhp, preds, j, use_base, tune=None,
                 x=None):
    """base + residual model on training VDs tr -> predictions for each index set in preds."""
    base = _fit_base(bkind, P, tr, y, w, b, bhp, x)
    if use_base:
        b_tr = base.predict(P, tr)
        b_p = [base.predict(P, p) for p in preds]
    else:
        m0 = _wmean(y, w)
        b_tr = np.full(len(tr), m0)
        b_p = [np.full(len(p), m0) for p in preds]
    r = y - b_tr
    if tune is not None:
        rhp = resmodel.tune(P, tr, r, w, tune[0], tune[1], j)
    rp = resmodel.fit_predict(P, tr, r, w, preds, rhp, j)
    return [bb + q for bb, q in zip(b_p, rp)], base, rhp


def _run(case, counted, rng, resmodel, use_base=True, level_all=False, base="muni",
         n_draws=S_DRAWS, greg="muni", shift_cal=False, use_jack=False, infl=1.0):
    """Shared engine: per target and history regime fit base + residual model on the counted
    VDs, ward-grouped CV for out-of-fold residuals / jackknife, GREG calibration, draws."""
    t0 = time.time()
    P = _prep(case)
    rng = np.random.default_rng(12345)
    ci = np.where(counted)[0]
    ui = np.where(~counted)[0]
    n, nu = P.n, len(ui)
    Yc = case.Y[ci]                                  # only counted rows are ever read
    Tc = _to_targets(Yc)
    bY = _bvar(Yc)
    w_c = P.size[ci]
    active = np.ones(K, bool)
    for k in range(K):
        if not P.has[:, k].any() and Yc[:, k].sum() == 0:
            active[k] = False
    fold_c, G = _folds(P, ci, G_FOLDS, rng)
    yhat = np.zeros((n, J))                          # projected target (uncounted rows)
    jack = np.zeros((G, nu, J)) if use_jack else None
    oof = np.zeros((len(ci), J))                     # out-of-fold residual, counted VDs
    base_dev = np.zeros((n_draws, nu, J))            # posterior deviations of the base effects
    base_dev_c = np.zeros((n_draws, len(ci), J))     # same draws for the counted VDs (for GREG)
    info = {"hp": {}, "w_hist": {}}
    skip = ("mu",) if greg else ()
    for j in range(J):
        if j < K and not active[j]:
            yhat[:, j] = -30.0
            continue
        hm = P.has[:, j]
        regimes = []
        if level_all:
            regimes.append(("level", np.ones(n, bool)))
        else:
            if hm.any():
                regimes.append(("swing", hm))
            if (~hm).any():
                regimes.append(("level", ~hm))
        a = _baseline_weight(P, ci[hm[ci]], Tc[hm[ci], j], j, w_c[hm[ci]]) if hm[ci].any() else 1.0
        info["w_hist"][j] = a
        B = _baseline(P, j, a)
        bH = np.where(P.has1[:, j], P.b1[:, j], P.b2[:, j])
        for rname, rmask in regimes:
            sel_c = rmask[ci]
            sel_u = rmask[ui]
            if not sel_u.any() and not sel_c.any():
                continue
            swing = rname == "swing" and not level_all
            if sel_c.sum() < MIN_TRAIN:
                sel_c = hm[ci] if (swing and hm[ci].any()) else np.ones(len(ci), bool)
            tr = ci[sel_c]
            off = B if swing else np.zeros(n)
            y = Tc[sel_c, j] - off[tr]
            b = bY[sel_c, j] + (bH[tr] if swing else 0.0)
            w = w_c[sel_c]
            f_tr = fold_c[sel_c]
            pu = ui[sel_u]
            xs = (B - B[hm].mean()) if swing else None
            (pred_u,), bfit, rhp = _fit_predict(P, base, tr, y, w, b, None, resmodel, None, [pu], j,
                                                use_base, tune=(f_tr, G), x=xs)
            bhp = getattr(bfit, "hp", None)
            info["hp"][f"{j}_{rname}"] = rhp
            if bhp is not None:
                info["hp"][f"{j}_{rname}_base"] = np.round(np.sqrt(bhp), 3).tolist()
                if getattr(bfit, "slope_x", None) is not None:
                    info["hp"][f"{j}_{rname}_slope"] = round(1 + float(bfit.u[bfit.slope_col]), 3)
            yhat[pu, j] = pred_u + off[pu]
            pc_reg = ci[rmask[ci]]
            fc_reg = fold_c[rmask[ci]]
            if use_base:
                dd = bfit.draw_dev(P, np.r_[pu, pc_reg], n_draws, rng, skip=skip)
                base_dev[:, sel_u, j] = dd[:, :len(pu)]
                base_dev_c[:, rmask[ci], j] = dd[:, len(pu):]
            # folds: OOF residuals for this regime's counted VDs (+ jackknife for uncounted)
            for g in range(G):
                keep = f_tr != g
                te_c = pc_reg[fc_reg == g]
                if keep.sum() < 3:
                    if use_jack:
                        jack[g, sel_u, j] = yhat[pu, j]
                    continue
                outs, _, _ = _fit_predict(P, base, tr[keep], y[keep], w[keep], b[keep], bhp, resmodel,
                                          rhp, [pu, te_c], j, use_base, x=xs)
                if use_jack:
                    jack[g, sel_u, j] = outs[0] + off[pu]
                if len(te_c):
                    pos = np.searchsorted(ci, te_c)
                    oof[pos, j] = Tc[pos, j] - (outs[1] + off[te_c])
    # ------------------------------------------------------------------ point + draws
    V = np.zeros((n, 7))
    V[ci] = Yc
    Vp = _to_counts(yhat[ui], P.R[ui], active)
    oofC = _to_counts(Tc - oof, P.R[ci], active)
    if greg:
        # ratio (GREG-type) calibration: the out-of-fold point predictions of the counted VDs,
        # summed, must match their actual totals (per municipality, shrunk to the province)
        corr = _greg_ratio(P, ci, ui, Yc, oofC, greg)
        info["greg"] = np.round(np.exp(np.log(corr).mean(0)), 3).tolist()
        Vp = Vp * corr
        Vp *= (P.R[ui] / np.maximum(Vp.sum(-1), 1e-9))[:, None]
    D = np.zeros((n_draws, n, 7))
    D[:, ci] = Yc[None]
    shock = _shift_shock(P, ci, ui, Yc, active, info) if shift_cal else None
    Du = _draw2(P, ci, ui, Yc, Tc, yhat, oof, oofC, Vp, base_dev, base_dev_c, jack, active, fold_c, G,
                greg, rng, n_draws, info, infl=infl, shock=shock,
                ward_in_base=use_base and base != "muni" and "ward" in BASES[base.replace("_slope", "")])
    D[:, ui] = Du
    V[ui] = Vp
    info["seconds"] = round(time.time() - t0, 2)
    return Projection(V, D, info)


def _bvar(C):
    """Poisson part of the sampling variance of the 7 targets given (expected) counts C."""
    C = np.asarray(C, float)
    valid = C[..., :K].sum(-1)
    b = np.empty(C.shape[:-1] + (J,))
    b[..., :K] = 1.0 / (C[..., :K] + 0.5)
    b[..., K] = 1.0 / (valid + 0.5) + 1.0 / (C[..., K] + 0.5)
    return b


def _phi_robust(r, b):
    """extra-Poisson variance phi such that median(r^2 / (phi + b)) = median of chi2_1."""
    q = r ** 2
    if np.all(q == 0):
        return 0.0
    f = lambda ph: np.median(q / (ph + b)) - 0.4549
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while f(hi) > 0 and hi < 1e3:
        hi *= 4
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _draw2(P, ci, ui, Yc, Tc, yhat, oof, oofC, Vp, base_dev, base_dev_c, jack, active, fold_c, G,
           greg, rng, S, info, infl=1.0, shock=None, ward_in_base=False):
    """Draws of the uncounted VDs' counts (S, nu, 7), centred on Vp.

    realisation : target-space noise with extra-Poisson variance phi_j (robust, from the
                  out-of-fold residuals) + the Poisson part 1/mu, split into a ward-correlated
                  share rho_j (from residual covariance of touching counted VDs) and a VD part
                  (resampled standardised OOF residual vectors: heavy tails, cross-party corr.)
    municipal   : N(0, tau^2 (1 - s_m)^2) of the shrunken municipality swing
    estimation  : Gaussian-multiplier ward-grouped jackknife of the GREG ratios (per category,
                  per municipality) -> the provincial / municipal swing error
    infl scales the correlated parts (ward, municipal, estimation)."""
    nu, nc = len(ui), len(ci)
    bc = _bvar(oofC)
    bu = _bvar(Vp)
    phi = np.zeros(J)
    rho = np.zeros(J)
    jj = [j for j in range(J) if j == K or active[j]]
    # touching pairs of counted VDs
    pos = -np.ones(P.n, int)
    pos[ci] = np.arange(nc)
    e = np.array(P.A.nonzero()).T
    e = e[(e[:, 0] < e[:, 1])]
    e = e[(pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)]
    pa, pb = pos[e[:, 0]], pos[e[:, 1]]
    for j in jj:
        phi[j] = _phi_robust(oof[:, j], bc[:, j])
        if len(pa) >= 5 and phi[j] > 0:
            ra, rb = oof[pa, j], oof[pb, j]
            # robust covariance: trim extreme products
            prod = ra * rb
            lim = np.quantile(np.abs(prod), 0.95)
            cv = np.mean(np.clip(prod, -lim, lim))
            r_hat = np.clip(cv / phi[j], 0.0, 0.9)
            npairs = len(pa)
            rho[j] = (npairs * r_hat + 30 * 0.3) / (npairs + 30)
        else:
            rho[j] = 0.3
    info["phi"] = np.round(phi, 3).tolist()
    info["rho"] = np.round(rho, 3).tolist()
    z = np.zeros_like(oof)
    for j in jj:
        z[:, j] = oof[:, j] / np.sqrt(phi[j] + bc[:, j])
        z[:, j] -= np.median(z[:, j])
    # realisation
    wu = P.ward_idx[ui]
    uw, winv = np.unique(wu, return_inverse=True)
    ward = rng.standard_normal((S, len(uw), J)) * np.sqrt(rho * phi)[None, None]
    pick = rng.integers(0, nc, size=(S, nu))
    vd = z[pick] * np.sqrt((1 - rho) * phi + bu)[None]
    if ward_in_base:
        ward[:] = 0.0          # ward effects are drawn from the mixed model's posterior instead
    T = yhat[ui][None] + infl * (ward[:, winv] + base_dev) + vd
    if jack is not None:
        dev = jack - jack.mean(0, keepdims=True)
        xi = rng.standard_normal((S, G))
        T = T + infl * np.sqrt((G - 1) / G) * np.einsum("sg,gnj->snj", xi, dev)
    if shock is not None:
        T = T + rng.standard_normal((S, 1, J)) * shock[None, None]
    T[..., :K] = np.where(active, T[..., :K], -30.0)
    C = _to_counts(T, P.R[ui][None, :], active)
    if greg:
        # propagate the base-effect draws through the GREG calibration: perturb the counted VDs'
        # out-of-fold predictions with the same effect draws and recompute the ratios per draw,
        # so a municipality-wide shift is absorbed by the count while cell / type / ward effects
        # that differ between counted and uncounted VDs are not
        Vp0 = _to_counts(yhat[ui], P.R[ui], active)
        C *= (Vp0 / np.maximum(C.mean(0), 1e-9))[None]
        Tcs = (Tc - oof)[None] + infl * base_dev_c
        Tcs[..., :K] = np.where(active, Tcs[..., :K], -30.0)
        Cc = _to_counts(Tcs, P.R[ci][None, :], active)
        Cc *= (oofC / np.maximum(Cc.mean(0), 1e-9))[None]
        C *= _greg_ratio(P, ci, ui, Yc, Cc, greg)
    else:
        C *= (Vp / np.maximum(C.mean(0), 1e-9))[None]
    # estimation error of the swing / GREG ratios
    if greg:
        L = np.zeros((G, nu, 7))
        for g in range(G):
            k = fold_c != g
            L[g] = np.log(_greg_ratio(P, ci[k], ui, Yc[k], oofC[k], greg))
        dev = L - L.mean(0, keepdims=True)
        xi = rng.standard_normal((S, G))
        eps = np.sqrt((G - 1) / G) * np.einsum("sg,gnc->snc", xi, dev)
        C *= np.exp(infl * eps)
    C *= (P.R[ui][None] / np.maximum(C.sum(-1), 1e-9))[..., None]
    return C


def _cheap_projection(P, tr, Ytr, pr, active, greg="muni"):
    """EB municipality swing + GREG (in-sample), used for the pseudo-arrival checks."""
    Ttr = _to_targets(Ytr)
    w = P.size[tr]
    yh = np.full((len(pr), J), -30.0)
    yin = np.full((len(tr), J), -30.0)
    for j in range(J):
        if j < K and not active[j]:
            continue
        hm = P.has[:, j]
        a = (_baseline_weight(P, tr[hm[tr]], Ttr[hm[tr], j], j, w[hm[tr]])
             if hm[tr].sum() > 2 else 1.0)
        B = _baseline(P, j, a)
        for swing, mask in ((True, hm), (False, ~hm)):
            st, sq = mask[tr], mask[pr]
            if not sq.any() and not st.any():
                continue
            if st.sum() < 5:
                if swing and hm[tr].sum() >= 1:
                    st = hm[tr]
                else:
                    st, swing = np.ones(len(tr), bool), False
            y = Ttr[st, j] - (B[tr[st]] if swing else 0.0)
            eb = EBMuni().fit(P, tr[st], y, w[st])
            yh[sq, j] = eb.predict(P, pr[sq]) + (B[pr[sq]] if swing else 0.0)
            yin[mask[tr], j] = eb.predict(P, tr[mask[tr]]) + (B[tr[mask[tr]]] if swing else 0.0)
    C = _to_counts(yh, P.R[pr], active)
    Cin = _to_counts(yin, P.R[tr], active)
    if greg:
        C = C * _greg_ratio(P, tr, pr, Ytr, Cin, greg)
        C *= (P.R[pr] / np.maximum(C.sum(-1), 1e-9))[:, None]
    return C


def _agg_shares(C):
    t = C.sum(0)
    return np.r_[t[:K] / max(t[:K].sum(), 1e-9), t[:K].sum() / max(t.sum(), 1e-9)]


def _shift_shock(P, ci, ui, Yc, active, info, frac=0.3, n_rand=6, max_ratio=4.0):
    """Order-shift (ACI-style pseudo-arrival) check. A propensity model (counted vs uncounted on
    pre-count features) ranks the counted VDs; the 30% least 'counted-like' are treated as the
    late arrivals and projected from the rest with the cheap swing+GREG model. Their aggregate
    error is compared with that of random splits of the same size; the excess, scaled by how much
    further the real uncounted VDs lie along the propensity axis, becomes a province-wide shock
    sd per target (log-share scale). In a live run, the real arrival order replaces the
    propensity ranking (score each new batch against its pre-arrival projection)."""
    from sklearn.linear_model import LogisticRegression
    X = P.F_lin
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    yb = np.zeros(P.n)
    yb[ci] = 1
    lr = LogisticRegression(C=0.05, max_iter=2000)
    lr.fit(Xs, yb, sample_weight=np.where(yb == 1, 0.5 / len(ci), 0.5 / len(ui)) * P.n)
    eta = Xs @ lr.coef_.ravel()
    ec = eta[ci]
    order = np.argsort(-ec)
    nl = max(int(frac * len(ci)), 5)
    late, early = order[-nl:], order[:-nl]
    d_pseudo = ec[early].mean() - ec[late].mean()
    d_real = ec.mean() - eta[ui].mean()
    ratio = float(np.clip(d_real / max(d_pseudo, 1e-6), 0.0, max_ratio))
    def err(e_idx, l_idx):
        C = _cheap_projection(P, ci[e_idx], Yc[e_idx], ci[l_idx], active)
        return _agg_shares(C) - _agg_shares(Yc[l_idx])
    e_shift = err(early, late)
    rng = np.random.default_rng(7)
    er = []
    for _ in range(n_rand):
        perm = rng.permutation(len(ci))
        er.append(err(perm[:-nl], perm[-nl:]))
    s2 = np.mean(np.square(er), 0)
    excess = np.sqrt(np.maximum(e_shift ** 2 - s2, 0.0)) * ratio
    tot = _agg_shares(Yc)
    sd = np.zeros(J)
    sd[:K] = np.where(active, excess[:K] / np.maximum(tot[:K], 0.01), 0.0)
    to = tot[K]
    sd[K] = excess[K] / max(to * (1 - to), 0.05)
    info["shift"] = {"ratio": round(ratio, 2), "excess_pp": np.round(100 * excess, 2).tolist()}
    return sd


def _greg_ratio(P, ci, ui, Yc, oofC, mode, kappa=20.0):
    """(..., len(ui), 7) multiplicative correction actual/predicted of the counted VDs' totals.
    oofC (..., nc, 7) are out-of-fold predicted counts of the counted VDs (leading dims = draws).
    mode 'prov': one ratio per category; 'muni': per municipality, shrunk to the province ratio
    with a pseudo-count of kappa average VDs."""
    tot_y = Yc.sum(0)
    tot_p = np.maximum(oofC.sum(-2), 1e-9)
    r0 = np.where(tot_y > 0, tot_y / tot_p, 1.0)
    if mode == "prov":
        return np.broadcast_to(r0[..., None, :], r0.shape[:-1] + (len(ui), 7))
    lead = oofC.shape[:-2]
    out = np.broadcast_to(r0[..., None, :], lead + (P.M, 7)).copy()
    avg = oofC.sum(-1).mean(-1)
    mix = oofC.sum(-2) / np.maximum(oofC.sum((-2, -1)), 1e-9)[..., None]
    prior = kappa * avg[..., None] * mix
    mc = P.muni[ci]
    for m in range(P.M):
        s = mc == m
        if not s.any():
            continue
        a = Yc[s].sum(0) + prior * r0
        b = oofC[..., s, :].sum(-2) + prior
        out[..., m, :] = np.where(b > 0, a / np.maximum(b, 1e-9), 1.0)
    return out[..., P.muni[ui], :]


# ----------------------------------------------------------------------------- methods

def ml_swing_muni(case, counted, rng):
    """EB municipality log-ratio swing (+ level model for parties with no history), GREG
    calibration, conformal/posterior draws. The simplest member of the family."""
    return _run(case, counted, rng, NoRes())


def ml_mixed(case, counted, rng):
    """REML crossed random-effects swing: municipality + area type + municipality x area type +
    profile cluster + ward, GREG calibration. Captures swing that varies by area type x muni."""
    return _run(case, counted, rng, NoRes(), base="mixed")


def ml_mixed_nw(case, counted, rng):
    """(ablation) ml_mixed without the ward effect."""
    return _run(case, counted, rng, NoRes(), base="mixed_nw")


def ml_mixed_slope(case, counted, rng):
    """Mixed swing with a learned (damped) slope on the history baseline, prior N(1, 0.3^2)."""
    return _run(case, counted, rng, NoRes(), base="mixed_slope")


def ml_mixed_look(case, counted, rng):
    """Mixed swing + k-nearest look-alike (history profile, same muni preferred) smoothing."""
    return _run(case, counted, rng, LookalikeRes(), base="mixed")


def ml_mixed_slope_look(case, counted, rng):
    """Mixed swing with learned slope + look-alike kNN smoothing of what is left."""
    return _run(case, counted, rng, LookalikeRes(), base="mixed_slope")


def ml_look(case, counted, rng):
    """Municipality swing + k-nearest look-alike smoothing."""
    return _run(case, counted, rng, LookalikeRes())


def ml_knn(case, counted, rng):
    """Municipality swing + kNN distance-weighted smoothing of counted VDs' swing residuals."""
    return _run(case, counted, rng, KNNRes())


def ml_gmrf(case, counted, rng):
    """Municipality swing + contiguity-graph (GMRF) propagation of swing residuals."""
    return _run(case, counted, rng, GMRFRes())


def ml_krig(case, counted, rng):
    """Municipality swing + Gaussian-process (kriging) of swing residuals on lon/lat."""
    return _run(case, counted, rng, KrigRes())


def ml_mixed_knn(case, counted, rng):
    """Mixed (type x muni, ward) swing + kNN smoothing of what is left."""
    return _run(case, counted, rng, KNNRes(), base="mixed")


def ml_mixed_gmrf(case, counted, rng):
    """Mixed (type x muni, ward) swing + contiguity-graph propagation of what is left."""
    return _run(case, counted, rng, GMRFRes(), base="mixed")


def ml_mixed_krig(case, counted, rng):
    """Mixed (type x muni, ward) swing + kriging of what is left."""
    return _run(case, counted, rng, KrigRes(), base="mixed")


def ml_ridge(case, counted, rng):
    """Municipality swing + ridge regression of the residual on features (no interactions)."""
    return _run(case, counted, rng, LinRes(0.0))


def ml_ridge_int(case, counted, rng):
    """Municipality swing + ridge with area-type, cluster, type x muni and shares x metro
    interactions."""
    return _run(case, counted, rng, LinRes(0.0, "int"))


def ml_enet(case, counted, rng):
    """Municipality swing + elastic net (l1_ratio 0.5) of the residual, interaction features."""
    return _run(case, counted, rng, LinRes(0.5, "int"))


def ml_mixed_ridge(case, counted, rng):
    """Mixed swing + ridge (interaction features) of what is left."""
    return _run(case, counted, rng, LinRes(0.0, "int"), base="mixed")


def ml_gbm(case, counted, rng):
    """Hybrid: municipality swing + LightGBM of the residual (trees chosen by CV, 0 allowed)."""
    return _run(case, counted, rng, GBMRes())


def ml_mixed_gbm(case, counted, rng):
    """Hybrid: mixed swing + LightGBM of what is left (trees by CV, 0 allowed)."""
    return _run(case, counted, rng, GBMRes(), base="mixed")


def ml_gbm_krig(case, counted, rng):
    """Hybrid: municipality swing + LightGBM residual + kriging of what is left."""
    return _run(case, counted, rng, Chain(GBMRes(), KrigRes()))


def ml_ridge_krig(case, counted, rng):
    """Hybrid: municipality swing + ridge (interactions) + kriging of what is left."""
    return _run(case, counted, rng, Chain(LinRes(0.0, "int"), KrigRes()))


def ml_gbm_change(case, counted, rng):
    """LightGBM directly on the change vs history (no municipality base; muni as a feature)."""
    g = GBMRes()
    g.iters = (25, 50, 100, 200, 400, 800)
    return _run(case, counted, rng, g, use_base=False)


def ml_gbm_level(case, counted, rng):
    """LightGBM directly on the level of each target (no base, no history offset)."""
    g = GBMRes()
    g.iters = (25, 50, 100, 200, 400, 800)
    return _run(case, counted, rng, g, use_base=False, level_all=True)


def ml_swing_nogreg(case, counted, rng):
    """(ablation) municipality swing without the GREG ratio calibration."""
    return _run(case, counted, rng, NoRes(), greg=None)


def ml_swing_shift(case, counted, rng):
    """Municipality swing + order-shift (pseudo-arrival, ACI-style) shock in the draws."""
    return _run(case, counted, rng, NoRes(), shift_cal=True)


def ml_mixed_shift(case, counted, rng):
    """Mixed swing + order-shift (pseudo-arrival, ACI-style) shock in the draws."""
    return _run(case, counted, rng, NoRes(), base="mixed", shift_cal=True)


# ----------------------------------------------------------------------------- WaPo-style


def _wq(x, w, q):
    """weighted quantile."""
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(cw, q), len(x) - 1)]


def _wapo_features(P, j, B, swing, interactions=False):
    """compact covariate set in the spirit of the WaPo model: own baseline, previous-LGE shares
    and turnout, VD size, municipality fixed effects."""
    sh1 = P.F_tree[:, :K]
    to1 = P.F_tree[:, 2 * K]
    cols = [sh1, to1[:, None], np.log(P.size)[:, None], P.onehot[:, 1:]]
    if interactions:
        cols.append((P.cell[:, None] == np.unique(P.cell)[None]).astype(float))
    if swing:
        cols.insert(0, B[:, None])
    return np.column_stack(cols)


def ml_wapo(case, counted, rng, alpha_q=0.02, interactions=False):
    """Faithful-as-possible Washington Post (Cherian & Bronner elex-model) analogue:
    * each VD is compared with its own previous result: target = change in log valid share per
      party (level for parties without history) and change in turnout logit;
    * point: ridge regression of the change on covariates (own baseline, previous shares and
      turnout, VD size, municipality fixed effects), weighted by VD size (WaPo 2024);
    * unit intervals: conformalized quantile regression (WaPo 2020): linear quantile regressions
      (5%, 95%) on a random half of the counted VDs (ward-grouped split), size-weighted conformal
      correction from the other half;
    * aggregation: the unit intervals define a sd per VD; errors are equicorrelated within a
      municipality (rho estimated by one-way ANOVA of the standardised calibration scores) plus a
      contest-level (province) random effect, as in WaPo's precinct update / 2024 bootstrap.
    No GREG calibration and no feature clamping (those are our additions in the engine).
    interactions=True adds municipality x area-type fixed effects (ml_wapo_int)."""
    from sklearn.linear_model import QuantileRegressor
    t0 = time.time()
    P = _prep(case)
    rng = np.random.default_rng(12345)
    ci, ui = np.where(counted)[0], np.where(~counted)[0]
    n, nu = P.n, len(ui)
    Yc = case.Y[ci]
    Tc = _to_targets(Yc)
    w_c = P.size[ci]
    active = np.array([P.has[:, k].any() or Yc[:, k].sum() > 0 for k in range(K)])
    fold_c, G = _folds(P, ci, G_FOLDS, rng)
    half = np.isin(fold_c, np.arange(0, G, 2))          # train half / calibration half
    yhat = np.full((n, J), -30.0)
    sd_u = np.zeros((nu, J))
    oof = np.zeros((len(ci), J))
    rho = np.zeros(J)
    rho0 = np.zeros(J)
    ridge = LinRes(0.0, clamp=False)
    for j in range(J):
        if j < K and not active[j]:
            continue
        hm = P.has[:, j]
        a = _baseline_weight(P, ci[hm[ci]], Tc[hm[ci], j], j, w_c[hm[ci]]) if hm[ci].any() else 1.0
        B = _baseline(P, j, a)
        for swing, mask in ((True, hm), (False, ~hm)):
            sc, su = mask[ci], mask[ui]
            if not su.any() and not sc.any():
                continue
            if sc.sum() < MIN_TRAIN:
                if swing and hm[ci].sum() > 0:
                    sc = hm[ci]
                else:
                    sc, swing = np.ones(len(ci), bool), False
            X = _wapo_features(P, j, B, swing, interactions)
            off = B if swing else np.zeros(n)
            tr = ci[sc]
            y = Tc[sc, j] - off[tr]
            w = w_c[sc]
            mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xs = lambda idx: (X[idx] - mu) / sdv
            # ridge point (alpha by grouped CV)
            Pt = Prep()
            Pt.F_lin = (X - mu) / sdv
            alpha = ridge.tune(Pt, tr, y, w, fold_c[sc], G, j)
            pu = ui[su]
            yhat[pu, j] = ridge.fit_predict(Pt, tr, y, w, [pu], alpha, j)[0] + off[pu]
            # CQR on the halves
            h = half[sc]
            if h.sum() < 8 or (~h).sum() < 8:
                h = np.arange(len(tr)) % 2 == 0
            qs = []
            for q in (0.05, 0.95):
                m = QuantileRegressor(quantile=q, alpha=alpha_q, solver="highs")
                m.fit(Xs(tr[h]), y[h], sample_weight=w[h] / w[h].mean())
                qs.append(m)
            cal = tr[~h]
            lo, hi = qs[0].predict(Xs(cal)), qs[1].predict(Xs(cal))
            E = np.maximum(lo - y[~h], y[~h] - hi)
            Q = _wq(E, w[~h], min(0.9 * (1 + 1 / len(E)), 1.0))
            lo_u, hi_u = qs[0].predict(Xs(pu)), qs[1].predict(Xs(pu))
            width = np.maximum(hi_u - lo_u + 2 * Q, 1e-3)
            sd_u[su, j] = width / (2 * 1.645)
            # standardised calibration scores for the correlation model
            pr_cal = ridge.fit_predict(Pt, tr[h], y[h], w[h], [cal], alpha, j)[0]
            sd_cal = np.maximum(qs[1].predict(Xs(cal)) - qs[0].predict(Xs(cal)) + 2 * Q, 1e-3) / 3.29
            zc = (y[~h] - pr_cal) / sd_cal
            pos = np.searchsorted(ci, cal)
            oof[pos, j] = zc
            # one-way random effects ANOVA by municipality -> equicorrelation
            mu_m = np.array([zc[P.muni[cal] == m].mean() if (P.muni[cal] == m).sum() > 1 else np.nan
                             for m in range(P.M)])
            nm = np.array([(P.muni[cal] == m).sum() for m in range(P.M)])
            ok = nm > 1
            if ok.sum() >= 2:
                msw = np.sum([np.sum((zc[P.muni[cal] == m] - mu_m[m]) ** 2) for m in np.where(ok)[0]]) / max(
                    nm[ok].sum() - ok.sum(), 1)
                n0 = (nm[ok].sum() - np.sum(nm[ok] ** 2) / nm[ok].sum()) / max(ok.sum() - 1, 1)
                msb = np.sum(nm[ok] * (mu_m[ok] - zc.mean()) ** 2) / max(ok.sum() - 1, 1)
                vb = max((msb - msw) / max(n0, 1), 0.0)
                rho[j] = float(np.clip(vb / (vb + msw), 0.0, 0.9))
            # contest-level effect: variance of the overall mean score beyond iid
            rho0[j] = float(np.clip(zc.mean() ** 2 - 1 / len(zc), 0.0, 0.25))
    S = S_DRAWS
    zp = rng.standard_normal((S, 1, J)) * np.sqrt(rho0)[None, None]
    zm = rng.standard_normal((S, P.M, J))[:, P.muni[ui]] * np.sqrt(rho)[None, None]
    zi = rng.standard_normal((S, nu, J)) * np.sqrt(1 - rho)[None, None]
    T = yhat[ui][None] + sd_u[None] * (zp + zm + zi)
    T[..., :K] = np.where(active, T[..., :K], -30.0)
    Du = _to_counts(T, P.R[ui][None], active)
    Vp = _to_counts(yhat[ui], P.R[ui], active)
    info = {"rho": np.round(rho, 3).tolist(), "rho0": np.round(rho0, 3).tolist()}
    V = np.zeros((n, 7))
    V[ci] = Yc
    V[ui] = Vp
    D = np.zeros((S, n, 7))
    D[:, ci] = Yc[None]
    D[:, ui] = Du
    info["seconds"] = round(time.time() - t0, 2)
    return Projection(V, D, info)


def ml_wapo_int(case, counted, rng):
    """WaPo-style ridge + CQR with municipality x area-type fixed effects (interactions)."""
    return ml_wapo(case, counted, rng, interactions=True)


def ml_mixed_look_shift(case, counted, rng):
    """Runner-up (no ML). Mixed swing (municipality + area type + muni x type + profile cluster +
    ward, REML) + k-nearest look-alike smoothing of what is left + GREG calibration, with the
    order-shift (pseudo-arrival, ACI-style) shock added to the draws."""
    return _run(case, counted, rng, LookalikeRes(), base="mixed", shift_cal=True)


def ml_mixed_gbm_shift(case, counted, rng):
    """RECOMMENDED. Mixed swing (municipality + area type + muni x type + profile cluster + ward,
    REML) + LightGBM of what is left (trees by grouped CV, 0 allowed) + GREG calibration, with
    posterior/conformal draws and the order-shift (pseudo-arrival, ACI-style) shock."""
    return _run(case, counted, rng, GBMRes(), base="mixed", shift_cal=True)
