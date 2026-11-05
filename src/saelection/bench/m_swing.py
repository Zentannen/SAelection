"""Classical swing, regression and cluster methods for the live-projection bake-off (family: swing).

Swing methods project each uncounted voting district (VD) from *its own history* (the two
predictor elections on `case`) plus a *swing* learned from the VDs counted so far. They differ in

* the scale the swing lives on
    add    additive (uniform) swing on shares of the electorate, p_ik = h_ik + s_k
    ratio  proportional swing, p_ik ∝ h_ik * r_k (aggregate ratio estimator), renormalised
    logit  multinomial-logit swing of each category vs ABSTAIN,
           alr(p_i) = alr(h_i) + s, alr = log(p_k / p_ABSTAIN)
* where the swing is pooled: province, municipality, ward, or a past-profile cluster, with
  empirical-Bayes (James-Stein) shrinkage down the chain province -> municipality -> ward
  (or province + cluster + municipality + ward, crossed, by backfitting);
* whether the swing is a constant per unit or a ridge regression on pre-count covariates
  (Curtice-Firth: previous shares in both elections, turnout, VD size, municipality).

Because ABSTAIN = registered - valid is one of the 7 categories, the swing covers turnout too.
`greben_level` is the CSIR fuzzy-cluster LEVEL estimator (Greben et al. 2006) for reference.

Best method: `hybrid_cf` = Curtice-Firth regression for uncounted VDs that have counted
look-alikes in past-profile space, `cluster_muni_ratio` (ratio swing pooled by area type x
municipality x ward with EB shrinkage) for VDs where the regression would extrapolate.
`FINAL_DRAWS` are the methods with draws; `COMPARE` are point-only comparison variants.

New parties (no history: ActionSA in LGE2021, MK in PE2024): either the cluster average of
their log-ratio vs ABSTAIN (province level + EB cluster/municipality/ward effects; the default,
best in the backtest) or a substitution ridge regression on the other parties' past shares.

Uncertainty (draws, S=200, float32), in log-ratio space:
 1. fixed part (province swing / regression): Bayesian block bootstrap over counted wards
    (exponential weights per ward), refitted per draw;
 2. EB posterior of every random effect; groups with no counted VD draw a fresh N(0, tau^2);
    draws at the cluster and municipality levels share a common N(0,1) per party with
    weight rho = 0.5 (unreached clusters tend to miss the back-off in the same direction);
 3. extrapolation penalty: a variogram of the swing in standardised past-profile (+ size)
    space, fitted on counted VD pairs, gives each uncounted VD an extra variance growing with
    its distance to the nearest counted VDs beyond the typical counted-counted distance
    (near zero under random counting, large under stronghold-first orders); scaled by 0.7
    and drawn as one common N(0,1) per party and draw (fully correlated across VDs);
 4. VD residuals resampled from the counted VDs (heavy tails kept), rescaled by each VD's
    sampling variance, cross-party correlation kept.
The two calibration constants (rho, the 0.7 scale) were tuned on the benchmark for 90%
province coverage (see notes_swing.md).

Pre-count work (features, clusters, fuzzy memberships) is cached per case, so a projection
with 200 draws costs about 0.2-0.6 s on one idle core.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from ..features import K
from .harness import Projection

NCAT = K + 1          # 7 categories, ABSTAIN last
SM = 0.5              # count smoothing for log-ratios
S_DRAWS = 200
_CACHE: dict = {}


# ----------------------------------------------------------------------------- small helpers
def _alr(c):
    c = c + SM
    return np.log(c[..., :K]) - np.log(c[..., [K]])


def _alr_var(c):
    c = c + SM
    return 1.0 / c[..., :K] + 1.0 / c[..., [K]]


def _softmax_counts(eta, reg):
    """eta (..., n, K) alr vs abstain -> counts (..., n, 7) summing to reg."""
    z = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,))], -1)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return reg[:, None] * e / e.sum(-1, keepdims=True)


def _gsum(lab, x, G):
    """sum rows of x (n, C) by integer label -> (G, C)."""
    out = np.zeros((G,) + x.shape[1:])
    np.add.at(out, lab, x)
    return out


def _zscore(F):
    mu, sd = F.mean(0), F.std(0)
    sd[sd < 1e-9] = 1.0
    return (F - mu) / sd


# ----------------------------------------------------------------------------- pre-count prep
class _Prep:
    pass


def _prep(case) -> _Prep:
    """Everything that is known before counting, computed once per case and cached."""
    key = (case.name, case.n, id(case))
    if key in _CACHE:
        return _CACHE[key]
    p = _Prep()
    n = case.n
    p.n = n
    p.reg = case.registered.astype(float)
    L = case.prev_lge.astype(float)
    P = case.prev_pe.astype(float)
    s1 = L[:, :K].sum(0) > 0
    s2 = P[:, :K].sum(0) > 0
    p.s1, p.s2 = s1, s2
    p.hist = s1 | s2                       # party has a past result in at least one election
    A1, A2 = _alr(L), _alr(P)
    V1, V2 = _alr_var(L), _alr_var(P)
    A1[:, ~s1] = A2[:, ~s1]
    A2[:, ~s2] = A1[:, ~s2]
    V1[:, ~s1] = V2[:, ~s1]
    V2[:, ~s2] = V1[:, ~s2]
    p.A1, p.A2, p.V1, p.V2 = A1, A2, V1, V2

    def elec_shares(C, own, other):
        H = C / np.maximum(C.sum(1, keepdims=True), 1)
        Ho = other / np.maximum(other.sum(1, keepdims=True), 1)
        for k in np.where(~own & p.hist)[0]:          # absent in this election -> borrow
            H[:, k] = Ho[:, k]
        return H / H.sum(1, keepdims=True)

    p.H1 = elec_shares(L, s1, P)            # shares of the electorate, previous LGE
    p.H2 = elec_shares(P, s2, L)            # previous PE

    def vshare(C):
        return C[:, :K] / np.maximum(C[:, :K].sum(1, keepdims=True), 1)

    def tlogit(C):
        t = (C[:, :K].sum(1) + SM) / (C.sum(1) + 2 * SM)
        return np.log(t / (1 - t))

    VS1, VS2 = vshare(L), vshare(P)
    p.VS1, p.VS2 = VS1, VS2
    p.T1, p.T2 = tlogit(L), tlogit(P)

    # municipality / ward indices
    p.muni = case.muni.astype(int)
    p.M = int(p.muni.max()) + 1
    wu, widx = np.unique(case.ward, return_inverse=True)
    p.ward = widx.astype(int)
    p.W = len(wu)

    # regression features (all known pre-count), standardised over all VDs.
    #   'share': bounded valid-vote shares of each party in both elections (safe to extrapolate)
    #   'all'  : shares + past log-ratios (sharper fit, but explodes when extrapolated)
    sh_cols, sh_names, alr_cols, alr_names = [], [], [], []
    for k in range(K):
        if s1[k]:
            sh_cols.append(VS1[:, k]); sh_names.append(f"vs1_{k}")
            alr_cols.append(A1[:, k]); alr_names.append(f"alr1_{k}")
        if s2[k]:
            sh_cols.append(VS2[:, k]); sh_names.append(f"vs2_{k}")
            alr_cols.append(A2[:, k]); alr_names.append(f"alr2_{k}")
    gen = [p.T1, p.T2, np.log(p.reg + 1), np.log((p.reg + 1) / (L.sum(1) + 1)),
           np.log((p.reg + 1) / (P.sum(1) + 1)), case.hist_present.astype(float)]
    gen_names = ["to1", "to2", "logsize", "growth1", "growth2", "hist_present"]
    D = np.eye(p.M)[p.muni][:, 1:]          # municipality dummies (first = reference)
    p.feats = {
        "share": np.column_stack([_zscore(np.column_stack(sh_cols + gen)), D]),
        "all": np.column_stack([_zscore(np.column_stack(sh_cols + alr_cols + gen)), D]),
    }
    p.fnames = {"share": sh_names + gen_names + [f"muni_{m}" for m in range(1, p.M)],
                "all": sh_names + alr_names + gen_names + [f"muni_{m}" for m in range(1, p.M)]}

    # clustering profile: valid shares + turnout in both predictor elections
    prof = [VS1[:, s1], VS2[:, s2], p.T1[:, None], p.T2[:, None]]
    p.profile = np.column_stack(prof)
    p.clusters = {}
    _CACHE[key] = p
    return p


def _clusters(p: _Prep, k: int, kind: str = "kmeans") -> np.ndarray:
    """CSIR-style VD clusters on past voting profile (cached). kind 'kmeans', 'gmm' or 'ward'
    (agglomerative Ward tree cut at k: cuts at different k are nested, so a fine cluster with
    no counted VD can back off to its coarse parent cluster)."""
    key = (k, kind)
    if key not in p.clusters:
        X = _zscore(p.profile)
        if kind == "ward":
            if "tree" not in p.clusters:
                from scipy.cluster.hierarchy import linkage
                p.clusters["tree"] = linkage(X, method="ward")
            from scipy.cluster.hierarchy import fcluster
            lab = fcluster(p.clusters["tree"], k, criterion="maxclust") - 1
        elif kind == "gmm":
            from sklearn.mixture import GaussianMixture
            lab = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=2).fit_predict(X)
        else:
            from sklearn.cluster import KMeans
            lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        p.clusters[key] = lab.astype(int)
    return p.clusters[key]


# ----------------------------------------------------------------------------- empirical Bayes
def _eb_level(r, w, lab, G):
    """One level of empirical-Bayes shrinkage of residuals r (nc, C) by group label.

    Group raw swing = weighted mean of r in the group. Between-group variance tau^2 by the
    unbalanced one-way ANOVA moment estimator with Kish effective sizes; the group effect is
    shrunk by B = tau^2 n_eff / (tau^2 n_eff + sigma^2) (James-Stein). Returns the shrunk
    effect (G, C), its posterior variance (G, C) (tau^2 for groups with no counted VD),
    tau^2 (C,) and sigma^2 (C,).
    """
    C = r.shape[1]
    sw = _gsum(lab, w, G)
    swr = _gsum(lab, w * r, G)
    sw2 = _gsum(lab, w * w, G)
    obs = sw > 0
    m = np.where(obs, swr / np.where(obs, sw, 1), 0.0)
    neff = np.where(obs, sw ** 2 / np.where(sw2 > 0, sw2, 1), 0.0)
    dev = r - m[lab]
    N = neff.sum(0)
    Gobs = obs.sum(0)
    s2 = (w * dev ** 2).sum(0) / np.maximum(w.sum(0), 1e-12) * N / np.maximum(N - Gobs, 1)
    num = (neff * m ** 2).sum(0) - np.maximum(Gobs - 1, 0) * s2
    den = N - (neff ** 2).sum(0) / np.maximum(N, 1e-12)
    tau2 = np.where(den > 0, np.maximum(num / np.where(den > 0, den, 1), 0.0), 0.0)
    B = tau2 * neff / np.maximum(tau2 * neff + s2, 1e-12)
    eff = B * m
    pvar = tau2 * (1 - B)
    return eff, pvar, tau2, s2


def _eb_fit(d, w, groups, n_iter=3):
    """Random effects for a list of groupings [(lab_counted, G), ...] by backfitting EB levels.
    Nested chains (muni then ward) and crossed ones (cluster, muni) are both handled."""
    nc, C = d.shape
    effs = [np.zeros((G, C)) for _, G in groups]
    pvars = [np.zeros((G, C)) for _, G in groups]
    taus = [np.zeros(C) for _ in groups]
    iters = n_iter if len(groups) > 1 else 1
    for _ in range(iters):
        for gi, (lab, G) in enumerate(groups):
            other = np.zeros_like(d)
            for gj, (labj, _) in enumerate(groups):
                if gj != gi:
                    other += effs[gj][labj]
            effs[gi], pvars[gi], taus[gi], _ = _eb_level(d - other, w, lab, G)
    resid = d - sum(e[lab] for e, (lab, _) in zip(effs, groups)) if groups else d
    return effs, pvars, taus, resid


# ----------------------------------------------------------------------------- ridge regression
def _ridge(X, y, w, lam):
    """Weighted ridge with unpenalised intercept. Returns (intercept, beta)."""
    sw = w.sum()
    xm = (w @ X) / sw
    ym = (w @ y) / sw
    Xc = X - xm
    A = (Xc * w[:, None]).T @ Xc + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, (Xc * w[:, None]).T @ (y - ym))
    return ym - xm @ b, b


def _ridge_cv(X, y, w, lams, folds=None):
    """Choose the ridge penalty: exact weighted LOO (folds=None) or group K-fold (folds = fold
    label per row). Returns the best lambda and the CV curve."""
    sw = w.sum()
    if folds is None:
        xm = (w @ X) / sw
        ym = (w @ y) / sw
        sq = np.sqrt(w)
        Xc = (X - xm) * sq[:, None]
        yc = (y - ym) * sq
        U, s, _ = np.linalg.svd(Xc, full_matrices=False)
        Uy = U.T @ yc
        U2 = U ** 2
        errs = []
        for lam in lams:
            f = s ** 2 / (s ** 2 + lam)
            fit = U @ (f * Uy)
            h = U2 @ f + w / sw
            loo = (yc - fit) / np.maximum(1 - h, 1e-6)
            errs.append(np.mean(loo ** 2))
    else:
        errs = []
        fl = np.unique(folds)
        for lam in lams:
            e = 0.0
            for f in fl:
                tr = folds != f
                if tr.sum() < X.shape[1] // 2 + 3 or (~tr).sum() == 0:
                    continue
                a, b = _ridge(X[tr], y[tr], w[tr], lam)
                e += (w[~tr] * (y[~tr] - a - X[~tr] @ b) ** 2).sum()
            errs.append(e)
    errs = np.asarray(errs)
    return lams[int(np.argmin(errs))], errs


# ----------------------------------------------------------------------------- base history
def _transfer(p: _Prep, Yc, counted, src="lge", wkind="reg"):
    """Ecological transfer (Goodman regression) base: the target share of category k in VD i
    is sum_j pi_jk h_ij, with h_ij the VD's past shares of the electorate (sources j = past
    parties + ABSTAIN of one or both predictor elections) and pi_jk >= 0 the fraction of past
    category j that goes to k now, fitted by NNLS on the counted VDs. New parties need no
    history (MK draws from past ANC/EFF/abstainers, ActionSA from DA/ANC/...), and because
    predictions are convex combinations of past shares they stay bounded when the counted VDs
    are unrepresentative (extrapolation is linear in shares, not in log-ratios)."""
    from scipy.optimize import nnls
    has = np.append(p.hist, True)
    srcs = []
    if src in ("lge", "both"):
        srcs.append(p.H1[:, has & np.append(p.s1, True)])
    if src in ("pe", "both"):
        srcs.append(p.H2[:, has & np.append(p.s2, True)])
    Hs = np.column_stack(srcs)
    yc = Yc / p.reg[counted, None]
    sw = np.sqrt(p.reg[counted]) if wkind == "reg" else np.ones(counted.sum())
    A = Hs[counted] * sw[:, None]
    Pi = np.zeros((Hs.shape[1], NCAT))
    for k in range(NCAT):
        if yc[:, k].sum() > 0:
            Pi[:, k] = nnls(A, yc[:, k] * sw, maxiter=50 * Hs.shape[1])[0]
    Q = Hs @ Pi
    Q = np.maximum(Q, 1e-6)
    Q = Q / Q.sum(1, keepdims=True)
    return Q, Pi


def _base(p: _Prep, base: str, ac, wc, counted, Yc=None):
    """Past log-ratio per VD to swing from (n, K). 'lge' / 'pe' = one predictor election,
    'mix' = per-party convex blend w*LGE + (1-w)*PE with w fitted on the counted VDs
    (the blend whose swing varies least across counted VDs), 'tr_lge' / 'tr_pe' / 'tr_both' =
    ecological transfer prediction (see _transfer)."""
    if base.startswith("tr_"):
        Q, _ = _transfer(p, Yc, counted, base[3:])
        return _alr(Q * p.reg[:, None]), 0.5 * (p.V1 + p.V2)
    if base == "lge":
        return p.A1.copy(), p.V1
    if base == "pe":
        return p.A2.copy(), p.V2
    B = np.empty_like(p.A1)
    for k in range(K):
        x = (p.A1[counted, k] - p.A2[counted, k])
        y = ac[:, k] - p.A2[counted, k]
        ww = wc[:, k]
        xm, ym = (ww @ x) / ww.sum(), (ww @ y) / ww.sum()
        vx = ww @ (x - xm) ** 2
        lam = float(np.clip((ww @ ((x - xm) * (y - ym))) / vx, 0, 1)) if vx > 0 else 0.5
        B[:, k] = lam * p.A1[:, k] + (1 - lam) * p.A2[:, k]
    return B, 0.5 * (p.V1 + p.V2)


# ----------------------------------------------------------------------------- core model
class _Fit:
    pass


def _groups_for(p, spec, cl=None):
    out = []
    for g in spec:
        if g == "muni":
            out.append((p.muni, p.M))
        elif g == "ward":
            out.append((p.ward, p.W))
        elif g == "cluster":
            out.append((cl, int(cl.max()) + 1))
        elif g == "clmuni":                        # cluster x municipality cells
            out.append((cl * p.M + p.muni, int(cl.max() + 1) * p.M))
        elif g.startswith("cm"):                   # cm10: k-means(10) area type x municipality cell
            lab = _clusters(p, int(g[2:]), "kmeans") * p.M + p.muni
            out.append((lab, (int(lab.max()) // p.M + 1) * p.M))
        elif g.startswith("hc") or g.startswith("km"):   # e.g. hc8 / km20: own clustering
            lab = _clusters(p, int(g[2:]), "ward" if g.startswith("hc") else "kmeans")
            out.append((lab, int(lab.max()) + 1))
        else:
            raise ValueError(g)
    return out


def _knn_index(Z, counted, kk):
    """k nearest counted VDs (in profile space Z) of every uncounted VD, and leave-one-out
    neighbours of every counted VD. Tricube weights on an adaptive bandwidth."""
    Zc, Zu = Z[counted], Z[~counted]
    kk = int(min(kk, counted.sum() - 1))
    nc2 = (Zc ** 2).sum(1)

    def nn(Q, self_ex):
        D2 = (Q ** 2).sum(1)[:, None] + nc2[None] - 2 * Q @ Zc.T
        if self_ex:
            np.fill_diagonal(D2, np.inf)
        idx = np.argpartition(D2, kk, axis=1)[:, :kk + 1]
        dd = np.take_along_axis(D2, idx, 1)
        o = np.argsort(dd, 1)
        idx, dd = np.take_along_axis(idx, o, 1), np.sqrt(np.maximum(np.take_along_axis(dd, o, 1), 0))
        h = dd[:, -1:] * 1.0001 + 1e-9
        kern = (1 - (dd / h) ** 3) ** 3
        return idx[:, :kk], kern[:, :kk] + 1e-6

    iu, ku = nn(Zu, False)
    ic, kc = nn(Zc, True)
    return iu, ku, ic, kc


def _knn_avg(idx, kern, d, w):
    """kernel x weight average of d (nc,) over neighbour index rows."""
    ww = kern * w[idx]
    return (ww * d[idx]).sum(1) / ww.sum(1)


def _fit_logit(case, counted, *, base="lge", fixed="const", groups=("muni", "ward"),
               ncl=40, cl_kind="kmeans", new_party="knn", cv="loo", lam=None, feats="share",
               clip=True, kk=20, geo=0.0, weights="prec"):
    """Fit the log-ratio model on the counted VDs.

    eta_ik = B_ik + f_k(x_i) + sum_g u_{g(i),k}
      B      past log-ratio of the party (base 'lge', 'pe' or blended 'mix'); 0 for new parties
      f      fixed part: 'const' (province swing), 'ridge' (Curtice-Firth regression on
             covariates, penalty by CV, fitted values clipped to the counted range) or 'knn'
             (kernel average of the swing of the kk most similar counted VDs in past-profile
             space; constant, not linear, beyond the counted range)
      u      EB random effects for the groupings in `groups` (muni, ward, cluster, clmuni)
    New parties use f = new_party ('reg' = ridge substitution regression on the other parties'
    past shares, 'knn' = similar-VD average of their log-ratio, 'const').
    """
    p = _prep(case)
    ft = _Fit()
    ft.p = p
    Yc = case.Y[counted].astype(float)
    ac = _alr(Yc)
    vc = _alr_var(Yc)
    need_cl = any(g in ("cluster", "clmuni") for g in groups)
    cl = _clusters(p, ncl, cl_kind) if need_cl else None
    ft.cl = cl
    w0 = 1.0 / (vc + 0.05)
    B, VB = _base(p, base, ac, w0, counted, Yc)
    has_base = np.ones(K, bool) if base.startswith("tr_") else p.hist
    if not base.startswith("tr_"):
        B[:, ~p.hist] = 0.0
        if new_party.startswith("tr_") and (~p.hist).any():
            # new parties: base = ecological-transfer prediction of their share
            Q, _ = _transfer(p, Yc, counted, new_party[3:])
            B[:, ~p.hist] = _alr(Q * p.reg[:, None])[:, ~p.hist]
            has_base = np.ones(K, bool)
    d = ac - B[counted]
    # precision weights: VD sampling variance (+ base variance) + extra VD-level variance
    s2 = np.maximum(np.average((d - np.average(d, 0, w0)) ** 2, 0, w0)
                    - np.average(vc + VB[counted], 0, w0), 0.02)
    wp = 1.0 / (vc + VB[counted] * has_base + s2)
    if weights == "votes":
        # votes at stake: the party's expected count in the VD (base election share x electorate)
        Hb = p.H1 if base == "lge" else (p.H2 if base == "pe" else 0.5 * (p.H1 + p.H2))
        E = Hb[counted, :K] * p.reg[counted, None]
        E[:, ~p.hist] = p.reg[counted, None]
        w = E / E.mean(0)
    elif weights == "reg":
        w = np.repeat(p.reg[counted, None] / p.reg[counted].mean(), K, 1)
    elif weights == "vp":
        # votes-at-stake x precision (harmonic compromise)
        Hb = p.H1 if base == "lge" else (p.H2 if base == "pe" else 0.5 * (p.H1 + p.H2))
        E = Hb[counted, :K] * p.reg[counted, None]
        E[:, ~p.hist] = p.reg[counted, None]
        w = np.sqrt(E / E.mean(0) * wp / wp.mean(0))
    else:
        w = wp
    zero = Yc[:, :K].sum(0) == 0
    kinds = []
    for k in range(K):
        kd = fixed if has_base[k] else {"reg": "ridge", "knn": "knn", "const": "const"}[new_party]
        kinds.append("const" if zero[k] else kd)
    ft.kinds = kinds
    unc = ~counted
    if feats.endswith("_w"):
        # within-cluster covariates: the regression only learns slopes *inside* a cluster, so
        # it cannot extrapolate across clusters (between-cluster swing goes through the EB)
        X = p.feats[feats[:-2]]
        clw = cl if cl is not None else _clusters(p, ncl, cl_kind)
        G = int(clw.max()) + 1
        X = X - (_gsum(clw, X, G) / np.bincount(clw, minlength=G)[:, None])[clw]
    else:
        X = p.feats[feats]
    ft.lam = np.full(K, np.nan)
    ft.fix_c = np.zeros_like(d)              # fitted (LOO for knn) at counted VDs
    ft.fix_u = np.zeros((unc.sum(), K))      # prediction at uncounted VDs
    if "knn" in kinds:
        Z = _zscore(p.profile)
        if geo > 0:
            Z = np.column_stack([Z, geo * _zscore(np.column_stack([case.lon, case.lat]))])
        ft.knn = _knn_index(Z, counted, kk)
    folds = None
    if cv == "group" and "ridge" in kinds:
        folds = _clusters(p, 20, "kmeans")[counted] % 5
    for k in range(K):
        if kinds[k] == "ridge" and counted.sum() >= 8:
            sw = w[:, k].sum()
            if lam is not None:
                lk = lam
            else:
                lk, _ = _ridge_cv(X[counted], d[:, k], w[:, k], np.logspace(-3, 2, 11) * sw / 100, folds)
            a0, b0 = _ridge(X[counted], d[:, k], w[:, k], lk)
            fc, fu = a0 + X[counted] @ b0, a0 + X[unc] @ b0
            if clip:
                fu = np.clip(fu, fc.min(), fc.max())
            ft.lam[k] = lk
        elif kinds[k] == "knn":
            iu, ku, ic, kc = ft.knn
            fc, fu = _knn_avg(ic, kc, d[:, k], w[:, k]), _knn_avg(iu, ku, d[:, k], w[:, k])
        else:
            kinds[k] = "const"
            fc = np.full(counted.sum(), np.average(d[:, k], weights=w[:, k]))
            fu = np.full(unc.sum(), fc[0])
        ft.fix_c[:, k], ft.fix_u[:, k] = fc, fu
    ft.X, ft.clip, ft.d = X, clip, d
    r = d - ft.fix_c
    grp = _groups_for(p, groups, cl)
    ft.groups = grp
    effs, pvars, taus, resid = _eb_fit(r, w, [(lab[counted], G) for lab, G in grp])
    ft.effs, ft.pvars, ft.taus = effs, pvars, taus
    ft.resid, ft.w_c, ft.wp, ft.counted, ft.B = resid, w, wp, counted, B
    return ft


def _predict_eta(ft):
    """alr prediction for all VDs (counted rows are placeholders: B + mean fixed part)."""
    p = ft.p
    eta = ft.B.copy()
    eta[~ft.counted] += ft.fix_u
    eta[ft.counted] += ft.fix_c
    for (lab, _), e in zip(ft.groups, ft.effs):
        eta = eta + e[lab]
    return eta


def _refit_fixed_unc(ft, k, dnet, ww):
    """Refit party k's fixed part under bootstrap weights ww; prediction at uncounted VDs."""
    kind = ft.kinds[k]
    if kind == "ridge":
        Xc, Xu = ft.X[ft.counted], ft.X[~ft.counted]
        a, b = _ridge(Xc, dnet, ww, ft.lam[k])
        f = a + Xu @ b
        if ft.clip:
            fc = a + Xc @ b
            f = np.clip(f, fc.min(), fc.max())
        return f
    if kind == "knn":
        iu, ku, _, _ = ft.knn
        return _knn_avg(iu, ku, dnet, ww)
    return np.full((~ft.counted).sum(), np.average(dnet, weights=ww))


def _profile_Z(p, size=True):
    """standardised past profile (+ log VD size): the space in which 'far from the counted
    VDs' is measured for the extrapolation penalty."""
    key = ("Z", size)
    if key not in p.clusters:
        Z = _zscore(p.profile)
        if size:
            Z = np.column_stack([Z, _zscore(np.log(p.reg + 1)[:, None])])
        p.clusters[key] = Z
    return p.clusters[key]


def _extrap_sd(p, counted, r, wp, nn=5, npairs=40000, seed=0, size=True, return_dist=False):
    """Extrapolation penalty from a variogram of the swing in past-profile space.

    Among counted VDs, the semivariance of the swing residual r (nc, K) between two VDs grows
    with their distance h in the standardised past-profile space; fit gamma(h) = a + b h
    (b >= 0) on binned pairs. An uncounted VD whose nn nearest counted VDs are at mean
    distance h_i is further from the data than counted VDs are from each other (h0), so its
    swing carries an extra variance b (h_i - h0)+ that the cluster EB does not see.
    Returns the extra SD (n_unc, K)."""
    Z = _profile_Z(p, size)
    Zc, Zu = Z[counted], Z[~counted]
    nc = len(Zc)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, nc, npairs)
    j = rng.integers(0, nc, npairs)
    keep = i != j
    i, j = i[keep], j[keep]
    h = np.sqrt(((Zc[i] - Zc[j]) ** 2).sum(1))
    g = 0.5 * (r[i] - r[j]) ** 2
    ww = 1.0 / (1.0 / wp[i] + 1.0 / wp[j])               # down-weight noisy (small) VD pairs
    qs = np.quantile(h, np.linspace(0, 1, 13))
    b_idx = np.clip(np.searchsorted(qs, h, side="right") - 1, 0, 11)
    hb = np.array([h[b_idx == q].mean() for q in range(12)])
    gb = np.array([(ww[b_idx == q] * g[b_idx == q]).sum(0) / ww[b_idx == q].sum(0) for q in range(12)])
    hm = hb.mean()
    slope = ((hb - hm)[:, None] * (gb - gb.mean(0))).sum(0) / ((hb - hm) ** 2).sum()
    slope = np.maximum(slope, 0.0)

    def knn_mean_dist(Q, self_ex):
        D2 = (Q ** 2).sum(1)[:, None] + (Zc ** 2).sum(1)[None] - 2 * Q @ Zc.T
        if self_ex:
            np.fill_diagonal(D2, np.inf)
        k = min(nn, nc - 1)
        return np.sqrt(np.maximum(np.partition(D2, k, axis=1)[:, :k], 0)).mean(1)

    h0 = np.median(knn_mean_dist(Zc, True))
    hu = knn_mean_dist(Zu, False)
    if return_dist:
        return hu, h0
    return np.sqrt(np.maximum(hu - h0, 0)[:, None] * slope[None])


def _logit_projection(case, counted, rng, nd=S_DRAWS, boot=True, resid=True, infl=1.0,
                      boot_mult=1.0, smear=0, rho=0.0, n_rho=2, vario=0.0, vrho=1.0,
                      vsize=True, **kw):
    """Point projection + draws for the log-ratio model (see module docstring).

    nd draws; boot = block bootstrap of the fixed part; resid = VD residuals; rho = shared
    share of the cluster/municipality effect draws (first n_rho grouping levels); vario =
    scale of the extrapolation penalty (vrho = its common-factor share, vsize = include VD
    size in the distance); infl / boot_mult scale the EB / bootstrap terms; smear = number
    of residual draws for a Duan smearing point estimate (0 = off; it did not help)."""
    ft = _fit_logit(case, counted, **kw)
    p = ft.p
    eta = _predict_eta(ft)
    zero = case.Y[counted, :K].sum(0) == 0          # category not on the ballot this time
    eta[:, zero] = -40.0
    mean_counts = _softmax_counts(eta, p.reg)
    mean_counts[:, :K][:, zero] = 0.0                   # not on the ballot: exactly zero
    unc = ~counted
    nu = unc.sum()
    vcnt = _alr_var(case.Y[counted].astype(float))
    zc = ft.resid * np.sqrt(ft.wp)                    # standardised counted residuals
    s2 = np.median(1.0 / ft.wp - vcnt, 0).clip(0.01)
    if smear and len(zc) > 2:
        # smearing (Duan): the expected count is the mean over the residual distribution,
        # not the count at the mean log-ratio (exp is convex)
        sd_u = np.sqrt(_alr_var(mean_counts[unc]) + s2)
        jj = np.random.default_rng(1).integers(0, len(zc), size=(smear, nu))
        mean_counts[unc] = _softmax_counts(eta[unc][None] + zc[jj] * sd_u[None], p.reg[unc]).mean(0)
    info = {"lam": ft.lam.tolist(), "kinds": ft.kinds}
    if nd <= 0:
        return Projection(mean_counts, None, info)
    f32 = np.float32
    etas = np.repeat(eta[unc][None].astype(f32), nd, 0)
    # 1) fixed part: Bayesian block bootstrap over counted wards (refit swing / regression)
    if boot:
        wc = p.ward[counted]
        re_c = np.zeros_like(ft.d)
        for (lab, _), e in zip(ft.groups, ft.effs):
            re_c += e[lab[counted]]
        dnet = ft.d - re_c
        Gw = rng.exponential(size=(nd, p.W))[:, wc]           # (nd, nc) ward weights
        for k in range(K):
            if zero[k]:
                continue
            if ft.kinds[k] == "const":                       # vectorised weighted means
                ww = Gw * ft.w_c[:, k][None]
                f = (ww @ dnet[:, k]) / ww.sum(1)
                etas[:, :, k] += (boot_mult * (f - ft.fix_u[0, k]))[:, None].astype(f32)
            else:
                for s in range(nd):
                    f = _refit_fixed_unc(ft, k, dnet[:, k], ft.w_c[:, k] * Gw[s])
                    etas[s, :, k] += (boot_mult * (f - ft.fix_u[:, k])).astype(f32)
    # 2) random effects: EB posterior (groups with no counted VD draw a fresh N(0, tau^2))
    #    rho > 0 correlates the draws of the groups at a level (one common N(0,1) per party
    #    and draw): the swings of the clusters the count has not reached yet tend to miss the
    #    back-off value in the same direction (they are all "the other kind" of VD).
    for gi, ((lab, G), e, pv) in enumerate(zip(ft.groups, ft.effs, ft.pvars)):
        r_ = rho if gi < n_rho else 0.0
        z = (np.sqrt(1 - r_) * rng.standard_normal((nd, G, K), dtype=f32)
             + np.sqrt(r_) * rng.standard_normal((nd, 1, K), dtype=f32))
        z *= (np.sqrt(pv) * infl).astype(f32)[None]
        etas += z[:, lab[unc]]
    # 2b) extrapolation penalty (variogram in profile space), correlated across VDs: one
    #     common N(0,1) per party and draw (share vrho) + one per cluster
    if vario > 0 and counted.sum() > 10:
        r0 = ft.d - np.average(ft.d, 0, ft.wp)
        xs = vario * _extrap_sd(p, counted, r0, ft.wp, size=vsize)
        clx = ft.cl if ft.cl is not None else _clusters(p, 20, "kmeans")
        Gx = int(clx.max()) + 1
        z = (np.sqrt(1 - vrho) * rng.standard_normal((nd, Gx, K), dtype=f32)[:, clx[unc]]
             + np.sqrt(vrho) * rng.standard_normal((nd, 1, K), dtype=f32))
        etas += z * xs.astype(f32)[None]
    # 3) VD residual, resampled from counted VDs, standardised by sampling + extra variance
    if resid and len(zc) > 2:
        j = rng.integers(0, len(zc), size=(nd, nu))
        etas += zc.astype(f32)[j] * np.sqrt(_alr_var(mean_counts[unc]) + s2).astype(f32)[None]
    etas[..., zero] = -40.0
    draws = np.repeat(mean_counts[None].astype(f32), nd, 0)
    draws[:, unc] = _softmax_counts(etas, p.reg[unc].astype(f32))
    draws[..., :K][..., zero] = 0.0
    return Projection(mean_counts, draws, info)


# ----------------------------------------------------------------------------- share-scale swings
def _geo_mix(H1, H2):
    """geometric mean of two compositions (= the alr average), renormalised."""
    g = np.sqrt(np.maximum(H1, 1e-6) * np.maximum(H2, 1e-6))
    g[(H1 <= 0) & (H2 <= 0)] = 0
    return g / g.sum(1, keepdims=True)


def _mix_shares(p, Yc, regc, counted, grid=np.linspace(0, 1, 5)):
    """Per category blend lam*H_LGE + (1-lam)*H_PE of the two past compositions, lam chosen
    on the counted VDs to minimise the vote-weighted dispersion of log(actual / base)."""
    H = np.empty_like(p.H1)
    for k in range(NCAT):
        best, bl = np.inf, 0.5
        for lam in grid:
            h = lam * p.H1[counted, k] + (1 - lam) * p.H2[counted, k]
            E = h * regc
            ok = E > 0
            if ok.sum() < 3:
                continue
            lr = np.log((Yc[ok, k] + 0.5) / (E[ok] + 0.5))
            v = np.average((lr - np.average(lr, weights=E[ok])) ** 2, weights=E[ok])
            if v < best:
                best, bl = v, lam
        H[:, k] = bl * p.H1[:, k] + (1 - bl) * p.H2[:, k]
    return H


def _recentre(mean, src: Projection, counted):
    """Move a predictive distribution onto another point projection: every draw of `src` is a
    log-ratio deviation (vs ABSTAIN) from src.vd_mean; apply the same deviation to `mean`."""
    unc = ~counted
    eps = np.float32(1e-3)
    m0 = (src.vd_mean[unc] + eps).astype(np.float32)
    dev = np.log(src.draws[:, unc].astype(np.float32) + eps) - np.log(m0)[None]
    dev = dev - dev[..., [K]]                            # alr deviation, ABSTAIN fixed
    q = np.log(mean[unc] + eps).astype(np.float32)[None] + dev
    q = np.exp(q - q.max(-1, keepdims=True))
    reg = mean[unc].sum(1)
    q = q * (mean[unc] > 0)[None]                         # exact zeros stay zero
    draws = np.repeat(mean[None].astype(np.float32), src.draws.shape[0], 0)
    draws[:, unc] = q / q.sum(-1, keepdims=True) * reg.astype(np.float32)[None, :, None]
    return draws


def _logit_kw_for(levels):
    g = tuple("cluster" if lv.startswith("km") else ("clmuni" if lv.startswith("cm") else lv)
              for lv in levels)
    ncl = next((int(lv[2:]) for lv in levels if lv[:2] in ("km", "cm")), 20)
    return dict(groups=g, ncl=ncl)


def _share_projection(case, counted, rng, scale="ratio", base="avg", levels=("km20", "muni", "ward"),
                      nd=S_DRAWS, rho=0.5, vario=0.7, new_party="const", cv="loo"):
    """Additive or proportional swing on shares of the electorate with EB pooling.

    add:   d_ik = y_ik/reg_i - h_ik, swing = size-weighted mean (aggregate additive swing)
    ratio: d_ik = Y_ik / (reg_i h_ik), swing = E-weighted mean (= aggregate ratio sum Y / sum E),
           applied multiplicatively and renormalised
    h = past shares of the electorate ('lge', 'pe', or 'avg' of both). Pooled province ->
    `levels` (muni, ward, km<k> = k-means profile cluster) with EB shrinkage of each unit's
    swing toward its parent. New parties (h = 0): share from the log-ratio cluster model.
    Draws: the log-ratio model's predictive distribution with the same pooling levels
    (bootstrap + EB + extrapolation penalty + residuals), re-centred on this projection.
    """
    p = _prep(case)
    Yc = case.Y[counted].astype(float)
    regc = p.reg[counted]
    has = np.append(p.hist, True)
    if base == "mix":
        H = _mix_shares(p, Yc, regc, counted)
    else:
        H = {"lge": p.H1, "pe": p.H2, "avg": 0.5 * (p.H1 + p.H2), "geo": _geo_mix(p.H1, p.H2)}[base]
    if scale == "add":
        d = Yc / regc[:, None] - H[counted]
        w = np.repeat(regc[:, None], NCAT, 1)
    else:
        E = H[counted] * regc[:, None]
        d = np.where(E > 0, Yc / np.maximum(E, 1e-9), 1.0)
        w = np.maximum(E, 1e-9)
    w = w / w.mean()
    mu = (w * d).sum(0) / w.sum(0)
    grp = _groups_for(p, levels)
    effs, _, _, _ = _eb_fit(d - mu, w, [(lab[counted], G) for lab, G in grp])
    sw = mu + sum(e[lab] for e, (lab, _) in zip(effs, grp))
    q = np.clip(H + sw if scale == "add" else H * sw, 0, None)
    q[:, ~has] = 0.0
    lp = _logit_projection(case, counted, rng, nd=nd, base="mix", new_party=new_party, cv=cv,
                           rho=rho, vario=vario, **_logit_kw_for(levels))
    newk = [k for k in range(K) if not p.hist[k]]
    if newk:                       # new parties: share of the electorate from the logit model
        q[:, newk] = lp.vd_mean[:, newk] / p.reg[:, None]
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-12)
    mean = q * p.reg[:, None]
    if nd <= 0:
        return Projection(mean)
    return Projection(mean, _recentre(mean, lp, counted))


# ----------------------------------------------------------------------------- CSIR level estimator
def _recent(case):
    """The more recent predictor election: PE2019 for LGE2021, LGE2021 for PE2024 (and the
    previous PE for the live 2026 run)."""
    return case.prev_lge if case.name.startswith("PE") else case.prev_pe


def _fcm(X, C, m, seed=0, iters=150):
    """Fuzzy c-means (Bezdek): memberships u_ic = 1 / sum_j (d_ic / d_ij)^(2/(m-1)),
    centres = u^m-weighted means. Initialised from k-means."""
    from sklearn.cluster import KMeans
    cen = KMeans(C, n_init=4, random_state=seed).fit(X).cluster_centers_
    for _ in range(iters):
        D2 = ((X[:, None, :] - cen[None]) ** 2).sum(-1) + 1e-10
        lu = -np.log(D2) / (m - 1)
        lu -= lu.max(1, keepdims=True)
        U = np.exp(lu)
        U /= U.sum(1, keepdims=True)
        Um = U ** m
        new = (Um.T @ X) / Um.sum(0)[:, None]
        if np.abs(new - cen).max() < 1e-6:
            cen = new
            break
        cen = new
    return U


def _greben_U(case, p, C, m, feats):
    key = ("fcm", C, m, feats)
    if key not in p.clusters:
        if feats == "raw":
            # Greben et al. (2006): raw party percentages of valid votes in the previous election
            R = _recent(case)
            X = 100 * R[:, :K] / np.maximum(R[:, :K].sum(1, keepdims=True), 1)
            X = X[:, X.sum(0) > 0]
        else:
            # Ittmann et al. (2017) fix: every party's share standardised (small parties count as
            # much as big ones), both predictor elections, plus turnout
            X = _zscore(p.profile)
        p.clusters[key] = _fcm(X, C, m)
    return p.clusters[key]


def greben_level(case, counted, rng, C=20, m=1.2, feats="raw", nd=S_DRAWS, vario=0.7, rho=0.5):
    """CSIR election-night model (Greben, Elphinstone & Holloway 2006, ORiON 22(1)), a LEVEL
    estimator, not a swing one. VDs are clustered by fuzzy c-means on their past party
    percentages; each cluster's current composition (party shares *and* turnout, i.e. the
    7 shares of the electorate) is the membership- and size-weighted average over the counted
    VDs; an uncounted VD is the membership-weighted mix of its clusters' compositions. The old
    results define only the memberships. Draws (not part of CSIR, which publishes no
    intervals): Bayesian block bootstrap over counted wards + resampled VD residuals + the
    profile-space extrapolation penalty (variogram of the VD level, see _extrap_sd)."""
    p = _prep(case)
    U = _greben_U(case, p, C, m, feats)
    Yc = case.Y[counted].astype(float)
    regc = p.reg[counted]
    Uc = U[counted]

    def compose(g):
        num = (Uc * g[:, None]).T @ Yc                 # (C, 7) membership-weighted counts
        den = (Uc * g[:, None]).T @ regc               # membership-weighted electorate
        ok = den > 1e-9 * den.max()
        Pc = np.where(ok[:, None], num / np.where(ok, den, 1)[:, None], 0.0)
        Ue = U * ok[None]
        return (Ue @ Pc) / np.maximum(Ue.sum(1, keepdims=True), 1e-12)

    q = compose(np.ones(counted.sum()))
    mean = q * p.reg[:, None]
    if nd <= 0:
        return Projection(mean)
    unc = ~counted
    wc = p.ward[counted]
    res = Yc / regc[:, None] - q[counted]               # counted VD residuals (share scale)
    xs = None
    if vario > 0 and counted.sum() > 10:
        # extrapolation penalty: variogram of the VD *level* (log-ratio) in profile space
        ac = _alr(Yc)
        wp = 1.0 / (_alr_var(Yc) + 0.05)
        xs = vario * _extrap_sd(p, counted, ac - np.average(ac, 0, wp), wp)
        clx = _clusters(p, 20, "kmeans")[unc]
    draws = np.repeat(mean[None], nd, 0)
    for s in range(nd):
        qs = compose(rng.exponential(size=p.W)[wc])[unc]
        qs = np.clip(qs + res[rng.integers(0, len(res), unc.sum())], 0, None)
        if xs is not None:
            z = np.sqrt(1 - rho) * rng.standard_normal((20, K))[clx] + np.sqrt(rho) * rng.standard_normal(K)
            qs[:, :K] *= np.exp(z * xs)
        draws[s, unc] = qs / np.maximum(qs.sum(1, keepdims=True), 1e-12) * p.reg[unc, None]
    return Projection(mean, draws)


# ----------------------------------------------------------------------------- method registry
UNC = dict(rho=0.5, vario=0.7)        # calibration of the draws (see _logit_projection)
CLUSTER_KW = dict(base="mix", fixed="const", ncl=20, groups=("cluster", "muni", "ward"),
                  new_party="const")
NEWP = dict(new_party="reg", cv="group")    # substitution regression for new parties


def _make(fn, doc, **kw):
    def method(case, counted, rng):
        return fn(case, counted, rng, **kw)
    method.__doc__ = doc
    return method


_LEVELS = {"prov": (), "muni": ("muni",), "ward": ("muni", "ward")}
for _lv, _g in _LEVELS.items():
    _pool = "province-wide" if not _g else " -> ".join(("province",) + _g) + " (EB shrinkage)"
    _nd = {} if _lv == "ward" else {"nd": 0}      # prov / muni: comparison, point projection only
    _pt = "" if _lv == "ward" else " Comparison variant: point projection only (no draws)."
    globals()[f"swing_add_{_lv}"] = _make(
        _share_projection, f"Additive (uniform) swing on shares of the electorate vs each VD's "
        f"previous-LGE result, pooled {_pool}; new parties by substitution regression.{_pt}",
        scale="add", base="lge", levels=_g, **NEWP, **UNC, **_nd)
    globals()[f"swing_ratio_{_lv}"] = _make(
        _share_projection, f"Proportional (ratio) swing vs each VD's previous-LGE result, "
        f"pooled {_pool}; new parties by substitution regression.{_pt}",
        scale="ratio", base="lge", levels=_g, **NEWP, **UNC, **_nd)
    globals()[f"swing_logit_{_lv}"] = _make(
        _logit_projection, f"Multinomial-logit swing (log-ratio vs ABSTAIN) vs each VD's "
        f"previous-LGE result, pooled {_pool}; new parties by substitution regression.{_pt}",
        base="lge", fixed="const", groups=_g, **NEWP, **UNC, **_nd)

reg_cf = _make(
    _logit_projection, "Curtice-Firth: ridge regression of each VD's log-ratio change (vs a "
    "per-party blend of its two past results) on past shares and log-ratios in both elections, "
    "turnout, size, growth and municipality dummies; penalty by leave-cluster-group-out CV; "
    "fitted values clipped to the counted range; EB municipality + ward effects on top; new "
    "parties by the same ridge (substitution) regression.",
    base="mix", fixed="ridge", feats="all", cv="group", groups=("muni", "ward"),
    new_party="reg", **UNC)

cluster_swing = _make(
    _logit_projection, "CSIR-style cluster swing on the log-ratio scale: k-means (20) on "
    "standardised past shares + turnout in both elections; swing of each uncounted VD = "
    "province swing + EB-shrunk cluster, municipality and ward effects (crossed, backfitted), "
    "so a cluster with few counted VDs backs off to province/municipality. Base = per-party "
    "blend of the two past results. New parties: province level + the same cluster/muni/ward "
    "effects (a cluster-average of their log-ratio).",
    **CLUSTER_KW, **UNC)

cluster_ratio = _make(
    _share_projection, "Cluster ratio swing (best single method): proportional swing of the "
    "average of the two past compositions, aggregate ratio per k-means cluster (20) crossed "
    "with municipality and ward, EB shrinkage to the province ratio; new parties from the "
    "log-ratio cluster model; draws = log-ratio model's predictive distribution re-centred.",
    scale="ratio", base="avg", levels=("km20", "muni", "ward"), **UNC)

cluster_ratio_subst = _make(
    _share_projection, "cluster_ratio, but new parties by substitution ridge regression on the "
    "other parties' past shares (ActionSA ~ DA..., MK ~ ANC/EFF...) instead of cluster levels. "
    "Comparison variant: point projection only.",
    scale="ratio", base="avg", levels=("km20", "muni", "ward"), **NEWP, **UNC, nd=0)

greben_csir = _make(
    greben_level, "CSIR / Greben et al. (2006) fuzzy-cluster LEVEL estimator: fuzzy c-means "
    "(20 clusters, m = 1.2) on raw party percentages of the most recent past election; "
    "uncounted VD = membership-weighted mix of membership x size weighted cluster averages.",
    C=20, m=1.2, feats="raw", **UNC)

greben_std = _make(
    greben_level, "Greben level estimator with the Ittmann et al. (2017) fix: clustering on "
    "standardised per-party shares (both elections) + turnout, 40 clusters (K chosen by "
    "backtest in 10-40).", C=40, m=1.2, feats="std", **UNC)


CM_LEVELS = ("km20", "muni", "cm20", "ward")
CM_KW = dict(base="mix", fixed="const", ncl=20, groups=("cluster", "muni", "clmuni", "ward"),
             new_party="const")

cluster_muni_ratio = _make(
    _share_projection, "Hierarchical swing per party x area type x municipality (ratio scale): "
    "area types = k-means (20) on standardised past shares of both elections + turnout; "
    "aggregate ratio per (area type x municipality) cell, EB-shrunk to area type + "
    "municipality (crossed), then province; plus EB ward effects. New parties from the "
    "log-ratio model with the same levels.", scale="ratio", base="avg", levels=CM_LEVELS, **UNC)

cluster_muni_ratio_k6 = _make(
    _share_projection, "cluster_muni_ratio with 6 coarse area types (K comparison). "
    "Comparison variant: point projection only.",
    scale="ratio", base="avg", levels=("km6", "muni", "cm6", "ward"), **UNC, nd=0)

cluster_muni_swing = _make(
    _logit_projection, "Hierarchical swing per party x area type x municipality on the "
    "log-ratio scale (area type 20 + municipality + cell + ward EB effects).", **CM_KW, **UNC)


def _broken(p) -> np.ndarray:
    """VDs whose own history cannot be swung from: past turnout > 90% in either predictor
    election (s24A out-of-district votes inflate the count; ABSTAIN ~ 0 then makes any
    multiplicative or log-ratio swing project ~100% turnout) or no valid votes in one."""
    return (p.T1 > np.log(9)) | (p.T2 > np.log(9)) | (p.VS1.sum(1) == 0) | (p.VS2.sum(1) == 0)


def _lookalike_fix(case, counted, mean, k=8, muni_w=2.0):
    """Replace the projection of uncounted broken-history VDs by a look-alike LEVEL: the
    size-weighted composition (shares of the electorate) of the k most similar counted,
    non-broken VDs on standardised past valid-vote shares of both elections, with the
    municipality as an extra coordinate (same municipality preferred)."""
    p = _prep(case)
    br = _broken(p)
    fix = br & ~counted
    donors = counted & ~br
    if not fix.any() or donors.sum() < k:
        return mean, fix
    if "lookalike_Z" not in p.clusters:
        Z = _zscore(np.column_stack([p.VS1[:, p.s1], p.VS2[:, p.s2]]))
        p.clusters["lookalike_Z"] = np.column_stack([Z, muni_w * np.eye(p.M)[p.muni]])
    Z = p.clusters["lookalike_Z"]
    Zd, Zf = Z[donors], Z[fix]
    D2 = (Zf ** 2).sum(1)[:, None] + (Zd ** 2).sum(1)[None] - 2 * Zf @ Zd.T
    nn = np.argpartition(D2, k - 1, axis=1)[:, :k]
    Yd = case.Y[donors].astype(float)
    comp = Yd[nn].sum(1) / np.maximum(p.reg[donors][nn].sum(1), 1)[:, None]
    comp = comp / comp.sum(1, keepdims=True)
    out = mean.copy()
    out[fix] = comp * p.reg[fix, None]
    return out, fix


def _hybrid(case, counted, rng, c=1.0, nd=S_DRAWS, reg_groups=("muni", "ward"), fix_broken=False):
    """Regression where it interpolates, cluster swing where it would extrapolate.

    Per uncounted VD, a = clip(1 - (h_i - h0) / (c h0), 0, 1), with h_i the mean distance to
    its 5 nearest counted VDs in past-profile (+ size) space and h0 the typical
    counted-to-counted distance. Projection = a * Curtice-Firth regression + (1 - a) *
    cluster_muni_ratio (count space). Draws: cluster_muni model's predictive distribution
    re-centred on the blend."""
    p = _prep(case)
    unc = ~counted
    rp = _share_projection(case, counted, rng, scale="ratio", base="avg", levels=CM_LEVELS,
                           nd=nd, **UNC)
    gp = _logit_projection(case, counted, rng, nd=0, base="mix", fixed="ridge", feats="all",
                           cv="group", groups=reg_groups, new_party="reg")
    if counted.sum() > 10:
        hu, h0 = _extrap_sd(p, counted, np.zeros((counted.sum(), 1)), np.ones((counted.sum(), 1)),
                            return_dist=True)
        a = np.clip(1 - (hu - h0) / (c * h0), 0, 1)
    else:
        a = np.zeros(unc.sum())
    mean = rp.vd_mean.copy()
    mean[unc] = a[:, None] * gp.vd_mean[unc] + (1 - a[:, None]) * rp.vd_mean[unc]
    info = {"mean_a": float(np.average(a, weights=p.reg[unc]))}
    if fix_broken:
        mean, fixed = _lookalike_fix(case, counted, mean)
        info["n_lookalike"] = int(fixed.sum())
    if nd <= 0:
        return Projection(mean, None, info)
    return Projection(mean, _recentre(mean, rp, counted), info)


hybrid_cf = _make(
    _hybrid, "Curtice-Firth regression where the uncounted VD lies within the counted VDs' "
    "past-profile range, cluster_muni_ratio where it would extrapolate (per-VD weight from "
    "the distance to the nearest counted VDs, c = 2: full trust at the typical "
    "counted-to-counted distance, none beyond 3x); draws from cluster_muni_ratio.", c=2.0)

hybrid_cf_c1 = _make(_hybrid, "hybrid_cf with a faster hand-over to the cluster model (c = 1). "
                     "Comparison variant: point projection only.", c=1.0, nd=0)


hybrid_live = _make(
    _hybrid, "hybrid_cf + look-alike fallback for VDs with broken history (past turnout > 90% "
    "from s24A votes, or no valid votes in a predictor election): their projection is the "
    "composition of the 8 most similar counted VDs (past valid shares, same municipality "
    "preferred). Tested on the live check: slightly WORSE than hybrid_cf (the regression part "
    "already absorbs the inflated past turnout), so not in FINAL.", c=2.0, fix_broken=True)


def swing_ens(case, counted, rng):
    """Ensemble: 2/3 cluster_muni_ratio + 1/3 cluster_muni_swing (point projections averaged);
    draws = the log-ratio model's predictive distribution re-centred on the average."""
    lp = _logit_projection(case, counted, rng, **CM_KW, **UNC)
    rp = _share_projection(case, counted, rng, scale="ratio", base="avg", levels=CM_LEVELS,
                           nd=0, **UNC)
    mean = (2 * rp.vd_mean + lp.vd_mean) / 3
    return Projection(mean, _recentre(mean, lp, counted))


FINAL_DRAWS = ["swing_add_ward", "swing_ratio_ward", "swing_logit_ward", "reg_cf",
               "cluster_swing", "cluster_ratio", "cluster_muni_ratio", "cluster_muni_swing",
               "hybrid_cf", "greben_csir", "greben_std", "swing_ens"]
COMPARE = ["swing_add_prov", "swing_add_muni", "swing_ratio_prov", "swing_ratio_muni",
           "swing_logit_prov", "swing_logit_muni", "cluster_ratio_subst", "cluster_muni_ratio_k6",
           "hybrid_cf_c1"]                             # point projection only
FINAL = FINAL_DRAWS + COMPARE
