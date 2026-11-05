"""Party baselines from the right election (family: v_asa). Best: `hybrid_asa`.

Problem addressed: ActionSA's history features lean on PE2024, where it polled 4.2% in Gauteng
(6.2 JHB, 3.8 TSH, 2.9 EKU; under 1% in 56 wards), although the comparable *local* election,
LGE2021, had it at 18.1 / 9.3 / 7.4%. In Sedibeng (GT421-3) and the West Rand (GT481/4/5) it
did not stand in 2021, so LGE2021 holds a structural zero there, and 53 metro VDs hold a
spurious ~0 in PE2024 (2-27% in LGE2021; the counted ones polled 4-50% in 2026).

Every method here is `m_swing.hybrid_cf` (Curtice-Firth ridge on counted look-alikes + the
area-type x municipality ratio swing, with its draws), run on a *re-based* copy of the case
(`_rebase`), optionally with a district level:

1. LGE-equivalent imputation (`impute`, strength `gamma`): for every party with votes in both
   predictor elections, the municipalities where it did not stand in the previous LGE (muni
   LGE share < 0.1% while its PE share there is >= 0.2%) get s_LGE*(i) = gamma r(i) s_PE(i),
   r(i) = aggregate LGE/PE ratio over the 150 most similar VDs (PE profile) where it stood.
   Other parties' LGE counts are scaled down so the VD's valid vote is unchanged. gamma='live'
   learns it from the counted VDs (`_live_gamma`).
2. Cross-election low-base floor (`floor` = phi): a VD whose share in one predictor election is
   below phi x what the other implies (other share x province LGE/PE ratio) gets that floor.
   Applied to every VD (scope 'all': in the not-stood municipalities this is a gamma = phi
   imputation) or only where the party stood (scope 'stood').
3. Neighbour floor (`nb_floor`): share >= nb_floor x touching VDs' share, same election.
4. Forced LGE weighting (`force_lge`): PE column := LGE-equivalent / r for that party.
5. Ratio-part base (`rbase`): 'avg' (hybrid_cf) or 'mix' (per-category blend learned live).
6. District level (`dist`): metros pooled / Sedibeng (GT42x) / West Rand (GT48x) as an EB level
   in the ratio swing and as dummies in the ridge (municipalities with few counted VDs borrow
   their district's swing; in Gauteng the district is also "where ActionSA stood in 2021").

Findings (live check at the 15:18 snapshot; notes_v_asa.md): a full LGE-equivalent base for
ActionSA over-predicts it by 6-10 pp in Sedibeng / West Rand municipality hold-outs (its 2026
vote grew much less where it had no 2021 local presence), forcing LGE weighting is worse, and
the live data itself weights PE2024 over LGE2021 for ActionSA. What helps is a *partial*
imputation (phi = 0.3 floor) plus the district level: `hybrid_asa`.

Benchmark fallback: ActionSA has no prior LGE or PE in the LGE2021 case and no prior PE in the
PE2024 case, so the floor / imputation are no-ops there and hybrid_asa = hybrid_cf + district.

Rules kept: case.Y is read only at `counted` (inside m_swing, and in `_live_gamma`); the
re-based history uses pre-count data only (previous elections, electorate, municipality).
The 'dist' grouping is added to m_swing's grouping function at import (runtime only; every other
grouping is delegated unchanged, m_swing.py is not edited).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses

import numpy as np

from ..features import K
from . import m_swing as ms
from .harness import Projection

ASA = 2
_CASES: dict = {}


# ----------------------------------------------------------------------------- re-basing
def _vshare(C):
    return C[:, :K] / np.maximum(C[:, :K].sum(1, keepdims=True), 1e-9)


def _set_share(C, k, s):
    """Set party k's valid-vote share of every VD in counts C (n, 7) to s (n,), scaling the
    other parties so the valid vote (and ABSTAIN) is unchanged."""
    C = C.copy()
    valid = C[:, :K].sum(1)
    new_k = s * valid
    rest_old = valid - C[:, k]
    rest_new = valid - new_k
    f = np.where(rest_old > 0, rest_new / np.maximum(rest_old, 1e-9), 0.0)
    for j in range(K):
        if j != k:
            C[:, j] = C[:, j] * f
    C[:, k] = new_k
    return C


def _muni_share(C, muni, k, M):
    num = np.bincount(muni, C[:, k], minlength=M)
    den = np.bincount(muni, C[:, :K].sum(1), minlength=M)
    return num / np.maximum(den, 1e-9)


def _lookalike_ratio(case, L, P, k, donors, targets, nn=150):
    """Aggregate LGE/PE ratio of party k's valid share among the nn most similar donor VDs
    (standardised PE valid shares + PE turnout) of every target VD."""
    vsP = _vshare(P)
    t2 = P[:, :K].sum(1) / np.maximum(P.sum(1), 1)
    Z = ms._zscore(np.column_stack([vsP[:, P[:, :K].sum(0) > 0], t2]))
    Zd, Zt = Z[donors], Z[targets]
    nn = int(min(nn, donors.sum()))
    D2 = (Zt ** 2).sum(1)[:, None] + (Zd ** 2).sum(1)[None] - 2 * Zt @ Zd.T
    idx = np.argpartition(D2, nn - 1, axis=1)[:, :nn]
    Lk = L[donors, k][idx].sum(1)
    Pk = (vsP[donors, k] * L[donors, :K].sum(1))[idx].sum(1)     # PE share x LGE valid
    return Lk / np.maximum(Pk, 1e-9)


def _nb_share(C, k, edges):
    """valid-vote-weighted share of party k over each VD's touching neighbours (own share if
    the VD has no neighbour)."""
    valid = C[:, :K].sum(1)
    n = len(C)
    num = np.zeros(n)
    den = np.zeros(n)
    for a, b in ((0, 1), (1, 0)):
        np.add.at(num, edges[:, a], C[edges[:, b], k])
        np.add.at(den, edges[:, a], valid[edges[:, b]])
    own = C[:, k] / np.maximum(valid, 1e-9)
    return np.where(den > 0, num / np.maximum(den, 1e-9), own)


def _rebase(case, impute=True, floor=0.0, floor_parties=(ASA,), force_lge=(), nn=150,
            stand=0.002, absent=0.001, nb_floor=0.0, gamma=1.0, scope="all"):
    """Return (prev_lge*, prev_pe*, info) for the re-based history (see module docstring).
    scope='stood': the floors only touch VDs of municipalities where the party stood in the
    election being floored (both elections for the cross-election floor), so a structural
    zero (did not stand) is kept as such and only VD-level zeros are floored."""
    L = case.prev_lge.astype(float).copy()
    P = case.prev_pe.astype(float).copy()
    muni = case.muni.astype(int)
    M = int(muni.max()) + 1
    info = {}
    both = (L[:, :K].sum(0) > 0) & (P[:, :K].sum(0) > 0)
    ratio, stood_both = {}, {}
    for k in np.where(both)[0]:
        sL, sP = _muni_share(L, muni, k, M), _muni_share(P, muni, k, M)
        stood = (sL >= stand) & (sP > 0)
        if not stood.any():
            continue
        don = stood[muni]
        r_prov = L[don, k].sum() / max((_vshare(P)[don, k] * L[don, :K].sum(1)).sum(), 1e-9)
        ratio[k] = r_prov
        stood_both[k] = stood[muni]
        miss = (sL < absent) & (sP >= stand)
        if impute and miss.any():
            tgt = miss[muni]
            r = _lookalike_ratio(case, L, P, k, don, tgt, nn)
            s = np.clip(gamma * r * _vshare(P)[tgt, k], 0, 0.95)
            Lt = _set_share(L[tgt], k, s)
            L[tgt] = Lt
            info[f"imputed_{k}"] = {"munis": [case.muni_codes[m] for m in np.where(miss)[0]],
                                    "ratio_median": float(np.median(r)), "ratio_prov": float(r_prov),
                                    "gamma": float(gamma)}
    if floor > 0:
        for k in floor_parties:
            if k not in ratio:
                continue
            r = ratio[k]
            sL, sP = _vshare(L)[:, k], _vshare(P)[:, k]
            fP = np.maximum(sP, floor * sL / r)
            fL = np.maximum(sL, floor * sP * r)
            if scope == "stood":
                fP = np.where(stood_both[k], fP, sP)
                fL = np.where(stood_both[k], fL, sL)
            nP, nL = int((fP > sP + 1e-12).sum()), int((fL > sL + 1e-12).sum())
            P = _set_share(P, k, np.clip(fP, 0, 0.95))
            L = _set_share(L, k, np.clip(fL, 0, 0.95))
            info[f"floored_{k}"] = (nL, nP)
    if nb_floor > 0:
        # neighbour floor: a VD's share may not fall below nb_floor x its neighbours' share in
        # the same election (a zero in an area where the party polls is treated as noise)
        for k in floor_parties:
            for nm in ("L", "P"):
                C = L if nm == "L" else P
                if C[:, k].sum() <= 0:
                    continue
                s, sn = _vshare(C)[:, k], _nb_share(C, k, case.edges)
                f = np.maximum(s, nb_floor * sn)
                if scope == "stood":
                    f = np.where((_muni_share(C, muni, k, M) >= stand)[muni], f, s)
                info[f"nbfloor_{nm}{k}"] = int((f > s + 1e-12).sum())
                C = _set_share(C, k, np.clip(f, 0, 0.95))
                if nm == "L":
                    L = C
                else:
                    P = C
    for k in force_lge:
        if k not in ratio:
            continue
        P = _set_share(P, k, np.clip(_vshare(L)[:, k] / ratio[k], 0, 0.95))
        info[f"forced_{k}"] = ratio[k]
    return L, P, info


def _rebased_case(case, _tag="", **kw):
    """Re-based copy of the case, cached so m_swing's per-case prep cache (keyed by id) hits."""
    key = (id(case), case.name, case.n, _tag, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _CASES:
        L, P, info = _rebase(case, **kw)
        c2 = dataclasses.replace(case, prev_lge=L, prev_pe=P)
        _CASES[key] = (c2, info, case)        # keep the original alive so its id is not reused
    return _CASES[key][0], _CASES[key][1]


def _live_gamma(case, counted, prior=0.5, n0=20.0, lo=0.1, hi=1.5, step=0.05, stand=0.002,
                absent=0.001):
    """Strength of the LGE-equivalent imputation learned from the count (counted VDs only).

    For each party with an imputation, rho_s = (2026 share / previous-LGE share) over counted VDs
    in municipalities where it stood, rho_n = (2026 share / (r_prov x previous-PE share)) over
    counted VDs where it did not; gamma = rho_n / rho_s makes the swing on the imputed base equal
    in both regions. Shrunk to `prior` with n0 pseudo-VDs, clipped, rounded to `step` (so the
    re-based case, and m_swing's prep cache, can be reused across calls). Returns {party: gamma}."""
    L, P = case.prev_lge.astype(float), case.prev_pe.astype(float)
    muni = case.muni.astype(int)
    M = int(muni.max()) + 1
    Yc = case.Y[counted].astype(float)
    out = {}
    both = (L[:, :K].sum(0) > 0) & (P[:, :K].sum(0) > 0)
    for k in np.where(both)[0]:
        sL, sP = _muni_share(L, muni, k, M), _muni_share(P, muni, k, M)
        stood = (sL >= stand) & (sP > 0)
        miss = (sL < absent) & (sP >= stand)
        if not stood.any() or not miss.any():
            continue
        don = stood[muni]
        r_prov = L[don, k].sum() / max((_vshare(P)[don, k] * L[don, :K].sum(1)).sum(), 1e-9)
        cs_, cn_ = stood[muni][counted], miss[muni][counted]
        g = prior
        if cs_.sum() >= 5 and cn_.sum() >= 1:
            ys = Yc[cs_, k].sum() / max(Yc[cs_, :K].sum(), 1)
            ls = L[counted][cs_, k].sum() / max(L[counted][cs_, :K].sum(), 1)
            yn = Yc[cn_, k].sum() / max(Yc[cn_, :K].sum(), 1)
            pn = P[counted][cn_, k].sum() / max(P[counted][cn_, :K].sum(), 1)
            if ls > 0 and pn > 0 and ys > 0:
                g_obs = (yn / (r_prov * pn)) / (ys / ls)
                n = float(cn_.sum())
                g = (n * g_obs + n0 * prior) / (n + n0)
        out[int(k)] = float(np.round(np.clip(g, lo, hi) / step) * step)
    return out


# ----------------------------------------------------------------------------- district level
def _district(codes):
    """District of each municipality: the metros pooled ('METRO'), Sedibeng (GT42x -> DC42),
    West Rand (GT48x -> DC48). In Gauteng this is also 'where ActionSA stood in 2021'."""
    out = []
    for c in codes:
        out.append("DC42" if c.startswith("GT42") else "DC48" if c.startswith("GT48") else "METRO")
    return out


_orig_groups_for = ms._groups_for


def _groups_for_dist(p, spec, cl=None):
    """m_swing._groups_for plus a 'dist' grouping (district, p.dist). Installed into m_swing at
    import (runtime only; every other grouping is delegated unchanged, so other methods are
    unaffected)."""
    out = []
    for g in spec:
        if g == "dist":
            out.append((p.dist, int(p.dist.max()) + 1))
        else:
            out.extend(_orig_groups_for(p, (g,), cl))
    return out


if getattr(ms._groups_for, "__name__", "") != "_groups_for_dist":
    ms._groups_for = _groups_for_dist


def _add_district(cs, p):
    """district labels on the prep + district dummies (metros = reference) appended to the
    ridge features, once per re-based case."""
    if getattr(p, "dist_added", False):
        return
    dcodes = _district(cs.muni_codes)
    uniq = sorted(set(dcodes))
    lab_m = np.array([uniq.index(d) for d in dcodes])
    p.dist = lab_m[p.muni]
    Dd = np.eye(len(uniq))[p.dist]
    keep = [j for j, u in enumerate(uniq) if u != "METRO"] if "METRO" in uniq else list(range(1, len(uniq)))
    if keep:
        for f in list(p.feats):
            p.feats[f] = np.column_stack([p.feats[f], Dd[:, keep]])
            p.fnames[f] = p.fnames[f] + [f"dist_{uniq[j]}" for j in keep]
    p.dist_added = True


# ----------------------------------------------------------------------------- hybrid
def _hybrid(case, counted, rng, c=2.0, nd=ms.S_DRAWS, rbase="avg", lbase="mix", dist=False,
            dist_after_muni=False, **rb):
    """m_swing._hybrid on the re-based case, with a choice of base for the ratio part and an
    optional district level (EB level in the ratio swing + dummies in the regression)."""
    if rb.get("gamma") == "live":
        g = _live_gamma(case, counted)
        rb = dict(rb, gamma=g.get(ASA, 1.0))     # (only ActionSA is ever imputed in Gauteng)
    cs, info = _rebased_case(case, _tag="dist" if dist else "", **rb)
    p = ms._prep(cs)
    levels = ms.CM_LEVELS
    if dist:
        _add_district(cs, p)
        levels = (("km20", "muni", "dist", "cm20", "ward") if dist_after_muni
                  else ("km20", "dist", "muni", "cm20", "ward"))
    unc = ~counted
    rp = ms._share_projection(cs, counted, rng, scale="ratio", base=rbase, levels=levels,
                              nd=nd, **ms.UNC)
    gp = ms._logit_projection(cs, counted, rng, nd=0, base=lbase, fixed="ridge", feats="all",
                              cv="group", groups=("dist", "muni", "ward") if dist else ("muni", "ward"),
                              new_party="reg")
    if counted.sum() > 10:
        hu, h0 = ms._extrap_sd(p, counted, np.zeros((counted.sum(), 1)), np.ones((counted.sum(), 1)),
                               return_dist=True)
        a = np.clip(1 - (hu - h0) / (c * h0), 0, 1)
    else:
        a = np.zeros(unc.sum())
    mean = rp.vd_mean.copy()
    mean[unc] = a[:, None] * gp.vd_mean[unc] + (1 - a[:, None]) * rp.vd_mean[unc]
    info = dict(info, mean_a=float(np.average(a, weights=p.reg[unc])))
    if nd <= 0:
        return Projection(mean, None, info)
    return Projection(mean, ms._recentre(mean, rp, counted), info)


def _make(doc, **kw):
    def method(case, counted, rng):
        return _hybrid(case, counted, rng, **kw)
    method.__doc__ = doc
    return method


vasa_ref = _make("hybrid_cf re-implemented here without any re-basing (sanity check: must equal "
                 "m_swing:hybrid_cf).", impute=False)
vasa_imp = _make("hybrid_cf on the LGE-equivalent history: ActionSA (any party) imputed in the "
                 "municipalities where it did not stand in the previous LGE from PE x look-alike "
                 "LGE/PE ratio.", impute=True)
vasa_imp_mix = _make("vasa_imp + ratio-part base blend learned live per category (rbase='mix').",
                     impute=True, rbase="mix")
vasa_imp_force = _make("vasa_imp + forced LGE weighting for ActionSA (PE column replaced by "
                       "LGE-equivalent / ratio).", impute=True, force_lge=(ASA,))
vasa_imp_floor = _make("vasa_imp + low-base floor 0.3 for ActionSA (a share below 0.3x what the "
                       "other election implies is raised to that).", impute=True, floor=0.3)
vasa_floor = _make("hybrid_cf + low-base floor 0.3 for ActionSA, no imputation.", impute=False,
                   floor=0.3)
vasa_imp_floor_all = _make("vasa_imp + low-base floor 0.2 for every party with both elections.",
                           impute=True, floor=0.2, floor_parties=(0, 1, 2, 4, 5))
vasa_imp_dist = _make("vasa_imp + district level (metros / Sedibeng / West Rand): EB level in the "
                      "ratio swing and dummies in the regression, so a municipality with few counted "
                      "VDs borrows from its district (where ActionSA did / did not stand in 2021).",
                      impute=True, dist=True)
vasa_dist = _make("hybrid_cf + district level only (no re-basing).", impute=False, dist=True)
vasa_imp_dist_nb = _make("vasa_imp_dist + neighbour floor 0.3 for ActionSA (share >= 0.3 x the "
                         "touching VDs' share, both elections).", impute=True, dist=True, nb_floor=0.3)
vasa_imp_nb = _make("vasa_imp + neighbour floor 0.3 for ActionSA.", impute=True, nb_floor=0.3)
vasa_g03 = _make("LGE-equivalent imputation at strength gamma = 0.3 (s_LGE* = 0.3 r s_PE).",
                 impute=True, gamma=0.3)
vasa_glive = _make("LGE-equivalent imputation with its strength gamma learned live from the "
                   "count (counted VDs where ActionSA stood vs did not stand in 2021).",
                   impute=True, gamma="live")
vasa_glive_dist = _make("vasa_glive + district level.", impute=True, gamma="live", dist=True)
vasa_glive_floor = _make("vasa_glive + cross-election floor 0.3 for ActionSA.", impute=True,
                         gamma="live", floor=0.3)
vasa_fnb = _make("hybrid_cf + cross-election floor 0.3 + neighbour floor 0.3 for ActionSA (no "
                 "explicit imputation; the cross-election floor gives the not-stood municipalities "
                 "an LGE base of 0.3 r s_PE).", impute=False, floor=0.3, nb_floor=0.3)
vasa_fnb_dist = _make("vasa_fnb + district level.", impute=False, floor=0.3, nb_floor=0.3, dist=True)
vasa_floor_dist = _make("vasa_floor + district level.", impute=False, floor=0.3, dist=True)
vasa_vdf = _make("No imputation (the not-stood zero is kept as a region indicator); VD-level "
                 "low-base floors for ActionSA only where it stood: cross-election floor 0.3 and "
                 "neighbour floor 0.3.", impute=False, floor=0.3, nb_floor=0.3, scope="stood")
vasa_vdf_dist = _make("vasa_vdf + district level.", impute=False, floor=0.3, nb_floor=0.3,
                      scope="stood", dist=True)
vasa_vdf_nb = _make("Neighbour floor 0.3 for ActionSA only where it stood (no cross-election "
                    "floor, no imputation).", impute=False, nb_floor=0.3, scope="stood")
vasa_vdf_x = _make("Cross-election floor 0.3 for ActionSA only where it stood (no neighbour "
                   "floor, no imputation).", impute=False, floor=0.3, scope="stood")
vasa_f02_dist = _make("vasa_floor_dist with floor 0.2.", impute=False, floor=0.2, dist=True)
vasa_f05_dist = _make("vasa_floor_dist with floor 0.5.", impute=False, floor=0.5, dist=True)
vasa_floor_dist2 = _make("vasa_floor_dist with the district level after the municipality in the "
                         "ratio swing (the correlated draws then stay on cluster + municipality, "
                         "as in hybrid_cf).", impute=False, floor=0.3, dist=True, dist_after_muni=True)


# ----------------------------------------------------------------------------- final
hybrid_asa = _make(
    "BEST of this family (live check + benchmark): hybrid_cf on a re-based history plus a "
    "district level. (a) Cross-election low-base floor 0.3 for ActionSA: a VD whose ActionSA "
    "share in one predictor election is below 0.3x what the other implies (other share x the "
    "province LGE2021/PE2024 ratio, 2.79) is raised to that; in Sedibeng / West Rand, where "
    "ActionSA did not stand in LGE2021, this gives a partial LGE-equivalent base 0.3 x 2.79 x "
    "PE2024 share (a full LGE-equivalent over-predicts there: ActionSA grows less where it had "
    "no 2021 local organisation), and 73 metro VDs with a spurious PE2024 zero get a floor. "
    "(b) District level (metros / Sedibeng / West Rand): EB level in the ratio swing and dummies "
    "in the ridge, so a municipality with few counted VDs borrows its district's swing. "
    "Benchmarks: no party has both a previous LGE and a previous PE with a gap, so (a) is a no-op "
    "and only (b) acts.", impute=False, floor=0.3, dist=True)
