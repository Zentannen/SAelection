"""Judge's combinations of the v_asa (ActionSA baseline) and v_spatial / v_ward (local pull)
variants (family: v_best).

The three variants fix different things and do not overlap:

* ``m_vasa.hybrid_asa``  - the *level*: hybrid_cf on a re-based vote history (ActionSA low-base
  floor 0.3 between LGE2021 and PE2024) plus a district level (metros / Sedibeng / West Rand).
  It changes municipal and area-type swings, not the within-ward pattern.
* ``m_vspatial``         - the *local pattern*: simple kriging of out-of-fold residuals (same
  ward + tapered exponential in centroid distance), so counted ward-mates and neighbours pull
  an uncounted VD; ``spatial_rake_half`` rakes the ward-free pattern to spatial_pull's
  municipal totals.
* ``m_vward``            - the same idea with ward-blocked folds and raking to the no-ward base.

WINNER (judge, notes_v_judge.md): ``asa_rake_half`` = spatial_rake_half on the hybrid_asa base.
Live check (15:18, 29 tasks) vs hybrid_cf: group 1.638 -> 1.568, random 0.520 -> 0.445, time
0.122 -> 0.111, muni 2.048 -> 1.888, group cov90 0.892 -> 0.902; benchmark random + real2026:
province 0.251 -> 0.242, muni 0.743 -> 0.720, ward 1.768 -> 1.690, CRPS 0.335 -> 0.326.

The combinations below run the v_spatial / v_ward machinery unchanged on hybrid_asa's re-based
case and base model. Nothing is copied: the re-based case comes from ``m_vasa._rebased_case``
(+ ``_add_district``), the hybrid base with configurable pooling levels from
``m_vward._hybrid_base``, and the kriging / raking / conditional simulation from
``m_vspatial._spatial`` / ``_raked``. m_vspatial hard-wires its base model in two module
functions (``_base_mean`` / ``_base_draws``); ``_asa_base`` swaps them for hybrid_asa's
equivalents for the duration of one call (runtime only, restored in ``finally``; m_vspatial.py
is not edited and every other caller sees the original functions).

Rules kept: case.Y is read only at ``counted`` (all reads are inside the imported modules, which
were checked); the re-based history is pre-count data; out-of-fold residuals mean a counted
VD's own result never enters the base prediction it is compared with.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from contextlib import contextmanager

from . import m_swing as ms
from . import m_vasa as va          # importing installs the 'dist' grouping in m_swing
from . import m_vspatial as vs
from . import m_vward as vw
from .harness import Projection

ASA_RB = dict(impute=False, floor=0.3)        # hybrid_asa's re-basing (floor only)
# hybrid_asa's pooling (district before municipality, as in m_vasa.hybrid_asa) ...
FULL = dict(levels=("km20", "dist", "muni", "cm20", "ward"), reg_groups=("dist", "muni", "ward"))
NW = dict(levels=("km20", "dist", "muni", "cm20"), reg_groups=("dist", "muni"))
# ... and with the district after the municipality (vasa_floor_dist2: the correlated draw of the
# EB levels stays on cluster + municipality, as in hybrid_cf). Live point forecasts are almost
# the same, but on the benchmark the backfitting order matters: fitted after the municipality,
# the district level absorbs ~nothing (LGE2021 muni MAE 1.312 -> 1.387), so the "2" variants lose.
FULL2 = dict(levels=("km20", "muni", "dist", "cm20", "ward"), reg_groups=("dist", "muni", "ward"))
NW2 = dict(levels=("km20", "muni", "dist", "cm20"), reg_groups=("dist", "muni"))


def asa_case(case):
    """hybrid_asa's re-based copy of the case with the district level installed (cached in
    m_vasa, so m_swing's prep cache is shared with m_vasa.hybrid_asa)."""
    cs, _ = va._rebased_case(case, _tag="dist", **ASA_RB)
    va._add_district(cs, ms._prep(cs))
    return cs


@contextmanager
def _asa_base(full=FULL, nw=NW):
    """Point m_vspatial's base model at hybrid_asa (full) / hybrid_asa without ward (nw)."""
    old_mean, old_draws = vs._base_mean, vs._base_draws

    def base_mean(case, counted, base="nw", c=2.0):
        kw = nw if base == "nw" else full
        return vw._hybrid_base(case, counted, None, c=c, nd=0, **kw).vd_mean

    def base_draws(case, counted, rng, base, nd):
        if base == "nw":     # log-ratio model without ward level and without VD residual
            lp = ms._logit_projection(case, counted, rng, nd=nd, base="mix", new_party="const",
                                      cv="loo", resid=False, **ms.UNC,
                                      **ms._logit_kw_for(nw["levels"]))
        else:                # hybrid_asa's draws (its ratio model's predictive distribution)
            lp = ms._share_projection(case, counted, rng, scale="ratio", base="avg",
                                      levels=full["levels"], nd=nd, **ms.UNC)
        return lp.vd_mean, lp.draws

    vs._base_mean, vs._base_draws = base_mean, base_draws
    try:
        yield
    finally:
        vs._base_mean, vs._base_draws = old_mean, old_draws


def asa_hybrid(case, counted, rng):
    """hybrid_asa rebuilt from the parts used here (check: must equal m_vasa:hybrid_asa)."""
    return vw._hybrid_base(asa_case(case), counted, rng, nd=ms.S_DRAWS, **FULL)


def asa_spatial_pull(case, counted, rng):
    """hybrid_asa + m_vspatial's spatial pull: simple kriging of hybrid_asa's out-of-fold
    (5 random folds) log-ratio residuals, same-ward + tapered exponential (R = 5 km) covariance
    fitted on cross-fold counted pairs; draws = hybrid_asa's draws re-centred."""
    with _asa_base():
        return vs._spatial(asa_case(case), counted, rng, base="full", F=5)


def asa_rake_half(case, counted, rng):
    """WINNER. m_vspatial.spatial_rake_half on hybrid_asa: VD pattern = log-ratio average of
    [hybrid_asa without ward level + ward / two-scale spatial kriging] and asa_spatial_pull,
    raked to asa_spatial_pull's municipal totals; draws = the no-ward model's predictive
    distribution + kriging posterior of the field (conditional simulation), re-centred."""
    with _asa_base():
        return vs._raked(asa_case(case), counted, rng, w_nw=0.5)


def asa_rake_half2(case, counted, rng):
    """asa_rake_half with the district level after the municipality (vasa_floor_dist2 order):
    same idea, the correlated EB draw stays on cluster + municipality."""
    with _asa_base(FULL2, NW2):
        return vs._raked(asa_case(case), counted, rng, w_nw=0.5)


def asa_spatial_pull2(case, counted, rng):
    """asa_spatial_pull with the district level after the municipality (draw order of
    vasa_floor_dist2)."""
    with _asa_base(FULL2, NW2):
        return vs._spatial(asa_case(case), counted, rng, base="full", F=5)


def asa_ward_pull(case, counted, rng):
    """m_vward.ward_pull on hybrid_asa: hybrid_asa without ward level as base, ward-blocked
    out-of-fold residuals, ward + exponential spatial kriging, raked to the base's municipal
    totals; draws = base draws + kriging posterior, raked per draw."""
    return vw._pull(asa_case(case), counted, rng, spatial=True, rake="muni", base_kw=NW)


FINAL = ["asa_spatial_pull", "asa_rake_half", "asa_rake_half2", "asa_spatial_pull2", "asa_ward_pull"]
