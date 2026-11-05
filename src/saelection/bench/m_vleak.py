"""Leakage / overfitting audit of the judge's winner m_vbest:asa_rake_half (family: v_leak).

Nothing here is a new projection method. The wrappers run an existing method on a *sealed*
copy of the case, so that any read of the answer outside ``counted`` is either impossible or
visible:

* ``strict_*``  : case.Y outside ``counted`` is overwritten with random garbage counts (seeded),
  i.e. the method physically cannot see the held-out (or never-counted) VDs' results. If the
  method never reads case.Y[~counted], the projection is bit-identical to the unsealed run.
* ``strict2_*`` : as strict, and in addition the held-out VDs' 2026 electorate (which the live
  check takes from the count itself, = Y.sum(1) for every reported VD) is replaced with the
  pre-count estimate the live run uses for VDs that have not reported (crosswalk registered
  population x growth learned from the counted VDs only). This removes the only per-VD
  quantity the live check derives from held-out results (a harness artefact shared by every
  method, not a leak specific to one method), so the method ranking can be re-checked without it.

Each call builds a new Case object, so every cache keyed by id(case) (m_swing prep, m_vasa
re-based case, m_vspatial / m_vward out-of-fold caches) is rebuilt from the sealed case: no
cached residual or base fit computed on another task's counted set can enter.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses

import numpy as np
import pandas as pd

from . import m_swing as ms
from . import m_vbest as vb

_KEEP: list = []          # keep sealed cases alive so no id() is reused within a process


def _reg_precount(case, counted):
    """Pre-count electorate for every non-counted VD: crosswalk registered population x the
    growth factor learned on counted VDs only (what live_check uses for unreported VDs)."""
    from ..data import PROCESSED
    cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")
    base = cw.groupby("vd")["vd_regpop"].first().reindex(case.vd).to_numpy(float)
    ok = counted & np.isfinite(base)
    growth = case.registered[ok].sum() / base[ok].sum()
    reg = case.registered.astype(float).copy()
    reg[~counted] = base[~counted] * growth
    return reg


def seal(case, counted, seed=12345, reg_precount=False):
    """Copy of the case with Y[~counted] replaced by random garbage counts."""
    Y = case.Y.astype(float).copy()
    g = np.random.default_rng(seed)
    nu = int((~counted).sum())
    Y[~counted] = g.integers(0, 3000, size=(nu, Y.shape[1])).astype(float)
    kw = dict(Y=Y)
    if reg_precount:
        kw["registered"] = _reg_precount(case, counted)
    cs = dataclasses.replace(case, **kw)
    _KEEP.append(cs)
    if len(_KEEP) > 200:
        del _KEEP[:100]
    return cs


def _wrap(fn, reg_precount=False):
    def method(case, counted, rng):
        return fn(seal(case, counted, reg_precount=reg_precount), counted, rng)
    method.__doc__ = f"{fn.__name__} on a sealed case (Y[~counted] = garbage" + (
        ", held-out electorate = pre-count estimate)." if reg_precount else ").")
    return method


strict_asa_rake_half = _wrap(vb.asa_rake_half)
strict_hybrid_cf = _wrap(ms.hybrid_cf)
strict2_asa_rake_half = _wrap(vb.asa_rake_half, reg_precount=True)
strict2_hybrid_cf = _wrap(ms.hybrid_cf, reg_precount=True)
