"""Blends of the best methods from each family.

blend_cf_ml: equal-weight mixture of m_swing.hybrid_cf and m_ml.ml_mixed_gbm_shift. The mean is
the average of the two means; the draws are a 50/50 mixture of both methods' draws, so the
interval also carries the disagreement between the two methods.
"""
from __future__ import annotations

import numpy as np

from .harness import Projection
from .m_ml import ml_mixed_gbm_shift
from .m_swing import hybrid_cf


def _mix(projs, rng, weights=None):
    w = np.full(len(projs), 1 / len(projs)) if weights is None else np.asarray(weights, float)
    mean = sum(wi * p.vd_mean for wi, p in zip(w, projs))
    draws = None
    if all(p.draws is not None for p in projs):
        S = min(p.draws.shape[0] for p in projs)
        pick = rng.choice(len(projs), size=S, p=w)
        draws = np.stack([projs[k].draws[rng.integers(projs[k].draws.shape[0])] for k in pick])
    return Projection(mean, draws, {"blend": [p.info for p in projs]})


def blend_cf_ml(case, counted, rng):
    return _mix([hybrid_cf(case, counted, rng), ml_mixed_gbm_shift(case, counted, rng)], rng)
