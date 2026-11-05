"""Design matrices for one 'transition': target election <- (previous LGE, previous national/provincial).

All quantities live on the additive-log-ratio (alr) scale against ABSTAIN, i.e. for VD i and
party k:  log(p_ik / p_i,abstain), where p are shares of the *registered* electorate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CATS, PARTIES, PROCESSED, load_history
from .geo import load_graph

K = len(PARTIES)
SMOOTH = 0.5


def alr(counts: np.ndarray) -> np.ndarray:
    """counts: (n, 7) in CATS order -> (n, 6) alr vs ABSTAIN, with +0.5 smoothing."""
    c = counts + SMOOTH
    return np.log(c[:, :K]) - np.log(c[:, [K]])


def alr_var(counts: np.ndarray) -> np.ndarray:
    """Delta-method sampling variance of the empirical alr."""
    c = counts + SMOOTH
    return 1.0 / c[:, :K] + 1.0 / c[:, [K]]


@dataclass
class Design:
    name: str
    vd: np.ndarray            # (n,) VD numbers in graph order
    muni: np.ndarray          # (n,) municipality index
    muni_codes: list
    edges: np.ndarray         # (m, 2) contiguity edges
    X1: np.ndarray            # (n, K) previous-LGE alr (centred per party)
    X2: np.ndarray            # (n, K) previous-PE alr (centred per party)
    has_history: np.ndarray   # (K,) party has any previous result in this transition
    imputed: np.ndarray       # (n,) VD history imputed from touching neighbours
    Y: np.ndarray | None = None        # (n, 7) target counts (CATS order)
    contested: np.ndarray | None = None  # (K,) party stood in the target election
    registered: np.ndarray | None = None
    centres: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.vd)


def _neighbour_fill(values: np.ndarray, known: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fill rows with known==False by iteratively averaging touching neighbours."""
    v = values.copy()
    known = known.copy()
    v[~known] = 0.0
    for _ in range(50):
        if known.all():
            break
        s = np.zeros_like(v)
        c = np.zeros(len(v))
        for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            kb = known[b]
            np.add.at(s, a[kb], v[b[kb]])
            np.add.at(c, a[kb], 1)
        newly = (~known) & (c > 0)
        v[newly] = s[newly] / c[newly][:, None]
        known = known | newly
    return v


def _predictor(hist, key, vds, edges):
    h = hist[hist.election == key].set_index("vd")
    present = np.isin(vds, h.index.to_numpy())
    counts = h.reindex(vds)[CATS].fillna(0).to_numpy(float)
    x = alr(counts)
    stood = h[PARTIES].sum().to_numpy() > 0
    x = _neighbour_fill(x, present, edges)
    return x, stood, present


def build_design(target: str | None, prev_lge: str, prev_pe: str, geom_year: int,
                 name: str | None = None) -> Design:
    hist = load_history()
    nodes, edges = load_graph(geom_year)
    vds = nodes["vd"].to_numpy()

    x1, s1, p1 = _predictor(hist, prev_lge, vds, edges)
    x2, s2, p2 = _predictor(hist, prev_pe, vds, edges)
    # party absent from one predictor election -> use the other; absent from both -> no history
    x1 = np.where(s1, x1, x2)
    x2 = np.where(s2, x2, x1)
    has_hist = s1 | s2
    centres = {"X1": x1.mean(0), "X2": x2.mean(0)}
    X1 = np.where(has_hist, x1 - centres["X1"], 0.0)
    X2 = np.where(has_hist, x2 - centres["X2"], 0.0)

    # municipality of each VD
    if geom_year == 2026:
        cw = pd.read_csv(PROCESSED / "vd_ward_crosswalk_2026.csv")
        mmap = cw.groupby("vd")["muni"].first()
    else:
        src = target if target else prev_pe
        mmap = hist[hist.election == src].groupby("vd")["muni"].first()
    muni_s = pd.Series(vds).map(mmap)
    codes = sorted(muni_s.dropna().unique())
    muni = muni_s.map({c: i for i, c in enumerate(codes)}).to_numpy()

    d = Design(name=name or f"{target}<-{prev_lge}+{prev_pe}", vd=vds, muni=muni.astype(int),
               muni_codes=codes, edges=edges, X1=X1, X2=X2, has_history=has_hist,
               imputed=~(p1 & p2), centres=centres)
    if target is not None:
        h = hist[hist.election == target].set_index("vd").reindex(vds)
        d.Y = h[CATS].fillna(0).to_numpy(float)
        d.registered = h["electorate"].fillna(0).to_numpy(float)
        d.contested = h[PARTIES].sum().to_numpy() > 0
    return d
