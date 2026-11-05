"""Voting-district geometry: contiguity (touching) graphs, 2026 VD->ward crosswalk.

Spatial structure in the model is *only* contiguity: two VDs are neighbours iff their
polygons touch (queen contiguity, with a 5 m tolerance for digitising slivers/gaps).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .data import PROCESSED, ROOT

GEO = ROOT / "data" / "geo"
METRIC_CRS = "EPSG:32735"  # UTM 35S, metres
TOUCH_TOL_M = 5.0

GEO_FILES = {
    2021: GEO / "2021_voting-district.shapefile" / "2021" / "vd2021.shp",
    2024: GEO / "2024_voting-district.shapefile" / "2024" / "vd2024.shp",
    2026: GEO / "2026_voting-district.gpkg" / "VotingDistricts2026.gpkg",
}


def load_vd_polygons(year: int) -> gpd.GeoDataFrame:
    f = GEO_FILES[year]
    if year == 2021:
        g = gpd.read_file(f).rename(columns={"CODE": "vd"})[["vd", "geometry"]]
        g["vd"] = g["vd"].astype(np.int64)
        h = pd.read_csv(PROCESSED / "vd_history_gauteng.csv", usecols=["vd", "election"])
        g = g[g["vd"].isin(h.loc[h.election == "LGE2021", "vd"])]
    elif year == 2024:
        g = gpd.read_file(f, where="Province='Gauteng'").rename(columns={"VDNumber": "vd"})[["vd", "geometry"]]
    else:
        g = gpd.read_file(f, where="Province='Gauteng'").rename(columns={"VDNumber": "vd"})[["vd", "geometry"]]
    g["vd"] = g["vd"].astype(np.int64)
    g = g.to_crs(METRIC_CRS)
    g["geometry"] = g.geometry.make_valid()
    g = g.dissolve(by="vd").reset_index()  # split VDs come as several pieces
    return g


def contiguity_edges(g: gpd.GeoDataFrame) -> np.ndarray:
    """Undirected edges (i<j, row positions in g) between VDs whose polygons touch."""
    buf = g.geometry.buffer(TOUCH_TOL_M / 2)
    left, right = gpd.GeoSeries(buf).sindex.query(buf, predicate="intersects")
    m = left < right
    return np.unique(np.stack([left[m], right[m]], axis=1), axis=0)


def graph_summary(edges: np.ndarray, n: int) -> dict:
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    a = a + a.T
    ncomp, lab = connected_components(a, directed=False)
    deg = np.asarray(a.sum(1)).ravel()
    return {"n": n, "edges": len(edges), "components": ncomp,
            "islands": int((deg == 0).sum()), "mean_degree": float(deg.mean()), "labels": lab}


def bym2_scale(edges: np.ndarray, n: int) -> float:
    """Riebler et al. (2016) scaling factor: geometric mean of the marginal variances of the
    ICAR generalised inverse, per connected component (singletons excluded)."""
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).toarray()
    a = a + a.T
    q = np.diag(a.sum(1)) - a
    _, lab = connected_components(a, directed=False)
    logs = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if len(idx) < 2:
            continue
        qc = q[np.ix_(idx, idx)]
        w, v = np.linalg.eigh(qc)
        keep = w > 1e-8
        cov = (v[:, keep] / w[keep]) @ v[:, keep].T
        logs.append(np.log(np.diag(cov)))
    return float(np.exp(np.mean(np.concatenate(logs))))


def ward_crosswalk_2026() -> pd.DataFrame:
    """One row per (VD, ward) piece of the 2026 delimitation, with the share of the VD's
    registered voters that falls in that ward (from the MDB/IEC WardPop field)."""
    v = gpd.read_file(GEO_FILES[2026], where="Province='Gauteng'", read_geometry=False)
    v = v.rename(columns={"VDNumber": "vd", "WardID": "ward", "CAT_B": "muni", "WardNo": "ward_no",
                          "REGPOP": "vd_regpop", "WardPop": "piece_regpop", "VotingStat": "station"})
    v["vd"] = v["vd"].astype(np.int64)
    v["ward"] = v["ward"].astype(np.int64)
    v = v.groupby(["vd", "ward"], as_index=False).agg(
        muni=("muni", "first"), ward_no=("ward_no", "first"), station=("station", "first"),
        vd_regpop=("vd_regpop", "first"), piece_regpop=("piece_regpop", "sum"))
    tot = v.groupby("vd")["piece_regpop"].transform("sum")
    n = v.groupby("vd")["piece_regpop"].transform("size")
    v["ward_weight"] = np.where(tot > 0, v["piece_regpop"] / tot.where(tot > 0, 1), 1.0 / n)
    return v


def build_geo_cache() -> None:
    """Write VD order, centroids and contiguity edges for each year to data/processed."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for year in (2021, 2024, 2026):
        g = load_vd_polygons(year)
        e = contiguity_edges(g)
        s = graph_summary(e, len(g))
        cen = g.geometry.centroid.to_crs("EPSG:4326")
        pd.DataFrame({"vd": g["vd"], "lon": cen.x.round(6), "lat": cen.y.round(6),
                      "area_km2": (g.geometry.area / 1e6).round(4)}).to_csv(
            PROCESSED / f"vd_nodes_{year}.csv", index=False)
        pd.DataFrame({"i": e[:, 0], "j": e[:, 1]}).to_csv(PROCESSED / f"vd_edges_{year}.csv", index=False)
        print(year, {k: v for k, v in s.items() if k != "labels"})
        if year == 2026:
            simp = g.copy()
            simp["geometry"] = simp.geometry.simplify(20)
            simp.to_crs("EPSG:4326").to_file(PROCESSED / "vd_2026_gauteng.gpkg", driver="GPKG")
    w = gpd.read_file(GEO / "2026_ward.gpkg" / "Wards2026.gpkg", where="Province='Gauteng'")
    w = w.rename(columns={"WardID": "ward", "CAT_B": "muni", "WardNo": "ward_no"})[["ward", "muni", "ward_no", "geometry"]]
    w["ward"] = w["ward"].astype(np.int64)
    w = w.to_crs(METRIC_CRS)
    w["geometry"] = w.geometry.make_valid().simplify(20)
    w.to_crs("EPSG:4326").to_file(PROCESSED / "wards_2026_gauteng.gpkg", driver="GPKG")
    cw = ward_crosswalk_2026()
    cw.to_csv(PROCESSED / "vd_ward_crosswalk_2026.csv", index=False)
    print("crosswalk", cw.shape, cw.vd.nunique(), "VDs", cw.ward.nunique(), "wards")


def load_graph(year: int) -> tuple[pd.DataFrame, np.ndarray]:
    nodes = pd.read_csv(PROCESSED / f"vd_nodes_{year}.csv")
    edges = pd.read_csv(PROCESSED / f"vd_edges_{year}.csv")[["i", "j"]].to_numpy()
    return nodes, edges


if __name__ == "__main__":
    build_geo_cache()
