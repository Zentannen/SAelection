"""Build public/data/wards.geojson for Gauteng from the 2026 ward GeoPackage.

Usage: python3 scripts/build_gauteng_wards.py <wards_2026_gauteng.gpkg> public/data/wards.geojson
The GeoPackage is on the `main` branch at data/processed/wards_2026_gauteng.gpkg.
Needs shapely (pip install shapely).
"""
import json
import sqlite3
import struct
import sys

from shapely import wkb
from shapely.geometry import mapping

MUNIS = {
    "JHB": ("City of Johannesburg", "City of Johannesburg"),
    "TSH": ("City of Tshwane", "City of Tshwane"),
    "EKU": ("Ekurhuleni", "Ekurhuleni"),
    "GT421": ("Emfuleni", "Sedibeng"),
    "GT422": ("Midvaal", "Sedibeng"),
    "GT423": ("Lesedi", "Sedibeng"),
    "GT481": ("Mogale City", "West Rand"),
    "GT484": ("Merafong City", "West Rand"),
    "GT485": ("Rand West City", "West Rand"),
}
TOLERANCE = 0.0004  # degrees, ~40 m: keeps shared borders recognisable at map scale


def gpkg_geom(blob: bytes):
    # GeoPackage binary header: 'GP', version, flags, srs_id, then an envelope sized by flags
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    return wkb.loads(blob[8 + env:])


def rounded(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 5), round(coords[1], 5)]
    return [rounded(c) for c in coords]


def main(src, dst):
    con = sqlite3.connect(src)
    feats = []
    for blob, ward, muni, ward_no in con.execute(
        "select geom, ward, muni, ward_no from wards_2026_gauteng order by ward"
    ):
        name, district = MUNIS[muni]
        g = gpkg_geom(blob).simplify(TOLERANCE, preserve_topology=True)
        m = mapping(g)
        feats.append({
            "type": "Feature",
            "properties": {
                "ward_id": str(ward),
                "name": f"{name} Ward {ward_no}",
                "municipality": name,
                "district": district,
                "province": "Gauteng",
            },
            "geometry": {"type": m["type"], "coordinates": rounded(m["coordinates"])},
        })
    with open(dst, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"))
    print(f"wrote {len(feats)} wards to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
