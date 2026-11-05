"""Stage 0: tidy VD-level history + geometry/contiguity caches (writes data/processed/)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from saelection.data import build_history
from saelection.geo import build_geo_cache

if __name__ == "__main__":
    build_history()
    build_geo_cache()
