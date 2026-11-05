"""Stage 1: learn model structure from 2016-2024 Gauteng results (writes artifacts/)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from saelection.train import main

if __name__ == "__main__":
    main()
