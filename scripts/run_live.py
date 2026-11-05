"""Stage 2: project the 2026 Gauteng count. Reads data/observed_2026/*.csv, writes output/live/."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from saelection.live import run

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None,
                    help="'module:function' bench method (e.g. m_swing:hybrid_cf) or 'bayes'; default: config/live.json 'method'")
    m = ap.parse_args().method or json.loads((pathlib.Path(__file__).resolve().parents[1] / "config" / "live.json").read_text()).get("method", "bayes")
    meta, prov = run(method=None if m == "bayes" else m)
    print("method:", meta["model"]["inference"].get("method"))
    print(f"{meta['pct_vds_counted']}% of VDs counted ({meta['n_vds_counted_exact']} with VD-level results, {meta['n_progress_units']} ward/municipal progress rows)")
    print(f"{'party':9s} {'proj. votes':>12s} {'90% interval':>23s} {'share':>7s} {'P(1st)':>7s}")
    for p, x in prov["parties"].items():
        v, s = x["votes"], x["share"]
        print(f"{p:9s} {v['mean']:12,d} [{v['p05']:10,d}, {v['p95']:10,d}] {100*s['mean']:6.1f}% {x['p_first']:7.2f}")
    print("turnout (valid/registered):", prov["turnout_valid"])
