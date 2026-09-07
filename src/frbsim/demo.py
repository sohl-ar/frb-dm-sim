"""Explicit L0 engineering smoke demo; host DM uses NATURAL LOG.

Default command requests full acceptance and exits nonzero while blocked.
Use --smoke for a labeled 5000-burst L0 fixture plus implemented gates.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
from .catalog import generate_catalog, canonical_json
from .schema import Parameters
from .selection import Survey
from .population import LuminosityFunction
from .mw import FixedISMFixture

FIXTURE_SOURCE = "SPEC-FINAL engineering smoke fixture; not an empirical survey or LF default"

def fixture_config():
    surveys = (Survey("fixture-localized", 0., 1., 0., 1., FIXTURE_SOURCE),
               Survey("fixture-unlocalized", 0., 0., 0., 1., FIXTURE_SOURCE))
    # Unit-L delta test mode; complete detection deliberately removes survey
    # claims from this engineering smoke. Fluence stays dimensionally correct.
    lf = LuminosityFunction("delta", 1., 1., 2., FIXTURE_SOURCE)
    mw = FixedISMFixture(30., "Macquart Table-1 replication subtraction; test fixture only")
    return surveys, lf, mw

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20200924)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    results = root/"results"
    results.mkdir(exist_ok=True)
    if not args.smoke:
        return subprocess.call([sys.executable, "-m", "pytest", "-m", "gates", "-q"], cwd=root)
    start = time.perf_counter()
    surveys, lf, mw = fixture_config()
    catalog = generate_catalog(5000, Parameters(), surveys, lf, args.seed, mw_model=mw)
    payload = catalog.simulator_payload()
    (results/"demo-simulator.json").write_text(canonical_json(payload), encoding="utf-8")
    (results/"demo-observations.json").write_text(canonical_json(catalog.observation_payload()), encoding="utf-8")
    summary = {"scope": "L0 engineering smoke; no empirical survey defaults; Phase 1 incomplete",
               "elapsed_generation_seconds": time.perf_counter()-start,
               "n_bursts": len(catalog.observations),
               "n_localized": sum(x.is_localized for x in catalog.observations),
               "metadata": catalog.metadata}
    (results/"demo-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}))
    return subprocess.call([sys.executable, "-m", "pytest", "-m", "gates", "--level", "l0", "-q"], cwd=root)

if __name__ == "__main__":
    raise SystemExit(main())
