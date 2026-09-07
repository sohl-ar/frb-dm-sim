"""L0 engineering fixture. Host DM uses NATURAL LOG. Not full L2 G7."""
import time
import numpy as np
import pytest
from frbsim.catalog import generate_catalog, canonical_json
from frbsim.schema import Parameters
from frbsim.demo import fixture_config
from frbsim import tolerances as t

pytestmark = pytest.mark.gates

def test_g7_l0_fixture(gate):
    surveys, lf, mw = fixture_config()
    start = time.perf_counter()
    a = generate_catalog(10_000, Parameters(), surveys, lf, 7001, mw_model=mw)
    elapsed = time.perf_counter()-start
    b = generate_catalog(10_000, Parameters(), surveys, lf, 7001, mw_model=mw)
    same = canonical_json(a.simulator_payload()) == canonical_json(b.simulator_payload())
    gate("G7", "L0_fixture_seed_replay", {"bit_identical": same, "n": len(a.observations)},
         {"bit_identical": True}, same)
    gate("G7", "L0_fixture_runtime", elapsed, {"seconds_max": t.CATALOG_SECONDS}, elapsed <= t.CATALOG_SECONDS)
    numeric = [[o.ra_deg, o.dec_deg, o.dm_obs, o.fluence_jy_ms, g.z_true,
                g.dm_cosmic, g.dm_host_rest, g.dm_mw_ism, g.dm_mw_halo, g.L]
               for o, g in zip(a.observations, a.ground_truth)]
    count = int(np.count_nonzero(~np.isfinite(numeric)))
    gate("G7", "L0_fixture_nonfinite", count, {"equals": 0}, count == 0)
    statistics = []
    for left, right in zip([.05, .4, .8, 1.2], [.4, .8, 1.2, 1.5]):
        selected = [g.dm_cosmic for g in a.ground_truth if left <= g.z_true <= right]
        statistics.append({"z_bin": [left, right], "count": len(selected),
                           "dm_mean": float(np.mean(selected)),
                           "dm_5_95": np.quantile(selected, [.05, .95]).tolist()})
    gate("G7", "L0_fixture_statistics_report", statistics,
         {"report_only": True, "pending": "comparison in full L2 pipeline"}, True, hard=False)
