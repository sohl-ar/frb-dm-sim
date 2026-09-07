"""Pretraining RNG gates; these do not certify a trained posterior."""
import numpy as np
import pytest
from frbsbi.preflight import SeedRange, assert_disjoint_seed_ranges, run_t1
from frbsim.fields import sample_cosmic

pytestmark = pytest.mark.phase2a


def test_t1_production_sampling_statistics(request):
    result = run_t1()
    request.config.phase2a_t1 = result
    assert result["status"] == "pass", f"STOP: T1 sampling discrepancy: {result}"


def test_t1_output_follows_rng_order():
    # Metamorphic test: permuting the supplied random stream must permute the
    # output identically. A deterministic percentile grid cannot pass this.
    class Stream:
        def __init__(self, values):
            self.values, self.calls = values, 0

        def random(self, size):
            assert size == len(self.values)
            self.calls += 1
            return self.values.copy()

    rng = np.random.default_rng(60_000)
    uniforms = rng.random(10_000)
    permutation = rng.permutation(len(uniforms))
    original, permuted = Stream(uniforms), Stream(uniforms[permutation])
    z = np.full(len(uniforms), 0.5)
    a, b = sample_cosmic(z, original), sample_cosmic(z, permuted)
    assert original.calls == permuted.calls == 1
    assert np.unique(a).size > 1
    assert np.array_equal(a[permutation], b)


def test_t1_seed_overlap_is_hard_error():
    assert_disjoint_seed_ranges({"train": SeedRange(0, 10), "val": SeedRange(10, 20)})
    for validation in (SeedRange(9, 20), SeedRange(0, 10), SeedRange(2, 5)):
        with pytest.raises(ValueError, match="seed-space overlap"):
            assert_disjoint_seed_ranges({"train": SeedRange(0, 10), "val": validation})
