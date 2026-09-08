"""Explicit input-composition augmentation; no physical-parameter prior changes.

The original Beta/Bernoulli policy remains the generator default. Expanded
datasets use disjoint quota groups assigned independently of catalog RNGs.
All fractions here describe engineering allocations authorized by the human.
"""
from functools import lru_cache
import numpy as np

LEGACY_COMPOSITION = "beta-2-6-v1"
EXPANDED_COMPOSITION = "expanded-localization-v2"
GROUP_NAMES = ("beta", "near_localized", "all_localized")
COMPOSITION_RNG_NAMESPACE = 82001
NEAR_PERCENT = 5
ALL_PERCENT = 2
NEAR_MIN_NUMERATOR, NEAR_MIN_DENOMINATOR = 9, 10


def quota_counts(total):
    """Round supplemental quotas up, so minima hold even for small splits."""
    if type(total) is not int or total < 2:
        raise ValueError("expanded composition needs at least two catalog seeds")
    near = (NEAR_PERCENT * total + 99) // 100
    all_local = (ALL_PERCENT * total + 99) // 100
    return {"beta": total - near - all_local,
            "near_localized": near, "all_localized": all_local}


@lru_cache(maxsize=16)
def composition_plan(start, stop):
    """Per-split permutation: quotas do not depend on shard boundaries or N."""
    counts = quota_counts(stop - start)
    rng = np.random.default_rng(np.random.SeedSequence(
        [COMPOSITION_RNG_NAMESPACE, start, stop]))
    order = rng.permutation(stop - start)
    groups = np.zeros(stop - start, dtype=np.int8)
    n_all, n_near = counts["all_localized"], counts["near_localized"]
    groups[order[:n_all]] = GROUP_NAMES.index("all_localized")
    groups[order[n_all:n_all+n_near]] = GROUP_NAMES.index("near_localized")
    groups.setflags(write=False)
    return groups


def expanded_mask(seed, start, stop, uniforms, beta_probability):
    group = GROUP_NAMES[int(composition_plan(start, stop)[seed-start])]
    if group == "beta":
        return uniforms < beta_probability, group
    if group == "all_localized":
        return np.ones(len(uniforms), dtype=np.bool_), group
    # Sample the realized count, not a Bernoulli probability that can miss 0.9.
    rng = np.random.default_rng(np.random.SeedSequence(
        [COMPOSITION_RNG_NAMESPACE, seed, 1]))
    minimum = (NEAR_MIN_NUMERATOR*len(uniforms)+NEAR_MIN_DENOMINATOR-1)//NEAR_MIN_DENOMINATOR
    count = int(rng.integers(minimum, len(uniforms)+1))
    mask = np.zeros(len(uniforms), dtype=np.bool_)
    mask[np.argsort(uniforms, kind="stable")[:count]] = True
    return mask, group


def composition_config(policy):
    if policy == LEGACY_COMPOSITION:
        return {"policy": policy, "localization": "Beta(2,6) then Bernoulli"}
    if policy != EXPANDED_COMPOSITION:
        raise ValueError("unknown composition policy")
    return {"policy": policy, "allocation_percent": {
                "beta": 100-NEAR_PERCENT-ALL_PERCENT,
                "near_localized": NEAR_PERCENT, "all_localized": ALL_PERCENT},
            "assignment": "independent seeded permutation over the complete split",
            "rng_namespace": COMPOSITION_RNG_NAMESPACE,
            "near_localized": "uniform integer count from ceil(9*N/10) through N; random subset",
            "all_localized": "exactly N localized",
            "N": "unchanged DiscreteUniform inclusive 1..512 in every group",
            "theta_prior": "unchanged; composition assignment independent of physical parameters"}
