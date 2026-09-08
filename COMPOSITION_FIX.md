# Composition and encoding fix — code preparation only

The human's immediate instruction limits this session to implementation and
documentation. Data generation, normalization fitting, retraining, tests and
physics/posterior validation have NOT run. Existing measured results remain
historical evidence; Phase 2a is still incomplete.

## Composition design

The expanded policy assigns disjoint groups over each complete split: 5% to
near-localized catalogs, 2% to all-localized catalogs, and the remaining 93%
to the original Beta(2,6)-then-Bernoulli mechanism. The approximate 92.5%
remainder in the request is resolved to 93% so the disjoint allocations sum
to 100%. The all-localized group also contributes to the requested minimum
fraction of catalogs with realized ell >= 0.9.

Each group retains the original DiscreteUniform(1,512) size draw. Group
assignment is a separately seeded permutation of the split's catalog IDs,
independent of the physical-parameter and size RNG stream. Supplemental
quotas round upward for split lengths not divisible by 100. Group membership
does not depend on batching and is recorded only as provenance, never as a
network feature.

The near-localized group draws a localized count uniformly from ceil(0.9*N)
through N, then chooses a random subset of that size. This guarantees the
requested realized fraction instead of only constraining its Bernoulli
probability. Small near-localized catalogs can also be exactly all-localized.
The dedicated all-localized group always exposes all redshifts. No all-
unlocalized quota was added beyond the specified mixture.

The default physics generator and existing gate-generated composition remain
Beta(2,6); augmentation is an explicit dataset policy. The expanded policy
applies to all three future dataset splits, including validation and test.
Parameter priors, population-family mixture, physics, N distribution and
selection model are unchanged. The changed distribution is the distribution
of observed catalog compositions used for training and held-out diagnostics.
Expanded-split aggregate calibration must be labeled as such; it is not an
unchanged Beta-only benchmark. Broader training support alone does not prove
either calibration or optimal information extraction.

## Deferred execution

Future dataset command (NOT executed):

```powershell
python -m frbsbi.data --composition expanded-localization-v2
```

It targets `work/training-data-v2/` and
`results/training-data-v2-manifest.json`, and refuses existing output. Original
split seed ranges are retained for paired comparisons; split separation is
unchanged. These will be regenerated composition variants, not a claim of
independent physical catalogs relative to the previous dataset. All physical
RNG draws are preserved; the supplementary mask RNG is separate.

The future manifest records shard hashes, policy, quotas and measured realized
counts. No future hashes or performance/calibration numbers exist yet.

Regression checks have been written but deliberately not run. Later checks
must cover quota counts, replay across batching, realized mask fractions,
preservation of latent/observable draws other than redshift availability,
all-size support, binary encoding and legacy-checkpoint replay. Acceptance
tolerances and criteria remain unchanged.
