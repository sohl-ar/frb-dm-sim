# Composition and encoding fix — code preparation only

The human's immediate instruction limits this session to implementation and
documentation. Data generation, normalization fitting, retraining, tests and
physics/posterior validation have NOT run. Existing measured results remain
historical evidence; Phase 2a is still incomplete.

Only static Python syntax parsing and Git diff/whitespace review were performed
in this session. They do not establish runtime correctness or scientific validity.

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

## Binary encoding and isolated run

New normalization reports use schema v2. `has_localized`, `has_unlocalized`
and `ols_available` have center zero and scale one, so the network receives
their original binary values. Continuous statistics retain training-only
mean/std scaling; constant continuous dimensions center to zero. There is
no clipping. This removes prevalence-dependent magnification of a binary
flag without discarding continuous information or changing network dimensions.

Identity normalization lives in the existing checkpoint buffers. The forward
operation does not override buffers when loading a historical checkpoint;
schema-v1 reports remain supported for historical replay. No existing
normalizer or checkpoint has been rewritten. The new fit has NOT executed.

Future preparation and training commands (NOT executed):

```powershell
python -m frbsbi.train_conditioned prepare --composition-v2
python -m frbsbi.train_conditioned launch --composition-v2
```

The first command scans the newly generated training split to fit normalization;
it is deferred along with the expensive work. The second uses the unchanged
training hyperparameters and statistics-bypass/ISAB/PMA architecture in
`results/conditioning-v2/`. The existing one-attempt guard remains in force.
The new run requires the expanded manifest and binary-identity normalization.
It records explicit dataset paths and normalization/manifest/shard hashes.

The acceptance reader now resolves dataset paths from the selected run's
training metadata, retaining legacy defaults for older runs. For a future
validation session, `FRBSBI_RUN_DIR=results/conditioning-v2` selects the new
run for the existing acceptance driver and pytest fixture. No gate has run,
and this variable has not been set in this session. Gate order, failure-stop
behavior, numerical comparisons, tolerances and required parameters are intact.
Contraction and other newly generated gate probes retain their established
Beta composition and overrides; cached validation/SBC uses the expanded split
and records that composition explicitly. Compare those populations by name.

Next execution session: run the deferred unit checks and generator replay
checks first, then generate the dataset, inspect recorded composition/size
coverage, fit normalization, train, and run acceptance. A future failure to
improve host contraction would still not by itself establish a physics limit;
a numerical reference posterior or justified information bound remains needed.

Older one-off scripts such as `check_statistics_bypass.py` and
`check_conditioning_ready.py` still target conditioning-v1; do not use them as
v2 evidence without explicitly updating their run and dataset paths. The
production prepare/launch/acceptance paths described above support v2.
