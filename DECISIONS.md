# Phase 2a pretraining decisions

Authority: user-supplied SPEC-02a FINAL, preserved in
[docs/SPEC-02a-request.txt](docs/SPEC-02a-request.txt), amended by
[the consolidated authorization](docs/SPEC-02a-authorization.txt) and
[anchor correction v2](docs/ANCHOR_CORRECTION_V2.md). Phase 1 and Phase 2a
retain separate acceptance ledgers. Dataset, model and training evidence
are recorded under `results/`; these do not imply posterior acceptance.

## T1: production RNG

The production path in `src/frbsim/fields.py`, `sample_cosmic`, calls
`rng.random(flat.size)` and passes those uniforms to the inverse CDF. It
does not stratify samples or rescale their sample mean. The earlier G4
mean check calls `CosmicPDF.sample(500*2000, rng)` per redshift, which also
uses RNG uniforms. Its deterministic percentile calls measure band shape,
not the sample mean. A small observed mean error is possible under IID
sampling; it is not evidence of stratification by itself.

`tests/phase2a/test_t1.py` checks production draws against finite-cutoff
moments and independently checks that permuting RNG inputs permutes outputs.
The mean standard error uses the truncated variance. The unbiased sample
variance standard error requires the fourth central moment as well:

    SE(mean) = sqrt(variance/N)
    SE(S^2) = sqrt((mu4 - (N-3)/(N-1)*variance^2)/N)

The far tail makes the second check broad. Its passing result is limited
evidence, not proof of a correct distribution or Gaussian sampling errors.
The requested tolerance is unchanged. Executed measurements, cutoff,
configuration, seeds and software provenance are in
[results/phase2a_gates.json](results/phase2a_gates.json).

Seed allocation is an engineering convention, enforced by
`assert_disjoint_seed_ranges`: half-open training, validation, test and
audit ranges are recorded in that JSON. The implemented `generate_batch`
entry point calls this guard before allocating RNGs; overlap rejection
is tested at that entry point.
T1 uses fixed-redshift cosmic draws, not a generated training catalog.

## T2: catalog design

- **D1 accepted as specified:** independent, repeat-free sightlines.
  Application to unclustered repeaters is prohibited. Future ingestion
  must group by host; a later simulator may share cosmic DM within a host
  while drawing burst-dependent host contributions.
- **D2 accepted as specified:** observed catalog size is discrete uniform
  on 1 through 512; size 1024 is diagnostic. Phase 2a's `generate_batch`
  now generates this observed count directly via conditional selection.
- **D3 accepted as specified:** localization probability is Beta(2,6),
  followed by per-burst Bernoulli draws; redshift appears iff localized.
- **D4 authorized:** F_min=5 Jy ms, CHIME Catalog 1 selection proxy;
  sensitivity value 3.5 Jy ms is a Catalog 2 lower-limit proxy. The old
  Phase 1 engineering demo stays a complete-detection fixture. Phase 2a
  receives this separate, approved configuration.
- **D5 authorized:** differential Schechter slope gamma=-1.16, with
  log10 E_scale=41.84 and log10 E_min=30 in erg, converted by the approved
  E/L=9.52e31 convention. Implemented in `frbsbi.selection`, citing James
  Eq.14 and Baptista Table 2. L_scale is not a hard maximum. The specified
  gamma and energy-scale sensitivity ranges are preserved in the authority
  document. At gamma=-1.16, the detected fraction inherits the stated
  approximate factor 1.45 per decade of E_min uncertainty.

**The print discrepancy is resolved by human correction v2.** The executed
z=0.2 recurrence, quadrature, series and rejection checks are recorded in
`results/selection-audit.json`. The independent series uses complete gamma
and its small-x expansion; it does not reuse the incomplete-gamma recurrence.
Knee fluences are computed from Astropy-checked distances. Sampling above
the knee is tested; there is no upper LF cutoff. Power-law extrapolation
without exponential suppression explains the original high-z overestimate.
The conditional joint generator and its naive-rejection comparison are now
implemented; `results/generator-audit.json` records validation and runtime.
A broad LF retains partial distance information;
the eventual exchange-rate study is separate from G-P8's sanity threshold.

## T3: prior-predictive validation

The specified priors remain f_d uniform on [0.6,1], F log-uniform on
[0.05,1], host median log-uniform on [20,200], and host sigma_ln uniform
on [0.2,2]. Host lognormal parameters use natural logs. Both population
families belong in training, with their flag confined to ground truth.
The executed `results/prior-predictive.json` records per-event CDFs from
both numerical quadrature and random draws, as well as the diagnostic grid.
All six events pass. T3 was reported to the human before training. The criterion is
the per-event interval as the central 99% [0.5%,99.5%], at each event's
own redshift, with CDF values and the diagnostic grid also reported.

## Accepted architecture and acceptance amendments

G-P8 is required and hard; G-P3 remains reported/soft. Batched attention
must ignore padding via key_padding_mask=True at padding only. Shuffle
real bursts before padding. Parameter transforms now use prior-CDF
normalization (log first for log-uniform priors), followed by logit to
the real line; invert with sigmoid and the prior's inverse map. Physical
parameter space is used for ranks and coverage. SBC uses 1000 posterior
samples per trial, batched over shared conditioning. The native ISAB/PMA
and conditional flow-matching model is implemented in `frbsbi.model`;
only ODE integration uses torchdiffeq. The architecture audit tests
padding, shuffling, missing-embedding gradients, reversible transforms,
ODE step doubling, and bit-identical short-run weights and samples.

The smoke test will use full Macquart Table 1: sky coordinates, DM,
fluence and redshift. Those additional columns have not yet been
transcribed into a simulator fixture; none are fabricated.

The human has verified all six original table rows. The authorized
190102 redshift correction is implemented; every current test consumer
was located and rerun. `results/table1-amendment.json` links the before
and after ledgers and records changes to means, corrections and coverage.

Generator performance and pretraining model checks have been measured.
Training progress and checkpoints are recorded independently of posterior
acceptance. Phase 1's L2 and Milky Way work remains open.

## Print-revision rule and selection interpretation

Anchor PRINTS are human estimates and may be revised by explicit human
sign-off with recorded multi-method provenance. Gate tolerances and physics
anchors are not revisable this way. This creates no precedent for changing
G-P4 coverage bands or any other acceptance tolerance. The generated table
below supersedes the stale prints in the original stored specification.

The "quiet validation" observation is retired entirely. The prior
z<=0.5-dominance claim was an interpretation error: a per-redshift
selection fraction was confused with the population-weighted detected
density. Measurement caught the error. The approved interpretation is
z>0.5 dominance, with the mixed population peaking at moderate redshift,
approximately z=0.8-1.0. The generated fractions below give the exact
below-z=0.5 shares. The Phase 3 exchange-rate result and paper framing
must describe this moderate-redshift regime. No independent real-data
validation is claimed, and no population parameters were changed.

## Training engineering configuration

Architecture widths, optimizer settings and epoch limits are numerical
design choices, not literature-derived physical constants. They are
explicit configurations in `ModelConfig` and the training run's JSON.
Use float32 for network operations, float64 for the simulator and physical
transform boundaries, deterministic CPU operations, no mixed precision,
and disjoint catalog seeds. Shards live under ignored `work/training-data`;
their complete hashes, counts, seeds and generation config are tracked in
`results/training-data-manifest.json`. Checkpoints and logs are tracked.
Short-run determinism is an engineering check, not posterior calibration.

<!-- GENERATED_ANCHOR_EVIDENCE -->

Source: `results/selection-audit.json`; human print correction v2.

| z | Authorized comparison print (%) | Computed probability (%) |
|---|---:|---:|
| 0.1 | 2 | 2.054519 |
| 0.2 | 1-1.5 | 1.305892 |
| 0.5 | 0.5-0.7 | 0.537451 |
| 1 | 0.151 | 0.150759 |
| 1.4 | 0.0494 | 0.049359 |

z=0.2: x_t=0.00891841450745; integrand x^gamma exp(-x)=236.479285066.
Quadrature=0.013058924316535; series=0.013058924316535; rejection=0.012798
with standard error 0.00011352704. The row status is pass.

Knee fluence at z=1: 12.49720195 Jy ms.
Knee fluence at z=1.4: 5.45737152 Jy ms.
At z=1.4 the knee is above the cut: detection starts just below the knee and includes the exponentially suppressed bright tail.

The executed below-z=0.5 fractions (quadrature / rejection) are:

- sfr: 16.9358% / 17.2643%.
- constant_comoving: 32.2382% / 32.4903%.

Human third checks at z=1 and 1.4 agree at 10-15%; at z=0.2 approximately 1%.
These are labeled corroboration. The executed numerical checks are the arbiter.
<!-- END_GENERATED_ANCHOR_EVIDENCE -->

<!-- GENERATED_REPOSITORY_AUDIT -->

## Repository audit and training outcome (generated from execution JSON)

Training recorded 19 epochs, with best epoch 13 and validation loss 2.845229963.
G-P1 and G-P2 pass; G-P4 fails on F central-68% overcoverage. Three other
parameters pass the recorded marginal SBC criteria, which is not proof of
informative conditional learning.

| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Recorded result |
|---|---|---|---|---|
| f_d | 0.664 | 0.944 | 1 | pass |
| F | 0.722 | 0.950 | 0.77273 | fail |
| host_median | 0.687 | 0.956 | 0.236142 | pass |
| host_sigma_ln | 0.665 | 0.942 | 1 | pass |

H1's earlier-plateau claim cannot be assessed because per-parameter loss
histories were not recorded. Width ratios from the authorized replay and
archived rank/prior correlations show prior-like predictions across all
parameters. H2 shows true-F-position dependence but does not establish a
logit-gradient defect. H3 does not support recovery in localized-heavy
catalogs. Architectural versus physical cause is unresolved; no physics
finding, model repair, new training run or tolerance change is authorized
by this diagnosis. Details and conditional bounds are in AUDIT_REPORT.md.

The resume restored states and reproduced the update/validation mathematics,
but continued after the configured patience boundary. This is flagged for
human decision; the evaluated best model remains unchanged by the extra
epoch according to the saved metadata and loss comparison. Source and
provenance defects are archived in repository-audit.json, not erased from
the original record. The claimed Colab notebook/badge additions are absent
from this repository; no decision to use them is recorded as verified.

Git EOL normalization changes original JSON byte hashes while preserving
their values. The checkpoint's manifest hash uses the Windows CRLF bytes;
a fresh LF checkout fails that guard. This portability defect is flagged
for human-approved repair without rewriting the scientific evidence.

The quiet-validation observation remains retired. The detected population
is dominated by z>0.5; the eventual exchange-rate framing must be moderate-z.
The future actual CHIME Catalog 1 DM-histogram comparison remains unrun.
Anchor prints may be revised by explicit human sign-off and evidence; gate
tolerances, physics anchors and coverage criteria cannot be revised that way.

## Conditioning implementation and G-P7 authorization — 2026-09-08

The human authorized implementation of the observable statistics bypass,
missing gate infrastructure and execution handoff, with quick checks only in
the architect session. The original ISAB/PMA path is retained; train-only
standardization feeds observable statistics directly to the velocity input.
No simulator source, physical parameter or existing gate tolerance is changed.
The separate conditioned launch preserves baseline files and allows one attempt,
with no resume or automatic retry. See CONDITIONING_FIX.md and HANDOFF.md.

Posterior contraction compares matching central-68% physical intervals of the
posterior and the actual prior, including log-uniform priors. The latest human
request's strict inequalities apply to every parameter. The number of catalogs
for width/soft probes is fixed in the named configuration file as an engineering
allocation where the spec does not prescribe one; the SBC and TARP sample sizes
remain those prescribed by SPEC-02a. These choices are recorded before training.

The human explicitly accepted **per-family coverage plus the selection-ignored
probe for G-P7, soft and reported-only**, and requested the cross-family
deferral be recorded in the ledger. Both families occur in this network's
training mixture, so those strata are not a wrong-family-trained-model test.
A true cross-trained wrong-family comparison is deferred to Phase 2b. The
authorization is present in the new ledger template and G-P7 result.

New-run manifest hashing canonicalizes parsed JSON for cross-platform line
endings; source-cache hashing also normalizes text line endings. Binary shard,
normalizer and checkpoint byte hashes remain checked. Old provenance and gate
evidence are not rewritten. Full execution will archive baseline preflight
reports before regenerating them. Cached posterior evidence is reusable only
under the same checkpoint, data and implementation identity.

Quick controls verify implementation mechanics, including positive and negative
gate controls. They do not establish posterior calibration or information gain.
No full retraining, full SBC or full gate suite ran in this architect session.

## Authorized inference continuation — 2026-09-08

After local training completed and contraction stopped at the first prescribed
catalog size, the human explicitly authorized the remaining size and independent
SBC, even though the earlier gate failed. This is an execution-order exception,
not a change to any criterion. The source, tests, seeds, counts, solver and
checkpoint remain unchanged. No retraining was performed.

The original stopped ledger is archived in the conditioned run's
authorized-followup directory. Existing methods produced the new measurements;
the canonical ledger includes both contraction sizes and SBC while preserving
the original failure and incomplete acceptance. CONDITIONING_FOLLOWUP.md is
generated from the recorded measurements and verifies them against saved samples.

The existing SBC subtraction/comparison falsely flags f_d at the inclusive
upper coverage boundary because of floating-point representation. It is
documented as an implementation issue; the stored failure status is retained.
No comparator, tolerance or criterion was changed. Other measured failures
remain independent of that issue. Human review precedes any fix or spec change.
