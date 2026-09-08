# Phase 2a pretraining decisions

Authority: user-supplied SPEC-02a FINAL, preserved in
[docs/SPEC-02a-request.txt](docs/SPEC-02a-request.txt), amended by
[the consolidated authorization](docs/SPEC-02a-authorization.txt) and
[anchor correction v2](docs/ANCHOR_CORRECTION_V2.md). Phase 1 and Phase 2a
retain separate acceptance ledgers. No training catalogs or network exist.

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

## T3: priors recorded, validation not run

The specified priors remain f_d uniform on [0.6,1], F log-uniform on
[0.05,1], host median log-uniform on [20,200], and host sigma_ln uniform
on [0.2,2]. Host lognormal parameters use natural logs. Both population
families belong in training, with their flag confined to ground truth.
No prior-predictive acceptance is claimed. The authorization now defines
the per-event interval as the central 99% [0.5%,99.5%], at each event's
own redshift, with CDF values and the diagnostic grid also reported.

## Accepted architecture and acceptance amendments

G-P8 is required and hard; G-P3 remains reported/soft. Batched attention
must ignore padding via key_padding_mask=True at padding only. Shuffle
real bursts before padding. Parameter transforms now use prior-CDF
normalization (log first for log-uniform priors), followed by logit to
the real line; invert with sigmoid and the prior's inverse map. Physical
parameter space is used for ranks and coverage. SBC uses 1000 posterior
samples per trial, batched over shared conditioning. No network is built.

The smoke test will use full Macquart Table 1: sky coordinates, DM,
fluence and redshift. Those additional columns have not yet been
transcribed into a simulator fixture; none are fabricated.

The human has verified all six original table rows. The authorized
190102 redshift correction is implemented; every current test consumer
was located and rerun. `results/table1-amendment.json` links the before
and after ledgers and records changes to means, corrections and coverage.

Training and posterior validation have not started. Generator performance
has been measured. Phase 1's L2 and Milky Way work remains open.

## Print-revision rule and selection interpretation

Anchor PRINTS are human estimates and may be revised by explicit human
sign-off with recorded multi-method provenance. Gate tolerances and physics
anchors are not revisable this way. This creates no precedent for changing
G-P4 coverage bands or any other acceptance tolerance. The generated table
below supersedes the stale prints in the original stored specification.

Comparison to similarly selected ASKAP samples would be selection
consistency, not independent validation. However, the claimed low-z
dominance does not hold for this authorized simulator: the generated
fractions below show a minority below z=0.5 in both families. Do not claim
agreement with Macquart/James ASKAP redshift distributions or a low-z
dominated exchange-rate regime from these simulations. Phase 3 framing
must use the actual selected distribution; the population remains unchanged.

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
