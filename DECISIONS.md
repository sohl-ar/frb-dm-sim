# Phase 2a pretraining decisions

Authority: user-supplied SPEC-02a FINAL, preserved in
[docs/SPEC-02a-request.txt](docs/SPEC-02a-request.txt), amended by
[the consolidated authorization](docs/SPEC-02a-authorization.txt). Phase 1 and Phase 2a
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
audit ranges are recorded in that JSON. This guard must also be called by
the future training-data entry point; that entry point is not implemented.
T1 uses fixed-redshift cosmic draws, not a generated training catalog.

## T2: catalog design

- **D1 accepted as specified:** independent, repeat-free sightlines.
  Application to unclustered repeaters is prohibited. Future ingestion
  must group by host; a later simulator may share cosmic DM within a host
  while drawing burst-dependent host contributions.
- **D2 accepted as specified:** observed catalog size is discrete uniform
  on 1 through 512; size 1024 is diagnostic. Existing Phase 1 generation
  takes an intrinsic count, so it is not yet this selected-size generator.
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

**Execution is stopped on the selection print comparison**, not missing
authorization. `results/selection-audit.json` records direct quadrature,
independent intrinsic rejection, conditional-L sampling, detected fractions
and population-integrated selection probabilities. At the final diagnostic
redshift the calculated fraction is just outside the supplied factor-two
comparison boundary. No LF parameter or print boundary has been altered.
The joint conditional z,L generator and its naive-rejection comparison
remain unimplemented pending this STOP report. A broad LF retains partial distance information;
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

Training, performance measurements and posterior validation are stopped
at the pretraining boundary. Phase 1's L2 and Milky Way work remains open.
