# Phase 2a pretraining decisions

Authority: user-supplied SPEC-02a FINAL, preserved in
[docs/SPEC-02a-request.txt](docs/SPEC-02a-request.txt). Phase 1 and Phase 2a
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
- **D4 BLOCKED:** `frbsim.demo.fixture_config` sets both fluence thresholds
  to zero. Every positive-fluence burst passes. Reusing those exact values
  cannot implement the requested nontrivial selection. A human-approved
  positive threshold, coordinated with D5, is required. No value was changed.
- **D5 BLOCKED:** the existing LF is a unit-valued delta fixture. The user
  must supply the literature-supported slope and lower/upper bounds, with
  citation, units and differential-versus-cumulative slope convention.
  The Phase 1 variable L has units Jy ms Mpc^2 and is defined by
  fluence = L/(4*pi*D_L^2). Published emitted-energy limits cannot be
  copied into L without an explicit, approved conversion convention.
  No training on the current delta fixture is authorized.

The detected fraction remains unmeasured for the future training config
because D4/D5 are unresolved. Once specified, report it and flag degenerate
selection. A broad LF is expected to retain partial distance information;
the eventual exchange-rate study is separate from G-P8's sanity threshold.

## T3: priors recorded, validation not run

The specified priors remain f_d uniform on [0.6,1], F log-uniform on
[0.05,1], host median log-uniform on [20,200], and host sigma_ln uniform
on [0.2,2]. Host lognormal parameters use natural logs. Both population
families belong in training, with their flag confined to ground truth.
No prior-predictive acceptance is claimed. Before a binary coverage gate
is implemented, define what probability counts as "non-trivial mass"
and evaluate each event at its own redshift as well as reporting the
requested diagnostic grid; the spec does not quantify that criterion.

## Other specification discrepancies to resolve before acceptance

G-P8 is explicitly HARD in its definition but omitted from the final
acceptance list. This ledger retains it as hard and unrun; no omission is
used to claim acceptance. G-P3 remains soft as explicitly defined.

The real-data smoke test requests DM and z, while the proposed encoder
also requires sky position and fluence. These missing observable inputs
must be sourced or their missing-data treatment approved, not fabricated.

Training, performance measurements and posterior validation are stopped
at the pretraining boundary. Phase 1's L2 and Milky Way work remains open.
