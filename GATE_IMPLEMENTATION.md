# Conditioned-run gate implementation

The new runner is `frbsbi.acceptance`, exposed by the posterior tests in
`pytest -m phase2a`. Existing preflight assertions execute first. New numerical
thresholds are transcriptions of the user authorization in `frbsbi/tolerances.py`;
existing tolerances and `src/frbsim` are unchanged.

Contraction uses the median over catalogs of matching central-68% physical
posterior/prior interval-width ratios, separately for every parameter and N.
For log-uniform F and host-median priors, the prior interval is obtained from
their actual quantiles, not the full support or a uniform-prior approximation.
Thus a returned prior has a width ratio close to one. The strict limits in the
latest request apply to every parameter. Information contrast uses the median
of paired unlocalized/localized f_d width ratios. Generator overrides consume
the original random draws before overriding N/composition, retaining default
bit replay and matched DM, fluence, position and theta between each pair.

TARP uses the pinned `sbi==0.27.0` sample API `_run_tarp`:
https://sbi.readthedocs.io/en/latest/_modules/sbi/diagnostics/tarp.html
and Lemos et al., https://arxiv.org/abs/2302.03026. A small adapter supplies fixed
histogram edges across the probability interval including the two gate points;
sbi's default empirical-range bins would move those points. Its histogram
uses float32 distance ranks even for float64 inputs, so the edges match that
dtype. Integer ECP counts are recovered before comparing the immutable coverage
criterion. Physical parameters are affinely scaled by their prior support
widths for distance calculations; no flow-logit distance is used. Independently
seeded uniform reference points and the complete ECP curve are recorded.
The prescribed TARP catalogs reuse the corresponding held-out SBC posterior
samples, avoiding another expensive inference pass without changing the test.

G-P3 reports per-parameter calibration metrics across the prescribed N grid,
including extrapolation. G-P7 reports per-family coverage and the
selection-ignored probe against a matched-N baseline, with coverage deltas and
biases. **Human authorization on 2026-09-08 accepts this as the soft G-P7**;
a true cross-trained wrong-family experiment is explicitly deferred to Phase
2b in the gate ledger. Scientific degradation is reported, not converted into
a new hard threshold. An execution exception still stops the run.

The Macquart smoke fixture uses all six Table-1 events as one catalog, including
the tentative association. DM, observed redshift, sky coordinates and fluence
come from https://arxiv.org/html/2005.13161 (Table 1). No observed SNR is invented;
the current encoder does not consume SNR. It reports means and interval widths
only, without a science interpretation or acceptance threshold.

Posterior artifacts preserve samples, features and simulation truth for future
audits, with per-artifact SHA256, seeds, draw counts and solver settings.
The real-data artifact's theta array is only an unused shape placeholder and
is not a measured or inferred truth. Checkpoint, canonical dataset and source
hashes bind cached evidence. Failed or interrupted gates cannot restart
automatically. Successful contraction/SBC results can be reused by full pytest
only with identical provenance. Missing gates never count as passes.

Both `results/conditioning-v1/phase2a_gates.json` and the canonical
`results/phase2a_gates.json` are written during execution. The old baseline is
archived before promotion. Full acceptance requires all hard gates passed,
soft diagnostics and smoke reported, all preflight tests selected and passed,
and no execution failure. Test collection alone has no report side effects.

Quick unit controls exercise prior-return failure, contraction success,
information-contrast success/failure, TARP diagonal/biased distance ranks,
actual small untrained-model inference, the smoke fixture, generator pairing,
ledger writes and fail/stop behavior. These checks do not certify the trained
posterior; the full statistical run remains unexecuted in this session.
