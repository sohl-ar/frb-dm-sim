# Verification ledger

Phase 1 is incomplete. An environmental blocker is not a passed gate or
human sign-off to omit an acceptance requirement.

## Verified or authorized

- Human Table-1 verification is complete for all six rows, per the
  consolidated authorization. FRB 190102 now uses z=0.291. The executed
  before/after comparison is in `results/table1-amendment.json`.
- Phase 2a D4/D5 defaults, unit conversion and Schechter form are now
  explicitly authorized in `docs/SPEC-02a-authorization.txt`. Numerical
  sampling checks pass. Human anchor correction v2 resolves the comparison
  prints and adds the executed z=0.2 evidence row; see
  `results/selection-audit.json`. The selected generator passes its
  engineering checks. No training has started. The requested low-z
  dominance interpretation is not supported by the generated population.

- Madau & Dickinson Eq. 15 uses 2.7, 2.9, **5.6**. Source:
  https://arxiv.org/html/1403.0007 ; correction explicitly authorized.
- Proper path includes H0; observed shell DM includes delay weighting.
- G8a/G8b split and pseudo-luminosity units explicitly authorized.
- L2 amplitude calibration prescription explicitly authorized, not yet built.
- pygedm signature and pc/degree/Quantity conventions checked against docs
  and downloaded PyPI source. Full-column and pulsar accuracy are not verified.
- CAMB documents `NonLinearModel.set_params(halofit_version='mead2020')`:
  https://camb.readthedocs.io/en/latest/nonlinear.html . Not yet runtime-tested.
- Authors' original mean-DM functions run under matched assumptions;
  commit and source hash are stored alongside the vendored BSD source.

## Environment blockers

- Linux/Colab follow-up: supplied gates and environment record are in
  `results/colab/`. The pygedm build failed there too; the installed state
  and actual compiler error remain unverified. `PYGEDM_DIAGNOSIS.md`
  supplies verbose diagnostic and verification cells, plus an isolated
  compatibility candidate. Source inspection identifies C++/libf2c build
  requirements and a separate SciPy simps import incompatibility. Neither
  has been claimed as the proven cause of the unseen Linux build failure.

- pygedm 3.3.0 source build fails because Microsoft Visual C++ 14+ is absent.
- GLASS 2026.2 requires healpy; healpy 1.20.0's native build fails on this
  Windows environment while configuring native dependencies.
- WSL reports that Windows Subsystem for Linux is not installed.
- No native-model replacement has been introduced. A Linux environment
  with the required native build dependencies is the intended continuation.
  No OS installation or reboot was attempted.

## Pending science/build verification

- TODO-VERIFY[pygedm]: known-distance pulsar test, 30/50 kpc endpoint
  convergence, poles, and at least ten independent-distance high-latitude
  ATNF pulsars. Do not use model-inferred distances to validate that model.
- TODO-VERIFY[survey defaults]: CHIME's original approximate 5 Jy ms
  selection proxy is supported by Catalog 1's discussion at
  https://doi.org/10.3847/1538-4365/ac33ab . Catalog 2 reports a median
  95% sensitivity threshold near 3.5 Jy ms, explicitly a lower-limit
  quantity, not a universal hard cut:
  https://arxiv.org/html/2601.09399 . No threshold has been silently swapped.
  The DSA-like fluence threshold remains unverified; the official overview
  https://www.deepsynoptic.org/overview does not by itself establish a
  width-independent scalar fluence limit.
- TODO-VERIFY[instrumental configuration]: physical DM noise and SNR
  conventions remain unresolved for full Phase 1 acceptance. Phase 2a's
  LF and localization defaults have their separate explicit authorization. The API
  accepts these with provenance; the smoke config contains labeled
  numerical fixtures only. They are not empirical defaults.
- TODO-VERIFY[L2]: shell projection/boundaries; CAMB P(k) normalization
  including primordial-spectrum inputs; native field generation;
  Gaussian P(k) round-trip; calibrated amplitude and held-out redshifts;
  full end-to-end runtime. API lookups alone cannot certify these.
- TODO-VERIFY[G4(iii)]: noisy full-pipeline replication. The current six-event
  check exercises the explicit zero-noise limit and is not full acceptance.
- Full Table-1 sky and fluence columns remain to be transcribed for the
  Phase 2a smoke test; the human sign-off on the six rows is recorded above.

No tolerance changes or additional deferrals have been signed off.
