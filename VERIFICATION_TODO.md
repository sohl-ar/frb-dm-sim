# Verification ledger

Phase 1 is incomplete. An environmental blocker is not a passed gate or
human sign-off to omit an acceptance requirement.

## Verified or authorized

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
- TODO-VERIFY[LF and instrumental configuration]: literature-calibrated
  pseudo-luminosity bounds, differential slope, localization fractions,
  DM noise and SNR convention require explicit choices/sources. The API
  accepts these with provenance; the smoke config contains labeled
  numerical fixtures only. They are not empirical defaults.
- TODO-VERIFY[L2]: shell projection/boundaries; CAMB P(k) normalization
  including primordial-spectrum inputs; native field generation;
  Gaussian P(k) round-trip; calibrated amplitude and held-out redshifts;
  full end-to-end runtime. API lookups alone cannot certify these.
- TODO-VERIFY[G4(iii)]: noisy full-pipeline replication. The current six-event
  check exercises the explicit zero-noise limit and is not full acceptance.
- The human Table-1 spot-check remains a human acceptance step. Transcribed
  data are preserved; code does not claim to have performed human review.

No tolerance changes or additional deferrals have been signed off.
