# Completed Phase 2a diagnosis

Recommendation: Phase 2a remains INCOMPLETE. Retain the existing criteria. The authorized comparator bug is fixed; F and host_sigma_ln still fail the aggregate SBC coverage gate. The host probes reveal a model generalization problem, not evidence for an established physical information limit.

Checkpoint: `006325b912cbe4f121576e670301c80ca938832ea040501fe6612611037f54ae`; best epoch 18, validation loss 1.683310196. No retraining occurred.

## Exact aggregate coverage and boundary correction

| Parameter | 68% covered/trials | 68% coverage | Delta (pp) | 95% covered/trials | 95% coverage | Delta (pp) | Corrected status |
|---|---|---:|---:|---|---:|---:|---|
| f_d | 681/1000 | 68.1% | +0.1 | 970/1000 | 97.0% | +2.0 | PASS |
| F | 705/1000 | 70.5% | +2.5 | 961/1000 | 96.1% | +1.1 | FAIL |
| host_median | 662/1000 | 66.2% | -1.8 | 941/1000 | 94.1% | -0.9 | PASS |
| host_sigma_ln | 656/1000 | 65.6% | -2.4 | 938/1000 | 93.8% | -1.2 | FAIL |

All adjusted KS p-values still pass; only the f_d status changes. Every coverage value, KS result and rank is identical to the archived calculation. These are marginal aggregate deviations, but marginal does not mean a passed gate. F remains outside the 68% band by half a percentage point and host_sigma_ln by four tenths of a point.

The evaluator now compares the exact rational covered/trials fraction with the exact decimal nominal probability and tolerance. Both inclusive endpoints pass, and the immediate count outside either endpoint fails. The tolerance is still +/-2pp, without an epsilon or an arbitrary rounding precision. Changing a rejection test from > to >= would also reject the exact endpoint, so suggested option A would not fix the bug. The saved-sample SBC calculation was rerun with the corrected evaluator; no new Monte Carlo draws were needed.

## Stratified coverage

Each cell is empirical 68% / 95% coverage in percent. Fractions use realized localization masks, not the latent Bernoulli probability. Exact counts and binomial confidence intervals are in diagnosis.json. Subgroups are diagnostic and do not acquire new hard +/-2pp gates.

| Group | Trials | f_d | F | Host median | Host sigma_ln |
|---|---:|---:|---:|---:|---:|
| ell < 0.15 | 300 | 65.33 / 98.33 | 70.67 / 95.33 | 67.67 / 91.00 | 67.67 / 94.33 |
| 0.15 <= ell < 0.35 | 472 | 68.64 / 96.19 | 70.34 / 96.40 | 66.10 / 95.13 | 64.62 / 94.92 |
| ell >= 0.35 | 228 | 70.61 / 96.93 | 70.61 / 96.49 | 64.47 / 96.05 | 64.91 / 90.79 |
| N < 64 | 135 | 65.93 / 94.07 | 74.07 / 95.56 | 70.37 / 94.07 | 60.74 / 94.81 |
| N >= 64 | 865 | 68.44 / 97.46 | 69.94 / 96.18 | 65.55 / 94.10 | 66.36 / 93.64 |

F's central-68% overcoverage varies little across localization-fraction strata. The larger change appears by catalog size: small catalogs show more F overcoverage and more host-width undercoverage. This does not establish that unlocalized marginalization is the unique cause or that training overfit large N. Localization strata also show tail mismatches: low-ell host-median 95% coverage is low, while high-ell host-width 95% coverage is low. Aggregate coverage therefore conceals composition-dependent behavior.

The small-N group has 135 trials. Its binomial 95% interval for F's 68% coverage is [65.83, 81.23]%; for host_sigma_ln it is [51.97, 69.03]%. Both include the nominal coverage. These are unadjusted descriptive intervals, not evidence to waive a failed gate.

## Host diagnosis: matched localization experiment

The paired experiment uses 100 catalogs at N=256, with 1000 posterior draws each. Theta, DM, fluence, positions, sample seeds and solver settings are matched. Only observed-redshift availability changes. Localization supplies z, not a direct host-DM measurement: the observed DM still combines cosmic and host contributions. See [Macquart et al.](https://arxiv.org/html/2005.13161), Eq. 1.

| Composition | Host median width/prior | Host sigma width/prior | Host median coverage 68/95 (%) | Host sigma coverage 68/95 (%) |
|---|---:|---:|---:|---:|
| mixed | 0.534764 | 0.604669 | 71.00 / 96.00 | 75.00 / 94.00 |
| all_localized | 0.581187 | 0.851922 | 49.00 / 83.00 | 64.00 / 89.00 |
| all_unlocalized | 0.823843 | 0.553673 | 54.00 / 84.00 | 57.00 / 84.00 |

The mixed posterior shows parameter tradeoffs: median within-catalog Spearman correlation host median vs f_d is -0.355; host sigma vs F is -0.399. These learned correlations are compatible with confounding but do not establish the true posterior geometry or a Fisher bound.

Training support explains why the all-localized comparison cannot determine the host information ceiling: among 50000 training catalogs, only 35 were all-localized, and their largest N was 5. Among 43866 training catalogs with N>=64, there were 0 all-localized and 40 all-unlocalized catalogs.

The all-localized standardized availability flag is -37.783216 for has_unlocalized. The combination of this rare flag and large N is absent from training. This is a concrete distribution-support and encoding-sensitivity problem in the learned model.

A targeted counterfactual replaced only that standardized flag with the paired mixed-catalog value 0.026467; all actual observations and all parameters stayed fixed. It changed the all-localized response as follows:

| Parameter | Original all-localized ratio | Counterfactual ratio | Original coverage 68/95 (%) | Counterfactual coverage 68/95 (%) |
|---|---:|---:|---:|---:|
| f_d | 0.667495 | 0.297979 | 45.00 / 84.00 | 77.00 / 96.00 |
| F | 1.299316 | 0.296193 | 77.00 / 97.00 | 70.00 / 96.00 |
| host_median | 0.581187 | 0.472841 | 49.00 / 83.00 | 73.00 / 92.00 |
| host_sigma_ln | 0.851922 | 0.593155 | 64.00 / 89.00 | 79.00 / 94.00 |

The counterfactual directly demonstrates large sensitivity to a single availability feature. It is an intentionally inconsistent summary, not a valid posterior, a recalibration method, or a gate pass. Coverage is still imperfect. It identifies a model mechanism in this stress regime without proving why every mixed-catalog host posterior remains wide.

## Decision and proposed next work

Keep Phase 2a INCOMPLETE and retain all gate criteria. The aggregate errors are in the request's marginal range, but Scenario A's condition that all science parameters pass is false because F still fails. The endpoint-composition probes reveal substantial additional model failures. No theoretical lower bound or numerical reference posterior was established, so Scenario D and a physics-limited host claim are not supported. A nuisance-only gate revision would also not resolve F's calibration failure.

For a separately authorized model-fix session, prioritize an experiment that preserves availability bits as binary inputs instead of standardizing rare flags, and gives the training distribution explicit coverage of the catalog compositions required by validation. Test their contributions in a controlled comparison, rather than assuming either alone is sufficient. Neither change nor any retraining is implemented here. Use a numerical reference posterior on a small fixed set of representative mixed/localized catalogs before changing the host-contraction criterion. The current results cannot justify a bound of the form 'host parameters require N>X'.

Passing marginal parameter SBC is not proof that all data information was extracted; this limitation is studied by [Modrak et al.](https://arxiv.org/abs/2211.02383). Thus even a future corrected aggregate pass would not replace the remaining contraction, TARP, information-content and engineering gates. No Phase 2a acceptance is declared.

Evidence: results/conditioning-v1/diagnosis-v2/diagnosis.json, training-support.json, flag-ablation.json and saved NPZs. The original assessment and ledger are archived in that directory. The canonical G-P4 row has the new assessment provenance while retaining the original sampling identity; unchanged caches are not relabeled as newly generated. The full runner's source-identity guard remains strict and no automatic failed-run restart is enabled.
