# Authorized inference followup

No retraining. Checkpoint and all source, gate criteria and tolerances unchanged.

Saved training selected epoch 18 of 20; validation loss 1.683310196.

| Parameter | N=16 ratio (<0.7) | Status | N=256 ratio (<0.5) | Status |
|---|---:|---|---:|---|
| f_d | 0.600766 | PASS | 0.317099 | PASS |
| F | 0.659368 | PASS | 0.330395 | PASS |
| host_median | 0.876256 | FAIL | 0.534764 | FAIL |
| host_sigma_ln | 0.828174 | FAIL | 0.604669 | FAIL |

SBC: 1000 mixed-N validation catalogs, 1000 draws per catalog.

| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Recorded status |
|---|---:|---:|---:|---|
| f_d | 68.100% | 97.000% | 1.000000 | FAIL |
| F | 70.500% | 96.100% | 1.000000 | FAIL |
| host_median | 66.200% | 94.100% | 0.025746 | PASS |
| host_sigma_ln | 65.600% | 93.800% | 1.000000 | FAIL |

The f_d FAIL is a floating-point boundary artifact in the existing comparator: 0.97 - 0.95 evaluates to 0.020000000000000018, which compares greater than 0.02. The measured coverage is on the inclusive upper boundary. This is reported without changing its recorded status. F exceeds its 68% band; host_sigma_ln is below its 68% band. Every adjusted KS p passes. The host-median SBC row passes.

Assessment: the severe prior-like behavior has improved, particularly for f_d and F, but full acceptance still fails. The host-parameter contraction failures and the non-boundary calibration failures remain substantive under the current spec. These checks do not establish whether the host thresholds are attainable for the selected catalogs or whether the network is still losing information.

I would retain the present thresholds for now, flag the comparator boundary issue for a separately reviewed implementation correction, and compare a few fixed catalogs against a numerical reference posterior before proposing a host-contraction spec change. Calibration stratified by catalog size would also help; this SBC result averages over the validation N distribution and is not separate calibration at N=16 and N=256.

G-P6 remains failed and acceptance remains incomplete. The original stopped ledger is archived beside results.json; both new results were recomputed from saved posterior samples by this reporting script.
