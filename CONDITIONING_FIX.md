# Conditioning experiment implementation

The original collapsed attention checkpoint is preserved. The new experiment
adds observable distribution statistics alongside the unchanged ISAB/PMA
branch. Its configuration, training-only normalizer and quick-check evidence
are in `results/conditioning-v1/`. Training has not been launched.

`src/frbsbi/statistics.py` defines the ordered statistic schema. DM and fluence
are recovered from the existing observable feature cache at its recorded
float32 precision; reductions and normalization use float64. This avoids
regenerating catalogs and never reads targets or hidden redshifts. The branch
includes DM/fluence distributions, localized DM-z moments and OLS residuals,
observed-redshift bins, counts and availability flags. Residuals are empirical
OLS residuals, with no physical subtraction or fiducial parameter assumption.

Every training catalog contributes once to fixed population-mean/std scaling.
Validation catalogs only measure variance and descriptive correlations. An
exactly constant training statistic has scale one and centers to zero. Empty
localized subsets use zero values with availability/count features; singleton
standard deviations are zero. These conventions are tested, including poisoned
missing-z placeholders and padded rows.

`PosteriorFlow.condition()` concatenates both branches immediately before the
existing velocity MLP. The default configuration remains backward compatible;
the new launch explicitly enables the bypass. No optimizer step is taken by
the quick check. The baseline checkpoint can still load with strict keys.

The launch retains the original training hyperparameters and seeds, creates a
separate output directory and refuses a second training attempt. New manifest
provenance hashes canonical parsed JSON, so Windows/Linux line endings do not
invalidate equivalent manifests. Shard bytes remain SHA256-checked. Historical
run provenance is untouched. The normalization artifact uses fixed LF endings.

This is an implementable experiment, not evidence that collapse is solved.
The hard calibration, contraction and information-content gates decide that
after the single authorized training run.
