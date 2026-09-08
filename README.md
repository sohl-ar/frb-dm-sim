# FRB dispersion-measure simulation and amortized inference

Latest code preparation: [composition and binary-encoding fix](COMPOSITION_FIX.md)
is implemented for conditioning-v2. Data generation, normalization fitting,
training and validation are deferred; the new code has not passed runtime tests.
Existing measured results refer to the preserved earlier runs.

This research repository models FRB catalogs and infers the diffuse-ionized
baryon fraction, cosmic-DM fluctuation parameter and host-DM distribution.
Background cosmology is pinned. L0 uses the published asymmetric cosmic-DM
distribution; a native Set Transformer conditions a flow-matching posterior
on localized and unlocalized observations.

**The conditioning run completed; acceptance remains failed.** Both catalog-size
contraction checks and independent SBC are now recorded in
[CONDITIONING_FOLLOWUP.md](CONDITIONING_FOLLOWUP.md). Do not relaunch training.
The authorized boundary fix makes f_d pass SBC; F and host_sigma_ln still fail.
The completed diagnosis includes exact coverage, composition/size strata and
evidence of a rare-flag/training-support problem in extreme compositions.
See [CONDITIONING_FIX.md](CONDITIONING_FIX.md) and
[GATE_IMPLEMENTATION.md](GATE_IMPLEMENTATION.md) for implementation conventions.

**Baseline run: training complete; acceptance failed.** Training recorded 19 epochs, with best epoch 13 and validation loss 2.845229963.
G-P1/G-P2 pass. Three parameters pass the recorded marginal SBC checks;
F is underconfident at central-68% coverage. The audit additionally finds
largely prior-like predictions across all four parameters, so these passes
must not be advertised as successful informative inference. Phase 3 is blocked.

| Parameter | 68% coverage | 95% coverage | Adjusted KS p | Recorded result |
|---|---|---|---|---|
| f_d | 0.664 | 0.944 | 1 | pass |
| F | 0.722 | 0.950 | 0.77273 | fail |
| host_median | 0.687 | 0.956 | 0.236142 | pass |
| host_sigma_ln | 0.665 | 0.942 | 1 | pass |

Start with [PROJECT_STATUS.md](PROJECT_STATUS.md) and the
[verification and diagnostic report](AUDIT_REPORT.md). Phase 1 remains
independently incomplete: L2 is unbuilt and pygedm validation is open.

## Quick start: inspect the saved checkpoint

Python with the dependencies in pyproject.toml is required; the tested
environment is recorded in requirements-training-lock.txt. From a clone:

```sh
python -m pip install -e '.[inference]'
python -c "import json; print(json.load(open('results/training.json'))['status'])"
```

No retraining is needed to inspect committed evidence. Checkpoints, seeds,
config hashes and reports are tracked. Large training shards live under
ignored work/training-data and are reproducible from the tracked manifest.
See [TRAINING_RUN.md](TRAINING_RUN.md) before authorizing a new run; training
and evaluation commands can overwrite run artifacts.

## Verification architecture

T1–T3 gate the RNG path, selected generator and prior predictive. Physics
gates verify mean-DM formulas, units, asymmetric sampling and replication
fixtures. Posterior gates separately check invariance, replay, calibration,
contraction and information content. Failures trigger STOP; tolerances and
physical anchors require explicit human authorization to change.

The separate saved ledgers are [Phase 1](results/gates.json) and
[Phase 2a](results/phase2a_gates.json). `pytest -m gates` remains nonzero for
incomplete L2 acceptance. `pytest -m phase2a` reruns preflight checks and
merges existing posterior evidence; it remains nonzero and does not
implement or execute the missing posterior gates. L0 smoke inputs are
engineering fixtures, not empirical survey calibrations.

## Colab and remaining work

Colab can execute the same repository without keeping scientific logic in
cells. Follow [COLAB_RUN.md](COLAB_RUN.md). The claimed repository notebook
`notebooks/frb_dm_sim_colab.ipynb` is absent; no working badge is claimed.
The human-run native-dependency diagnosis is [PYGEDM_DIAGNOSIS.md](PYGEDM_DIAGNOSIS.md).

G-P5/TARP, G-P6, G-P8, the G-P3/G-P7 diagnostics and real-data smoke remain
unimplemented/unrun. G-P4 and weak contraction must be resolved. L2/GLASS
and the Milky Way model remain a separate Phase 1 task. The actual CHIME
Catalog 1 DM-histogram comparison is deferred to Phase 2b/3.

See [CONVENTIONS.md](CONVENTIONS.md), [DECISIONS.md](DECISIONS.md),
[VERIFICATION_TODO.md](VERIFICATION_TODO.md) and the
[historical document index](docs/README.md). No physics or gate code was
changed by the repository audit.
