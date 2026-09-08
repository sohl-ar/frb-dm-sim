# Reproduce the L0-native Phase 2a training run

Physics, prior and selection decisions are in `DECISIONS.md`. T3 is a hard
prerequisite; its event CDFs were reported before training began. This
workflow uses the repository's Python files, not notebook execution state.

```sh
python -m pip install -e '.[inference]'
python -m frbsbi.prior_predictive
python scripts/model_audit.py
python -m frbsbi.data
python -m frbsbi.train
```

Use the generated `requirements-training-lock.txt` to reproduce this run's
library versions on a compatible host. The network runs on CPU in float32;
simulator arithmetic and physical parameter transform boundaries remain
float64. Seeds and all numerical model/optimizer choices are explicit.

The data command creates the prescribed training/validation/test splits
under `work/training-data`. Shards are regenerable and intentionally kept
out of Git. `results/training-data-manifest.json` tracks each shard's hash,
counts, seeds, and generator configuration. Training verifies every shard
hash before fitting. Test data do not influence fitting or checkpoint choice.

The training command runs up to the configured epoch limit with validation
patience, records loss and elapsed time, and saves the best validation
checkpoint plus a last-epoch recovery checkpoint. The best checkpoint is
`results/checkpoints/phase2a-best.pt`; its exact model/run config is embedded.
`results/training.json` records the dataset manifest hash and progress.

After training finishes, the implemented posterior checks run with:

```sh
python -m frbsbi.evaluate
python scripts/update_phase2a_status.py
```

The evaluator checks the checkpoint and validation-shard hashes, then runs
G-P1, G-P2 and G-P4 in that order. A hard failure stops subsequent checks.
It writes `results/posterior-gates.json`; it exits nonzero because this
subset cannot by itself close full acceptance. The status script merges
saved evidence and generates the summary without rerunning any experiment.

Finishing training does not pass SBC, TARP, contraction, information-content
or real-data checks. The full Phase 2a command remains nonzero until all
its requirements are implemented and passed. Phase 1 stays independently
incomplete. The future CHIME Catalog 1 DM-histogram comparison is a real-data
selection consistency task for Phase 2b/3, not evidence already obtained.
