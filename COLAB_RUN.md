# Colab executes the repository

The current trained checkpoint and failed/incomplete acceptance reports are
already saved in Git. Inspect them before authorizing any new computation.
Colab is an optional Linux execution host; all science remains in Python
modules and pytest. This guide was reviewed as source, not executed in a
new Colab runtime during the repository audit.

The claimed `notebooks/frb_dm_sim_colab.ipynb` and Colab badges are absent
from this checkout and available Git history. Their correctness cannot be
verified. Recover the original file if it exists; no replacement notebook
or working badge is silently assumed.

## Load a fresh checkout and inspect saved results

Upload the current `frb-dm-sim.bundle` to `/content/`, then use a Python cell:

```python
from pathlib import Path
import json
import subprocess

repo = Path('/content/frb-dm-sim')
if not repo.exists():
    subprocess.run(['git', 'clone', '/content/frb-dm-sim.bundle', str(repo)], check=True)
subprocess.run(['git', 'status', '--short'], cwd=repo, check=True)
subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True)
training = json.loads((repo / 'results/training.json').read_text())
ledger = json.loads((repo / 'results/phase2a_gates.json').read_text())
print(training['status'], training['best_epoch'], training['best_validation_loss'])
print({name: item['status'] for name, item in ledger['gates'].items()})
```

An existing checkout is inspected, not reset or overwritten. A bundle has
only the commits present when it was created; check its SHA against the
current GitHub repository. Authenticated GitHub cloning is another option
for this private repository; never put credentials in a committed notebook.
No runtime connection or installation is necessary merely to read reports.

## Install only for newly authorized execution

For the L0-native inference code, run a separate Python cell:

```python
import sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', f'{repo}[inference]'], check=True)
```

This installs the core/inference extras, not pygedm or L2. Exact historical
versions are recorded in requirements-training-lock.txt, which includes a
Windows-local editable path and is not a portable pip requirements file.
Use compatible Python and explicit version pins when reproducing that
environment; successful installation on the current Colab image is unverified.

The audit also found a raw-hash portability issue: Git normalizes the original
JSON manifest to LF, while the checkpoint references its Windows CRLF bytes.
The content is identical, but the evaluator's manifest guard will reject a
fresh checkout until this provenance issue is explicitly repaired. Do not
bypass it. Reading saved reports does not require passing that runtime guard.

Training, full SBC and new simulations require authorization beyond the
completed repository audit. See TRAINING_RUN.md for their artifact and
overwrite behavior. All commands should use an explicit checkout path and
fail on errors; execution order should not contain hidden scientific state.

## Native dependencies remain a separate human task

The prior Linux install aborted at pygedm. Follow PYGEDM_DIAGNOSIS.md to
capture the verbose failure and verify installed packages. Avoid repeating
the combined `.[l2,mw]` transaction without diagnosing that failure.
Installation success would not implement the missing L2 physics or gates.

When a fresh gate run is authorized, `python -m pytest -m gates` is the
strict Phase 1 command and `python -m pytest -m phase2a` is the separate
Phase 2a command. Both are expected to remain nonzero at this checkpoint.
The Phase 2a pytest command merges saved posterior evidence; it does not
execute missing gates. Preserve original reports and export any new
results before disconnecting the disposable runtime.
