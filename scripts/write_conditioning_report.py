"""Render frozen conditioning evidence, including the full correlation matrix."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def table(headers,rows):
    def escaped(items): return ' | '.join(str(x).replace('|','\\|') for x in items)
    return '\n'.join(['| '+escaped(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+escaped(row)+' |' for row in rows])


def main():
    d=json.loads((ROOT/'results/conditioning-diagnosis.json').read_text())
    p=json.loads((ROOT/'results/pma-collapse-detail.json').read_text())
    init=json.loads((ROOT/'results/conditioning-initialization-reference.json').read_text())
    corr=np.array(d['physical_correlation_matrix']); names=d['parameter_labels']
    def cell(value):
        text=f'{value:.6f}'
        return '**'+text+'**' if abs(value)>.3 else text
    matrix=table(['Summary dimension',*names],[[label,*[cell(v) for v in row]]
        for label,row in zip(d['summary_labels'],corr)])
    stages=table(['Stage','Initialized catalog variation RMS','Trained catalog variation RMS'],
        [[key,f"{v['initialized_variation_RMS']:.8g}",f"{v['trained_variation_RMS']:.8g}"] for key,v in init['stages'].items()])
    internal=table(['PMA internal stage','Float32 catalog variation RMS','Float64 catalog variation RMS'],
        [[key,f"{v['catalog_variation_RMS']:.8g}",f"{p['precisions']['float64']['activations'][key]['catalog_variation_RMS']:.8g}"]
         for key,v in p['precisions']['float32']['activations'].items()])
    probe=next(r for r in d['velocity_sensitivity'] if r['state_index']==0 and r['t']==.5)
    velocity=table(['Condition replacement','Mean ||v_real − v_alternative||','Maximum'],
        [[key,f"{v['mean']:.9g}",f"{v['max']:.9g}"] for key,v in probe['alternatives'].items()])
    loss=table(['Condition','Mean CFM loss','Alternative − correct','Paired catalog SE'],
        [['correct',f"{d['loss']['total_real']:.10f}",'—','—']]+
        [[key,f"{v['total_loss']:.10f}",f"{v['delta_alternative_minus_real']:.9g}",f"{v['paired_catalog_SE']:.9g}"] for key,v in d['loss']['alternatives'].items()])
    perparam=table(['Condition',*names],[['correct',*[f'{x:.8f}' for x in d['loss']['per_parameter_real']]]]+
        [[key,*[f'{x:.8f}' for x in value['per_parameter_loss']]] for key,value in d['loss']['alternatives'].items()])
    maxima=table(['Parameter','Largest |r|','Dimension'],[[name,f'{abs(corr[:,j]).max():.6f}',d['summary_labels'][int(abs(corr[:,j]).argmax())]] for j,name in enumerate(names)])
    factor=init['stages']['PMA_summary']['initialized_variation_RMS']/init['stages']['PMA_summary']['trained_variation_RMS']
    report=f"""# Frozen-network conditioning diagnosis

**The connected encoder has learned an almost catalog-independent pooled
summary.** Attenuation begins in the ISAB stack and becomes extreme in PMA.
The velocity decoder receives the condition correctly, but the scientific
pooled features vary too little to influence it materially. Almost all
measured catalog-to-catalog velocity sensitivity comes from log(1+N) and
the observed localized fraction. The exact optimizer dynamics that caused
this learned representation collapse are not established.

Scope: {len(d['catalog_seeds'])} existing validation catalogs, seeds
{min(d['catalog_seeds'])}–{max(d['catalog_seeds'])}, checkpoint epoch {d['checkpoint_epoch']}.
Checkpoint SHA256: `{d['checkpoint_sha256']}`. Audit source commit:
`{d['git_sha']}`. No training, optimizer steps, new simulations, ODE/posterior
sampling, src/ edits, gate changes or acceptance reruns occurred.

## 1. Exact conditioning pathway

- [data.py:25](src/frbsbi/data.py#L25) encodes log10 DM, Galactic unit xyz,
  log10(fluence + fixture noise floor), localized z or placeholder, and
  the missing-z bit into seven stored features. These are existing shards.
- [model.py:91](src/frbsbi/model.py#L91) projects z into eight components
  or substitutes a learned missing vector, and adds a four-component type
  embedding. The missing bit is retained separately.
- [model.py:101](src/frbsbi/model.py#L101) concatenates these into
  {d['architecture']['per_burst_projection_input']} features and applies a
  **single linear projection**, not a separate multilayer per-burst MLP,
  to {d['architecture']['per_burst_width']} dimensions.
- [model.py:103](src/frbsbi/model.py#L103): two ISAB blocks, each with
  sixteen inducing points and four attention heads. True padding positions
  are ignored in inducing attention; real unlocalized bursts remain elements.
- [model.py:105](src/frbsbi/model.py#L105): PMA uses one learned query;
  its MAB output is a {d['architecture']['pooled_width']}-component summary.
- [model.py:108](src/frbsbi/model.py#L108): append log(1+N) and realized
  localized fraction, giving {d['architecture']['condition_width']} condition dimensions.
- [model.py:125](src/frbsbi/model.py#L125): concatenate x_t, time, summary.
  Velocity linear dimensions are {d['architecture']['velocity_linear_layers']},
  with SiLU between hidden layers. No FiLM or repeated conditioning injection.
- [model.py:128](src/frbsbi/model.py#L128) uses that path for CFM loss;
  [model.py:138](src/frbsbi/model.py#L138) repeats each summary for its
  posterior draws and closes over it in the ODE vector field.

**Connectivity check: PASS.** There is no detach, discarded return value,
unconditional residual branch around the decoder, silent dimension broadcast,
or missing summary argument. Autograd finds nonzero condition derivatives.
The MAB residual/LayerNorm operations at lines 60–61 can attenuate learned
variation, which is observed below; their gains are not identically zero.

## 2. Does the summary contain signal?

{maxima}

{len(d['flag_abs_r_gt_0_3'])} matrix entries exceed the requested descriptive
|r|>0.3 flag. This is not a multiple-testing-corrected significance test.
Raw physical-parameter correlations are shown; correlations with normalized
prior coordinates, including logarithmic F/host-median coordinates, are
also saved in JSON. Linear correlation is not a complete information test.

Crucially, correlation is invariant to rescaling: a tiny surviving signal
can have substantial correlation yet be effectively unused by the decoder.
The stage measurements expose that problem:

{stages}

Before PMA these measurements are means over real elements of each catalog;
they are a comparable catalog-level diagnostic, not the entire set activation
tensor. The PMA measurement is its actual output. Initial weights were
reconstructed from the recorded seed, with forward evaluation only.
Pooled variation has fallen by a factor of {factor:.6g} from initialization.
The per-burst projection still contains parameter-associated variation.

### PMA internals

{internal}

The attention output has a large nearly constant component; its value RMS
is {p['precisions']['float32']['activations']['attention_output']['value_RMS']:.8g}.
The reductions across the first normalization and final residual/normalization
are substantial. Learned normalization gain ranges are
{p['precisions']['float32']['norm1_gain_min_max']} and
{p['precisions']['float32']['norm2_gain_min_max']}, so this is not a zero-gain
disconnect. Value/output projection weights are also nonzero. Uniform-value
projection variation is similar to actual attention-output variation;
attention weighting alone is not the sole bottleneck.

Float64 evaluation of the same weights reproduces the collapse. Float32
roundoff affects the smallest variations but is not the origin of the problem.
The endpoint comparison identifies learned attenuation; it cannot reconstruct
when training entered this regime or which optimizer/normalization interaction
caused it. Intermediate encoder histories were not saved.

## 3. Does the velocity field use catalog conditioning?

At fixed x_t=0 and t={probe['t']}:

{velocity}

The JSON additionally records a fixed nonzero x_t and other times, with the
same qualitative separation. Full shuffling changes the two metadata features
as well as the pooled vector; isolating those paths shows that metadata
accounts for essentially the entire effect. The mean condition is another
control for catalog variation. Zeroing is deliberately off-distribution and
its larger effect does not demonstrate use of scientific catalog information.

At the first velocity layer, the varying pooled contribution has RMS
{d['first_velocity_layer']['pooled_varying_contribution_RMS']:.9g}, versus
{d['first_velocity_layer']['extras_varying_contribution_RMS']:.9g} from the
two metadata features. The mean-condition contribution has RMS
{d['first_velocity_layer']['mean_condition_contribution_RMS']:.9g}.
The decoder has nonzero weights and derivatives for the pooled features,
but their natural variation reaches it at negligible amplitude.

## 4. Conditioning benefit in the frozen-model loss

{loss}

Each catalog uses {d['loss_noise_draws_per_catalog']} common random x0/time
pairs. Comparisons retain identical targets and noise; the averaged shuffled
control uses {d['shuffle_count']} derangements. Uncertainty is computed across
catalog-average paired differences, not by treating all repeated noise draws
as independent catalogs. These are diagnostics, not changed gate tolerances.

{perparam}

Shuffling only the pooled vector leaves loss unchanged to numerical precision.
Full-condition shuffling produces a small benefit of correct conditioning,
mostly attributable to the two metadata features. These finite-sample
differences do not establish generalizable scientific information extraction.
The large zeroing penalty reflects removal of the learned baseline as well
as any conditioning signal.

**No usable learned pooled-condition benefit is not the same as no incentive
in the CFM objective.** A frozen collapsed model cannot tell us what benefit
an informative representation or exact posterior could achieve. Neither
absence of physical information nor lack of gradient pressure is proven.

## 5. Bottleneck and proposed repair — not implemented

The measured bottleneck is learned encoder representation collapse, strongest
at PMA and its residual/normalization stages, after earlier ISAB attenuation.
It is not a missing conditioning connection. A purely decoder-side FiLM
replacement would still receive an almost constant scientific condition and
is therefore not the first targeted repair supported by these measurements.

Proposed next-session experiment, subject to human approval:

1. Add a parallel permutation-invariant observable-statistics branch **before
   the ISAB/PMA stack**, concatenated directly with the final condition.
   Include mean/dispersion and robust quantiles of log DM and log fluence;
   localized z/log-DM covariance and scatter summaries in fixed redshift bins;
   per-bin counts and explicit missing masks. Use only observable values.
   Define features/configuration explicitly; retain the natural-log host
   convention, original priors, selection law and physics unchanged.
2. Standardize this branch with statistics fitted on the training split only,
   save them with the checkpoint, and reject nonfinite/degenerate scales.
   Keep the existing pooled path and decoder for an interpretable ablation.
   Do not amplify float32-scale PMA noise by blindly dividing by its tiny std.
3. Once training is separately approved, compare the original and bypass
   versions from the same recorded initialization, catalogs and noise seeds
   in a short controlled run. Record stage variation, gradient norms and
   per-parameter correct-versus-shuffled loss curves. Require evidence that
   the scientific statistics branch influences the output before spending
   another full training budget.
4. If the noncollapsed branch is still unused, test repeated conditioning
   injection/FiLM as a separate decoder ablation. Consider a high-information
   curriculum only after that experiment supports a training-pressure issue.
   Do not combine all changes at once or claim the proposed bypass guarantees
   calibrated/informative posteriors.
5. Preserve all original acceptance tests. Posterior contraction and
   calibration still need independent verification, ideally including an
   informative reference posterior on existing catalogs. This diagnosis
   neither runs missing gates nor closes Phase 2a or Phase 1.

## Full summary–physical-parameter correlation matrix

Bold entries satisfy |r|>0.3. The final two rows are appended metadata,
not pooled features. Unrounded values and dimension standard deviations are
in [the JSON](results/conditioning-diagnosis.json).

{matrix}

## Artifacts and preservation

- [Primary diagnostic JSON](results/conditioning-diagnosis.json)
- [PMA internal and float64 check](results/pma-collapse-detail.json)
- [Initialization reference](results/conditioning-initialization-reference.json)
- [Saved summary/loss arrays](results/conditioning-diagnostic-arrays.npz)
- Reproduction scripts under scripts/: diagnose_conditioning.py,
  inspect_pma_collapse.py, conditioning_initialization_reference.py and
  write_conditioning_report.py. The report generator reads saved JSON only.

The measured model state and checkpoint hash are unchanged. Existing training,
SBC and acceptance reports are preserved. The earlier newline-provenance and
resume findings remain open; this session does not silently repair them.
"""
    (ROOT/'CONDITIONING_DIAGNOSIS.md').write_text(report,encoding='utf-8',newline='\n')
    print(json.dumps({'max_abs_correlations':maxima,'flagged_entries':d['flag_abs_r_gt_0_3'],
        'initialized_over_trained_PMA_variation':factor},indent=2))


if __name__=='__main__': main()
