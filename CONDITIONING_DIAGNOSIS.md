# Frozen-network conditioning diagnosis

**The connected encoder has learned an almost catalog-independent pooled
summary.** Attenuation begins in the ISAB stack and becomes extreme in PMA.
The velocity decoder receives the condition correctly, but the scientific
pooled features vary too little to influence it materially. Almost all
measured catalog-to-catalog velocity sensitivity comes from log(1+N) and
the observed localized fraction. The exact optimizer dynamics that caused
this learned representation collapse are not established.

Scope: 100 existing validation catalogs, seeds
50000–50099, checkpoint epoch 13.
Checkpoint SHA256: `3260ef3eb126e81913f037971c734732672b126e58db711bf5e04d4817fcd802`. Audit source commit:
`98c3da30c2c72bd43310002235c1bc024fd26920`. No training, optimizer steps, new simulations, ODE/posterior
sampling, src/ edits, gate changes or acceptance reruns occurred.

## 1. Exact conditioning pathway

- [data.py:25](src/frbsbi/data.py#L25) encodes log10 DM, Galactic unit xyz,
  log10(fluence + fixture noise floor), localized z or placeholder, and
  the missing-z bit into seven stored features. These are existing shards.
- [model.py:91](src/frbsbi/model.py#L91) projects z into eight components
  or substitutes a learned missing vector, and adds a four-component type
  embedding. The missing bit is retained separately.
- [model.py:101](src/frbsbi/model.py#L101) concatenates these into
  18 features and applies a
  **single linear projection**, not a separate multilayer per-burst MLP,
  to 64 dimensions.
- [model.py:103](src/frbsbi/model.py#L103): two ISAB blocks, each with
  sixteen inducing points and four attention heads. True padding positions
  are ignored in inducing attention; real unlocalized bursts remain elements.
- [model.py:105](src/frbsbi/model.py#L105): PMA uses one learned query;
  its MAB output is a 64-component summary.
- [model.py:108](src/frbsbi/model.py#L108): append log(1+N) and realized
  localized fraction, giving 66 condition dimensions.
- [model.py:125](src/frbsbi/model.py#L125): concatenate x_t, time, summary.
  Velocity linear dimensions are [[71, 128], [128, 128], [128, 128], [128, 4]],
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

| Parameter | Largest \|r\| | Dimension |
|---|---|---|
| f_d | 0.542128 | pooled_58 |
| F | 0.325354 | pooled_23 |
| host_median | 0.423780 | pooled_32 |
| host_sigma_ln | 0.237802 | pooled_62 |

27 matrix entries exceed the requested descriptive
|r|>0.3 flag. This is not a multiple-testing-corrected significance test.
Raw physical-parameter correlations are shown; correlations with normalized
prior coordinates, including logarithmic F/host-median coordinates, are
also saved in JSON. Linear correlation is not a complete information test.

Crucially, correlation is invariant to rescaling: a tiny surviving signal
can have substantial correlation yet be effectively unused by the decoder.
The stage measurements expose that problem:

| Stage | Initialized catalog variation RMS | Trained catalog variation RMS |
|---|---|---|
| per_burst_projection_mean | 0.060375196 | 0.057472813 |
| ISAB_1_element_mean | 0.095120124 | 0.0030653066 |
| ISAB_2_element_mean | 0.090634246 | 0.00011340999 |
| PMA_summary | 0.077100768 | 1.4816416e-07 |

Before PMA these measurements are means over real elements of each catalog;
they are a comparable catalog-level diagnostic, not the entire set activation
tensor. The PMA measurement is its actual output. Initial weights were
reconstructed from the recorded seed, with forward evaluation only.
Pooled variation has fallen by a factor of 520374 from initialization.
The per-burst projection still contains parameter-associated variation.

### PMA internals

| PMA internal stage | Float32 catalog variation RMS | Float64 catalog variation RMS |
|---|---|---|
| attention_output | 4.8553084e-05 | 4.8517456e-05 |
| after_norm1 | 1.6762382e-06 | 1.676223e-06 |
| feedforward_output | 4.1128942e-06 | 4.0997179e-06 |
| after_norm2 | 1.4816416e-07 | 1.4458788e-07 |
| uniform_value_projection | 4.8542926e-05 | 4.8512055e-05 |

The attention output has a large nearly constant component; its value RMS
is 12.417583.
The reductions across the first normalization and final residual/normalization
are substantial. Learned normalization gain ranges are
[0.43624716997146606, 1.6688932180404663] and
[0.1528734564781189, 0.8224412798881531], so this is not a zero-gain
disconnect. Value/output projection weights are also nonzero. Uniform-value
projection variation is similar to actual attention-output variation;
attention weighting alone is not the sole bottleneck.

Float64 evaluation of the same weights reproduces the collapse. Float32
roundoff affects the smallest variations but is not the origin of the problem.
The endpoint comparison identifies learned attenuation; it cannot reconstruct
when training entered this regime or which optimizer/normalization interaction
caused it. Intermediate encoder histories were not saved.

## 3. Does the velocity field use catalog conditioning?

At fixed x_t=0 and t=0.5:

| Condition replacement | Mean \|\|v_real − v_alternative\|\| | Maximum |
|---|---|---|
| zero | 0.159035077 | 0.168236136 |
| mean | 0.00791215113 | 0.0332293846 |
| shuffled | 0.0110785802 | 0.0343747623 |
| pooled_shuffled_only | 7.07627288e-08 | 1.40971594e-07 |
| extras_shuffled_only | 0.0110785783 | 0.0343748406 |

The JSON additionally records a fixed nonzero x_t and other times, with the
same qualitative separation. Full shuffling changes the two metadata features
as well as the pooled vector; isolating those paths shows that metadata
accounts for essentially the entire effect. The mean condition is another
control for catalog variation. Zeroing is deliberately off-distribution and
its larger effect does not demonstrate use of scientific catalog information.

At the first velocity layer, the varying pooled contribution has RMS
8.34634325e-08, versus
0.0739500228 from the
two metadata features. The mean-condition contribution has RMS
0.726009731.
The decoder has nonzero weights and derivatives for the pooled features,
but their natural variation reaches it at negligible amplitude.

## 4. Conditioning benefit in the frozen-model loss

| Condition | Mean CFM loss | Alternative − correct | Paired catalog SE |
|---|---|---|---|
| correct | 2.9265616417 | — | — |
| zero | 3.3878953415 | 0.4613337 | 0.0345575431 |
| mean | 2.9296203703 | 0.00305872858 | 0.00109568861 |
| shuffled | 2.9306613812 | 0.00409973949 | 0.00136415708 |
| pooled_shuffled_only | 2.9265616503 | 8.64267349e-09 | 1.16875639e-08 |
| extras_shuffled_only | 2.9306613711 | 0.00409972936 | 0.00136415739 |
| shuffled_average_8 | 2.9297039299 | 0.00314228822 | 0.00114638092 |

Each catalog uses 256 common random x0/time
pairs. Comparisons retain identical targets and noise; the averaged shuffled
control uses 8 derangements. Uncertainty is computed across
catalog-average paired differences, not by treating all repeated noise draws
as independent catalogs. These are diagnostics, not changed gate tolerances.

| Condition | f_d | F | host_median | host_sigma_ln |
|---|---|---|---|---|
| correct | 2.93098906 | 3.13722311 | 2.86084698 | 2.77718741 |
| zero | 3.36416259 | 3.55854274 | 3.35151259 | 3.27736344 |
| mean | 2.93505564 | 3.14034921 | 2.86271295 | 2.78036368 |
| shuffled | 2.93752097 | 3.14068894 | 2.86473156 | 2.77970404 |
| pooled_shuffled_only | 2.93098904 | 3.13722314 | 2.86084699 | 2.77718743 |
| extras_shuffled_only | 2.93752095 | 3.14068888 | 2.86473159 | 2.77970407 |
| shuffled_average_8 | 2.93581295 | 3.13909321 | 2.86328670 | 2.78062285 |

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

| Summary dimension | f_d | F | host_median | host_sigma_ln |
|---|---|---|---|---|
| pooled_00 | 0.174017 | -0.106089 | -0.163368 | -0.148032 |
| pooled_01 | -0.051991 | -0.052370 | -0.265137 | -0.190118 |
| pooled_02 | -0.100615 | -0.049951 | -0.273317 | -0.167691 |
| pooled_03 | -0.021834 | -0.061042 | -0.266387 | -0.187512 |
| pooled_04 | 0.011469 | 0.021641 | 0.228071 | 0.194394 |
| pooled_05 | 0.131017 | -0.184502 | -0.067627 | -0.036667 |
| pooled_06 | -0.229368 | 0.181981 | 0.095959 | 0.105961 |
| pooled_07 | -0.131481 | 0.183935 | -0.073880 | -0.028274 |
| pooled_08 | -0.290192 | 0.208616 | 0.032189 | 0.069133 |
| pooled_09 | 0.132516 | 0.058094 | 0.257626 | 0.141351 |
| pooled_10 | -0.069598 | 0.135228 | 0.125764 | 0.062395 |
| pooled_11 | -0.161791 | 0.036972 | **-0.325436** | -0.210100 |
| pooled_12 | **0.316039** | -0.233304 | -0.053366 | -0.086667 |
| pooled_13 | 0.255430 | -0.183149 | 0.005628 | -0.085194 |
| pooled_14 | -0.201084 | 0.019251 | **-0.355291** | -0.187111 |
| pooled_15 | -0.108348 | 0.055282 | 0.174161 | 0.096965 |
| pooled_16 | **-0.319049** | 0.125447 | -0.244713 | -0.058996 |
| pooled_17 | 0.009711 | 0.065209 | 0.258748 | 0.157120 |
| pooled_18 | -0.256422 | 0.132013 | 0.091217 | 0.137436 |
| pooled_19 | **-0.308091** | 0.079488 | 0.059294 | 0.175136 |
| pooled_20 | 0.285096 | 0.030716 | 0.281363 | 0.124125 |
| pooled_21 | 0.188881 | -0.163224 | -0.097713 | -0.088672 |
| pooled_22 | **0.446958** | -0.172477 | 0.002043 | -0.126510 |
| pooled_23 | **0.535719** | **-0.325354** | 0.197953 | -0.003358 |
| pooled_24 | 0.243888 | -0.093505 | 0.198633 | 0.168695 |
| pooled_25 | 0.029379 | -0.081228 | -0.172644 | -0.196430 |
| pooled_26 | -0.048474 | 0.139925 | 0.208724 | 0.081507 |
| pooled_27 | -0.001837 | -0.178369 | -0.098912 | 0.028915 |
| pooled_28 | -0.090236 | 0.115446 | 0.199933 | 0.129025 |
| pooled_29 | -0.146488 | 0.125270 | 0.187399 | 0.138376 |
| pooled_30 | 0.071474 | -0.030395 | -0.040335 | -0.149171 |
| pooled_31 | **0.356818** | -0.086339 | **0.327254** | 0.084998 |
| pooled_32 | **-0.524646** | 0.168715 | **-0.423780** | -0.201643 |
| pooled_33 | **0.307000** | -0.017792 | **0.362148** | 0.169421 |
| pooled_34 | -0.102151 | 0.153639 | 0.165696 | 0.074430 |
| pooled_35 | **0.481781** | **-0.309581** | 0.168694 | 0.097299 |
| pooled_36 | **-0.346052** | 0.119862 | -0.120008 | 0.068275 |
| pooled_37 | -0.173618 | 0.146409 | 0.124300 | 0.116004 |
| pooled_38 | 0.029157 | 0.063304 | 0.285808 | 0.160508 |
| pooled_39 | 0.146646 | -0.196971 | -0.040132 | 0.000262 |
| pooled_40 | 0.039228 | -0.084676 | -0.227045 | -0.211718 |
| pooled_41 | -0.078443 | 0.127436 | 0.182374 | 0.138027 |
| pooled_42 | **-0.321815** | 0.295480 | 0.002782 | -0.085497 |
| pooled_43 | -0.159513 | 0.192558 | 0.127250 | 0.026028 |
| pooled_44 | -0.251268 | 0.192496 | 0.025700 | 0.026338 |
| pooled_45 | -0.108959 | 0.136502 | 0.007534 | 0.098066 |
| pooled_46 | -0.107866 | -0.048436 | -0.285075 | -0.155227 |
| pooled_47 | **0.471212** | -0.184012 | **0.369504** | 0.125067 |
| pooled_48 | **0.370541** | -0.246150 | 0.230954 | 0.141218 |
| pooled_49 | 0.002330 | 0.039063 | 0.288307 | 0.186938 |
| pooled_50 | **0.315420** | -0.281022 | 0.096874 | 0.025822 |
| pooled_51 | 0.134931 | -0.155594 | -0.179371 | -0.078066 |
| pooled_52 | -0.019391 | 0.045108 | 0.241208 | 0.188468 |
| pooled_53 | 0.133948 | -0.055907 | -0.216785 | -0.205856 |
| pooled_54 | -0.065572 | 0.144818 | 0.192461 | 0.016312 |
| pooled_55 | -0.129458 | 0.037925 | **-0.319520** | -0.203369 |
| pooled_56 | -0.118523 | 0.027105 | -0.284488 | -0.136082 |
| pooled_57 | 0.073511 | -0.065630 | -0.227452 | -0.198409 |
| pooled_58 | **0.542128** | -0.117071 | **0.303392** | 0.126122 |
| pooled_59 | 0.208390 | -0.075504 | 0.245713 | 0.154690 |
| pooled_60 | 0.012267 | -0.083052 | -0.243182 | -0.175197 |
| pooled_61 | -0.081685 | 0.149832 | 0.152931 | 0.082190 |
| pooled_62 | -0.097789 | 0.015189 | **-0.300575** | -0.237802 |
| pooled_63 | 0.195031 | 0.029117 | **0.319240** | 0.137155 |
| log1p_N | 0.081032 | 0.125546 | 0.118216 | -0.020197 |
| localized_fraction | 0.149386 | -0.152605 | -0.132968 | -0.111556 |

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
