"""Generate the current report from saved diagnostic counts and posterior draws."""
import json
import numpy as np
from frbsbi.data import ROOT
from frbsbi.gate_metrics import PARAMETERS
from frbsbi.train import write_json
from frbsbi.train_conditioned import RUN_DIR

out = RUN_DIR/"diagnosis-v2"
r = json.loads((out/"diagnosis.json").read_text())
support = json.loads((out/"training-support.json").read_text())
ablation = json.loads((out/"flag-ablation.json").read_text())
# These probes reused inference settings, not the baseline artifact's timing.
# Remove inherited baseline timing metadata; no measurement is replaced.
for item in r["host_probe"]["artifacts"].values():
    item.pop("seconds",None)
ablation["sampling"].pop("seconds",None)
write_json(out/"diagnosis.json",r)
write_json(out/"flag-ablation.json",ablation)
training = json.loads((RUN_DIR/"training.json").read_text())
pair = lambda row: f"{100*row['0.68']['coverage']:.2f} / {100*row['0.95']['coverage']:.2f}"
lines = ["# Completed Phase 2a diagnosis","",
    "Recommendation: Phase 2a remains INCOMPLETE. Retain the existing criteria. "
    "The authorized comparator bug is fixed; F and host_sigma_ln still fail the aggregate SBC coverage gate. "
    "The host probes reveal a model generalization problem, not evidence for an established physical information limit.","",
    f"Checkpoint: `{r['checkpoint_sha256']}`; best epoch {training['best_epoch']}, "
    f"validation loss {training['best_validation_loss']:.9f}. No retraining occurred.","",
    "## Exact aggregate coverage and boundary correction","",
    "| Parameter | 68% covered/trials | 68% coverage | Delta (pp) | 95% covered/trials | 95% coverage | Delta (pp) | Corrected status |",
    "|---|---|---:|---:|---|---:|---:|---|"]
for row,metric in zip(r["coverage_exact"],r["corrected_SBC"]):
    a,b = row['0.68'],row['0.95']
    lines.append(f"| {row['parameter']} | {a['covered']}/{a['trials']} | {100*a['coverage']:.1f}% | {a['deviation_pp']:+.1f} "
                 f"| {b['covered']}/{b['trials']} | {100*b['coverage']:.1f}% | {b['deviation_pp']:+.1f} | {metric['status'].upper()} |")
lines += ["", "All adjusted KS p-values still pass; only the f_d status changes. "
    "Every coverage value, KS result and rank is identical to the archived calculation. "
    "These are marginal aggregate deviations, but marginal does not mean a passed gate. "
    "F remains outside the 68% band by half a percentage point and host_sigma_ln by four tenths of a point.","",
    "The evaluator now compares the exact rational covered/trials fraction with the exact decimal nominal probability "
    "and tolerance. Both inclusive endpoints pass, and the immediate count outside either endpoint fails. "
    "The tolerance is still +/-2pp, without an epsilon or an arbitrary rounding precision. "
    "Changing a rejection test from > to >= would also reject the exact endpoint, so suggested option A would not fix the bug. "
    "The saved-sample SBC calculation was rerun with the corrected evaluator; no new Monte Carlo draws were needed.","",
    "## Stratified coverage","",
    "Each cell is empirical 68% / 95% coverage in percent. Fractions use realized localization masks, "
    "not the latent Bernoulli probability. Exact counts and binomial confidence intervals are in diagnosis.json. "
    "Subgroups are diagnostic and do not acquire new hard +/-2pp gates.","",
    "| Group | Trials | f_d | F | Host median | Host sigma_ln |",
    "|---|---:|---:|---:|---:|---:|"]
names = {"low_localized":"ell < 0.15","medium_localized":"0.15 <= ell < 0.35", "high_localized":"ell >= 0.35",
         "small_N":"N < 64","large_N":"N >= 64"}
for name,row in r["groups"].items():
    lines.append(f"| {names[name]} | {row['n_trials']} | "+" | ".join(pair(v) for v in row['coverage'])+" |")
small = r["groups"]["small_N"]
lines += ["", "F's central-68% overcoverage varies little across localization-fraction strata. "
    "The larger change appears by catalog size: small catalogs show more F overcoverage and more host-width undercoverage. "
    "This does not establish that unlocalized marginalization is the unique cause or that training overfit large N. "
    "Localization strata also show tail mismatches: low-ell host-median 95% coverage is low, while high-ell host-width "
    "95% coverage is low. Aggregate coverage therefore conceals composition-dependent behavior.","",
    f"The small-N group has {small['n_trials']} trials. Its binomial 95% interval for F's 68% coverage is "
    f"[{100*small['coverage'][1]['0.68']['binomial_95_interval'][0]:.2f}, "
    f"{100*small['coverage'][1]['0.68']['binomial_95_interval'][1]:.2f}]%; "
    "for host_sigma_ln it is "
    f"[{100*small['coverage'][3]['0.68']['binomial_95_interval'][0]:.2f}, "
    f"{100*small['coverage'][3]['0.68']['binomial_95_interval'][1]:.2f}]%. "
    "Both include the nominal coverage. These are unadjusted descriptive intervals, not evidence to waive a failed gate.","",
    "## Host diagnosis: matched localization experiment","",
    f"The paired experiment uses {r['host_probe']['trials']} catalogs at N={r['host_probe']['N']}, "
    f"with {r['host_probe']['draws']} posterior draws each. Theta, DM, fluence, positions, sample seeds and solver settings "
    "are matched. Only observed-redshift availability changes. Localization supplies z, not a direct host-DM measurement: "
    "the observed DM still combines cosmic and host contributions. See [Macquart et al.](https://arxiv.org/html/2005.13161), Eq. 1.","",
    "| Composition | Host median width/prior | Host sigma width/prior | Host median coverage 68/95 (%) | Host sigma coverage 68/95 (%) |",
    "|---|---:|---:|---:|---:|"]
for name in ("mixed","all_localized","all_unlocalized"):
    row = r["host_probe"][name]
    w = row['contraction_diagnostic']['median_posterior_prior_ratio']
    lines.append(f"| {name} | {w[2]:.6f} | {w[3]:.6f} | {pair(row['coverage'][2])} | {pair(row['coverage'][3])} |")
lines += ["", "The mixed posterior shows parameter tradeoffs: median within-catalog Spearman correlation "
    f"host median vs f_d is {r['host_probe']['mixed']['median_within_posterior_spearman'][2][0]:.3f}; "
    f"host sigma vs F is {r['host_probe']['mixed']['median_within_posterior_spearman'][3][1]:.3f}. "
    "These learned correlations are compatible with confounding but do not establish the true posterior geometry or a Fisher bound.","",
    f"Training support explains why the all-localized comparison cannot determine the host information ceiling: "
    f"among {support['training_catalogs']} training catalogs, only {support['all_localized_count']} were all-localized, "
    f"and their largest N was {support['all_localized_N_max']}. Among {support['N_ge64_count']} training catalogs with N>=64, "
    f"there were {support['all_localized_N_ge64']} all-localized and {support['all_unlocalized_N_ge64']} all-unlocalized catalogs.","",
    "The all-localized standardized availability flag is "
    f"{support['standardized_probe_features']['all_localized'][0]['median_standardized_value']:.6f} "
    "for has_unlocalized. The combination of this rare flag and large N is absent from training. "
    "This is a concrete distribution-support and encoding-sensitivity problem in the learned model.","",
    "A targeted counterfactual replaced only that standardized flag with the paired mixed-catalog value "
    f"{ablation['replacement_standardized_value']:.6f}; all actual observations and all parameters stayed fixed. "
    "It changed the all-localized response as follows:","",
    "| Parameter | Original all-localized ratio | Counterfactual ratio | Original coverage 68/95 (%) | Counterfactual coverage 68/95 (%) |",
    "|---|---:|---:|---:|---:|"]
for j,name in enumerate(PARAMETERS):
    a,b = r['host_probe']['all_localized'],ablation['result']
    lines.append(f"| {name} | {a['contraction_diagnostic']['median_posterior_prior_ratio'][j]:.6f} "
                 f"| {b['contraction_diagnostic']['median_posterior_prior_ratio'][j]:.6f} "
                 f"| {pair(a['coverage'][j])} | {pair(b['coverage'][j])} |")
lines += ["", "The counterfactual directly demonstrates large sensitivity to a single availability feature. "
    "It is an intentionally inconsistent summary, not a valid posterior, a recalibration method, or a gate pass. "
    "Coverage is still imperfect. It identifies a model mechanism in this stress regime without proving why every "
    "mixed-catalog host posterior remains wide.","",
    "## Decision and proposed next work","",
    "Keep Phase 2a INCOMPLETE and retain all gate criteria. The aggregate errors are in the request's marginal range, "
    "but Scenario A's condition that all science parameters pass is false because F still fails. "
    "The endpoint-composition probes reveal substantial additional model failures. "
    "No theoretical lower bound or numerical reference posterior was established, so Scenario D and a physics-limited "
    "host claim are not supported. A nuisance-only gate revision would also not resolve F's calibration failure.","",
    "For a separately authorized model-fix session, prioritize an experiment that preserves availability bits as binary "
    "inputs instead of standardizing rare flags, and gives the training distribution explicit coverage of the catalog "
    "compositions required by validation. Test their contributions in a controlled comparison, rather than assuming "
    "either alone is sufficient. Neither change nor any retraining is implemented here. "
    "Use a numerical reference posterior on a small fixed set of representative mixed/localized catalogs before changing "
    "the host-contraction criterion. The current results cannot justify a bound of the form 'host parameters require N>X'.", "",
    "Passing marginal parameter SBC is not proof that all data information was extracted; this limitation is studied by "
    "[Modrak et al.](https://arxiv.org/abs/2211.02383). Thus even a future corrected aggregate pass would not replace the "
    "remaining contraction, TARP, information-content and engineering gates. No Phase 2a acceptance is declared.","",
    "Evidence: results/conditioning-v1/diagnosis-v2/diagnosis.json, training-support.json, flag-ablation.json and saved NPZs. "
    "The original assessment and ledger are archived in that directory. The canonical G-P4 row has the new assessment "
    "provenance while retaining the original sampling identity; unchanged caches are not relabeled as newly generated. "
    "The full runner's source-identity guard remains strict and no automatic failed-run restart is enabled.",""]
(ROOT/"CONDITIONING_FOLLOWUP.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
print(json.dumps({"report":"CONDITIONING_FOLLOWUP.md","status":"written","acceptance_complete":False}))
