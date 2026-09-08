"""SPEC-02a T1 audit of the existing production L0 RNG, without training.

Finite-cutoff PDF moments determine both standard errors. F/sqrt(z) is
never used as an RMS. For unbiased S^2, independence gives exactly
Var(S^2) = [mu4 - (N-3)/(N-1) * variance^2] / N. This is a sampling
variance identity, not an assumption that the heavy-tailed statistic is
Gaussian. Passing a three-standard-error check is not proof of RNG quality.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from frbsim.fields import cosmic_pdf, sample_cosmic
from frbsim.mean_dm import transcription
from .tolerances import T1_SAMPLING_SIGMAS


@dataclass(frozen=True)
class SeedRange:
    """Half-open range of top-level catalog seeds, not physical parameters."""
    start: int
    stop: int

    def __post_init__(self):
        if (type(self.start) is not int or type(self.stop) is not int
                or not 0 <= self.start < self.stop):
            raise ValueError("seed ranges require integer 0 <= start < stop")


def assert_disjoint_seed_ranges(ranges):
    ordered = sorted(ranges.items(), key=lambda item: item[1].start)
    for (left_name, left), (right_name, right) in zip(ordered, ordered[1:]):
        if left.stop > right.start:
            raise ValueError(f"STOP: seed-space overlap: {left_name} and {right_name}")


# Engineering allocation only; no training catalogs have been generated.
SEED_RANGES = {"training": SeedRange(0, 50_000),
               "validation": SeedRange(50_000, 55_000),
               "test": SeedRange(55_000, 60_000),
               "audit": SeedRange(60_000, 70_000)}
T1_CONFIG = {"n": 10_000, "z": 0.5, "F": 0.32, "f_d": 0.85,
             "seed": SEED_RANGES["audit"].start}


def run_t1():
    assert_disjoint_seed_ranges(SEED_RANGES)
    n, z, F, fd, seed = (T1_CONFIG[k] for k in ("n", "z", "F", "f_d", "seed"))
    pdf = cosmic_pdf(F / np.sqrt(z))
    dm_scale = transcription(z, f_d=fd)
    delta = np.exp(pdf.log_delta)
    mean_delta = pdf.mean
    variance_delta = pdf.truncated_std**2
    fourth_delta = float(np.trapezoid(
        (delta - mean_delta)**4 * pdf.density_in_log_delta, pdf.log_delta))
    target_mean = dm_scale * mean_delta
    target_variance = dm_scale**2 * variance_delta
    mean_se = np.sqrt(target_variance / n)
    variance_se = dm_scale**2 * np.sqrt(
        (fourth_delta - (n - 3) / (n - 1) * variance_delta**2) / n)
    draws = sample_cosmic(np.full(n, z, dtype=np.float64),
                          np.random.default_rng(seed), f_d=fd, F=F)
    mean, variance = float(draws.mean()), float(draws.var(ddof=1))
    mean_error_se = abs(mean - target_mean) / mean_se
    variance_error_se = abs(variance - target_variance) / variance_se
    passed = (bool(np.all(np.isfinite(draws)))
              and mean_error_se <= T1_SAMPLING_SIGMAS
              and variance_error_se <= T1_SAMPLING_SIGMAS)
    return {"name": "T1", "status": "pass" if passed else "fail",
            "config": T1_CONFIG, "dtype": str(draws.dtype),
            "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
            "delta_max": pdf.delta_max, "truncated_mean_delta": mean_delta,
            "truncated_variance_delta": variance_delta,
            "truncated_fourth_central_moment_delta": fourth_delta,
            "analytic_mean_dm": target_mean, "sample_mean_dm": mean,
            "analytic_variance_dm": target_variance, "sample_variance_dm": variance,
            "mean_standard_error": float(mean_se),
            "variance_standard_error": float(variance_se),
            "mean_error_in_standard_errors": float(mean_error_se),
            "variance_error_in_standard_errors": float(variance_error_se),
            "tolerance": {"sampling_standard_errors": T1_SAMPLING_SIGMAS},
            "seed_ranges": {k: asdict(v) for k, v in SEED_RANGES.items()},
            "limitation": "Finite-cutoff fourth moment makes the variance gate broad; "
                          "three standard errors do not imply Gaussian coverage."}


def write_phase2a_report(root, t1, tests_passed, failures):
    """Keep the incomplete Phase 2a ledger separate from Phase 1 evidence."""
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    config = {"T1": T1_CONFIG, "seed_ranges": {k: asdict(v) for k, v in SEED_RANGES.items()}}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": 1, "scope": "Phase 2a preflight only",
              "git_sha": git("rev-parse", "HEAD"),
              "git_dirty": bool(git("status", "--porcelain")),
              "config_hash": config_hash, "python": platform.python_version(),
              "numpy": np.__version__, "acceptance_complete": False, "exit_status": 1,
              "preflight_tests_passed": tests_passed, "test_failures": failures,
              "pretraining": {"T1": t1 or {"status": "not_run"},
                  "T2": {"status": "not_run", "reason": "D4/D5 authorized; validation report required"},
                  "T3": {"status": "not_run"}},
              "gates": {f"G-P{i}": {"status": "not_run", "hard": i not in (3, 7)}
                        for i in range(1, 9)},
              "smoke_test": {"status": "not_run"}, "trained_checkpoint": None}
    selection_path = root / "results" / "selection-audit.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        report["pretraining"]["T2"] = {
            "status": "blocked" if not selection["prints_within_factor_two"] else "partial",
            "D4_D5_authorization": "resolved: docs/SPEC-02a-authorization.txt",
            "reason": "STOP: selection print comparison failed" if not selection["prints_within_factor_two"]
                      else "Selection validated; conditional joint generator validation remains",
            "selection_evidence": "results/selection-audit.json",
            "selection_evidence_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}
        report["pretraining"]["T2"]["z02_verified"] = selection.get("z02_verification",{}).get("status")=="pass"
        generator_path = root / "results" / "generator-audit.json"
        if generator_path.exists():
            generated = json.loads(generator_path.read_text(encoding="utf-8"))
            report["training_data_generator"] = {"status":generated["status"],"evidence":"results/generator-audit.json",
                "cold_bursts_per_second":generated["cold_bursts_per_second"],
                "required_bursts_per_second":generated["performance_tolerance"],
                "sha256":hashlib.sha256(generator_path.read_bytes()).hexdigest()}
            ready = (selection["prints_within_factor_two"] and selection["numeric_checks_pass"]
                     and report["pretraining"]["T2"]["z02_verified"] and generated["status"]=="pass")
            report["pretraining"]["T2"]["status"] = "pass" if ready else "blocked"
            report["pretraining"]["T2"]["reason"] = "Authorized selected generator validated" if ready else "Pretraining validation incomplete"
        report["open_findings"] = [{"name":"low_redshift_dominance_claim","status":"corrected_by_human_authorization",
            "claim":"selected catalogs are dominated by z<=0.5",
            "actual_quadrature_fraction_below_z05":{k:v["quadrature_fraction_detected_below_z05"]
                for k,v in selection["population_naive_rejection"].items()},
            "action":"Retire quiet validation; record moderate-z dominance and Phase 3 framing. CHIME Catalog 1 DM comparison is a future consistency check."}]
    prior_path = root / "results/prior-predictive.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        report["pretraining"]["T3"] = {"status":prior["status"],"evidence":"results/prior-predictive.json",
            "event_CDFs":{e["frb"]:e["quadrature_CDF"] for e in prior["events"]},
            "sha256":hashlib.sha256(prior_path.read_bytes()).hexdigest()}
    target = root / "results" / "phase2a_gates.json"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
