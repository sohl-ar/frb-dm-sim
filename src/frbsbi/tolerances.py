"""Immutable SPEC-02a T1 tolerance; changes require human sign-off."""

T1_SAMPLING_SIGMAS = 3.0

# Consolidated authorization: approximate print comparisons stop beyond 2x.
SELECTION_PRINT_FACTOR = 2.0
# New numerical checks for the newly authorized tail implementation; these
# do not replace or relax any Phase 1 or Phase 2a acceptance threshold.
TAIL_QUADRATURE_REL = 1e-8
TAIL_REJECTION_SE = 3.0
CONDITIONAL_KS_P_MIN = .001
GENERATOR_BURSTS_PER_SECOND = 8600.0  # SPEC-02a and consolidated authorization; immutable.
T3_CDF_INTERVAL = (.005,.995)  # Consolidated authorization, central 99%.
PERMUTATION_W1 = 1e-4  # SPEC-02a G-P1, physical parameters normalized by prior widths.
TRANSFORM_ROUNDTRIP_REL = 1e-6  # New float32 engineering check; no physics tolerance replacement.
SBC_TRIALS = 1000
SBC_SAMPLES = 1000
SBC_KS_ALPHA = .01
COVERAGE_ABS = .02
