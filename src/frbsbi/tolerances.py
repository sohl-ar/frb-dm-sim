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
