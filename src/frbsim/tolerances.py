"""Immutable acceptance thresholds from SPEC-FINAL v2.1 + user amendments.

Changing these requires human sign-off. Numerical solver settings live elsewhere.
"""
FORMULA_REL = .001
AUTHOR_REL = .01
K_RANGE = (950., 1000.)
INTEGRAL_RANGE = (1.10, 1.12)
DM1_RANGE = (880., 940.)
LOW_Z_ANCHOR = 973.
DISTANCE_REL = 1e-6
UNIFORM_RAY_REL = 1e-6
SHELL_CONVERGENCE_REL = .005
POWER_REL = .05
FIELD_MEAN_REL = .01
DM_MEAN_REL = .01
BAND_REL = .30
CALIBRATION_SHAPE_REL = .20
PREDICTIVE_REQUIRED = 5
MW_MEDIAN_REL = .30
POLE_RANGE = (15., 45.)
CATALOG_SECONDS = 300.
COUNT_SLOPE_ABS = .1
GATE_NAMES = tuple(f"G{i}" for i in range(1, 10))
