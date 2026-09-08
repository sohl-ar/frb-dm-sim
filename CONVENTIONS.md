# Units and conventions audit

Phase 2a's consolidated authorization supplies a separate Schechter LF
and selection proxy. The fixed conversion is E[erg]=9.52e31 L for the
specified 1 GHz bandwidth; L already includes 4*pi. This is a declared
pseudo-luminosity convention, not a new spectral/redshift correction.
The 1.3 GHz energy reference and 1 GHz bandwidth versus native observing
bands remain a misspecification axis. The Schechter exponential scale is
not the brightest possible burst. See `DECISIONS.md` and the executed
`results/selection-audit.json`; training remains stopped at its print check.

This implements the mean-DM and L0 portions of SPEC-FINAL v2.1 with the
user's five amendments. L2 acceptance is **not complete**. Fixed paper
anchors and gate tolerances have not been revised. Computed results live
in `results/`, written by executable runs rather than copied into this audit.

## 1. Density and the scale-factor audit

The density appearing in the dispersion integral is physical number density.
A fixed comoving volume has a physical volume smaller by a factor
`a^3 = (1+z)^-3`. Conserving its electron count therefore gives
`n_e,phys = n_e,0 (1+z)^3`. The present-day mean is
`n_e,0 = f_d Omega_b [3 H0^2/(8 pi G)] (1-Y_He/2)/m_p`.
For the specified fully ionized mixture, hydrogen supplies 0.75 electrons
per proton mass of baryons and helium supplies `2*0.25/4 = 0.125`, totaling
0.875. This is a mass-fraction calculation, not a helium number fraction.

## 2. Proper path length

From `1+z=1/a`, differentiating with respect to cosmic proper time gives
`|dz/dt|=(1+z)H(z)`. A photon travels the local proper distance `dl=c|dt|`.
Consequently `dl/dz=c/[H0 (1+z) E(z)]`. H0 supplies the inverse-time unit;
E is dimensionless. Comoving distance instead obeys `dchi=(1+z)dl`.

## 3. Observed dispersion weighting

The contribution to observed DM is `n_e,phys dl/(1+z)`. The second factor
of `1/(1+z)` is the dispersion-delay observation weighting, distinct from
the proper-length conversion. Density contributes three powers, proper
length removes one, and the delay weighting removes one: the net
integrand contains `(1+z)^1`. Ray integration must accumulate this observed
contribution inside each shell; dividing an entire column by the source
redshift would assign the wrong weight to foreground material.

## 4. Closure with Macquart Eq. (2)

Before substituting density, the expression is
`dDM = c n_e,phys dz/[H0 (1+z)^2 E(z)]`.
Substituting the critical density cancels one power of H0 and leaves
`3 c H0 Omega_b f_d (0.875)/(8 pi G m_p)` times
`(1+z) dz/E(z)`. SI evaluation gives an electron column in m^-2;
the single conversion to pc cm^-3 is centralized in `constants.py`.
The four checks use SI redshift quadrature, CGS scale-factor quadrature,
a comoving-radius sphere ray ODE, and original upstream density/summation
functions. Shared fixed physical inputs are intentional; numerical
integration implementations are distinct.

The authors' repository has its own evolving diffuse-fraction model.
For the formula comparison, boundary callbacks explicitly supply constant
f_d and full H/He ionization, and its cosmology argument receives the
specified background. Its `ne_cosmic` and `average_DM` function bodies
are unmodified. Full original source, BSD license, commit and SHA256 are
vendored. This checks matched assumptions; it does not claim that our
constant-f_d curve reproduces the upstream default evolving census.

The background contains matter and Lambda only. The Astropy oracle is an
explicit `FlatLambdaCDM(H0=67.4, Om0=.315, Ob0=.049, Tcmb0=0)`, avoiding
radiation or neutrino additions from an unrelated convenience cosmology.

## 5. Distances, pseudo-luminosity, population and selection

Comoving chi determines volume; `DL=(1+z)chi` enters the authorized
phenomenological fluence relation. L has units **Jy ms Mpc^2** and DL is
in Mpc, so `L/(4 pi DL^2)` is in Jy ms. L is not emitted energy in erg.
Bandwidth, spectral K corrections and bolometric redshift factors belong
to the expressly deferred physical energy model. This phenomenological
choice is authorized by the user; no equivalence to a particular zdm
implementation has been independently established here.

The two redshift shapes follow the literal spec `p(z) proportional to
R(z) dVc/dz`. SFR uses Madau & Dickinson Eq. 15 with exponent **5.6**;
the constant normalization cancels. This is a rate-shape proxy as specified.
No additional source-clock event-rate interpretation is silently imposed.
The default interval is 0.05 to 1.5. Below z=0.05, peculiar velocities
and local structure compromise the smooth-cosmology model. An explicit
nondefault lower limit remains possible for controlled experiments.

G8a uses an independent Euclidean radial fixture with a negligible inner
excluded volume over its fit interval. It does not use the cosmic redshift
floor. Its delta-function L gives the volume-law slope -3/2. G8b holds
distance fixed and samples a bounded `dN/dL proportional to L^-alpha`,
testing the cumulative slope `-(alpha-1)` away from both luminosity bounds.
The constants specifying these synthetic numerical fixtures are not
empirical survey or luminosity-function defaults.

All sky draws are uniform in RA and sin(dec). Mixing surveys equally is
an explicit simulation design, not a claim about exposure or detection
rates. Localization is a configured Bernoulli probability, independent of
other properties. DM error and noise-equivalent fluence are supplied by
the survey configuration. The basic SNR proxy is fluence divided by that
noise-equivalent fluence. Redshift measurement errors are not modeled.
No uncited empirical LF bounds, localization fractions or noise levels
are installed as scientific defaults.

## 6. Host logarithm convention

We draw `ln(DM_host,rest) ~ Normal(ln(median), sigma_ln^2)`.
The specified median is not mu itself. The observation receives
`DM_host,rest/(1+z_host)`. Rest-frame median and width have no redshift
evolution in v1. Comparing a base-10 grid requires converting **both**
parameters: `mu_ln=ln(10)*mu_10`, `sigma_ln=ln(10)*sigma_10`.
Macquart Eq. 3 and the spec supply the natural-log convention and ranges;
the exact fiducial median and width are the spec's choices within those
ranges, not measurements attributed to the paper.

## 7. Diffuse baryons and halos

The parameter is **f_d**, the fraction in diffuse ionized gas, including
ionized halo gas. Stars, stellar remnants and neutral ISM are excluded.
An f_IGM partition excluding halos is a different quantity and cannot
replace f_d numerically. There is no additional cosmic halo mean term.
The Milky Way halo is a local foreground term, distinct from intervening
cosmic halos. It uses the specified fiducial and sensitivity range.
At fixed background the cosmic mean is proportional to f_d Omega_b;
pinning Omega_b does not observationally resolve that degeneracy.

L0 preserves Macquart's asymmetric alpha=beta=3 PDF. Its dimensionless
shape parameter is F/sqrt(z). This is **not** its ordinary standard
deviation. A log-Delta grid from 1e-4 to 1e4 resolves the core and long
tail; C0 enforces unit mean on that domain, and A normalizes it. At large
Delta, missing probability decreases as Delta_max^-2 and missing first
moment as Delta_max^-1, whereas the second moment grows logarithmically.
The cutoff doubling check measures stability of the 5th/95th quantiles.
The truncated standard deviation is reported separately. Heterogeneous
draws interpolate quantiles between sigma nodes; they never rescale a
sample to force its mean. L0 has no spatial correlations.

## 8. GLASS shells and amended L2 calibration — pending implementation

The official GLASS interface uses `RadialWindow` objects and angular
spectra in triangular ordering. These are projected shell statistics,
not a direct three-dimensional P(k) array. Shell-boundary integration,
projection, k units and Gaussian-field power validation remain to be
implemented and exercised with the actual library. This section is not
marked verified merely because the API documentation is reachable.

The authorized L2 prescription is to scale Gaussian fluctuation amplitude
and calibrate realized sightline standard deviation at z=0.5 to
`F z^-0.5 <DM(z)>`. Validate its shape at z=0.2 and 1.0 within 20%.
This is **phenomenological calibration, not derivation**. It replaces
G4(ii)'s old approximate scatter-band comparison as authorized. The
L0 PDF-width parameter must not be silently reinterpreted as L0's RMS
to make this calibration. The L2 calibration convention and L0 PDF
convention are recorded separately. A physical mapping to evacuated
halo mass or feedback physics is out of scope. Electron bias b_e=1 is
a stated baseline limitation. No L2 results exist yet.

## Observable boundary and reproducibility

Frozen, slotted observations expose only the specified observable fields.
Unlocalized `z_obs` must be None; violations raise even under Python -O.
Ground truth and theta are exported separately. The observation-only
export excludes simulator metadata because it contains theta. Simulator
metadata includes seed, git SHA, dirty flag, full config, and config hash.
Bit reproducibility is scoped to the same code, dependency versions,
platform and configuration. Floating computations use float64;
identifiers, booleans and missing-value None retain their native types.

## Primary sources

- Macquart et al.: https://arxiv.org/abs/2005.13161
- Authors' code: https://github.com/FRBs/FRB (vendored commit in provenance.json)
- Madau & Dickinson Eq. 15: https://arxiv.org/html/1403.0007
- Planck: https://arxiv.org/abs/1807.06209
- GLASS: https://glass.readthedocs.io/stable/reference/fields.html
- pygedm: https://pygedm.readthedocs.io/en/latest/pygedm.html
