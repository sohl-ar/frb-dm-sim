# Statistics bypass quick-check evidence

Generated from statistics-quickcheck.json and quick-unit-tests.xml; no training was performed.

Validation catalogs: 100. Statistics: 34; varying dimensions: 31. Availability flags constant in this validation subset are retained because absent/singleton subsets occur elsewhere.

Correlations are descriptive and unadjusted for multiple comparisons. They do not demonstrate posterior learning.

| Statistic | Raw variance | Standardized variance | f_d r | F r | Host median r | Host sigma_ln r |
|---|---:|---:|---:|---:|---:|---:|
| dm_mean | 44987.5 | 1.2045 | 0.386464 | 0.008575 | 0.405065 | 0.397544 |
| dm_std | 911901 | 0.946098 | -0.019613 | 0.244333 | 0.192323 | 0.362735 |
| dm_median | 20825 | 1.01416 | 0.628393 | -0.426510 | 0.329475 | 0.139815 |
| dm_q25 | 12955.2 | 1.01388 | 0.527639 | -0.409908 | 0.388768 | 0.082898 |
| dm_q75 | 32237.5 | 0.996493 | 0.692624 | -0.334638 | 0.360416 | 0.190257 |
| dm_min | 3007.72 | 0.546017 | 0.107127 | -0.224348 | 0.367552 | -0.134781 |
| dm_max | 1.42649e+08 | 0.587838 | -0.012794 | 0.245620 | 0.237225 | 0.342761 |
| fluence_mean | 161.553 | 0.689867 | 0.088304 | -0.034013 | -0.036499 | 0.031908 |
| fluence_std | 10017.5 | 0.967121 | 0.111403 | -0.045844 | 0.030011 | 0.039170 |
| fluence_median | 2.30837 | 0.473459 | -0.150965 | -0.018786 | -0.121198 | 0.048577 |
| localized_dm_mean | 266462 | 2.9089 | 0.191568 | 0.130283 | 0.256494 | 0.136444 |
| localized_dm_std | 2.16961e+06 | 2.66865 | 0.025787 | 0.273026 | 0.195922 | 0.142664 |
| localized_z_mean | 0.00843353 | 0.44397 | 0.071678 | -0.008183 | 0.050430 | -0.072884 |
| localized_count | 2101.22 | 0.648177 | 0.163208 | -0.039180 | -0.004618 | -0.035900 |
| unlocalized_count | 15190.4 | 1.07498 | -0.098049 | 0.225822 | 0.151855 | 0.097437 |
| has_localized | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| has_unlocalized | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ols_available | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| localized_z_std | 0.0024512 | 0.514185 | 0.100123 | 0.013515 | 0.078785 | -0.044775 |
| dm_z_slope | 1.38805e+06 | 0.0299243 | 0.194288 | 0.088244 | -0.026284 | -0.215058 |
| dm_z_intercept | 1.12796e+06 | 0.0202924 | -0.118088 | -0.032436 | 0.140314 | 0.311359 |
| residual_std | 2.20329e+06 | 2.74553 | 0.004759 | 0.290354 | 0.211604 | 0.170466 |
| residual_q25 | 240677 | 3.05279 | -0.023659 | -0.306125 | -0.228545 | -0.166739 |
| residual_median | 161702 | 5.35351 | 0.042186 | -0.280526 | -0.220904 | -0.238563 |
| residual_q75 | 134458 | 7.31049 | 0.093003 | -0.165383 | -0.140115 | -0.193604 |
| zbin0_count | 162.26 | 0.648252 | 0.171630 | -0.006800 | -0.080269 | -0.027763 |
| zbin0_dm_mean | 108088 | 0.526511 | 0.176133 | -0.061902 | 0.303647 | 0.330945 |
| zbin0_dm_std | 419255 | 0.55543 | 0.115182 | 0.032441 | 0.187905 | 0.354462 |
| zbin1_count | 518.414 | 0.700528 | 0.133673 | -0.049834 | -0.009037 | -0.046459 |
| zbin1_dm_mean | 340513 | 2.89784 | 0.061016 | 0.139091 | 0.257103 | 0.238461 |
| zbin1_dm_std | 2.78568e+06 | 4.64744 | -0.078905 | 0.245029 | 0.202942 | 0.241677 |
| zbin2_count | 187.153 | 0.607502 | 0.164578 | -0.042007 | 0.074309 | -0.017117 |
| zbin2_dm_mean | 527556 | 1.88662 | 0.249209 | 0.121978 | 0.210826 | 0.033408 |
| zbin2_dm_std | 2.37123e+06 | 2.00465 | 0.033382 | 0.318605 | 0.163448 | 0.059558 |

Forward/backward test loss (untrained model): 8.24051762.
Bypass first-layer gradient norm: 0.112810396.
Summary dimension: 100; velocity input dimension: 105.
Unit checks: 12 tests, 0 failures, 0 errors.

The full correlation matrix and all absolute-correlation flags are retained in the source JSON.
