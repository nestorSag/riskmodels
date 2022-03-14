"""
This package implements one and two-dimensional extreme value models. Synthatic sugar is provided for univariate models, making it possible to compute conditional distributions of the form `X | X > x` using the `>` and `>=` operators; distributions with integer support can be convolved with other continuous and integer distributions, which is useful for applications in energy procurement; see the `riskmodels.univariate.Empirical` class and `riskmodels.powersys.iid.convgen` module. 
Univariate models consist of MLE-based and Bayesian generalised Pareto tail models and the resulting semiparametric models with a lower empirical component and a higher parametric tail model. Bivariate models implement Gaussian and Gumbel-Hougaard copulas for exceedances; see the `riskmodels.bivariate` module.
Univariate and bivariate time-collapsed models for power surplus distributions are implemented in the `riskmodels.powersys.iid.surplus` module.
Sequential models for univariate and bivariate power surplus distributions are implemented along with functionality for large-scale simulation using a map-reduce and multi-core patterns; see the `riskmodels.powersys.ts` module.
"""
