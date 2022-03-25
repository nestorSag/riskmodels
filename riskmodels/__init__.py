"""
This package implements one and two-dimensional extreme value models. Synthatic sugar is provided for univariate models, making it possible to compute conditional distributions of the form `X | X > x` using the `>` and `>=` operators; distributions with integer support can be convolved with other continuous and integer distributions, which is useful for applications in energy procurement; see the `riskmodels.univariate.Empirical` class and `riskmodels.adequacy.acg_models` module. 
Univariate models consist of MLE-based and Bayesian generalised Pareto tail models and the resulting semiparametric models with a lower empirical component and a higher parametric tail model. Bivariate models implement Gaussian and Gumbel-Hougaard copulas for exceedances; see the `riskmodels.bivariate` module.
Efficient implementations of sequential and non-sequential capacity surplus models for power system adequacy assessment are implemented in the `riskmodels.adequacy.capacity_models` module
"""
__version__ = "1.0.0-dev"
