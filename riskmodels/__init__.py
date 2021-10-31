"""
This package implements one and two-dimensional extreme value models. Synthatic sugar is provided for univariate models, making it possible to compute conditional distributions of the form `X | X > x` using the `>` and `>=` operators; distributions with integer support can be convolved with other continuous and integer distributions, which is useful for applications in energy procurement. 
Univariate models consist of MLE-based and Bayesian generalised Pareto tail models and the resulting semiparametric models with a lower empirical component and a higher parametric tail model.
Bivariate models implement Gaussian and Gumbel-Hougaard copulas for exceedances.
"""