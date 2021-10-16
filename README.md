# riskmodels: a library for univariate and bivariate extreme value analysis (and applications to energy procurement)

This library focuses on extreme value models for risk analysis in one and two dimensions. MLE-based and Bayesian generalised Pareto tail models are available for one-dimensional data, while for two-dimensional data, logistic and Gaussian (MLE-based) extremal dependence models are also available. Logistic models are appropriate for data whose extremal occurrences are strongly associated, while a Gaussian model offer an asymptotically independent, but still parametric, copula.

The `powersys` submodule offers utilities for applications in energy procurement, with functionality to model available conventional generation (ACG) as well as to calculate loss of load expectation (LOLE) and expected energy unserved (EEU) indices on a wide set of models; efficient parallel time series based simulation for univariate and bivariate power surpluses is also available. 

## Requirements

This package works with Python >= 3.7

## Documentation

The Github pages of this repo contains the package's documentation

## Quickstart

### Extreme value analysis

Because this library stemmed from research in energy procurement, this example is related to that but the `univariate` and `bivariate` modules are quite general and can be applied to any kind of data. The following example analyses the co-occurrence of low wind generation between Great Britain and France. 

```py

```