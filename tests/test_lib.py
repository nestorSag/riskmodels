"""
This script is meant to be run on GitHub actions workflows
"""
import numpy as np
import pandas as pd
import scipy as sp

import riskmodels.univariate as univar
import riskmodels.bivariate as bivar

from riskmodels.powersys.ts import convgen

tol = 1e-6

def test_empirical():
  """Basic correctness tests for empirical distrubition objects.
  """
  n = 16
  data = np.arange(1,n+1)
  dist1 = univar.Empirical.from_data(data)

  for k in data:
    assert np.isclose(dist1.pdf(k), 1/n)
    assert np.isclose(dist1.cdf(k), k/n)
    assert np.isclose(dist1.ppf(k/n), k)

  assert np.isclose(dist1.mean(), np.mean(data))
  assert np.isclose(dist1.std(), np.std(data))

  dist2 = dist1 + 1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)

  dist2 = 1 + dist1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)

  dist2 = dist1 - 1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support -1)
  assert np.allclose(dist2.data, dist1.data -1)

  dist2 = -dist1
  assert np.allclose(dist2.pdf_values, np.flip(dist1.pdf_values))
  assert np.allclose(dist2.support,np.flip(-dist1.support))
  assert np.allclose(dist2.data, -dist1.data)

  dist2 = 2*dist1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support, 2*dist1.support)
  assert np.allclose(dist2.data, 2*dist1.data)

  dist2 = dist1 >= 8
  assert dist2.min == 8
  assert dist2.max == max(data)


def test_package():
  """Integration tests for the most common paths in the package's use. As the package offloads most of the basic computational building blocks to numpy, scipy and statsmodels, correctness is not unit tested at the moment.
  """
  def validate_vector(x):
    return True if (np.sum(np.isnan(x)) == 0 and np.sum(np.isinf(x)) == 0) else False

  np.random.seed(1)
  sample_size = 5000
  q_th = 0.95
  season_length = 3360
  scale_factor = 1000000
  cov = np.array([[1, 0.7],[0.7, 1]])
  sample = sp.stats.multivariate_normal.rvs(size=sample_size, cov = scale_factor*cov)

  # test univariate empirical constructors
  x, y = sample.T
  u, v = univar.Empirical.from_data(x), univar.Empirical.from_data(y)

  # test positive offset
  z = u + 100
  assert u.max + 100 == z.max and np.all(u.pdf_values == z.pdf_values) and np.all(u.support + 100 == z.support)

  # test negative offset
  z = u - 100
  assert u.max - 100 == z.max and np.all(u.pdf_values == z.pdf_values) and np.all(u.support - 100 == z.support)

  # test positive rescaling offset
  z = 2*u
  assert 2*u.max == z.max and np.all(u.pdf_values == z.pdf_values) and np.all(2*u.support == z.support)

  # test binning
  z = u.to_integer()
  assert isinstance(z, univar.Binned)

  # test conditioning
  z = u >= u.ppf(0.5)
  assert z.min >= u.ppf(0.5)
  assert z.max == u.max

  # test basic probabilistic functionality
  assert u.mean() and u.std() and u.cdf(0.5) and u.ppf(0.5) and u.cvar(q_th)

  # test tail fitting 
  mle = u.to_integer().fit_tail_model(threshold=u.ppf(0.9))
  bayesian = u.to_integer().fit_tail_model(threshold=u.ppf(0.9), bayesian=True)

  # test basic mle probabilistic functionality
  assert mle.mean() and mle.std() and mle.cdf(mle.mean()) and mle.ppf(1 - (1-q_th)/2) and mle.cvar(q_th)

  # test basic bayesian probabilistic functionality
  assert bayesian.mean() and bayesian.std() and bayesian.cdf(bayesian.mean()) and bayesian.ppf(1- (1-q_th)/2) and bayesian.cvar(1- (1-q_th)/2)

  # test specific bayesian functionality 
  a, b, c, d, e = bayesian.cdf(bayesian.mean(), return_all=True), bayesian.ppf(1-(1-q_th)/2, return_all=True), bayesian.cvar(1-(1-q_th)/2, return_all=True), bayesian.mean(return_all=True), bayesian.std(return_all=True)

  return validate_vector(a) and validate_vector(b) and validate_vector(c) and validate_vector(d) and validate_vector(e)
  # test simulation
  assert validate_vector(u.simulate(size=100))
  assert validate_vector(mle.simulate(size=100))
  assert validate_vecotr(bayesian.simulate(size=100))

  # only test lack of errors in what follows

  # test mixtures
  z = u.to_integer() + v.to_integer()
  z = u.to_integer() + mle
  z = u.to_integer() + bayesian

  # test conditioning in mixtures
  u_i, v_i = u.to_integer(), v.to_integer()
  z = u_i + v_i >= (u_i+v_i).mean()
  z = (u_i + mle) >= (u_i + mle).mean()
  z = (u_i + bayesian) >= (u_i+bayesian).mean()

  # test bivariate EV modelling code
  empirical_models = [univar.Empirical.from_data(v) for v in sample.T]
  ev_models = [model.fit_tail_model(threshold=model.ppf(q_th)) for model in empirical_models]
  bivariate_model = bivar.Empirical.from_data(sample)
  r = bivariate_model.test_asymptotic_dependence(q_th)
  bivariate_ev = bivariate_model.fit_tail_model(
    model = "gaussian",
    quantile_threshold = q_th,
    margin1 = ev_models[0],
    margin2 = ev_models[1])

  ## test C code for sequential generation
  gen_df = pd.DataFrame([{"availability": 0.95, "capacity": 250, "mttr": 50} for k in range(250)])
  gen = convgen.MarkovChainGenerationModel.from_generator_df(gen_df)
  s = gen.simulate_seasons(size=1000, season_length=season_length, seasons_per_trace=4)