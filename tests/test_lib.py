"""
This script is meant to be run on GitHub actions workflows
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import scipy as sp

import riskmodels.univariate as univar
import riskmodels.bivariate as bivar

from riskmodels.adequacy import acg_models, capacity_models


from scipy.stats import (
    gumbel_r as gumbel,
    norm as gaussian,
    multivariate_normal as mv_gaussian,
    rayleigh
)

season_length = 3360
tol = 1e-6
np.random.seed(1)

def test_empirical():
  """Basic correctness tests for empirical distribution objects.
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


def test_univariate():
  """Integration tests for the most common paths in the univariate module's use.
  """
  def is_valid_vector(x):
    return True if (np.sum(np.isnan(x)) == 0 and np.sum(np.isinf(x)) == 0) else False

  def are_valid_scalars(*args):
    return not np.any(np.isnan(np.array(args)))

  sample_size = 5000
  q_th = 0.95
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
  assert are_valid_scalars(u.mean(), u.std(), u.cdf(0.5), u.ppf(0.5), u.cvar(q_th))

  # test tail fitting 
  mle = u.to_integer().fit_tail_model(threshold=u.ppf(0.9))
  bayesian = u.to_integer().fit_tail_model(threshold=u.ppf(0.9), bayesian=True)

  # test basic mle probabilistic functionality
  assert are_valid_scalars(mle.mean(), mle.std(), mle.cdf(mle.mean()), mle.ppf(1 - (1-q_th)/2), mle.cvar(q_th))

  # test basic bayesian probabilistic functionality
  assert are_valid_scalars(bayesian.mean(), bayesian.std(), bayesian.cdf(bayesian.mean()), bayesian.ppf(1- (1-q_th)/2), bayesian.cvar(1- (1-q_th)/2))

  # test specific bayesian functionality 
  a, b, c, d, e = bayesian.cdf(bayesian.mean(), return_all=True), bayesian.ppf(1-(1-q_th)/2, return_all=True), bayesian.cvar(1-(1-q_th)/2, return_all=True), bayesian.mean(return_all=True), bayesian.std(return_all=True)

  return is_valid_vector(a) and is_valid_vector(b) and is_valid_vector(c) and is_valid_vector(d) and is_valid_vector(e)
  
  # test simulation
  assert is_valid_vector(u.simulate(size=100))
  assert is_valid_vector(mle.simulate(size=100))
  assert is_valid_vector(bayesian.simulate(size=100))

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

def test_bivariate():
  """Integration tests for the most common paths in the bivariate module's use.
  """
  gaussian_sampler = bivar.Gaussian(
    quantile_threshold = 0,
    params = np.array([0.7]),
    margin1 = gaussian,
    margin2 = gaussian)

  logistic_sampler = bivar.Logistic(
    quantile_threshold = 0,
    params = np.array([0.7]),
    margin1 = gumbel,
    margin2 = gumbel)

  asym_log_sampler = bivar.AsymmetricLogistic(
    quantile_threshold = 0,
    params = np.array([0.6, 0.6, 0.9]),
    margin1 = gumbel,
    margin2 = gumbel)

  samplers = [
    (gaussian_sampler, "gaussian"),
    (logistic_sampler, "logistic"), 
    (asym_log_sampler, "asymmetric logistic")]

  np.random.seed(1)
  # validate true parameters being within 95% contour bands of estimated parameters
  for sampler, name in samplers:
    s = sampler.simulate(size = 5000)
    x = bivar.Empirical.from_data(s)
    x = x.fit_tail_model(quantile_threshold=0.95, model = name)
    cov = x.tail.mle_cov
    mean = sampler.params
    mle = x.tail.params
    dist_bound = rayleigh.ppf(0.95)
    dist = (mean-mle).reshape((1,-1)).T
    isotropic_dist = np.sqrt(dist.T.dot(np.linalg.inv(cov).dot(dist)))
    assert isotropic_dist < dist_bound


def test_sequential_models():
  season_length = 3360
  ## test for exceptions in C code for sequential generation
  gen_df = pd.DataFrame([{"availability": 0.95, "capacity": 250, "mttr": 50} for k in range(250)])
  gen = acg_models.Sequential.from_generator_df(gen_df)
  s = gen.simulate_seasons(size=1000, season_length=season_length, seasons_per_trace=4)

  ## integration test, not unit tests
  out_folder = Path("tests") / "sequential_test_data"
  out_folder.mkdir(exist_ok=True, parents=True)
  #
  n_seasons = 5
  season_length = 1000
  demand = np.random.normal(loc = gen.mean(), scale = gen.std(), size=(n_seasons * season_length,))
  wind = np.random.normal(loc = 0.0, scale = gen.std(), size=(n_seasons * season_length,))
  #
  eu=capacity_models.UnivariateSequential.init(
    output_dir = str(out_folder),
    n_traces = 1000,
    n_files = 1,
    gen = gen,
    demand = demand,
    renewables = wind,
    season_length = season_length).simulate_eu()
  assert np.all(np.logical_not(np.isnan(eu)))
  #
  # test two-area sequential model
  # create new generator
  gen2 = acg_models.Sequential.from_generator_df(gen_df)
  itc_cap = 1000
  policy="veto"
  demands = np.random.normal(loc = gen.mean(), scale = gen.std(), size=(n_seasons * season_length,2))
  winds = np.random.normal(loc = 0.0, scale = gen.std(), size=(n_seasons * season_length,2))
  #
  eu=capacity_models.BivariateSequential.init(
    output_dir = str(out_folder),
    n_traces = 1000,
    n_files = 1,
    gens = [gen, gen2],
    demand = demands,
    renewables = winds,
    season_length = season_length).simulate_eu(itc_cap=itc_cap, policy=policy)
  assert np.all(np.logical_not(np.isnan(eu)))
  shutil.rmtree(out_folder)
