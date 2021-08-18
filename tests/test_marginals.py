import pytest as pt
import numpy as np
import scipy as sp
from riskmodels.base.marginals import *

tol = 1e-6
def test_empirical():
  n = 16
  data = np.arange(1,n+1)
  dist1 = Empirical.from_data(data)

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

  dist2 = 1 + dist
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)

  dist2 = dist - 1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support -1)
  assert np.allclose(dist2.data, dist1.data -1)

  dist2 = -dist
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

  dist2 = dist1 <= 8
  assert dist2.max == 8
  assert dist2.min == min(data)


def test_binned():
  n, p = 8, 0.5
  binom1 = sp.stats.binom(n,p)

  support = np.arange(n+1)
  pdf_vals = binom1.pmf(support)

  x = Binned(support = support, pdf_values = pdf_vals)
  y = Binned(support = support, pdf_values = pdf_vals)

  z = x + y

  binom_z =sp.stats.binom(2*n, p)

  for k in np.arange(2*n+1):
    assert np.isclose(z.pdf(k), binom_z.pmf(k), tol)
    assert np.isclose(z.cdf(k), binom_z.cdf(k), tol)
    assert np.isclose(z.ppf(k/(2*n)), binom_z.ppf(k/(2*n)), tol)

  assert np.isclose(z.mean(), binom_z.mean(), tol)
  assert np.isclose(z.std(), binom_z.std(), tol)


  dist2 = z + 1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)
  assert isinstance(dist2, Binned)

  dist2 = 1 + z
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)
  assert isinstance(dist2, Binned)

  dist2 = 1.0 + z
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support + 1)
  assert np.allclose(dist2.data, dist1.data + 1)
  assert isinstance(dist2, Empirical)

  dist2 = z - 1
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support,dist1.support -1)
  assert np.allclose(dist2.data, dist1.data -1)
  assert isinstance(dist2, Binned)

  dist2 = -z
  assert np.allclose(dist2.pdf_values, np.flip(dist1.pdf_values))
  assert np.allclose(dist2.support,np.flip(-dist1.support))
  assert np.allclose(dist2.data, -dist1.data)
  assert isinstance(dist2, Binned)

  dist2 = 2*z
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support, 2*dist1.support)
  assert np.allclose(dist2.data, 2*dist1.data)
  assert isinstance(dist2, Binned)

  dist2 = 2.0*z
  assert np.allclose(dist2.pdf_values, dist1.pdf_values)
  assert np.allclose(dist2.support, 2*dist1.support)
  assert np.allclose(dist2.data, 2*dist1.data)
  assert isinstance(dist2, Empirical)

  dist2 = z >= 8
  assert dist2.min == 8
  assert dist2.max == max(data)

  dist2 = z <= 8
  assert dist2.max == 8
  assert dist2.min == min(data)


def test_gp_tail():

  np.random.seed(1)
  data = sp.stats.genpareto.rvs(size = 5000, loc=0, scale=1, c = 0)

  threshold = 0
  dist = GPTail.fit(data, threshold)

  
# def test_tail_fitting():
#   np.random.seed(1)
#   p = 0.05
#   data = np.random.normal(size=50000)
#   threshold = np.quantile(data, p)
#   model = Empirical(data).fit_tail_model(threshold)

#   assert isinstance(model, EmpiricalWithGPTail)

#   bayesian_model = Empirical(data).fit_tail_model(threshold, bayesian=True)


  


