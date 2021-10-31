"""This module contains bivariate risk models to analyse exceedance dependence between components. Available exceedance models are inspired in bivariate generalised Pareto models, whose support is an inverted L-shaped subset of Euclidean space, where at least one component takes an extreme value above a specified threshold. Available parametric models in this module include the logistic model, equivalent to a Gumbel-Hougaard copula between exceedances, and a Gaussian model, equivalent to a Gaussian copula. The former exhibits asymptotic dependence and the latter asymtptotic independence, which characterises the dependence of extremes across components.
Finally, `Empirical` instances have methods to assess asymptotic dependence vs independence through hypothesis tests and visual inspection of the Pickands dependence function.
"""

from __future__ import annotations

import logging
import time
import typing as t
import traceback
import warnings
from argparse import Namespace
from abc import ABC, abstractmethod
from multiprocessing import Pool
from collections.abc import Iterable
import copy

import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib

import numpy as np
import emcee

from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.signal import fftconvolve
from scipy.special import lambertw
from scipy.stats import genpareto as gpdist, gumbel_r as gumbel, norm as gaussian, multivariate_normal as mv_gaussian, gaussian_kde

from pydantic import BaseModel, ValidationError, validator, PositiveFloat
from functools import reduce

import riskmodels.univariate as univar

from riskmodels.utils.tmvn import TruncatedMVN as tmvn

import emcee

class BaseDistribution(BaseModel, ABC):

  """Base interface for bivariate distributions
  """
  
  _allowed_scalar_types = (int, float, np.int64, np.int32, np.float32, np.float64)
  _figure_color_palette = ["tab:cyan", "deeppink"]
  _error_tol = 1e-6

  data: t.Optional[np.ndarray]

  class Config:
    arbitrary_types_allowed = True

  def __repr__(self):
    return "Base distribution object"

  def __str__(self):
    return self.__repr__()

  @abstractmethod
  def pdf(self, x: np.ndarray) -> float:
    """Evaluate probability density function
    
    """
    pass

  @abstractmethod
  def cdf(self, x: np.ndarray):
    """Evaluate cumulative distribution function
    
    """
    pass

  @abstractmethod
  def simulate(self, size: int):
    """Simulate from bivariate distribution
    
    """
    pass

  def plot(self, size: int = 1000) -> matplotlib.figure.Figure:
    """Sample distribution and produce scatterplots and histograms
    
    Args:
        size (int, optional): Sample size
    
    Returns:
        matplotlib.figure.Figure: figure
    """
    sample = self.simulate(size)

    x = sample[:,0]
    y = sample[:,1]

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(x, y, color = self._figure_color_palette[0], alpha=0.35)

    # now determine nice limits by hand:
    # binwidth = 0.25
    # lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    # ax_scatter.set_xlim((-lim, lim))
    # ax_scatter.set_ylim((-lim, lim))

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=25, color = self._figure_color_palette[0], edgecolor='white')
    #plt.title(f"Scatter plot from {np.round(size/1000,1)}K simulated samples")
    ax_histy.hist(y, bins=25, orientation='horizontal', color = self._figure_color_palette[0], edgecolor='white')

    #ax_histx.set_xlim(ax_scatter.get_xlim())
    #ax_histy.set_ylim(ax_scatter.get_ylim())
    plt.tight_layout()
    return fig



class Mixture(BaseDistribution):

  """Base interface for a bivariate mixture distribution
  """
  
  distributions: t.List[BaseDistribution]
  weights: np.ndarray

  def __repr__(self):
    return f"Mixture with {len(self.weights)} components"


  def simulate(self, size: int) -> np.ndarray:
    
    n_samples = np.random.multinomial(n=size, pvals = self.weights, size=1)[0]
    indices = (n_samples > 0).nonzero()[0]
    samples = [dist.simulate(size=k) for dist, k in zip([self.distributions[k] for k in indices], n_samples[indices])]
    return np.concatenate(samples, axis=0)

  def cdf(self, x:np.ndarray, **kwargs) -> float:
    vals = [w*dist.cdf(x,**kwargs) for w, dist in zip(self.weights, self.distributions)]
    return reduce(lambda x,y: x + y, vals)

  def pdf(self, x:np.ndarray, **kwargs) -> float:
    
    vals = [w*dist.pdf(x,**kwargs) for w, dist in zip(self.weights, self.distributions)]
    return reduce(lambda x,y: x + y, vals)


class ExceedanceModel(Mixture):

  """Interface for exceedance models
  """
  def __repr__(self):
    return f"Sempirametric model with {self.tail.__class__.__name__} exceedance dependence"

  def plot_diagnostics(self):

    return self.distributions[1].plot_diagnostics()

  @property
  def tail(self):
    return self.distributions[1]

  @property
  def empirical(self):
    return self.distributions[0]
  



class Independent(BaseDistribution):

  """Bivariate distribution with independent components
  """
  
  x: univar.BaseDistribution
  y: univar.BaseDistribution

  def __repr__(self):
    return f"Independent bivariate distribution with marginals:\nx:{x.__repr__()}\ny:{y.__repr__()}"

  def pdf(self, x: np.ndarray):
    x1, x2 = x
    return self.x.pdf(x1)*self.y.pdf(x2)

  def cdf(self, x: np.ndarray):
    x1, x2 = x
    return self.x.cdf(x1)*self.y.cdf(x2)

  def simulate(self, size: int):
    return np.concatenate([self.x.simulate(size).reshape((-1,1)), self.y.simulate(size).reshape((-1,1))], axis=1)




class ExceedanceDistribution(BaseDistribution):

  """Main interface for exceedance distributions, which are defined on a region of the form \\( $U \\nleq u$ \\), or equivalently \\( \\max\\{U_1,U_2\\} > u \\). 
  """
  quantile_threshold: float

  @classmethod
  @abstractmethod
  def fit(cls, data: np.ndarray, threshold: float):
    """Fit the model through maximum likelihood estimation
    
    Args:
        data (np.ndarray): Observed data
        threshold (float): Exceedance threshold
    """
    pass

  @abstractmethod
  def plot_diagnostics(self):
    """Plot diagnostics for the fitted model.
    """
    pass

  @classmethod
  def unbundle(cls, data: t.Union[np.ndarray,t.Iterable]) -> t.Tuple[t.Union[np.ndarray, float], t.Union[np.ndarray, float]]:
    """Unbundles matrix or iterables into separate components
    
    Args:
        data (t.Union[np.ndarray, t.Iterable]): dara
    
    """
    if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 2:
      x = data[:,0]
      y = data[:,1]
    elif isinstance(data, Iterable):
      # if iterable, unroll
      x, y = data
    else:
      raise TypeError("data must be an n x 2 numpy array or an iterable of length 2.")
    return x, y

  @classmethod
  def bundle(cls, x: t.Union[np.ndarray, float, int], y: t.Union[np.ndarray,float, int]) -> t.Tuple[t.Union[np.ndarray, float], t.Union[np.ndarray, float]]:
    """bundle a pair of arrays or primitives into n x 2 matrix
    
    Args:
        data (t.Union[np.ndarray, t.Iterable])
    
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and len(x) == len(y):
      z = np.concatenate([x.reshape((-1,1)), y.reshape((-1,1))], axis=1)
    elif issubclass(type(x), (float, int)) and issubclass(type(y), (float, int)):
      z = np.array([x,y]).reshape((1,2))
    else:
      raise TypeError("x, y must be 1-dimensional arrays or inherit from float or int.")
    return z


class Logistic(ExceedanceDistribution):

  """This model is equivalent to a Gumbel-Hougaard copula model restricted to the region of the form \\( \\mathbf{U} \\nleq u \\), or equivalently \\( \\max\\{\\mathbf{U}_1,\\mathbf{U}_2\\} > u \\), which represents threshold exceedances above \\( u \\) in at least one component. This model can also be thought as a pre-limit bivariate Generalised Pareto with logistic dependence in the context of extreme value theory. Consequently, under this model there is asymptotic dependence between components, this is, extreme values across components are strongly associated. Use it if there is strong evidence of asymptotic dependence in the data. 

  In Gumbel scale (this is, when both marginal distributions follow a standard Gumbel distribution), its cumulative probability function is given by

  $$ F(\\textbf{x}) - \\exp \\left( - \\left( \\exp(-\\textbf{x}_1/\\alpha) + \\exp(-\\textbf{x}_2/\\alpha) \\right)^\\alpha \\right), \\,\\,\\, \\textbf{x} \\nleq \\textbf{a}, \\,\\,\\, \\alpha \\in (0,1)$$

  where \\( \\textbf{a} \\) is the vector formed by the marginal quantiles of a probability threshold \\( p \\) such that exceedances with probability less than \\(1-p\\) are considered extreme (say \\(p = 0.95\\) )
  """
  
  alpha: float
  margin1: univar.BaseDistribution
  margin2: univar.BaseDistribution

  _model_marginal_dist = gumbel
  _marginal_model_name = "Gumbel"

  def __repr__(self):
    return f"{self.__class__.__name__} exceedance dependence model with alpha = {self.alpha} and quantile threshold {self.quantile_threshold}"

  @property
  def model_scale_threshold(self):
    return self._model_marginal_dist.ppf(self.quantile_threshold)

  @property
  def data_scale_threshold(self):
    return self.model_to_data_dist(self.bundle(self.model_scale_threshold, self.model_scale_threshold))

  @validator("alpha")
  def validate_alpha(cls, alpha):
    if alpha < 0 or alpha > 1:
      raise TypeError("alpha must be in the open interval (0,1) ")
    else:
      return alpha

  @validator("data")
  def validate_data(cls, data):
    if data is None or len(data.shape) != 2 or data.shape[1] != 2:
      raise ValueError("Data needs to be an n x 2 matrix array")
    else:
      return data

  def data_to_model_dist(self, data: t.Union[np.ndarray,t.Iterable]) -> np.ndarray:
    """Transforms original data scale to standard Gumbel scale
    
    Args:
        data (t.Union[np.ndarray, t.Iterable]): observations in original scale
    
    """
    x, y = self.unbundle(data)

    ## to copula scale
    x = self.margin1.cdf(x)
    y = self.margin2.cdf(y)

    # pass to Gumbel scale 
    x = self._model_marginal_dist.ppf(x)
    y = self._model_marginal_dist.ppf(y)

    return self.bundle(x,y)

  def model_to_data_dist(self, data: t.Union[np.ndarray,t.Iterable]) -> np.ndarray:
    """Transforms data in standard Gumbel scale to original data scale
    
    Args:
        x (np.ndarray): data from first component
        y (np.ndarray): data from second component
    
    Returns:
        np.ndarray
    """

    # copula scale
    x, y = self.unbundle(data)

    u = self._model_marginal_dist.cdf(x)
    w = self._model_marginal_dist.cdf(y)

    # data scale
    u = self.margin1.ppf(u)
    w = self.margin2.ppf(w)

    return self.bundle(u,w)

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates logpdf function for Gumbel exceedances
    
    
    Args:
        alpha (float): Dependence parameter
        threshold (float): Exceedance threshold in Gumbel scale
        data (t.Union[np.ndarray, t.Iterable]): Observed data in Gumbel scale

    """
    x, y = cls.unbundle(data)

    nlogp = (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha
    lognlogp = alpha*np.log(np.exp(-x/alpha) + np.exp(-y/alpha))
    rescaler = 1 - cls.logistic_gumbel_cdf(alpha, cls.bundle(threshold,threshold))

    #a = np.exp((x + y - nlogp*alpha)/alpha)
    log_a = (x + y)/alpha - nlogp

    #b = nlogp
    log_b = lognlogp

    #c = 1 + alpha*(nlogp - 1)
    log_c = np.log(1 + alpha*(nlogp - 1))

    #d = 1.0/(alpha*(np.exp(x/alpha) + np.exp(y/alpha))**2)
    log_d = -(np.log(alpha) + 2*np.log(np.exp(x/alpha) + np.exp(y/alpha)))

    log_density = log_a + log_b + log_c + log_d - np.log(rescaler)

    # density is 0 when both coordinates are below the threshold
    nil_density_idx = np.logical_and(x <= threshold, y <= threshold)
    log_density[nil_density_idx] = -np.Inf

    return log_density

  @classmethod
  def loglik(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates log-likelihood for Gumbel exceedances
    
    """
    return np.sum(cls.logpdf(alpha, threshold, data))

  @classmethod
  def logistic_gumbel_cdf(cls, alpha: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates unconstrained standard Gumbel CDF

    """
    x, y = cls.unbundle(data)
    return np.exp(-(np.exp(-x/alpha) + np.exp(-y/alpha))**(alpha))

  @classmethod
  def fit(
    cls, 
    data: t.Union[np.ndarray,t.Iterable], 
    quantile_threshold: float,
    margin1: univar.BaseDistribution = None,
    margin2: univar.BaseDistribution = None,
    return_opt_results = False,
    x0: float = None) -> Logistic:
    """Fits the model from provided data, threshold and marginal distributons
    
    Args:
        data (t.Union[np.ndarray, t.Iterable]): input data
        quantile_threshold (float): Description: quantile threshold over which observations are classified as extreme
        margin1 (univar.BaseDistribution, optional): Marginal distribution for first component
        margin2 (univar.BaseDistribution, optional): Marginal distribution for second component
        return_opt_results (bool, optional): If True, the object from the optimization result is returned
        x0 (float, optional): Initial point for the optimisation algorithm. Defaults to 0.5
    
    Returns:
        Logistic: Fitted model
    
    
    """
    if margin1 is None:
      margin1 = univar.empirical.from_data(data[:,0])
      warnings.warn("margin1 is None; using an empirical distribution", stacklevel=2)

    if margin2 is None:
      margin1 = univar.empirical.from_data(data[:,1])
      warnings.warn("margin1 is None; using an empirical distribution", stacklevel=2)

    if not isinstance(quantile_threshold, float) or quantile_threshold <= 0 or quantile_threshold >= 1:
      raise ValueError("quantile_threshold must be in the open interval (0,1)")

    mapped_data = cls(
      alpha=0.5, 
      margin1=margin1, 
      margin2=margin2, 
      quantile_threshold=quantile_threshold).data_to_model_dist(data)

    x,y = cls.unbundle(mapped_data)

    # get threshold exceedances
    model_scale_threshold = cls._model_marginal_dist.ppf(quantile_threshold)

    exs_idx = np.logical_or(x > model_scale_threshold, y > model_scale_threshold)
    x = x[exs_idx]
    y = y[exs_idx]

    mapped_exceedances = cls.bundle(x,y)


    def logistic(x):
      return 1.0/(1 + np.exp(-x))

    x0 = 0.5 if x0 is None else x0

    def loss(phi, data):
      alpha = logistic(phi)
      return -np.mean(cls.loglik(alpha, model_scale_threshold, data))

    res = minimize(
      fun=loss, 
      x0 = x0,
      method = "BFGS",
      args = (mapped_exceedances,))

    if return_opt_results:
      warn.warnings("Returning raw results for rescaled exceedance data (sdev ~ 1).")
      return res
    else:
      phi = res.x[0]
      alpha = logistic(phi)

    return cls(
      quantile_threshold = quantile_threshold,
      alpha = alpha,
      data = data[exs_idx,:],
      margin1 = margin1,
      margin2 = margin2)

  @classmethod
  def hessian(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates loglikelihood's second deriviative (i.e., negative estimator's precision)
    
    Args:
        alpha (float): dependence parameter
        threshold (float): Threshold in standard scale (i.e. Gumbel or Gaussian)
        data (t.Union[np.ndarray, t.Iterable]): Data in standard scale (i.e. Gumbel or Gaussian)
    
    Returns:
        TYPE: Description
    
    """
    delta = 1e-3
    n = len(data)
    return (cls.loglik(alpha - delta, threshold, data) -2*cls.loglik(alpha, threshold, data) + cls.loglik(alpha + delta, threshold, data))/(n*delta**2)

  def plot_diagnostics(self) -> matplotlib.figure.Figure:
    """Returns diagnostic plots for the fitted model
    
    Returns:
        matplotlib.figure.Figure: figure
    
    """

    x, y = self.unbundle(self.data)
    z1,z2= self.unbundle(self.data_to_model_dist(self.data))
    n = len(z1)

    fig, axs = plt.subplots(2, 2)

    ####### loglikelihood plot

    model_scale_threshold = self.model_scale_threshold
    sdev = np.sqrt(-1.0/self.hessian(self.alpha, model_scale_threshold, self.bundle(z1, z2)))
    grid = np.linspace(self.alpha - sdev, self.alpha + sdev, 100)
    grid = grid[np.logical_and(grid > 0, grid < 1)]
    ll = np.array([self.loglik(alpha, model_scale_threshold, self.bundle(z1,z2)) for alpha in grid])

    # filter to almost optimal values
    max_ll = max(ll)
    almost_optimal = np.abs(ll - max_ll) < np.abs(2*max_ll)
    ll = ll[almost_optimal]
    grid = grid[almost_optimal]

    axs[0,0].plot(grid, ll, color=self._figure_color_palette[0])
    axs[0,0].vlines(x=self.alpha, ymin=min(ll), ymax = max(ll), linestyle="dashed", colors = self._figure_color_palette[1])
    axs[0,0].title.set_text('Log-likelihood')
    axs[0,0].set_xlabel('Alpha')
    axs[0,0].set_ylabel('log-likelihood')

    #print("loglikelihood plot finished")

    ####### density plot
    z1_range = max(z1) - min(z1)
    z2_range = max(z2) - min(z2)

    x_range = np.linspace(min(z1) - 0.05*z1_range, max(z1) + 0.05*z1_range, 50)
    y_range = np.linspace(min(z2) - 0.05*z2_range, max(z2) + 0.05*z2_range, 50)

    X, Y = np.meshgrid(x_range, y_range)
    bundled_grid = self.bundle(X.reshape((-1,1)), Y.reshape((-1,1)))
    Z = self.logpdf(data=bundled_grid, threshold=model_scale_threshold, alpha=self.alpha).reshape(X.shape)
    axs[0,1].contourf(X,Y,Z)
    axs[0,1].scatter(z1,z2, color=self._figure_color_palette[1], s=0.9)
    axs[0,1].title.set_text(f'Model density ({self._marginal_model_name} scale)')
    axs[0,1].set_xlabel('x')
    axs[0,1].set_ylabel('y')

    ##### log odds plot
    cdf_values = self.cdf(self.data)
    model_logodds = np.log(cdf_values/(1-cdf_values))
    ecdf_values = Empirical.from_data(self.data).cdf(self.data)
    empirical_logodds = np.log(ecdf_values/(1-ecdf_values))

    axs[1,0].scatter(model_logodds, empirical_logodds, color=self._figure_color_palette[0])

    axs[1,0].title.set_text('Model vs data log-odds')
    axs[1,0].set_xlabel('Empirical log-odds')
    axs[1,0].set_ylabel('Model log-odds')
    axs[1,0].set_xlim(-5,5)
    axs[1,0].set_ylim(-5,5)
    min_e, max_e = max(-5,min(empirical_logodds)), min(5,max(empirical_logodds))
    axs[1,0].plot([min_e, max_e], [min_e, max_e], linestyle="--", color="black")

    plt.tight_layout()
    return fig

  def simulate(self, size: int):
    alpha = self.alpha
    exs_prob = 1 - self.quantile_threshold
    ### simulate in Gumbel scale maximum component: z = max(x1, x2) ~ Gumbel(loc=alpha*np.log(2)) using inverse function method
    q0 = gumbel.cdf(self.model_scale_threshold, loc=alpha*np.log(2)) # quantile of model's threshold in the maximum's distribution
    u = np.random.uniform(size=size, low=q0)
    maxima = gumbel.ppf(q=u, loc=alpha*np.log(2))

    ###simulate difference between maxima and minima r = max(x,y) - min(x,y) using inverse function method
    u = np.random.uniform(size=size)
    r = (alpha*np.log((-((alpha - 1)*np.exp(maxima)*lambertw(-(np.exp(-maxima - (2**alpha*np.exp(-maxima)*alpha)/(alpha - 1))*(-2**(alpha - 1)*(u - 1))**(alpha/(alpha - 1))*alpha)/(alpha - 1)))/alpha)**(1/alpha) - 1)).real

    minima = maxima - r

    #allocate maxima randomly between components
    max_indices= np.random.binomial(1,0.5,size)

    x = np.concatenate([
      maxima[max_indices==0].reshape((-1,1)),
      minima[max_indices==1].reshape((-1,1))],
      axis = 0)

    y = np.concatenate([
      minima[max_indices==0].reshape((-1,1)),
      maxima[max_indices==1].reshape((-1,1))],
      axis = 0)

    return self.model_to_data_dist(self.bundle(x,y))

  def cdf(self, data: np.ndarray):
    mapped_data = self.data_to_model_dist(data)
    gumbel_threshold = self.model_scale_threshold
    u = np.minimum(mapped_data, gumbel_threshold)
    norm_factor = float(1 - self.logistic_gumbel_cdf(self.alpha, self.bundle(gumbel_threshold, gumbel_threshold)))

    return (self.logistic_gumbel_cdf(self.alpha, mapped_data) - self.logistic_gumbel_cdf(self.alpha,u))/norm_factor
     

  def dx_dz(self, z: t.Union[float, np.ndarray], component: int):
    """Calculate analytically or otherwise, the derivative of the standardised marginal distributions with respect to original data scale. This is necessary to calculate pdf values in the original data scale
    
    Args:
        z (t.Union[float, np.ndarray]): values in original scale
        component (int): component index(0 or 1)
    
    """
    margin = self.margin1 if component == 0 else self.margin2
    if isinstance(margin, univar.EmpiricalWithGPTail):
      mu, sigma, xi = margin.tail.threshold, margin.tail.scale, margin.tail.shape
      p = self.quantile_threshold
      dx = -((1 - p)*((xi*(z - mu))/sigma + 1)**(-1/xi - 1))/(sigma*((1 - p)*(1 - ((xi*(z - mu))/sigma + 1)**(-1/xi)) + p)*np.log((1 - p)*(1 - ((xi*(z - mu))/sigma + 1)**(-1/xi)) + p))
    else:
      # estimate by finite differences
      eps = 1e-3
      dx = (margin.pdf(z+eps) - margin.pdf(z-eps))/(2*eps)
    return dx

  def pdf(self, data: t.Union[np.ndarray,t.Iterable]):
    z1, z2 = self.unbundle(data)
    model_scale_data = self.data_to_model_dist(data)
    return np.exp(self.logpdf(self.alpha, self.quantile_threshold, model_scale_data))*self.dx_dz(z1,0)*self.dx_dz(z2,1)






# This class inherits from Logistic for coding convenience, but they are not theoretically related

class Gaussian(Logistic):

  """This model is equivalent to a Gaussian copula model restricted to the region of the form \\( \\mathbf{U} \\nleq u \\), or equivalently \\( \\max\\{\\mathbf{U}_1,\\mathbf{U}_2\\} > u \\), which represents threshold exceedances above \\( u \\) in at least one component. This model can also be thought of as a pre-limit bivariate Generalised Pareto with Gaussian dependence, which is degenerate in the limit. Consequently, under this model there is asymptotic independence between components, this is, extreme values across components are weakly dependent, and occur more or less independently of each other. Use it if there is weak evidence for asymptotic dependence in the data.

  In Gaussian scale (i.e. when both marginal distributions are gaussian), its density function is given by a bivariate normal distribution restricted to the region described above, where \\( u \\) would be the quantile corresponding to a probability value \\( p \\) such that exceedances of probability less than \\(1 - p\\) are considered extreme.

  """
  
  alpha: float
  margin1: univar.BaseDistribution
  margin2: univar.BaseDistribution

  _model_marginal_dist = gaussian
  _marginal_model_name = "Gaussian"

  @property
  def cov(self):
    return np.array([[1,self.alpha],[self.alpha,1]])
  
  @classmethod
  def logistic_gumbel_cdf(cls, alpha: float, data: np.ndarray):
    """Calculates unconstrained standard Gaussian CDF

    """
    return mv_gaussian.cdf(data, cov = np.array([[1,alpha],[alpha,1]]))

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates logpdf for Gaussian exceedances
    
    """
    x, y = cls.unbundle(data)
    if isinstance(alpha, (list, np.ndarray)):
      alpha = alpha[0]
    norm_factor = 1 - mv_gaussian.cdf(cls.bundle(threshold,threshold), cov = np.array([[1,alpha],[alpha,1]]))
    density = mv_gaussian.logpdf(data, cov = np.array([[1,alpha],[alpha,1]])) - np.log(norm_factor)

    # density is 0 when both coordinates are below the threshold
    nil_density_idx = np.logical_and(x <= threshold, y<= threshold)
    density[nil_density_idx] = -np.Inf

    return density

  def simulate(self, size: int):
    """Simulate exceedances
    
    Args:
        size (int): Description
    
    Returns:
        TYPE: Description
    """

    # exceedance subregions:
    # r1 => exceedance in second component only, r2 => exceedance in both components, r3 => exceedance in first component only
    threshold = self.model_scale_threshold
    th = self.bundle(threshold,threshold)
    p1 = self.quantile_threshold - mv_gaussian.cdf(th, cov = self.cov)
    p2 = 1 - 2*self.quantile_threshold + mv_gaussian.cdf(th, cov = self.cov)
    p3 = 1 - mv_gaussian.cdf(th, cov = self.cov) - (p1+p2)

    p = np.array([p1,p2,p3])
    p = p/np.sum(p)

    # compute number of samples per subregion
    n1, n2, n3 = np.random.multinomial(n=size, pvals = p, size=1)[0].astype(np.int32)
    n1, n2, n3 = int(n1), int(n2), int(n3)

    r1_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = self.cov, 
      lb = np.array([-np.Inf, threshold]),
      ub = np.array([threshold, np.Inf])).sample(n1).T

    r2_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = self.cov, 
      lb = np.array([threshold, threshold]),
      ub = np.array([np.Inf, np.Inf])).sample(n2).T

    r3_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = self.cov, 
      lb = np.array([threshold, -np.Inf]),
      ub = np.array([np.Inf, threshold])).sample(n3).T

    samples = np.concatenate([r1_samples, r2_samples, r3_samples], axis = 0)

    return self.model_to_data_dist(samples)

  def dx_dz(z: t.Union[float, np.ndarray], component: int):
    """Calculate analytically or otherwise, the derivative of the standard Gumbel transform with respect to original data scale. This is necessary to calculate pdf values in the original data scale
    
    Args:
        z (t.Union[float, np.ndarray]): values in original scale
        component (int): component index(0 or 1)
    
    """
    margin = self.margin1 if component ==0 else margin2
    eps = 1e-3
    dx = (margin.pdf(z+eps) - margin.pdf(z-eps))/(2*eps)
    return dx








class Empirical(BaseDistribution):

  """Bivariate empirical distribution induced by a sample of observed data
  
  """
  
  data: np.ndarray
  pdf_values: np.ndarray

  _exceedance_models = {
    "logistic": Logistic,
    "gaussian": Gaussian
  }

  def __repr__(self):
    return f"Bivariate empirical distribution with {len(data)} points"

  @validator("pdf_values", allow_reuse=True)
  def check_pdf_values(cls, pdf_values):
    if np.any(pdf_values < -cls._error_tol):
      raise ValueError("There are negative pdf values")
    if not np.isclose(np.sum(pdf_values), 1, atol=cls._error_tol):
      print(f"sum: {np.sum(pdf_values)}, pdf vals: {pdf_values}")
      raise ValueError("pdf values don't sum 1")
    # pdf_values = np.clip(pdf_values, a_min = 0.0, a_max = 1.0)
    # # normalise
    # pdf_values = pdf_values/np.sum(pdf_values)

    return pdf_values


  @classmethod
  def from_data(cls, data: np.ndarray):
    """Instantiate an empirical distribution from an n x 2 data matrix
    
    Args:
        data (np.ndarray): observed data
    
    
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 2 or data.shape[1] != 2:
      raise ValueError("data must be an n x 2 numpy array")

    n = len(data)
    return Empirical(data=data, pdf_values = 1.0/n * np.ones((n,), dtype=np.float64))

  def pdf(self, x: np.ndarray):

    return np.mean(self.data == x.reshape((1,2)))

  def cdf(self, x: np.ndarray):
    if  len(x.shape) > 1:
      return np.array([self.cdf(elem) for elem in x])

    u = self.data <= x.reshape((1,2)) # componentwise comparison
    v = u.dot(np.ones((2,1))) >= 2 #equals 1 if and only if both components are below x
    return np.mean(v)

  def simulate(self, size: int):
    n = len(self.data)
    idx = np.random.choice(n, size=size)
    return self.data[idx]

  def get_marginals(self):

    return univar.Empirical.from_data(self.data[:,0]), univar.Empirical.from_data(self.data[:,1])

  def fit_tail_model(
    self,
    model: str,
    quantile_threshold: float,
    margin1: univar.BaseDistribution = None, 
    margin2: univar.BaseDistribution = None):
    """Fits a parametric model for threshold exceedances in the data. For a given threshold \\( u \\), exceedances are defined as vectors \\(U\\) such that \\( \\max\\{U_1,U_2\\} > u \\), this is, an exceedance in at least one component, and encompasses an inverted L-shaped subset of Euclidean space.
    Currently, logistic and Gaussian models are available, with the former exhibiting asymptotic dependence, a strong type of dependence between extreme occurrences across components, and the latter exhibiting asymptotic independence, in which extremes occur relatively independently across components.
    
    Args:
        model (str): name of selected model, currently one of 'gaussian' or 'logistic'
        margin1 (univar.BaseDistribution, optional): Marginal distribution for first component. If not provided, a semiparametric model with a fitted Generalised Pareto upper tail is used.
        margin2 (univar.BaseDistribution, optional): Marginal distribution for second component. If not provided, a semiparametric model with a fitted Generalised Pareto upper tail is used.
        quantile_threshold (float): Quantile threshold to use for the definition of exceedances
    
    Returns:
        ExceedanceModel
    
    """
    if model not in self._exceedance_models:
      raise ValueError(f"model must be one of {self._exceedance_models}")

    if margin1 is None:

      margin1, _ = self.get_marginals()
      margin1 = margin1.fit_tail_model(threshold=margin1.ppf(quantile_threshold))
      warnings.warn(f"First marginal not provided. Fitting tail model using provided quantile threshold ({quantile_threshold} => {margin1.ppf(quantile_threshold)})", stacklevel=2)

    if margin2 is None:

      _, margin2 = self.get_marginals()
      margin2 = margin2.fit_tail_model(threshold=margin2.ppf(quantile_threshold))
      warnings.warn(f"Second marginal not provided. Fitting tail model using provided quantile threshold ({quantile_threshold} => {margin2.ppf(quantile_threshold)})", stacklevel=2)

    data = self.data

    x = data[:,0]
    y = data[:,1]

    exceedance_idx = np.logical_or(x > margin1.ppf(quantile_threshold), y > margin2.ppf(quantile_threshold))

    exceedances = data[exceedance_idx]

    exceedance_model = self._exceedance_models[model].fit(
      data = exceedances, 
      quantile_threshold = quantile_threshold,
      margin1 = margin1,
      margin2 = margin2)

    empirical_model = Empirical.from_data(self.data[np.logical_not(exceedance_idx)])

    p = np.mean(np.logical_not(exceedance_idx))

    return ExceedanceModel(
      distributions = [empirical_model, exceedance_model],
      weights = np.array([p, 1-p]))

  def test_asymptotic_dependence(self, quantile_threshold: float = 0.95) -> float:
    """Computes the Savage-Dickey ratio for the coefficient of tail dependence \\($\\eta$\\) to test the hypothesis of asymptotic dependence (See 'Statistics of Extremes' by Beirlant, page 345-346). A uniform prior is placed on the hypothesis space \\($\\eta \\in [0,1]$\\) with \\($\\eta = 1$\\) corresponding to asymptotic dependence and vice versa. The posterior density is approximated through Gaussian Kernel density estimation.
    
    Args:
        quantile_threshold (float, optional): Quantile threshold over which the coefficient of tail dependence is to be estimated.
    
    Returns:
        float: Savage-Dickey ratio. If larger than 1, this favors the asymptotic dependence hypothesis and vice versa.
    """

    # Use generalised pareto to fit tails
    x, y = self.data.T
    x_dist, y_dist = univar.Empirical.from_data(x), univar.Empirical.from_data(y)
    x_dist, y_dist = x_dist.fit_tail_model(x_dist.ppf(quantile_threshold)), y_dist.fit_tail_model(y_dist.ppf(quantile_threshold))

    # transform to approximate standard Frechet margins and map to test data t
    u1, u2 = x_dist.cdf(x), y_dist.cdf(y)
    z1, z2 = -1/np.log(u1), -1/np.log(u2)
    t = np.minimum(z1,z2)

    ### compute savage-dickey density ratio

    # log prior for this particular problem, since we know the tail index to be in [0,1].
    def log_prior(theta):
      scale, shape = theta
      if scale > 0 and shape >= 0 and shape <= 1:
        return 0.0
      else:
        return -np.Inf

    t_dist = univar.Empirical.from_data(t)
    # Pass initial point that enforces theoretical constraints of 0 <= eta <= 1. Approximation inaccuracies from the transformation to Frechet margins and MLE estimation can violate the bounds.
    mle_tail = t_dist.fit_tail_model(t_dist.ppf(quantile_threshold))
    x0 = np.array([mle_tail.tail.scale, max(0,min(0.99, mle_tail.tail.shape))])
    # sample posterior
    t_dist = t_dist.fit_tail_model(t_dist.ppf(quantile_threshold), bayesian=True, log_prior=log_prior, x0 =x0)

    #approximate posterior distribution through Kernel density estimation. Evaluating it on 1 gives us the savage-dickey ratio
    return gaussian_kde(t_dist.tail.shapes).evaluate(1)[0]

  def plot_pickands(self, quantile_threshold: float = 0.95):
    """Returns a plot of the empirical Pickands dependence function induced by joint exceedances above the specified quantile threshold. The Pickands dependence function \\($A: [0,1] \\to [1/2,1]$\\) is convex and bounded by \\($\\max\\{t,1-t\\} \\leq A(t) \\leq 1$\\); it can be used to assess extremal dependence, as there is a one-to-one correspondence between \\($A(t)$\\) and extremal copulas; the closer it is to its lower bound, the stronger the extremal dependence. Conversely, for asymptotically independent data \\(A(t) = 1$\\).
    
    Args:
        quantile_threshold (float, optional): Quantile threshold over which Pickands dependence will be approximated.
    """
    fig = plt.figure(figsize=(5,5))

    def pickands(v: float) -> float:
      """Non-parametric Pickands dependence function approximation based on Hall and Tajvidi (2000).
      
      Args:
          v (float): Pickands dependence function argument
      
      Returns:
          float: Nonparametric estimate of the Pickands dependence function
      """
      # s and t arrays come from outer scope and are exponentially distributed maps of original data
      a = (n*s/np.sum(s))/(1-v)
      b = (n*t/np.sum(t))/v
      return 1.0/np.mean(np.minimum(a,b))

    # find joint extremes
    x, y = self.data.T
    joint_extremes_idx = np.logical_and(
      x > np.quantile(x, quantile_threshold),
      y > np.quantile(y, quantile_threshold))
    n = np.sum(joint_extremes_idx)

    # compute nonparametric scores
    x_exs, y_exs = x[joint_extremes_idx], y[joint_extremes_idx]
    x_exs_model, y_exs_model = univar.Empirical.from_data(x_exs), univar.Empirical.from_data(y_exs), 

    # normalise to avoid values on the border of the unit square
    u1, u2 = n/(n+1)*x_exs_model.cdf(x_exs), n/(n+1)*y_exs_model.cdf(y_exs)
    # map to exponential
    s, t = -np.log(u1), -np.log(u2)

    x = np.linspace(0,1,101)
    pk = np.array([pickands(_) for _ in x])

    plt.plot(x,pk,color="darkorange", label="Empirical")
    plt.plot(x,np.maximum(1-x,x), linestyle="dashed", color="black")
    plt.plot(x,np.ones((len(x),)), linestyle="dashed", color="black")
    plt.xlabel("t")
    plt.ylabel("A(t)")
    plt.title("Empirical Pickands dependence of joint exceedances")
    plt.grid()
    return fig





