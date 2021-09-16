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

import numpy as np
import emcee

from scipy.stats import genpareto as gpdist, gumbel_r as gumbel, norm as gaussian, multivariate_normal as mv_gaussian
from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.stats import gaussian_kde as kde
from scipy.signal import fftconvolve
from scipy.special import lambertw


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

  def plot(self, size: int = 1000):
    """Sample distribution and produce scatterplots and histograms
    
    Args:
        size (int, optional): Sample size
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
    plt.figure(figsize=(8, 8))

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
    plt.show()



class Mixture(BaseDistribution):

  """Base interface for a bivariate mixture distribution
  """
  
  distributions: t.List[BaseDistribution]
  weights: np.ndarray

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
  
  def plot_diagnostics(self):

    self.distributions[1].plot_diagnostics()

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

  def pdf(self, x: np.ndarray):
    x1, x2 = x
    return self.x.pdf(x1)*self.y.pdf(x2)

  def cdf(self, x: np.ndarray):
    x1, x2 = x
    return self.x.cdf(x1)*self.y.cdf(x2)

  def simulate(self, size: int):
    return np.concatenate([self.x.simulate(size).reshape((-1,1)), self.y.simulate(size).reshape((-1,1))], axis=1)




class ExceedanceDistribution(BaseDistribution):

  """Main interface for exceedance distributions, which are defined on a region of the form $U \\nleq u$, or equivalently $\\max\\{U_1,U_2\\} > u$. 
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

  """This model is equivalent to a Gumbel-Hougaard copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. This model can also be thought as a pre-limit bivariate Generalised Pareto with logistic dependence in the context of extreme value theory. Consequently, under this model there is asymptotic dependence between components, this is, extreme values across components are strongly associated. Use it if there is strong evidence of asymptotic dependence in the data.
  """
  
  alpha: float
  margin1: univar.BaseDistribution
  margin2: univar.BaseDistribution

  _model_marginal_dist = gumbel
  _marginal_model_name = "Gumbel"

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
    #x = np.array([self.margin1.cdf(z) for z in x]) #list comprehension
    #y = np.array([self.margin2.cdf(z) for z in y]) #list comprehension

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
    #u = np.array([self.margin1.ppf(v) for v in u]) #list comprehension
    #w = np.array([self.margin2.ppf(v) for v in w]) #list comprehension

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

    # nlogp = np.exp(-x/alpha) + np.exp(-y/alpha)
    # rescaler = 1 - cls.uncond_cdf(alpha, cls.bundle(threshold,threshold))
    # density = x/alpha - (nlogp)**alpha + y/alpha + alpha*np.log(nlogp) + np.log((-alpha + alpha*(nlogp)**alpha + 1)) - np.log(alpha) - 2*np.log(np.exp(x/alpha) + np.exp(y/alpha)) - np.log(rescaler)

    #density = x/alpha - (nlogp)**alpha + y/alpha + alpha*np.log(nlogp) + np.log((-alpha + alpha*(nlogp)**alpha + 1)) - np.log(alpha) - 2*np.log(np.exp(x/alpha) + np.exp(y/alpha)) - np.log(rescaler)

    nlogp = (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha
    lognlogp = alpha*np.log(np.exp(-x/alpha) + np.exp(-y/alpha))
    rescaler = 1 - cls.uncond_cdf(alpha, cls.bundle(threshold,threshold))

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
  def uncond_cdf(cls, alpha: float, data: t.Union[np.ndarray,t.Iterable]):
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
        data (t.Union[np.ndarray, t.Iterable])
        quantile_threshold (float): Description
        margin1 (univar.BaseDistribution, optional)
        margin2 (univar.BaseDistribution, optional)
        return_opt_results (bool, optional): If True, the object from the optimization result is returned
        x0 (float, optional): Initial point for the optimisation algorithm. Defaults to 0.5
    
    Returns:
        Logistic: Fitted model
    
    
    Raises:
        ValueError: Description
    
    
    """
    if margin1 is None:
      margin1 = univar.empirical.from_data(data[:,0])
      warnings.warn("margin1 is None; using an empirical distribution")

    if margin2 is None:
      margin1 = univar.empirical.from_data(data[:,1])
      warnings.warn("margin1 is None; using an empirical distribution")

    if not isinstance(quantile_threshold, float) or quantile_threshold <= 0 or quantile_threshold >= 1:
      raise ValueError("quantile_threshold must be in the open interval (0,1)")

    mapped_data = cls(
      alpha=0.5, 
      margin1=margin1, 
      margin2=margin2, 
      quantile_threshold=quantile_threshold).data_to_model_dist(data)

    x,y = cls.unbundle(mapped_data)

    n = len(x)

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
      return -cls.loglik(alpha, model_scale_threshold, data)/n

    res = minimize(
      fun=loss, 
      x0 = x0,
      method = "BFGS",
      args = (mapped_exceedances,))


    ##### optimization in copula scale
    # def copula_cdf(alpha, data):
    #   U, V = cls.unbundle(data)
    #   theta = 1.0/alpha
    #   h = np.power(-np.log(U), theta) + np.power(-np.log(V), theta)
    #   h = -np.power(h, 1.0 / theta)
    #   cdfs = np.exp(h)
    #   return cdfs

    # def copula_logpdf(alpha, data):
    #   U, V = cls.unbundle(data)
    #   theta = 1.0/alpha
    #   # a = np.power(np.multiply(U, V), -1)
    #   # tmp = np.power(-np.log(U), theta) + np.power(-np.log(V), theta)
    #   # b = np.power(tmp, -2 + 2.0 / theta)
    #   # c = np.power(np.multiply(np.log(U), np.log(V)), theta - 1)
    #   # d = 1 + (theta - 1) * np.power(tmp, -1.0 / theta)
    #   #return self.cumulative_distribution(X) * a * b * c * d

    #   log_a = -np.log(U*V)
    #   tmp = np.power(-np.log(U), theta) + np.power(-np.log(V), theta)
    #   log_b = (-2 + 2.0 / theta)* np.log(tmp)
    #   log_c = (theta-1)*np.log(np.log(U)*np.log(V))
    #   log_d = np.log(1 + (theta - 1) * np.power(tmp, -1.0 / theta))
    #   logcdf = np.log(copula_cdf(alpha, data))
    #   return logcdf + log_a + log_b + log_c + log_d

    # def loss(alpha, data):
    #   return -np.mean(copula_logpdf(alpha, data))

    # copula_exceedances = cls.bundle(cls._model_marginal_dist.cdf(x), cls._model_marginal_dist.cdf(y))

    # res = minimize(
    #   fun=loss, 
    #   x0 = x0,
    #   method = "BFGS",
    #   args = (copula_exceedances,))

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

  def plot_diagnostics(self):
    """Produce diagnostic plots for fitted model

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
    y_range = np.linspace(min(z1) - 0.05*z2_range, max(z1) + 0.05*z2_range, 50)

    X, Y = np.meshgrid(x_range, y_range)
    bundled_grid = self.bundle(X.reshape((-1,1)), Y.reshape((-1,1)))
    Z = self.logpdf(data=bundled_grid, threshold=model_scale_threshold, alpha=self.alpha).reshape(X.shape)
    axs[0,1].contourf(X,Y,Z)
    axs[0,1].scatter(z1,z2, color=self._figure_color_palette[1], s=0.9)
    axs[0,1].title.set_text(f'Model density ({self._marginal_model_name} scale)')
    axs[0,1].set_xlabel('x')
    axs[0,1].set_ylabel('y')

    ##### Q-Q plot
    cdf_values = self.cdf(self.data)
    model_logodds = np.log(cdf_values/(1-cdf_values))
    ecdf_values = Empirical.from_data(self.data).cdf(self.data)
    empirical_logodds = np.log(ecdf_values/(1-ecdf_values))

    axs[1,0].scatter(model_logodds, empirical_logodds, color=self._figure_color_palette[0])

    axs[1,0].title.set_text('Model vs data log-odds')
    axs[1,0].set_ylabel('Data log-odds')
    axs[1,0].set_ylabel('Model log-odds')
    axs[1,0].set_xlim(-5,5)
    min_e, max_e = max(-5,min(empirical_logodds)), min(5,max(empirical_logodds))
    axs[1,0].plot([min_e, max_e], [min_e, max_e], linestyle="--", color="black")

    plt.tight_layout()
    plt.show()

    # #print("density plot finished")
    # ####### Pickands function
    # x_grid = np.linspace(0,1,50)
    # logistic_pickands = (x_grid**(1.0/self.alpha) + (1-x_grid)**(1.0/self.alpha))**(self.alpha)
    # ## get data for empirical pickands function
    # # see Statistics of Extremes by Berlaint, page 315
    # th_margin1, th_margin2 = self.margin1.ppf(self.quantile_threshold), self.margin2.ppf(self.quantile_threshold)
    # exceedance_data = self.data[np.logical_and(x > th_margin1, y > th_margin2)]
    # exs1, exs2 = self.unbundle(exceedance_data)

    # tail = self.margin1 > th_margin1
    # xi = -np.log(tail.cdf(exs1))
    # #xi = -np.log(np.array([tail.cdf(x_) for x_ in exs1])) #list comprehension
    # tail = self.margin2 > th_margin2
    # eta = -np.log(tail.cdf(exs2))
    # #eta = -np.log(np.array([tail.cdf(y_) for y_ in exs2])) #list comprehension

    # finite_idx = np.logical_and(np.isfinite(xi), np.isfinite(eta))
    # xi = xi[finite_idx]
    # eta = eta[finite_idx]

    # def nonparametric_pickands(t):
    #   return 1.0/np.mean(np.minimum(xi/(np.mean(xi)*(1-t)), eta/(np.mean(eta)*t)))
    # empirical_pickands = np.array([nonparametric_pickands(t) for t in x_grid])

    # axs[1,1].plot(x_grid, logistic_pickands, color = self._figure_color_palette[0])
    # axs[1,1].plot(x_grid, empirical_pickands, color = self._figure_color_palette[1])
    # axs[1,1].plot(x_grid, np.ones((len(x_grid),)), linestyle="--", color="black")
    # axs[1,1].plot(x_grid, np.maximum(1-2*x_grid,2*x_grid-1), linestyle="--", color="black")
    # axs[1,1].title.set_text("model vs empirical Pickands function")
    # axs[1,1].set_xlabel("t")
    # axs[1,1].set_ylabel("")

  def simulate(self, size: int):
    alpha = self.alpha
    exs_prob = 1 - self.quantile_threshold
    ### simulate in Gumbel scale maximum component: z = max(x,y) ~ Gumbel(loc=np.log(2)) using inverse function method
    u = np.random.uniform(size=size)
    maxima = -np.log(-np.log(1 - exs_prob*(1-u))) + alpha*np.log(2)

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
    norm_factor = float(1 - self.uncond_cdf(self.alpha, self.bundle(gumbel_threshold, gumbel_threshold)))

    return (self.uncond_cdf(self.alpha, mapped_data) - self.uncond_cdf(self.alpha,u))/norm_factor
     

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

  """This model is equivalent to a Gaussian copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. This model can also be thought of as a pre-limit bivariate Generalised Pareto with Gaussian dependence, which is degenerate in the limit. Consequently, under this model there is asymptotic independence between components, this is, extreme values across components are weakly dependent, and occur more or less independently of each other. Use it if there is weak evidence for asymptotic dependence in the data.
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
  def uncond_cdf(cls, alpha: float, data: np.ndarray):
    """Calculates unconstrained standard Gaussian CDF

    """
    return mv_gaussian.cdf(data, cov = np.array([[1,alpha],[alpha,1]]))

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates logpdf for Gaussian exceedances
    
    """
    x, y = cls.unbundle(data)
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

  @property
  def pdf_lookup(self):
    """Mapping from values in the support to their probability mass
    
    """
    return {tuple(row): p for p, row in zip(self.pdf_values,self.data)}


  @classmethod
  def from_data(cls, data: np.ndarray):
    """Instantiate an empirical distribution from an array of bivariate data
    
    Args:
        data (np.ndarray): observed data
    
    
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
      raise ValueError("data must be a 2-dimensional numpy array")

    df = pd.DataFrame(data)
    df.columns = ["x","y"]
    df["counter"] = 1.0
    df = df.groupby(by=["x","y"]).agg(n = ("counter",np.sum)).reset_index()
    x, y, p = np.array(df["x"]), np.array(df["y"]), np.array(df["n"])

    pdf_values = p/np.sum(p)
    return Empirical(data=np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))],axis=1), pdf_values = pdf_values)

  def pdf(self, x: np.ndarray):

    if  len(x.shape) > 1:
      return np.array([self.pdf(elem) for elem in x])
    try:
      self.pdf_lookup[tuple(x)]
    except KeyError as e:
      return 0.0

  def cdf(self, x: np.ndarray):
    if  len(x.shape) > 1:
      return np.apply_along_axis(self.cdf, axis=1, arr=x)
    x1, x2 = x
    return np.mean(np.logical_and(self.data[:,0] <= x1, self.data[:,1] <=x2))

  def simulate(self, size: int):
    n = len(self.data)
    idx = np.random.choice(n, size=size)
    return self.data[idx]

  def fit_exceedance_model(
    self,
    model: str,
    quantile_threshold: float,
    margin1: univar.BaseDistribution = None, 
    margin2: univar.BaseDistribution = None):
    """Fits a parametric model for threshold exceedances in the data. For a given threshold $u$, exceedances are defined as vectors $U$ such that $\\max\\{U_1,U_2\\} > u$, this is, an exceedance in at least one component, and encompasses an inverted L-shaped subset of Euclidean space.
    Currently, logistic and Gaussian models are available, with the former exhibiting asymptotic dependence, a strong type of dependence between extremes across components, and the latter exhibiting asymptotic independence, in which extremes happen mostly independently across components.
    
    Args:
        model (str): name of selected model, currently one of 'gaussian' or 'logistic'
        margin1 (univar.BaseDistribution, optional): Marginal distribution for first component. If not provided, a semiparametric model with a fitted Generalised Pareto tail is used.
        margin2 (univar.BaseDistribution, optional): Marginal distribution for second component. If not provided, a semiparametric model with a fitted Generalised Pareto tail is used.
        quantile_threshold (float): Quantile threshold to use for the deffinition of exceedances
    
    Returns:
        ExceedanceModel
    
    """
    data = self.data

    x = data[:,0]
    y = data[:,1]

    if model not in self._exceedance_models:
      raise ValueError(f"model must be one of {self._exceedance_models}")

    if margin1 is None:
      warnings.warn("first marginal not provided. Fitting tail model for first component using provided quantile threshold.")
      margin1 = univar.Empirical.from_data(x).fit_tail_model(x.ppf(quantile_threshold))

    if margin2 is None:
      warnings.warn("second marginal not provided. Fitting tail model for second component using provided quantile threshold.")
      margin2 = univar.Empirical.from_data(y).fit_tail_model(x.ppf(quantile_threshold))

    exceedance_idx = np.logical_or(x > margin1.ppf(quantile_threshold), y > margin2.ppf(quantile_threshold))

    exceedances = data[exceedance_idx]

    # data in copula scale
    # u1 = np.array([margin1.ppf(x) for x in exceedances[:,0]]).reshape((-1,1))
    # u2 = np.array([margin2.ppf(x) for x in exceedances[:,1]]).reshape((-1,1))

    # if np.any(u1 == 0) or np.any(u1 == 1.0) or np.any(u2 == 0) or np.any(u2 == 1):
    #   raise ValueError("Some values are either 0 or 1 in copula scale.")

    exceedance_model = self._exceedance_models[model].fit(
      data = data, 
      quantile_threshold = quantile_threshold,
      margin1 = margin1,
      margin2 = margin2)

    empirical_model = Empirical.from_data(self.data[np.logical_not(exceedance_idx)])

    p = np.mean(np.logical_not(exceedance_idx))

    return ExceedanceModel(
      distributions = [empirical_model, exceedance_model],
      weights = np.array([p, 1-p]))

  # def test_exceedance_dependence(self):

  #   def tegpd_logp(x,eta,sigma):
  #     #returns the sum of log-liklihoods
  #     return tt.sum(-tt.log(sigma) - (1.0/eta+1)*tt.log(1+eta/sigma*x))

  #   def get_eta_posterior(obs,n_samples = 2500):
      
  #     if isinstance(obs,list):
  #       obs = numpy.array(obs)
        
  #     #Bayesian specification and inference
  #     eta_model = pm.Model()

  #     with eta_model:
  #       #uniform eta prior in (0,1)
  #       eta = pm.Uniform("eta",lower=0,upper=1)
  #       #flat sigma prior in positive line
  #       sigma = pm.HalfFlat("sigma")
  #       X = pm.DensityDist("tegpd",tegpd_logp,observed={"eta":eta,"sigma":sigma,"x":obs})
  #       trace = pm.sample(n_samples,random_seed = 1)
  #       #plots = pm.traceplot(trace)
        
  #     eta_samples = trace.get_values("eta")
  #     #sigma_samples = trace.get_values("sigma")
      
  #     #posterior_kde = arviz.plot_kde(eta_samples)
  #     posterior_AD_prob = sp.stats.gaussian_kde(eta_samples).evaluate(1)
      
  #     # get mean kernel density estimator for density at eta = 1
  #     #eta_posterior = posterior_kde.lines[0].get_ydata()[-1]
  #     #return 1
  #     return 1.0/(1+1.0/posterior_AD_prob) if posterior_AD_prob > 0 else 0
