from __future__ import annotations

import logging
import time
import typing as t
import traceback
import warnings
from argparse import Namespace
from abc import ABC, abstractmethod
from multiprocessing import Pool
import copy

import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import numpy as np
import emcee

from scipy.stats import genpareto as gpdist, gumbel_r as gumbel, normal as gaussian, multivariate_normal as mv_gaussian
from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.stats import gaussian_kde as kde
from scipy.signal import fftconvolve


from pydantic import BaseModel, ValidationError, validator, PositiveFloat
from functools import reduce

import riskmodels.univariate as univar

from riskmodels.utils.tmvn import TruncatedMVN as tmvn

class BaseDistribution(BaseModel):

  """Base interface for bivariate distributions
  """
  
  _allowed_scalar_types = (int, float, np.int64, np.int32, np.float32, np.float64)
  _figure_color_palette = ["tab:cyan", "deeppink"]
  _error_tol = 1e-6

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
    ax_scatter.scatter(x, y, color = color = self._figure_color_palette[0])

    # now determine nice limits by hand:
    # binwidth = 0.25
    # lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    # ax_scatter.set_xlim((-lim, lim))
    # ax_scatter.set_ylim((-lim, lim))

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=25, color = self._figure_color_palette[0])
    ax_histy.hist(y, bins=25, orientation='horizontal', color = self._figure_color_palette[0])

    #ax_histx.set_xlim(ax_scatter.get_xlim())
    #ax_histy.set_ylim(ax_scatter.get_ylim())

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



class Empirical(BaseDistribution):

  """Bivariate empirical distribution induced by a sample of observed data
  
  """
  
  data: np.ndarray
  pdf_values: np.ndarray

  exceedance_models = {
    "logistic": Gumbel,
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
    if not isisnstance(data, np.ndarray) or len(data.shape) != 2:
      raise ValueError("data must be a 2-dimensional numpy array")

    df = pd.DataFrame(data)
    df.columns = ["x","y"]
    df["counter"] = 1.0
    df = df.groupby(by=["x","y"]).agg(n = ("counter",np.sum)).reset_index()
    x, y, p = np.array(df["x"]), np.array(df["y"]), np.array(df["n"])

    pdf_values = p/np.sum(p)
    return Empirical(data=np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))],axis=1), pdf_values = pdf_values)

  def pdf(self, x: np.ndarray):
    return self.pdf_lookup(tuple(x))

  def cdf(self, x: np.ndarray):
    x1, x2 = x
    n = len(self.data)
    return np.sum(np.logical_and(self.data[:,0] <= x1, self.data[:,1] <=x2))/n

  def simulate(self, size: int):
    n = len(self.data)
    idx = np.choice(n, size=size)
    return self.data[idx]

  def fit_exceedance_model(
    cls,
    model: str,
    margin1: univar.BaseDistribution = None, 
    margin2: univar.BaseDistribution = None, 
    quantile_threshold: float):
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

    if model not in cls.exceedance_models:
      raise ValueError(f"model must be one of {cls.exceedance_models}")

    if margin1 is None:
      warnings.warn("first marginal not provided. Fitting tail model for first component using provided quantile threshold.")
      margin1 = univar.Empirical.from_data(x).fit_tail_model(x.ppf(quantile_threshold))

    if margin2 is None:
      warnings.warn("second marginal not provided. Fitting tail model for second component using provided quantile threshold.")
      margin2 = univar.Empirical.from_data(y).fit_tail_model(x.ppf(quantile_threshold))

    exceedance_idx = np.logical_or(x >= margin1.ppf(quantile_threshold), y>= margin2.ppf(quantile_threshold))

    exceedances = data[exceedance_idx]

    # data in copula scale
    # u1 = np.array([margin1.ppf(x) for x in exceedances[:,0]]).reshape((-1,1))
    # u2 = np.array([margin2.ppf(x) for x in exceedances[:,1]]).reshape((-1,1))

    # if np.any(u1 == 0) or np.any(u1 == 1.0) or np.any(u2 == 0) or np.any(u2 == 1):
    #   raise ValueError("Some values are either 0 or 1 in copula scale.")

    exceedance_model = cls.exceedance_models[model].fit(
      data = data, 
      quantile_threshold = quantile_threshold,
      margin1 = margin1,
      margin2 = margin2)

    empirical_model = Empirical.from_data(data[np.logical_not(exceedance_idx)])

    p = Empirical.from_data(data).cdf((margin1.ppf(quantile_threshold), margin2.ppf(quantile_threshold)))

    return ExceedanceModel(
      distributions = [empirical_model, exceedance_model],
      weights = [p, 1-p])



class Independent(BaseDistribution):

  """Bivariate distribution with independent components
  """
  
  x: BaseDistribution
  y: BaseDistribution

  def pdf(self, x: np.ndarray):
    x1, x2 = x
    return x.pdf(x1)*y.pdf(x2)

  def cdf(self, x: np.ndarray):
    x1, x2 = x
    return x.cdf(x1)*y.cdf(x2)

  def simulate(self, size: int):
    return np.concatenate([x.simulate(size).reshape((-1,1)), y.simulate(size).reshape((-1,1))], axis=1)




class ExceedanceDistribution(BaseDistribution):

  """Main interface for exceedance distributions, which are defined on a region of the form $U \\nleq u$, or equivalently $\\max\\{U_1,U_2\\} > u$. 
  """
  quantile_threshold: float
  #margin1: univar.BaseDistribution
  #margin2: univar.BaseDistribution

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
    """Unbundles matrix or iterableinto separate components
    
    Args:
        data (t.Union[np.ndarray, t.Iterable]): dara
    
    Returns:
        TYPE: Description
    """
    if isinstance(data, np.ndarray):
      x = data[:,0]
      y = data[:,1]
    else:
      # if iterable, unroll
      x, y = data
    return x, y


class Logistic(ExceedanceDistribution):

  """This model is equivalent to a Gumbel-Hougaard copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. This model can also be thought as a pre-limit bivariate Generalised Pareto with logistic dependence in the context of extreme value theory. Consequently, under this model there is asymptotic dependence between components, this is, extreme values across components are strongly associated. Use it if there is strong evidence of asymptotic dependence in the data.
  """
  
  alpha: float
  margin1: univar.BaseDistribution
  margin2: univar.BaseDistribution

  _model_marginal_dist = gumbel

  def data_to_model_marginal_dist(self, data: t.Union[np.ndarray,t.Iterable]) -> t.Tuple[t.Union[np.ndarray, float], t.Union[np.ndarray, float]]:
    """Map original data scale to standard Gumbel scale
    
    Args:
        data (t.Union[np.ndarray, t.Iterable]): observations in original scale
    
    """
    x, y = self.unbundle(data)

    ## to copula scale
    x = np.array([self.margin1.ppf(z) for z in x])
    y = np.array([self.margin1.ppf(z) for z in y])

    # pass to Gumbel scale 
    x = self._model_marginal_dist.ppf(x)
    y = self._model_marginal_dist.ppf(y)
    return x, y

  def model_to_data_dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maps stanrdard Gumbel scale to original data scale
    
    Args:
        x (np.ndarray): data from first component
        y (np.ndarray): data from second component
    
    Returns:
        np.ndarray
    """

    # copula scale
    x = self._model_marginal_dist.cdf(x)
    y = self._model_marginal_dist.cdf(x)

    # data scale
    x = np.array([self.margin1.ppf(u) for u in x])
    y = np.array([self.margin2.ppf(u) for u in y])

    return np.concatenate([x,y], axis=1)

  @validator("alpha")
  def validate_alpha(cls, alpha):
    if alpha < 0 or alpha > 1:
      raise TypeError("alpha must be in the open interval (0,1) ")

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates logpdf for Gumbel exceedances
    
    """
    x, y = self.unbundle(data)
    # density = (np.exp(x/alpha - (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha + y/alpha)*
    #   (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha*
    #   (-alpha + alpha*(np.exp(-x/alpha) + np.exp(-y/alpha))**alpha + 1))/
    # (alpha*(np.exp(x/alpha) + np.exp(y/alpha))**2)

    x = np.array(x)
    y = np.array(y)

    nlogp = np.exp(-x/alpha) + np.exp(-y/alpha)
    rescaler = 1 - self.uncond_cdf(alpha, threshold, threshold)
    density = x/alpha - (nlogp)**alpha + y/alpha + alpha*np.log(nlogp) + np.log((-alpha + alpha*(nlogp)**alpha + 1)) - np.log(alpha) - 2*np.log(np.exp(x/alpha) + np.exp(y/alpha)) - np.log(rescaler)

    # density is 0 when both coordinates are below the threshold
    nil_density_idx = np.logical_and(x <= threshold, y<= threshold)
    density[nil_density_idx] = 0.0

    return density

  @classmethod
  def loglik(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates log-likelihood for Gumbel exceedances
    
    """
    x, y = self.unbundle(data)

    return np.sum(logpdf(alpha, threshold, x,y))

  @classmethod
  def uncond_cdf(cls, alpha: float, x: np.ndarray, y: np.ndarray):
    """Calculates unconstrained standard Gumbel CDF

    """
    x, y = self.unbundle(data)
    return np.exp(-(np.exp(-x/alpha) + np.exp(-y/alpha))**(alpha))

  @classmethod
  def fit(
    cls, 
    data: t.Union[np.ndarray,t.Iterable], 
    quantile_threshold: float,
    margin1: univar.BaseDistribution = None,
    margin2: univar.BaseDistribution = None) -> Logistic:
    """Fits the model from provided data, threshold and marginal distributons
    
    Args:
        data (t.Union[np.ndarray, t.Iterable])
        quantile_threshold (float)
        margin1 (univar.BaseDistribution, optional)
        margin2 (univar.BaseDistribution, optional)
    
    Returns:
        Logistic: Fitted model
    
    """
    if margin1 is None:
      margin1 = univar.empirical.from_data(data[:,0])
      warnings.warn("margin1 is None; using an empirical distribution")

    if margin2 is None:
      margin1 = univar.empirical.from_data(data[:,1])
      warnings.warn("margin1 is None; using an empirical distribution")

    if not isinstance(quantile_threshold, float) or quantile_threshold <= 0 or quantile_threshold >= 1:
      raise ValueError("quantile_threshold must be in the open interval (0,1)")

    x, y = self.data_to_model_marginal_dist(data)
    n = len(x)

    # get threshold exceedances
    model_scale_threshold = cls.model_dist.ppf(quantile_threshold)

    exs_idx = np.logical_or(x >= model_scale_threshold, y >= model_scale_threshold)
    x = x[exs_idx]
    y = y[exs_idx]

    def logistic(x):
      return 1.0/(1 + np.exp(-x))

    def loss(phi, x, y):
      alpha = logistic(phi)
      return cls.loglik(alpha, quantile_threshold, x, y)/n

    res = minimize(
      fun=loss, 
      x0 = 0.0,
      method = "BFGS")

    if return_opt_results:
      warn.warnings("Returning raw results for rescaled exceedance data (sdev ~ 1).")
      return res
    else:
      phi = res.x[0]
      alpha = logistic(phi)
      data = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))], axis=1)

    return cls(
      quantile_threshold = quantile_threshold,
      alpha = alpha,
      data = data,
      margin1 = margin1,
      margin2 = margin2)

  @classmethod
  def hessian(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates loglikelihood's second deriviative (i.e., negative estimator's precision)
    
    """
    x, y = self.data_to_model_marginal_dist(data)

    delta = 1e-3
    fisher = (cls.loglik(alpha - delta, threshold, x, y) -2*cls.loglik(alpha, threshold, x, y) + cls.loglik(alpha + delta, threshold, x, y))/(n*delta**2)

    return -1.0/fisher

  def plot_diagnostics(self):
    """Produce diagnostic plots for fitted model

    """
    # x, y = self.unbundle(self.data)
    # n = len(x)
    # # map to standard Gumbel scale
    # z1 = -np.log(-np.log(x))
    # z2 = -np.log(-np.log(y))

    z1,z2= self.data_to_model_marginal_dist(self.data)
    n = len(z1)

    fig, axs = plt.subplots(3, 2)

    ####### loglikelihood plot
    sdev = np.sqrt(self.hessian(self.alpha, self.quantile_threshold, z1, z2))
    grid = np.linspace(self.alpha - sdev, self.alpha + sdev, 100)
    grid = grid[np.logical_and(grid > 0, grid < 1)]
    ll = np.array([self.loglik(alpha, self.quantile_threshold, z1, z2) for alpha in grid])

    axs[0,0].plot(grid, ll, color=self._figure_color_palette[0])
    axs[0,0].vlines(x=self.alpha, ymin=min(ll), ymax = max(ll), linestyle="dashed", colors = self._figure_color_palette[1])
    axs[0,0].title.set_text('Log-likelihood')
    axs[0,0].set_xlabel('Alpha')
    axs[0,0].set_ylabel('log-likelihood')

    ####### density plot
    z1_range = max(z1) - min(z1)
    z2_range = max(z2) - min(z2)

    x_range = np.linspace(min(z1) - 0.05*z1_range, max(z1) + 0.05*z1_range, 50)
    y_range = np.linspace(min(z1) - 0.05*z2_range, max(z1) + 0.05*z2_range, 50)

    X, Y = np.meshgrid(x_range, y_range)
    Z = self.pdf(X,Y)
    axs[0,1].contourf(X,Y,Z)
    axs[0,1].plot(z1,z2, color="darkorange", s=2)
    axs[0,1].title.set_text('Data with model density (Gumbel scale)')
    axs[0,1].set_xlabel('y')
    axs[0,1].set_ylabel('x')


    ####### Pickands function
    x_grid = np.linspace(0,1,50)
    logistic_pickands = (x_grid**(1.0/self.alpha) + (1-x_grid)**(1.0/self.alpha))**(self.alpha)
    ## get data for empirical pickands function
    # see Statistics of Extremes by Berlaint, page 315
    pickands_data = -np.log(self.data[np.logical_and(x > self.quantile_threshold, y > self.quantile_threshold)])
    xi = pickands_data[:,0]
    eta = pickands_data[:,1]

    def nonparametric_pickands(t):
      return 1.0/np.mean(np.minimum(xi/(1-t), eta/t))
    empirical_pickands = np.array([nonparametric_pickands(t) for t in x_grid])

    axs[1,0].plot(x_grid, logistic_pickands, color = self._figure_color_palette[0])
    axs[1,0].plot(x_grid, empirical_pickands, color = self._figure_color_palette[1])
    axs[1,0].plot(x_grid, np.ones((len(x_grid),)), linestyle="--")
    axs[1,0].plot(x_grid, np.maximum(1-2*x_grid,2*x_grid-1), linestyle="--")
    axs[1,0].title.set_text("Pickands function")
    axs[1,0].set_x_label("t")
    axs[1,0].set_y_label("")

    ##### Q-Q plots
    def marginal_ppf(q, component = 0):
      def target_func(x):
        x_ = np.Inf*np.ones((2,))
        x_[component] = x
        return self.cdf(x_) - q
      x0 = self._model_marginal_dist.ppf(q)
      return root_scalar(target_func, x0 = x0, x1 = x0+1, method="secant").root


    #### first component
    probability_range = np.linspace(0.01,0.99, 99)
    empirical_quantiles = np.quantile(z1, probability_range)
    model_quantiles = np.array([marginal_ppf(p) for p in probability_range])

    axs[1,1].scatter(model_quantiles, empirical_quantiles, color = self._figure_color_palette[0])
    min_x, max_x = min(model_quantiles), max(model_quantiles)
    #axs[0,1].set_aspect('equal', 'box')
    axs[1,1].title.set_text('Q-Q plot (1st component)')
    axs[1,1].set_xlabel('model quantiles')
    axs[1,1].set_ylabel('Data quantiles')
    axs[1,1].grid()
    axs[1,1].plot([min_x,max_x],[min_x,max_x], linestyle="--", color="black")

    #### second component
    empirical_quantiles = np.quantile(z2, probability_range)
    model_quantiles = np.array([marginal_ppf(p, component=1) for p in probability_range])

    axs[2,0].scatter(model_quantiles, empirical_quantiles, color = self._figure_color_palette[0])
    min_x, max_x = min(model_quantiles), max(model_quantiles)
    #axs[0,1].set_aspect('equal', 'box')
    axs[2,0].title.set_text('Q-Q plot (2nd component)')
    axs[2,0].set_xlabel('model quantiles')
    axs[2,0].set_ylabel('Data quantiles')
    axs[2,0].grid()
    axs[2,0].plot([min_x,max_x],[min_x,max_x], linestyle="--", color="black")

    plt.show()

  def simulate(self, size: int):
    alpha = self.alpha
    ### simulate in Gumbel scale maximum component: z = max(x,y) ~ Gumbel using inverse function method
    u = np.random.uniform(size=size)
    maxima = -np.log(-np.log(1 - exs_prob*(1-u))) + alpha*np.log(2)

    ###simulate difference between maxima and minima r = max(x,y) - min(x,y) using inverse function method
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

    return self.model_to_data_dist(x,y)

  def cdf(self, data: t.Union[np.ndarray,t.Iterable]):
    x, y = self.data_to_model_marginal_dist(data)

    u1 = np.clip(x, a_min=0, a_max = self._model_marginal_dist.ppf(self.quantile_threshold))
    u2 = np.clip(y, a_min=0, a_max = self._model_marginal_dist.ppf(self.quantile_threshold))

    return (self.uncond_cdf(self.alpha, x, y) - self.uncond_cdf(self.alpha, u1, u2))/(1 - self.uncond_cdf(self.alpha, self.quantile_threshold, self.quantile_threshold))
     

  def dx_dz(z: t.Union[float, np.ndarray], component: int):
    """Calculate analytically or otherwise, the derivative of the standardised marginal distributions with respect to original data scale. This is necessary to calculate pdf values in the original data scale
    
    Args:
        z (t.Union[float, np.ndarray]): values in original scale
        component (int): component index(0 or 1)
    
    """
    margin = self.margin1 if component ==0 else margin2
    if isinstance(univar.EmpiricalWithGPTail):
      mu, sigma xi = self.margin.tail.threshold, self.margin.tail.scale, self.margin.tail.shape
      p = self.quantile_threshold
      dx = -((1 - p)*((xi*(z1 - mu))/sigma + 1)**(-1/xi - 1))/(sigma*((1 - p)*(1 - ((xi*(z1 - mu))/sigma + 1)**(-1/xi)) + p)*np.log((1 - p)*(1 - ((xi*(z1 - mu))/sigma + 1)**(-1/xi)) + p))
    else:
      # estimate by finite differences
      eps = 1e-3
      dx = (margin.pdf(z+eps) - margin.pdf(z-eps))/(2*eps)
    return dx

  def pdf(self, data: t.Union[np.ndarray,t.Iterable]):
    z1, z2 = data[:,0], data[:,1]
    x, y = self.data_to_model_marginal_dist(data)
    return np.exp(self.logpdf(self.alpha, self.quantile_threshold, x, y))*dx_dz(z1,0)*dx_dz(z2,1)




class Gaussian(ExceedanceDistribution):

  """This model is equivalent to a Gaussian copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. This model can also be thought of as a pre-limit bivariate Generalised Pareto with Gaussian dependence, which is degenerate in the limit. Consequently, under this model there is asymptotic independence between components, this is, extreme values across components are weakly dependent, and occur more or less independently of each other. Use it if there is weak evidence for asymptotic dependence in the data.
  """
  
  alpha: float
  margin1: univar.BaseDistribution
  margin2: univar.BaseDistribution

  _model_marginal_dist = gaussian

  @classmethod
  def uncond_cdf(cls, alpha: float, x: np.ndarray, y: np.ndarray):
    """Calculates unconstrained standard Gumbel CDF

    """
    z = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))], axis=1)
    return mv_gaussian.cdf(data, cov = np.array([[1,alpha],[alpha,1]]))

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, data: t.Union[np.ndarray,t.Iterable]):
    """Calculates logpdf for Gumbel exceedances
    
    """
    x, y = self.unbundle(data)
    # density = (np.exp(x/alpha - (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha + y/alpha)*
    #   (np.exp(-x/alpha) + np.exp(-y/alpha))**alpha*
    #   (-alpha + alpha*(np.exp(-x/alpha) + np.exp(-y/alpha))**alpha + 1))/
    # (alpha*(np.exp(x/alpha) + np.exp(y/alpha))**2)

    x = np.array(x)
    y = np.array(y)

    z = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))], axis=1)
    density = mv_gaussian.logpdf(z, cov = np.array([[1,alpha],[alpha,1]])) - mv_gaussian.logcdf([threshold,threshold], cov = np.array([[1,alpha],[alpha,1]]))

    # density is 0 when both coordinates are below the threshold
    nil_density_idx = np.logical_and(x <= threshold, y<= threshold)
    density[nil_density_idx] = 0.0

    return density

  def plot_diagnostics(self):
    """Produce diagnostic plots for fitted model

    """
    # x, y = self.unbundle(self.data)
    # n = len(x)
    # # map to standard Gumbel scale
    # z1 = -np.log(-np.log(x))
    # z2 = -np.log(-np.log(y))

    z1,z2= self.data_to_model_marginal_dist(self.data)
    n = len(z1)

    fig, axs = plt.subplots(3, 2)

    ####### loglikelihood plot
    sdev = np.sqrt(self.hessian(self.alpha, self.quantile_threshold, z1, z2))
    grid = np.linspace(self.alpha - sdev, self.alpha + sdev, 100)
    grid = grid[np.logical_and(grid > 0, grid < 1)]
    ll = np.array([self.loglik(alpha, self.quantile_threshold, z1, z2) for alpha in grid])

    axs[0,0].plot(grid, ll, color=self._figure_color_palette[0])
    axs[0,0].vlines(x=self.alpha, ymin=min(ll), ymax = max(ll), linestyle="dashed", colors = self._figure_color_palette[1])
    axs[0,0].title.set_text('Log-likelihood')
    axs[0,0].set_xlabel('Alpha')
    axs[0,0].set_ylabel('log-likelihood')

    ####### density plot
    z1_range = max(z1) - min(z1)
    z2_range = max(z2) - min(z2)

    x_range = np.linspace(min(z1) - 0.05*z1_range, max(z1) + 0.05*z1_range, 50)
    y_range = np.linspace(min(z1) - 0.05*z2_range, max(z1) + 0.05*z2_range, 50)

    X, Y = np.meshgrid(x_range, y_range)
    Z = self.pdf(X,Y)
    axs[0,1].contourf(X,Y,Z)
    axs[0,1].plot(z1,z2, color="darkorange", s=2)
    axs[0,1].title.set_text('Data with model density (Gumbel scale)')
    axs[0,1].set_xlabel('y')
    axs[0,1].set_ylabel('x')

    ##### Q-Q plots
    def marginal_ppf(q, component = 0):
      def target_func(x):
        x_ = np.Inf*np.ones((2,))
        x_[component] = x
        return self.cdf(x_) - q
      x0 = self._model_marginal_dist.ppf(q)
      return root_scalar(target_func, x0 = x0, x1 = x0+1, method="secant").root


    #### first component
    probability_range = np.linspace(0.01,0.99, 99)
    empirical_quantiles = np.quantile(z1, probability_range)
    model_quantiles = np.array([marginal_ppf(p) for p in probability_range])

    axs[1,1].scatter(model_quantiles, empirical_quantiles, color = self._figure_color_palette[0])
    min_x, max_x = min(model_quantiles), max(model_quantiles)
    #axs[0,1].set_aspect('equal', 'box')
    axs[1,1].title.set_text('Q-Q plot (1st component)')
    axs[1,1].set_xlabel('model quantiles')
    axs[1,1].set_ylabel('Data quantiles')
    axs[1,1].grid()
    axs[1,1].plot([min_x,max_x],[min_x,max_x], linestyle="--", color="black")

    #### second component
    empirical_quantiles = np.quantile(z2, probability_range)
    model_quantiles = np.array([marginal_ppf(p, component=1) for p in probability_range])

    axs[2,0].scatter(model_quantiles, empirical_quantiles, color = self._figure_color_palette[0])
    min_x, max_x = min(model_quantiles), max(model_quantiles)
    #axs[0,1].set_aspect('equal', 'box')
    axs[2,0].title.set_text('Q-Q plot (2nd component)')
    axs[2,0].set_xlabel('model quantiles')
    axs[2,0].set_ylabel('Data quantiles')
    axs[2,0].grid()
    axs[2,0].plot([min_x,max_x],[min_x,max_x], linestyle="--", color="black")

    plt.show()

  def simulate(self, size: int):
    """Simulate exceedances
    
    Args:
        size (int): Description
    
    Returns:
        TYPE: Description
    """

    # exceedance subregions:
    # r1 => exceedance in second component only, r2 => exceedance in both componenrs, r3 => exceedance in first component only
    p1 = self._model_marginal_dist.cdf(threshold) - mv_gaussian.cdf([threshold,threshold], cov = cov)
    p2 = 1 - self._model_marginal_dist.cdf(threshold) - mv_gaussian.cdf([threshold,threshold], cov = cov) - p1
    p3 = 1 - mv_gaussian.cdf([threshold,threshold], cov = cov) - (p1+p2)

    p = np.array([p1,p2,p3])
    p = p/np.sum(p)

    # compute number of samples per subregion
    n1, n2, n3 = np.random.multinomial(n=size, pvals = p, size=1)[0]

    r1_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = cov, 
      lb = np.array([-np.Inf, threshold]),
      ub = np.array([threshold, np.Inf])).sample(n1)

    r2_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = cov, 
      lb = np.array([threshold, threshold]),
      ub = np.array([np.Inf, np.Inf])).sample(n2)

    r3_samples = tmvn(
      mu=np.zeros((2,)), 
      cov = cov, 
      lb = np.array([threshold, -np.Inf]),
      ub = np.array([np.Inf, threshold])).sample(n2)
    
    samples = np.concatenate([r1_samples, r2_samples, r3_samples], axis = 0)
    x = samples[:,0]
    y = samples[:,1]

    return self.model_to_data_dist(x,y)

  def cdf(self, data: t.Union[np.ndarray,t.Iterable]):
    x, y = self.data_to_model_marginal_dist(data)

    u1 = np.clip(x, a_min=0, a_max = self._model_marginal_dist.ppf(self.quantile_threshold))
    u2 = np.clip(y, a_min=0, a_max = self._model_marginal_dist.ppf(self.quantile_threshold))

    return (self.uncond_cdf(self.alpha, x, y) - self.uncond_cdf(self.alpha, u1, u2))/(1 - self.uncond_cdf(self.alpha, self.quantile_threshold, self.quantile_threshold))
     

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
