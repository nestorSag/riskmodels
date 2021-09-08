"""
This module contains copula models for exceedances of bivariate distributions. 
  
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
import copy

import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import numpy as np
import emcee

from scipy.stats import genpareto as gpdist
from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.stats import gaussian_kde as kde
from scipy.signal import fftconvolve


from pydantic import BaseModel, ValidationError, validator, PositiveFloat
from functools import reduce


class ExceedanceCopula(BaseModel):

  """Main interface for exceedance copulas, which are defined on a region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$. 
  """

  threshold: float
  data: np.ndarray

  _allowed_scalar_types = (int, float, np.int64, np.int32, np.float32, np.float64)
  _figure_color_palette = ["tab:cyan", "deeppink"]
  _error_tol = 1e-6

  class Config:
    arbitrary_types_allowed = True

  @abstractmethod
  def pdf(self, x: np.ndarray, y: np.ndarray):
    """Returns the pdf value evaluated on (x, y)
    
    Args:
        x (np.ndarray): Values at first component
        y (np.ndarray): Values at second component
    """
    pass

  @abstractmethod
  def cdf(self, x: np.ndarray, y: np.ndarray):
    """Returns the cdf value evaluated on (x, y)
    
    Args:
        x (np.ndarray): Values at first component
        y (np.ndarray): Values at second component
    """
    pass

  @abstractmethod
  def simulate(self, size: int):
    """Returns simulated values from the model
    
    Args:
        size (int): Number of values to simulate
  
    """
    pass

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



class Gumbel(ExceedanceCopula):

  """This model is a Gumbel-Hougaard copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. This model also corresponds to a bivariate logistic generalised Pareto distribution in the context of extreme value theory. Consequently, it is assumed that there is asymptotic dependence between components, this is, extreme values across components are strongly associated.
  """
  
  alpha: float

  @validator("alpha")
  def validate_alpha(cls, alpha):
    if alpha < 0 or alpha > 1:
      raise TypeError("alpha must be in the open interval (0,1) ")

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):
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
  def loglik(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):

    return np.sum(logpdf(alpha, threshold, x,y))

  @classmethod
  def uncond_cdf(cls, alpha: float, x: np.ndarray, y: np.ndarray):
    return np.exp(-(np.exp(-x/alpha) + np.exp(-y/alpha))**(alpha))

  def cdf(self, x: np.ndarray, y: np.ndarray):
    u1 = np.clip(x, a_min=0, a_max = self.threshold)
    u2 = np.clip(y, a_min=0, a_max = self.threshold)

    return (self.uncond_cdf(self.alpha, x, y) - self.uncond_cdf(self.alpha, u1, u2))/(1 - self.uncond_cdf(self.alpha, self.threshold, self.threshold))
     

  def pdf(self, x: np.ndarray, y:n p.ndarray):
    return np.exp(self.logpdf(self.alpha, self.threshold, x, y))

  @classmethod
  def fit(cls, data: np.ndarray, threshold: float, return_results):

    if not isinstance(threshold, float) or threshold <= 0 or threshold >= 1:
      raise ValueError("threshold must be in the open interval (0,1)")

    x = data[:,0]
    y = data[:,1]
    n = len(x)

    if min(x) < 0 or max(x) > 1 or min(y) < 0 or max(y) > 1:
      raise ValueError("Data must be in copula (i.e. unit uniform) scale")

    # get threshold exceedances
    exs_idx = np.logical_or(x >= threshold, y >=threshold)
    x = x[exs_idx]
    y = y[exs_idx]

    # map to standard Gumbel scale
    z1 = -np.log(-np.log(x))
    z2 = -np.log(-np.log(y))

    def loss(phi, x, y):
      alpha = 1.0/(1 + np.exp(-phi))
      return cls.loglik(alpha, threshold, x, y)/n

    res = minimize(
      fun=loss, 
      x0 = 0.0,
      method = "BFGS")

    if return_opt_results:
      warn.warnings("Returning raw results for rescaled exceedance data (sdev ~ 1).")
      return res
    else:
      phi = res.x[0]
      alpha = 1.0/(1 + np.exp(-phi))
      data = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))], axis=1)

    return cls(
      threshold = threshold,
      alpha = alpha,
      data = data)

  @classmethod
  def hessian(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):
    delta = 1e-3
    fisher = (cls.loglik(alpha - delta, threshold, x, y) -2*cls.loglik(alpha, threshold, x, y) + cls.loglik(alpha + delta, threshold, x, y))/(n*delta**2)

    return -1.0/fisher

  def plot_diagnostics(self):

    x = self.data[:,0]
    y = self.data[:,1]
    n = len(x)
    # map to standard Gumbel scale
    z1 = -np.log(-np.log(x))
    z2 = -np.log(-np.log(y))


    fig, axs = plt.subplots(2, 2)

    ####### loglikelihood plot
    sdev = np.sqrt(self.hessian(self.alpha, self.threshold, z1, z2))
    grid = np.linspace(self.alpha - sdev, self.alpha + sdev, 100)
    grid = grid[np.logical_and(grid > 0, grid < 1)]
    ll = np.array([self.loglik(alpha, self.threshold, z1, z2) for alpha in grid])

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
    pickands_data = -np.log(self.data[np.logical_and(x > self.threshold, y > self.threshold)])
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


    plt.show()

  def simulate(self, size: int):
    alpha = self.alpha
    ### simulate maximum component: z = max(x,y)
    u = np.random.uniform(size=size)
    maxima = -np.log(-np.log(1 - exs_prob*(1-u))) + alpha*np.log(2)

    ###simulate difference between maxima and minima r = max(x,y) - min(x,y)
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

    return np.concatenate([x,y], axis=1)








class Gaussian(ExceedanceCopula):

  """This model is a Gaussian copula model restricted to the region of the form $U \\nleq U$, or equivalently $\\max\\{U_1,U_2\\} > u$, which represents threshold exceedances above $u$ in at least one component. As the Gaussian copula is asymptotically independent for any correlation value lower than one, it is assumed that extreme values across components in the data are weakly associated and tend to occur progressively independently at more extreme levels.
  """
  
  alpha: float

  @validator("alpha")
  def validate_alpha(cls, alpha):
    if alpha < 0 or alpha > 1:
      raise TypeError("alpha must be in the open interval (0,1) ")

  @classmethod
  def logpdf(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):
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
  def loglik(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):

    return np.sum(logpdf(alpha, threshold, x,y))

  @classmethod
  def uncond_cdf(cls, alpha: float, x: np.ndarray, y: np.ndarray):
    return np.exp(-(np.exp(-x/alpha) + np.exp(-y/alpha))**(alpha))

  def cdf(self, x: np.ndarray, y: np.ndarray):
    u1 = np.clip(x, a_min=0, a_max = self.threshold)
    u2 = np.clip(y, a_min=0, a_max = self.threshold)

    return (self.uncond_cdf(self.alpha, x, y) - self.uncond_cdf(self.alpha, u1, u2))/(1 - self.uncond_cdf(self.alpha, self.threshold, self.threshold))
     

  def pdf(self, x: np.ndarray, y:n p.ndarray):
    return np.exp(self.logpdf(self.alpha, self.threshold, x, y))

  @classmethod
  def fit(cls, data: np.ndarray, threshold: float, return_results):

    if not isinstance(threshold, float) or threshold <= 0 or threshold >= 1:
      raise ValueError("threshold must be in the open interval (0,1)")

    x = data[:,0]
    y = data[:,1]
    n = len(x)

    if min(x) < 0 or max(x) > 1 or min(y) < 0 or max(y) > 1:
      raise ValueError("Data must be in copula (i.e. unit uniform) scale")

    # get threshold exceedances
    exs_idx = np.logical_or(x >= threshold, y >=threshold)
    x = x[exs_idx]
    y = y[exs_idx]

    # map to standard Gumbel scale
    z1 = -np.log(-np.log(x))
    z2 = -np.log(-np.log(y))

    def loss(phi, x, y):
      alpha = 1.0/(1 + np.exp(-phi))
      return cls.loglik(alpha, threshold, x, y)/n

    res = minimize(
      fun=loss, 
      x0 = 0.0,
      method = "BFGS")

    if return_opt_results:
      warn.warnings("Returning raw results for rescaled exceedance data (sdev ~ 1).")
      return res
    else:
      phi = res.x[0]
      alpha = 1.0/(1 + np.exp(-phi))
      data = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))], axis=1)

    return cls(
      threshold = threshold,
      alpha = alpha,
      data = data)

  @classmethod
  def hessian(cls, alpha: float, threshold: float, x: np.ndarray, y: np.ndarray):
    delta = 1e-3
    fisher = (cls.loglik(alpha - delta, threshold, x, y) -2*cls.loglik(alpha, threshold, x, y) + cls.loglik(alpha + delta, threshold, x, y))/(n*delta**2)

    return -1.0/fisher

  def plot_diagnostics(self):

    x = self.data[:,0]
    y = self.data[:,1]
    n = len(x)
    # map to standard Gumbel scale
    z1 = -np.log(-np.log(x))
    z2 = -np.log(-np.log(y))


    fig, axs = plt.subplots(2, 2)

    ####### loglikelihood plot
    sdev = np.sqrt(self.hessian(self.alpha, self.threshold, z1, z2))
    grid = np.linspace(self.alpha - sdev, self.alpha + sdev, 100)
    grid = grid[np.logical_and(grid > 0, grid < 1)]
    ll = np.array([self.loglik(alpha, self.threshold, z1, z2) for alpha in grid])

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
    pickands_data = -np.log(self.data[np.logical_and(x > self.threshold, y > self.threshold)])
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


    plt.show()

  def simulate(self, size: int):
    alpha = self.alpha
    ### simulate maximum component: z = max(x,y)
    u = np.random.uniform(size=size)
    maxima = -np.log(-np.log(1 - exs_prob*(1-u))) + alpha*np.log(2)

    ###simulate difference between maxima and minima r = max(x,y) - min(x,y)
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

    return np.concatenate([x,y], axis=1)