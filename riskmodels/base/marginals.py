from __future__ import annotations

import logging
import time
import typing as t
import traceback
import warnings
from argparse import Namespace
from abc import ABC, abstractmethod
from multiprocessing import Pool

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import emcee

from scipy.stats import genpareto as gpdist
from scipy.optimize import LinearConstraint, minimize

from pydantic import BaseModel, ValidationError, validator, PositiveFloat


# class BaseWrapper(BaseModel):

#   class Config:
#     arbitrary_types_allowed = True


class BaseDistribution(BaseModel):

  """Base interface for available data model types
  """
  _allowed_scalar_types = (int, float)
  _figure_color_palette = ["tab:cyan", "deeppink"]
  _error_tol = 1e-6

  class Config:
    arbitrary_types_allowed = True

  @abstractmethod
  def simulate(self, size: int) -> np.ndarray:
    """Produces simulated values from model
    
    Args:
        n (int): Number of samples
    """
    pass

  @abstractmethod
  def moment(self, n: int) -> float:
    """Calculates non-centered moments
    
    Args:
        n (int): moment order
    """
    pass

  @abstractmethod
  def ppf(self, q: float) -> float:
    """Calculates quantile for a given probability value
    
    Args:
        q (float): probability level
    """
    pass

  def mean(self) -> float:
    """Calculates the expected value
    
    """
    return self.moment(1)

  def std(self) -> float:
    """Calculates the standard deviation
    
    """
    return np.sqrt(self.moment(2) - self.mean()**2)

  @abstractmethod
  def cdf(self, x:float) -> float:
    """Evaluates the cumulative probability function
    
    Args:
        x (float): a point in the support
    """
    pass

  @abstractmethod
  def pdf(self, x:float) -> float:
    """Calculates the probability mass or probability density function
    
    Args:
        x (float): a point in the support
    """
    pass

  def histogram(self, size: int = 1000) -> None:
    """Plots a histogram of a simulated sample
    
    Args:
        size (int, optional): sample size
    
    """

    #show histogram from 1k samples
    samples = self.simulate(size=size)
    plt.hist(samples, bins=25, edgecolor="white", color=self._figure_color_palette[0])
    plt.title("Histogram from 1K simulated samples")
    plt.show()

  def cvar(self, p: float):
    """Calculates conditional value at risk for a probability level p, defined as the mean conditioned to an exceedance above the p-quantile.
    
    Args:
        p (float): Description
    
    Returns:
        TYPE: Description
    
    Raises:
        ValueError: Description
    """
    if p < 0 or p >= 1:
      raise ValueError("p must be in the open interval (0,1)")

    return (self >= self.ppf(p)).mean()





class GPTail(BaseDistribution):
  """Representation of a fitted Generalized Pareto distribution as a tail model
  
  Args:
      threshold (float): modeling threshold
      shape (float): fitted shape parameter
      scale (float): fitted scale parameter
      data (np.array, optional): exceedance
  """

  threshold: float
  shape: float
  scale: PositiveFloat
  data: t.Optional[np.ndarray]

  @property
  def endpoint(self):
    return self.threshold - self.scale/self.shape if self.shape < 0 else np.Inf
  

  @property
  def model(self):
    return gpdist(loc=self.threshold, c=self.shape, scale=self.scale)

  # @classmethod
  # def fit(cls, data: np.array, threshold: float) -> GPTail:
  #   exceedances = data[data > threshold]
  #   shape, _, scale = gpdist.fit(exceedances, floc=threshold)
  #   return cls(
  #     threshold = threshold,
  #     shape = shape,
  #     scale = scale,
  #     data = exceedances)

  def __add__(self, other: float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"+ is implemented for instances of types: {self._allowed_scalar_types}")

    return GPTail(
      threshold = self.threshold + other,
      shape = self.shape,
      scale = self.scale,
      data = self.data + other if self.data is not None else None)

  __radd__ = __add__

  def __sub__(self, other: float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"- is implemented for instances of types: {self._allowed_scalar_types}")

    return self.__add__(-other)

  __rsub__ = __sub__

  def __ge__(self, other: float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f">= and > are implemented for instances of types: {self._allowed_scalar_types}")

    if other >= self.endpoint:
      raise ValueError(f"No probability mass above endpoint ({self.endpoint}); conditional distribution X >= {other} does not exist")

    if other <= self.threshold:
      return self
    else:
      # condition on empirical data if applicable
      new_data = self.data[self.data >= other] if self.data is not None else None
      # if no observed data is above threshold, discards
      new_data = None if len(new_data) == 0 else new_data
      if new_data is None:
        warnings.warn(f"No observed data above {other}; setting data to None in conditional model.")
      
      return GPTail(
        threshold=other,
        shape = self.shape,
        scale = self.scale + self.shape*(other - self.threshold),
        data = new_data)

  def __gt__(self, other: float) -> GPTail:

    return self.__ge__(other)

  def __mul__(self, other: float) -> GPTail:

    if not isinstance(other, self._allowed_scalar_types) or other <= 0:
      raise TypeError(f"* is implemented for positive instances of: {self._allowed_scalar_types}")

    new_data = other*self.data if self.data is not None else None
    if new_data is None:
      warnings.warn(f"No observed data above {other}; setting data to None in conditional model.")

    return GPTail(
      threshold= other*self.threshold,
      shape = self.shape,
      scale = other*self.scale,
      data = new_data)

  __rmul__ = __mul__

  def simulate(self, size: int) -> np.ndarray:
    return self.model.rvs(size=size)

  def moment(self, n: int) -> float:
    return self.model.moment(n)

  def ppf(self, q: float) -> float:
    return self.model.ppf(q)

  def cdf(self, x:float) -> float:
    return self.model.cdf(x)

  def pdf(self, x:float) -> float:
    return self.model.pdf(x)

  def std(self):
    return self.model.std()

  def mle_cov(self) -> np.ndarray:
    """Returns the estimated parameter covariance matrix evaluated at the fitted parameters
    
    Returns:
        np.ndarray: Covariance matrix
    """

    if self.data is None:
      raise ValueError("exceedance data not provided for this instance of GPTail; covariance matrix can't be estimated")
    else:
      hess = self.loglik_hessian([self.scale,self.shape], threshold=self.threshold, data=self.data)
      return np.linalg.inv(-hess/len(self.data))

  @classmethod
  def loglik(cls, params: t.List[float], threshold: float, data: np.ndarray) -> float:
    """Returns the negative log-likelihood for a Generalised Pareto model
    
    Args:
        *params (t.List[float]): Vector parameter with scale and shape values, in that order
        threshold (float): model threshold
        data (np.ndarray): exceedance data
    
    Returns:
        float
    """
    scale, shape = params
    return np.sum(gpdist.logpdf(data, loc=threshold, c=shape, scale=scale))

  @classmethod
  def loglik_grad(cls, params: t.List[float], threshold: float, data: np.ndarray) -> np.ndarray:
    """Returns the gradient of the negative log-likelihood for a Generalised Pareto model
    
    Args:
        *params (t.List[float]): Vector parameter with scale and shape values, in that order
        threshold (float): model threshold
        data (np.ndarray): exceedance data
    
    Returns:
        np.ndarray: gradient array
    """
    scale, shape = params
    y = data - threshold

    if not np.isclose(shape, 0, atol=cls._error_tol):
      grad_scale = np.sum((y - scale)/(scale*(y*shape + scale)))
      grad_shape = np.sum(-((y*(1 + shape))/(shape*(y*shape + scale))) + np.log(1 + (y*shape)/scale)/shape**2)
    else:
      grad_scale = np.sum((y-scale)/scale**2)
      grad_shape = np.sum(y*(y-2*scale)/(2*scale**2))

    return np.array([grad_scale,grad_shape])


  @classmethod
  def loglik_hessian(cls, params: t.List[float], threshold: float, data: np.ndarray) -> np.ndarray:
    """Returns the Hessian matrix of the negative log-likelihood for a Generalised Pareto model
    
    Args:
        *params (t.List[float]): Vector parameter with scale and shape values, in that order
        threshold (float): model threshold
        data (np.ndarray): exceedance data above the threshold
    
    Returns:
        np.ndarray: hessian matrix array
    """
    scale, shape = params
    y = data - threshold

    if not np.isclose(shape, 0, atol=cls._error_tol):
      d2scale = np.sum(-(1/(shape*scale**2)) + (1 + shape)/(shape*(y*shape + scale)**2))

      #d2shape = (y (3 y ξ + y ξ^2 + 2 σ))/(ξ^2 (y ξ + σ)^2) - (2 Log[1 + (y ξ)/σ])/ξ^3
      d2shape = np.sum((y*(3*y*shape + y*shape**2 + 2*scale))/(shape**2*(y*shape + scale)**2) - (2*np.log(1 + (y*shape)/scale))/shape**3)

      #dscale_dshape = (y (-y + σ))/(σ (y ξ + σ)^2)
      dscale_dshape = np.sum((y*(-y + scale))/(scale*(y*shape + scale)**2))
    else:
      d2scale = np.sum((scale-2*y)/scale**3)
      dscale_dshape = np.sum(-y*(y-scale)/scale**3)
      d2shape = np.sum(y**2*(3*scale-2*y)/(3*scale**3))

    hessian = np.array([[d2scale,dscale_dshape],[dscale_dshape,d2shape]])

    return hessian

  @classmethod
  def logreg(cls, params: t.List[float]) -> float:
    """The regularisation terms are inspired in uninformative and zero-mean Gaussian priors for the scale and shape respectively, thus it is given by

    $$r(\\sigma, \\xi) = 0.5 \\cdot \\log(\\sigma) - 0.5 \\xi^2$$
    
    Args:
        params (t.List[float]): scale and shape parameters in that order
    
    Returns:
        float: Description
    """
    scale, shape = params

    return 0.5*(np.log(scale) + shape**2)

  @classmethod
  def logreg_grad(cls, params: t.List[float]) -> float:
    """Returns the gradient of the regularisation term
    
    Args:
        params (t.List[float]): scale and shape parameters in that order

    """
    scale, shape = params
    return 0.5*np.array([1.0/scale, 2*shape])

  @classmethod
  def logreg_hessian(cls, params: t.List[float]) -> float:
    """Returns the Hessian of the regularisation term
    
    Args:
        params (t.List[float]): scale and shape parameters in that order

    """
    scale, shape = params
    return 0.5*np.array([-1.0/scale**2,0,0,2]).reshape((2,2))

  @classmethod
  def loss(cls, params: t.List[float], threshold: float, data: np.ndarray) -> np.ndarray:
    """Calculate the loss function on the provided parameters; this is the sum of the (negative) data log-likelihood and (negative) regularisation terms for the scale and shape. Everything is divided by the number of data points.
    
    Args:
        params (t.List[float]): Description
        threshold (float): Model threshold; eequivalently, location parameter
        data (np.ndarray): exceedance data above threshold
    
    """
    scale, shape = params
    n = len(data)
    unnorm = cls.loglik(params, threshold, data) + cls.logreg(params)
    return -unnorm/n

  @classmethod
  def loss_grad(cls, params: t.List[float], threshold: float, data: np.ndarray) -> np.ndarray:
    """Calculate the loss function's gradient on the provided parameters
    
    Args:
        params (t.List[float]): Description
        threshold (float): Model threshold; eequivalently, location parameter
        data (np.ndarray): exceedance data above threshold
    
    """
    n = len(data)
    return -(cls.loglik_grad(params, threshold, data) + cls.logreg_grad(params))/n

  @classmethod
  def loss_hessian(cls, params: t.List[float], threshold: float, data: np.ndarray) -> np.ndarray:
    """Calculate the loss function's Hessian on the provided parameters
    
    Args:
        params (t.List[float]): Description
        threshold (float): Model threshold; eequivalently, location parameter
        data (np.ndarray): exceedance data above threshold
    
    """
    n = len(data)
    return -(cls.loglik_hessian(params, threshold, data) + cls.logreg_hessian(params))/n

  @classmethod
  def fit(
    cls, 
    data: np.ndarray, 
    threshold: float, 
    x0: np.ndarray = None,
    return_opt_results=False) -> t.Union[GPTail, sp.optimize.OptimizeResult]:
    """Fits a tail Generalised Pareto model using a constrained trust region method
    
    Args:
        data (np.ndarray): exceedance data above threshold
        threshold (float): Model threshold
        x0 (np.ndarray, optional): Initial guess for optimization; if None, the result of scipy.stats.genpareto.fit is used as a starting point.
        return_opt_results (bool, optional): If True, return the OptimizeResult object; otherwise return fitted instance of GPTail
    
    Returns:
        t.Union[GPTail, sp.optimize.OptimizeResult]: Description
    """
    exceedances = data[data > threshold]
    x_max = max(exceedances)

    constraints = LinearConstraint(
      A = np.array([[1/(x_max-threshold),1], [1, 0]]), 
      lb=np.zeros((2,)), 
      ub=np.Inf)

    if x0 is None:
      # use default scipy fitter to get initial estimate
      # this is almost always good enough
      shape, _, scale = gpdist.fit(exceedances, floc=threshold)
      x0 = np.array([scale, shape])

    loss_func = lambda params: GPTail.loss(params, threshold, exceedances)
    loss_grad = lambda params: GPTail. loss_grad(params, threshold, exceedances)
    loss_hessian = lambda params: GPTail.loss_hessian(params, threshold, exceedances)

    res = minimize(
      fun=loss_func, 
      x0 = x0,
      method = "trust-constr",
      jac = loss_grad,
      hess = loss_hessian)

    if return_opt_results:
      return res
    else:
      scale, shape = list(res.x)

      return cls(
        threshold = threshold,
        scale = scale,
        shape = shape,
        data = data)








class Discrete(BaseDistribution):

  """Model for an discrete (empirical) probability distribution, induced by a sample of data.

  Args:
      support (np.ndarray): distribution support
      pdf_values (np.ndarray): pdf array
      data (np.array, optional): data
  """

  support: np.ndarray
  pdf_values: np.ndarray
  data: t.Optional[np.ndarray]

  @validator("pdf_values", allow_reuse=True)
  def check_pdf_values(cls, pdf_values):
    if np.any(pdf_values < 0):
      raise ValidationError("There are negative pdf values")
    if not np.isclose(np.sum(pdf_values), 1, atol=cls._error_tol):
      raise ValidationError("pdf values don't sum 1")
    return pdf_values/np.sum(pdf_values)

  @property
  def is_valid(self):
    n = len(self.support)
    m = len(self.pdf_values)

    if n != m:
      raise ValueError("Lengths of support and pdf arrays don't match")

    if not np.all(np.roll(self.support, -1)[0:(n-1)] - self.support[0:(n-1)] > 0):
      raise ValueError("Support array must be in increasing order")
    return True 
  
  @property
  def pdf_lookup(self):
    """Mapping from values in the support to their probability mass
    
    """
    return {key: val for key, val in zip(self.support, self.pdf_values)}

  @property
  def min(self):
    """Minimum value in the support
    """
    return self.support[0]

  @property
  def max(self):
    """Maximum value in the support
    
    """
    return self.support[-1]

  @property
  def cdf_values(self):
    """Mapping from values in the support to their cumulative probability
    
    """
    x = np.cumsum(self.pdf_values)
    x[-1] = 1.0
    return x

  def __mul__(self, factor: float):

    if not isinstance(factor, self._allowed_scalar_types) or factor == 0:
      raise TypeError(f"multiplication is supported only for nonzero instances of type:{self._allowed_scalar_types}")

    return Discrete(support = factor*self.support, pdf_values = self.pdf_values, data = factor*self.data)

  __rmul__ = __mul__

  def __add__(self, other: t.Union[float, GPTail, Mixture]):

    if not isinstance(other, self._allowed_scalar_types + (GPTail, Mixture)):
      raise TypeError(f"+ is supported only for scalar, GPTail or Mixture instances")

    elif isinstance(other, self._allowed_scalar_types):
      return Discrete(support = self.support + other, pdf_values = self.pdf_values, data = self.data + other)

    elif isinstance(other, GPTail):

      indices = (self.pdf_values > 0).nonzero()[0]
      nn_support = self.pdf_values[indices]
      # mixed GPTails don't carry over exceedance data for efficient memory use
      dists = [GPTail(threshold=other.threshold + x, scale=other.scale, shape=other.shape) for x in nn_support]

      return Mixture(weights = nn_support, distributions = dists)

    else:
      return Mixture(weights=other.weights, distributions = [self + dist for dist in other.distributions])

  __radd__ = __add__

  def __neg__(self):

    return self.map(lambda x: -x)

  def __sub__(self, other: Discrete):

    if isinstance(other, [Discrete, float]):

      return self + (-other)

    else:
      raise TypeError("Subtraction is only defined for instances of Discrete or float ")

  __rsub__ = __sub__

  def __ge__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f">= is implemented for instances of types : {self._allowed_scalar_types}")

    index = self.support >= other

    return type(self)(
      support = self.support[index],
      pdf_values = self.pdf_values[index]/np.sum(self.pdf_values[index]), 
      data = self.data[self.data >= other])

  def __gt__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"> is implemented for instances of types: {self._allowed_scalar_types}")

    index = self.support > other

    return type(self)(
      support = self.support[index],
      pdf_values = self.pdf_values[index]/np.sum(self.pdf_values[index]), 
      data = self.data[self.data > other])

  def __le__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"<= is implemented for instances of type float: {self._allowed_scalar_types}")

    index = self.support <= other

    return type(self)(
      support = self.support[index],
      pdf_values = self.pdf_values[index]/np.sum(self.pdf_values[index]), 
      data = self.data[self.data <= other])

  def __lt__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"< is implemented for instances of type float: {self._allowed_scalar_types}")

    index = self.support < other

    return type(self)(
      support = self.support[index],
      pdf_values = self.pdf_values[index]/np.sum(self.pdf_values[index]), 
      data = self.data[self.data < other])


  def simulate(self, size: int):
    return np.random.choice(self.support, size=size, p=self.pdf_values)

  def moment(self, n: int):

    return np.sum(self.pdf_values * self.support**n)

  def ppf(self, q: float):

    if q < 0 or q > 1:
      raise ValueError(f"q needs to be in (0,1)")

    if q < self.cdf_values[0]:
      warnings.warn("q is lower than the smallest CDF value; returning -1.0")
      return -1.0
    else:
      return self.support[np.argmax(self.cdf_values >=q)]

  def cdf(self, x: int):

    if x < self.min:
      return 0.0
    elif x >= self.max:
      return 1.0
    else:
      first_nonlower = np.argmax(self.support >= x)
      return self.cdf_values[first_nonlower]


  def pdf(self, x: float):

    try:
      pdf_val = self.pdf_lookup[x]
      return pdf_val
    except KeyError as e:
      return 0.0

  def std(self):
    return np.sqrt(self.map(lambda x: x - self.mean()).moment(2))

  @classmethod
  def from_data(cls, data: np.array):

    support, unnorm_pdf = np.unique(data, return_counts=True)
    n = np.sum(unnorm_pdf)
    return cls(
      support=support, 
      pdf_values=unnorm_pdf/n, 
      data=data)

  def plot_mean_residual_life(self, threshold: float) -> None:
    """Produces a mean residual life plot for the tail of the distribution above a given threshold
    
    Args:
        threshold (float): threshold value
    
    """
    fitted = self.fit_tail_model(threshold)
    scale, shape = fitted.tail.scale, fitted.tail.shape

    x_vals = np.linspace(threshold, max(self.data))
    if shape >= 1:
      raise ValueError(f"Expectation is not finite: fitted shape parameter is {params.c}")
    y_vals = (scale + shape*x_vals)/(1-shape)
    plt.plot(x_vals,y_vals, color=self._figure_color_palette[0])
    plt.scatter(x_vals, y_vals, color=self._figure_color_palette[0])
    plt.title("Mean residual life plot")
    plt.xlabel("Threshold")
    plt.ylabel("Mean exceedance")
    plt.show()

  def fit_tail_model(self, threshold: float, bayesian=False, **kwargs) -> t.Union[DiscreteWithGPTail, DiscreteWithBayesianGPTail]:
    """Fits a tail GP model above a specified threshold and return the fitted semiparametric model
    
    Args:
        threshold (float): Threshold above which a Generalised Pareto distribution will be fitted
        bayesian (bool, optional): If True, fit model through Bayesian inference 
        **kwargs: Additional parameters passed to BayesianGPTail.fit or to GPTail.fit
    
    Returns:
        t.Union[DiscreteWithGPTail, DiscreteWithBayesianGPTail]
    
    """

    if threshold >= self.max:
      raise ValueError("Discrete pdf is 0 above the provided threshold. Select a lower threshold for estimation.")

    if self.data is None:
      raise ValueError("Data is not set for this distribution, so a tail model cannot be fitted. You can simulate from it and use the sampled data instead")
    else:
      data = self.data

    if bayesian:
      return DiscreteWithBayesianGPTail.from_data(data, threshold, **kwargs)
    else:
      return DiscreteWithGPTail.from_data(data, threshold, **kwargs)

  def map(self, f: t.Callable) -> Discrete:
    """Returns the distribution resulting from an arbitrary transformation
    
    Args:
        f (t.Callable): Target transformation; it should take a numpy array as input
    
    """
    dist_df = pd.DataFrame({"pdf": self.pdf_values, "support": f(self.support)})
    mapped_dist_df = dist_df.groupby("support").sum().reset_index().sort_values("support")

    return Discrete(
      support = np.array(mapped_dist_df["support"]),
      pdf_values = np.array(mapped_dist_df["pdf"]),
      data = f(self.data) if self.data is not None else None)

  def to_integer(self):
    """Convert to Integer distribution
    
    """
    integer_dist = self.map(lambda x: np.round(x))

    return Integer(
      support = integer_dist.support.astype(Integer._supported_types[0]),
      pdf_values = integer_dist.pdf_values,
      data = integer_dist.data)




class Integer(Discrete):

  """Discrete distribution with an integer support. This allows it to be convolved with other integer distribution to obtain the distribution of a sum of random variables, assuming independence between the summands. 
  """
  
  _supported_types = [np.int64, int]

  @validator("support", allow_reuse=True)
  def integer_support(cls, support):

    if support.dtype in cls._supported_types:
      return support
    else:
      raise ValidationError(f"Support entry types must be one of {self._supported_types}")

  
  def __add__(self, other: t.Union[float, int, Integer, GPTail, Mixture]):

    if isinstance(other, int):
      new_support = np.arange(min(self.support) + other, max(self.support) + other + 1)
      return Integer(
        support = new_support, 
        pdf_values = self.pdf_values, 
        data = self.data)

    if isinstance(other, Integer):
      new_support = np.arange(min(self.support) + min(other.support), max(self.support) + max(other.support) + 1)
      return Integer(
        support = new_support, 
        pdf_values = sp.signal.fftconvolve(self.pdf_values, other.pdf_values))

    else:
      return super().__add__(other)

  __radd__ = __add__

  @classmethod
  def from_data(cls, data: np.ndarray):

    data = np.array(data)
    if data.dtype not in self._supported_types:
      warnings.warn("Casting input data to integer values by rounding")
      data = data.astype(np.int64)

    return super().from_data(data).to_integer()

  def cdf(self, x: float):

    if x < self.min:
      return 0.0
    elif x >= self.max:
      return 1.0
    else:
      return self.cdf_values[int(x) - self.min]

  def pdf(self, x: float):

    if not isinstance(x, (int, np.int32, np.int64)) or x > self.max or x < self.min:
      return 0.0
    else:
      return self.pdf_values[x - self.min]







class Mixture(BaseDistribution):

  """This class represents a probability distribution given by a mixture of weighted continuous and discrete densities; as the base continuous densities can only be of class GPTail, this class is intended to represent either a semiparametric model with a Generalised Pareto tail, or the convolution of such a model with an integer distribution, as is the case for the power surplus distribution in power system reliability modeling.

  Args:
      distributions (t.List[BaseDistribution]): list of distributions that make up the mixture
      weights (np.ndarray): weights for each of the distribution. The weights must be a distribution themselves
  """
  
  distributions: t.List[BaseDistribution]
  weights: np.ndarray

  @validator("weights", allow_reuse=True)
  def check_weigths(cls, weights):
    if not np.isclose(np.sum(weights),1, atol=cls._error_tol):
      raise ValidationError(f"Weights don't sum 1 (sum = {np.sum(weights)})")
    elif np.any(weights <= 0):
      raise ValidationError("Negative or null weights are present")
    else:
      return weights


  def __mul__(self, factor: float):

    return Mixture(
      weights=self.weights, 
      distributions = [factor*dist for dist in self.distributions])

  def __add__(self, factor: float):

    return Mixture(
      weights=self.weights, 
      distributions = [factor + dist for dist in self.distributions])

  __rmul__ = __mul__

  def __ge__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f">= is implemented for instances of types : {self._allowed_scalar_types}")

    cond_weights = np.array([1 - dist.cdf(other) + (isinstance(dist,Discrete))*dist.pdf(other) for dist in self.distributions])
    new_weights = cond_weights*self.weights

    indices = (new_weights > 0).nonzero()[0]

    nz_weights = new_weights[indices]
    
    return Mixture(
      weights = nz_weights/np.sum(nz_weights), 
      distributions = [dist >= other for dist in self.distributions[nz_weights]])

  def __gt__(self, other:float):

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"> is implemented for instances of types : {self._allowed_scalar_types}")

    cond_weights = np.array([1 - dist.cdf(other) for dist in self.distributions])
    new_weights = cond_weights*self.weights

    indices = (new_weights > 0).nonzero()[0]

    nz_weights = new_weights[indices]
    
    return Mixture(
      weights = nz_weights/np.sum(nz_weights), 
      distributions = [dist > other for dist in self.distributions[nz_weights]])

    index = self.support > other

    return type(self)(
      self.support[index],
      self.pdf_values[index]/np.sum(self.pdf_values[index]), 
      self.data[self.data > other])


  def simulate(self, size: int) -> np.ndarray:
    
    n_samples = np.random.multinomial(n=size, pvals = self.weights, size=1)[0]
    indices = (n_samples > 0).nonzero()[0]
    samples = [dist.simulate(size=k) for dist, k in zip([self.distributions[k] for k in indices], n_samples[indices])]
    return np.concatenate(samples, axis=0)

  def moment(self, n: int) -> float:
    
    moments = [dist.moment(n) for dist in self.distributions]
    return np.dot(moments,self.weights)

  def ppf(self, q: float) -> float:

    def target_function(x):
      return self.cdf(x) - q

    ppfs =[dist.ppf(q) for dist in self.distributions]
    x0 = np.dot(self.weights, ppfs)

    return opt.root_scalar(target_function, x0 = x0, method="bisect").root
    
  def cdf(self, x:float) -> float:
    cdfs = [dist.cdf(x) for dist in self.distributions]
    return np.dot(cdfs,self.weights)

  def pdf(self, x:float) -> float:
    
    pdfs = [dist.pdf(x) for dist in self.distributions]
    return np.dot(pdfs,self.weights)








class DiscreteWithGPTail(Mixture):

  """Represents a semiparametric extreme value model with a fitted Generalized Pareto distribution above a certain threshold, and an discrete empirical distribution below it

  """

  @property
  def discrete(self):
    return self.distributions[0]

  @property
  def tail(self):
    return self.distributions[1]
  
  @property
  def threshold(self):
    return self.distributions[1].params.threshold

  @property
  def exs_prob(self):
    return self.weights[1]

  def ppf(self, q: float):
    if q <= 1 - self.exs_prob:
      return self.discrete.ppf(q/(1-self.exs_prob))
    else:
      return self.tail.ppf((q - (1-self.exs_prob))/self.exs_prob)

  @classmethod
  def from_data(cls, data: np.ndarray, threshold: float, **kwargs) -> discreteWithGPTail:
    """Fits a model from a given data array and threshold value
    
    Args:
        data (np.ndarray): Data 
        threshold (float): Threshold value to use for the tail model
        **kwargs: Additional arguments passed to GPTail.fit
    
    Returns:
        discreteWithGPTail: Fitted model
    """
    exs_prob = 1 - Discrete.from_data(data).cdf(threshold)

    exceedances = data[data > threshold]
    
    discrete = Discrete.from_data(data[data <= threshold])

    tail = GPTail.fit(data=exceedances, threshold=threshold, **kwargs)

    return cls(
      distributions = [discrete, tail],
      weights = np.array([1 - exs_prob, exs_prob]))

  def plot_diagnostics(self) -> None:
    """Produces diagnostic plots for the fitted model
    
    """
    if self.tail.data is None:
      raise ValueError("Exceedance data was not provided for this model.")

    fig, axs = plt.subplots(3, 2)


    #################### Profile log-likelihood

    # set profile intervals based on MLE variance
    mle_cov = self.tail.mle_cov()
    scale_bounds, shape_bounds = (self.tail.scale - np.sqrt(mle_cov[0,0]), self.tail.scale + np.sqrt(mle_cov[0,0])), (self.tail.shape - np.sqrt(mle_cov[1,1]), self.tail.shape + np.sqrt(mle_cov[1,1]))

    #set profile grids
    scale_grid = np.linspace(scale_bounds[0], scale_bounds[1], 50)
    shape_grid = np.linspace(shape_bounds[0], shape_bounds[1], 50)


    #declare profile functions
    scale_profile_func = lambda x: self.tail.loglik([x, self.tail.shape], self.tail.threshold, self.tail.data)
    shape_profile_func = lambda x: self.tail.loglik([self.tail.scale, x], self.tail.threshold, self.tail.data)

    loss_value = scale_profile_func(self.tail.scale)

    scale_profile = np.array([scale_profile_func(x) for x in scale_grid])
    shape_profile = np.array([shape_profile_func(x) for x in shape_grid])

    alpha = 2 if loss_value > 0 else 0.5

    # filter to almost-optimal values

    def filter_grid(grid, optimum):
      radius = 2*np.abs(optimum)
      return np.logical_and(np.logical_not(np.isnan(grid)), np.isfinite(grid), np.abs(grid - optimum) < radius)

    scale_filter = filter_grid(scale_profile, loss_value)
    shape_filter = filter_grid(shape_profile, loss_value)


    valid_scales = scale_grid[scale_filter]
    valid_scale_profile = scale_profile[scale_filter]

    axs[0,0].plot(valid_scales, valid_scale_profile, color=self._figure_color_palette[0])
    axs[0,0].vlines(x=self.tail.scale, ymin=min(valid_scale_profile), ymax = max(valid_scale_profile), linestyle="dashed", colors = self._figure_color_palette[1])
    axs[0,0].title.set_text('Profile scale log-likelihood')
    axs[0,0].set_xlabel('Scale')
    axs[0,0].set_ylabel('log-likelihood')

    valid_shapes = shape_grid[shape_filter]
    valid_shape_profile = shape_profile[shape_filter]

    axs[0,1].plot(valid_shapes, valid_shape_profile, color=self._figure_color_palette[0])
    axs[0,1].vlines(x=self.tail.shape, ymin=min(valid_shape_profile), ymax = max(valid_shape_profile), linestyle="dashed", colors = self._figure_color_palette[1])
    axs[0,1].title.set_text('Profile shape log-likelihood')
    axs[0,1].set_xlabel('Shape')
    axs[0,1].set_ylabel('log-likelihood')

    ######################## Log-likelihood surface ###############
    scale_grid, shape_grid = np.mgrid[
    min(valid_scales):max(valid_scales):2*np.sqrt(mle_cov[0,0])/50,
    shape_bounds[0]:shape_bounds[1]:2*np.sqrt(mle_cov[0,0])/50]

    scale_mesh, shape_mesh = np.meshgrid(
      np.linspace(min(valid_scales), max(valid_scales), 50), 
      np.linspace(min(valid_shapes), max(valid_shapes), 50))

    max_x = max(self.tail.data)

    z = np.empty(scale_mesh.shape)
    for i in range(scale_mesh.shape[0]):
      for j in range(scale_mesh.shape[1]):
        shape = shape_mesh[i,j]
        scale = scale_mesh[i,j]
        if shape < 0 and self.tail.threshold - scale/shape < max_x:
          z[i,j] = np.nan
        else:
          z[i,j] = self.tail.loglik([scale, shape], self.tail.threshold, self.tail.data)

    # negate z to recover true loglikelihood
    axs[1,0].contourf(scale_mesh, shape_mesh, z, levels = 15)
    axs[1,0].scatter([self.tail.scale], [self.tail.shape], color="darkorange", s=2)
    axs[1,0].annotate(text="MLE", xy = (self.tail.scale, self.tail.shape), color="darkorange")
    axs[1,0].title.set_text('Log-likelihood surface')
    axs[1,0].set_xlabel('Scale')
    axs[1,0].set_ylabel('Shape')



    ############## histogram vs density ################
    hist_data = axs[1,1].hist(self.tail.data, bins=25, edgecolor="white", color=self._figure_color_palette[0])

    range_min, range_max = min(self.tail.data), max(self.tail.data)
    x_axis = np.linspace(range_min, range_max, 100)
    pdf_vals = self.tail.pdf(x_axis)
    y_axis = hist_data[0][0] / pdf_vals[0] * pdf_vals

    axs[1,1].plot(x_axis, y_axis, color=self._figure_color_palette[1])
    axs[1,1].title.set_text("Data vs fitted density")
    axs[1,1].set_xlabel('Exceedance data')
    axs[1,1].yaxis.set_visible(False) # Hide only x axis
    #axs[0, 0].set_aspect('equal', 'box')


    ############# Q-Q plot ################
    probability_range = np.linspace(0.01,0.99, 99)
    discrete_quantiles = np.quantile(self.tail.data, probability_range)
    self_quantiles = self.tail.ppf(probability_range)

    axs[2,0].scatter(self_quantiles, discrete_quantiles, color = self._figure_color_palette[0])
    min_x, max_x = min(self_quantiles), max(self_quantiles)
    #axs[0,1].set_aspect('equal', 'box')
    axs[2,0].title.set_text('Q-Q plot')
    axs[2,0].set_xlabel('self quantiles')
    axs[2,0].set_ylabel('Data quantiles')
    axs[2,0].grid()
    axs[2,0].plot([min_x,max_x],[min_x,max_x], linestyle="--", color="black")

    ############ Mean return plot ###############
    scale, shape = self.tail.scale, self.tail.shape

    n_obs = len(self.discrete.data)+len(self.tail.data)

    exs_prob = self.exs_prob
    m = 10**np.linspace(np.log(1/exs_prob + 1)/np.log(10), 3,20)
    return_levels = self.tail.ppf(1 - 1/(exs_prob*m))

    axs[2,1].plot(m,return_levels,color=self._figure_color_palette[0])
    axs[2,1].set_xscale("log")
    axs[2,1].title.set_text('Return levels')
    axs[2,1].set_xlabel('1/frequency')
    axs[2,1].set_ylabel('Return level')
    axs[2,1].grid()

    try:
      #for this bit, look at An Introduction to Statistical selfing of Extreme Values, p.82
      mle_cov = self.tail.mle_cov()
      eigenvals, eigenvecs = np.linalg.eig(mle_cov)
      if np.all(eigenvals > 0):
        covariance = np.eye(3)
        covariance[1::,1::] = mle_cov
        covariance[0,0] = exs_prob*(1-exs_prob)/n_obs
        #
        return_stdevs = []
        for m_ in m:
          quantile_grad = np.array([
            scale*m_**(shape)*exs_prob**(shape-1),
            shape**(-1)*((exs_prob*m_)**shape-1),
            -scale*shape**(-2)*((exs_prob*m_)**shape-1)+scale*shape**(-1)*(exs_prob*m_)**shape*np.log(exs_prob*m_)
            ])
          #
          sdev = np.sqrt(quantile_grad.T.dot(covariance).dot(quantile_grad))
          return_stdevs.append(sdev)
        #
        axs[2,1].fill_between(m, return_levels - return_stdevs, return_levels + return_stdevs, alpha=0.2, color=self._figure_color_palette[1])
      else:
        warnings.warn("Covariance MLE matrix is not positive definite; it might be ill-conditioned")
    except Exception as e:
      warnings.warn(f"Confidence bands for return level could not be calculated; covariance matrix might be ill-conditioned; full trace: {traceback.format_exc()}")


    ############# MLE confidence regions #####################
    # try:
    #   mle_cov = self.tail.mle_cov()
    #   eigenvals, eigenvecs = np.linalg.eig(mle_cov)
    #   if np.all(eigenvals > 0):
    #     mean = np.array([self.tail.scale, self.tail.shape])
    #     #
    #     # bounds = [(
    #     #   sp.stats.norm.ppf(0.025, loc = mean[k], scale = mle_cov[k,k]),
    #     #   sp.stats.norm.ppf(0.975, loc = mean[k], scale = mle_cov[k,k])) for k in range(2)]
    #     #
    #     rayleigh_95ci = sp.stats.rayleigh.ppf(0.95)
    #     cov_sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ np.linalg.inv(eigenvecs)
    #     norm_sqrt = max(np.linalg.eig(cov_sqrt)[0])
    #     bound = norm_sqrt*rayleigh_95ci
    #     #
    #     bounds = [bound * np.array([-1,1]) for k in range(2)]
    #     grids = [np.linspace(b[0],b[1],50) for b in bounds]
    #     x, y = np.mgrid[
    #     bounds[0][0]:bounds[0][1]:(bounds[0][1]-bounds[0][0])/50, 
    #     bounds[1][0]:bounds[1][1]:(bounds[1][1]-bounds[1][0])/50]
    #     pos = np.dstack((x,y))
    #     #
    #     #find contour corresponding to 95% cumulative probability from the center
    #     rv = sp.stats.multivariate_normal(mean=mean, cov = mle_cov)
    #     ctr = axs[1,0].contour(x, y, rv.pdf(pos))
    #     fmt = {}
    #     for lvl in ctr.levels:
    #       if lvl > 0:
    #         mapped_norm = -np.log(lvl/(2*np.math.pi)**(-1)*np.linalg.det(mle_cov)**(0.5))
    #         fmt[lvl] = np.round(sp.stats.rayleigh.cdf(mapped_norm),2)
    #     #
    #     axs[1,0].clabel(ctr, ctr.levels[::2], inline=True, fmt=fmt, fontsize=10)
    #     axs[1,0].title.set_text('Approximate 95% MLE confidence region')
    #     axs[1,0].set_xlabel('Scale')
    #     axs[1,0].set_ylabel('Shape')
    #   else:
    #     warnings.warn("Covariance MLE matrix is not positive definite; it might be ill-conditioned")
    # except Exception as e:
    #   warnings.warn(f"MLE central confidence region could not be calculated; covariance matrix might be ill-conditioned; full trace: {traceback.format_exc()}")

    #plt.show()

    plt.tight_layout()
    plt.show()





class BayesianGPTail(BaseDistribution):

  """Generalised Pareto tail model which is fitted through Bayesian inference.

  Args:
      threshold (float): modeling threshold
      data (np.array, optional): exceedance data
      shape (np.ndarray): sample from posterior shape distribution
      scale (np.ndarray): sample from posterior scale distribution
  """
  
  threshold: float
  data: t.Optional[np.ndarray]
  shape_posterior: np.ndarray
  scale_posterior: np.ndarray

  #_self.posterior_trace = None

  @classmethod
  def fit(
    cls, 
    data: np.ndarray, 
    threshold: float, 
    max_posterior_samples: int = 1000,
    chain_length: int = 2000,
    plot_diagnostics: bool = True,
    n_walkers: int = 32,
    n_cores: int = 4,
    burn_in: int = 100,
    thinning: int = None) -> BayesianGPTail:
    """Fits a Generalised Pareto model through Bayesian inference using exceedance data, starting with flat, uninformative priors for both and sampling from posterior shape and scale parameter distributions.
    
    Args:
        data (np.ndarray): observational data
        threshold (float): modeling threshold; location parameter for Generalised Pareto model
        max_posterior_samples (int, optional): Maximum number of posterior samples to keep
        chain_length (int, optional): timesteps in each chain
        plot_diagnostics (bool, optional): If True, plots MCMC diagnostics
        n_walkers (int, optional): Number of concurrent paths to use
        n_cores (int, optional): Number of cores to use in parallelization
        burn_in (int, optional): Number of initial samples to discard
        thinning (int, optional): Thinning factor to reduce autocorrelation; if None, an automatic estimate from emcee's get_autocorr_time is used.
    
    Returns:
        BayesianGPTail: fitted model
    
    Deleted Parameters:
        parallel (bool, optional): If True, uses all cores in the machine to sample from posterior.
    """
    exceedances = data[data > threshold]
    x_max = max(data - threshold)

    # def log_likelihood(theta, data):
    #   scale, shape = theta
    #   return np.sum(gpdist.logpdf(data, c=shape, scale=scale, loc=threshold))

    def log_prior(theta):
      scale, shape = theta
      if shape > -scale/(x_max):
        return 0.0
      else:
        return -np.Inf

    def log_probability(theta, data):
      prior = log_prior(theta)
      if np.isfinite(prior):
        return prior + GPTail.loglik(theta, threshold, data)#log_likelihood(theta, data)
      else:
        return -np.Inf

    exceedances = data[data > threshold]
    ndim = 2
    # make initial guess
    shape, _, scale = gpdist.fit(exceedances, floc=threshold)
    x0 = np.array([scale, shape])

    # create random walkers
    pos =  x0 + 1e-4 * np.random.randn(n_walkers, ndim)

    with Pool(n_cores) as pool:
      sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=ndim, log_prob_fn=log_probability, args=(exceedances, ))
      sampler.run_mcmc(pos, chain_length, progress=True)

    samples = sampler.get_chain()

    tau = sampler.get_autocorr_time()
    thinning = int(np.round(np.mean(tau)))
    print(f"Using a thinning factor of {thinning} (from emcee.EnsembleSampler.get_autocorr_time)")
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning, flat=True)

    if flat_samples.shape[0] > max_posterior_samples:
      np.random.shuffle(flat_samples)
      flat_samples = flat_samples[0:max_posterior_samples,:]

    print(f"Got {flat_samples.shape[0]} posterior samples.")

    if plot_diagnostics:
      fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
      labels = ["scale", "shape"]
      for i in range(ndim):
          ax = axes[i]
          ax.plot(samples[:, :, i], alpha=0.3, color=cls._figure_color_palette[0])
          ax.set_xlim(0, len(samples))
          ax.set_ylabel(labels[i])
          if i == 0:
            ax.set_title("Chain mixing")
          ax.yaxis.set_label_coords(-0.1, 0.5)
      plt.show()


    scale_posterior = flat_samples[:,0]
    shape_posterior = flat_samples[:,1]

    return cls(
      threshold = threshold,
      data = exceedances,
      shape_posterior = shape_posterior,
      scale_posterior = scale_posterior)


  def __add__(self, other: float) -> BayesianGPTail:

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f"+ is implemented for instances of types: {self._allowed_scalar_types}")

    return BayesianGPTail(
      threshold = self.threshold + other,
      shape_posterior = self.shape,
      scale_posterior = self.scale,
      data = self.data + other if self.data is not None else None)

  __radd__ = __add__

  def __ge__(self, other: float) -> BayesianGPTail:

    if not isinstance(other, self._allowed_scalar_types):
      raise TypeError(f">= and > are implemented for instances of types: {self._allowed_scalar_types}")

    if other >= self.endpoint:
      raise ValueError(f"No probability mass above endpoint ({self.endpoint}); conditional distribution X >= {other} does not exist")

    if other <= threshold:
      return self
    else:
      # condition on empirical data if applicable
      new_data = self.data[self.data >= other] if self.data is not None else None
      # if no observed data is above threshold, discard
      new_data = None if len(new_data) == 0 else new_data

      if new_data is None:
        warnings.warn(f"No observed data above {other}; setting data to None in conditional model.")

      return BayesianGPTail(
        threshold=other,
        shape_posterior = self.shape,
        scale_posterior = self.scale_posterior + self.shape_posterior*(other - self.threshold),
        data = new_data)

  def __gt__(self, other: float) -> BayesianGPTail:

    return self.__ge__(other)

  def __mul__(self, other: float) -> BayesianGPTail:

    if not isinstance(other, self._allowed_scalar_types) or other <= 0:
      raise TypeError(f"* is implemented for positive instances of: {self._allowed_scalar_types}")

    new_data = other*self.data if self.data is not None else None
    if new_data is None:
      warnings.warn(f"No observed data above {other}; setting data to None in conditional model.")

    return BayesianGPTail(
      threshold= other*self.threshold,
      shape_posterior= self.shape_posterior,
      scale_posterior = other*self.scale_posterior,
      data = new_data)

  def sample_posterior(self, size: int) -> t.Tuple[np.array, np.array]:
    indices = np.random.choice(a = np.arange(len(self.shape_posterior)),size=size)

    shapes = self.shape_posterior[indices]
    scales = self.scale_posterior[indices]

    return scales, shapes

  def simulate(self, size: int) -> np.ndarray:
    scales, shapes = self.sample_posterior(size=size)
    return gpdist.rvs(size=size, c=shapes, scale=scales, loc=self.threshold)

  def ppf(self, q: float, return_samples=False) -> t.Union[float, np.array]:
    vals = gpdist.ppf(q, c=self.shape_posterior, scale=self.scale_posterior, loc=self.threshold)
    if return_samples:
      return vals
    else:
      return np.mean(vals)

  def cdf(self, x:float, return_samples=False) -> t.Union[float, np.array]:
    vals = gpdist.cdf(x, c=self.shape_posterior, scale=self.scale_posterior, loc=self.threshold)
    if return_samples:
      return vals
    else:
      return np.mean(vals)

  def pdf(self, x:float, return_samples=False) -> t.Union[float, np.array]:
    vals = gpdist.pdf(x, c=self.shape_posterior, scale=self.scale_posterior, loc=self.threshold)
    if return_samples:
      return vals
    else:
      return np.mean(vals)

  def std(self, return_samples=False) -> t.Union[float, np.array]:
    if np.any(self.shape_posterior >= 0.5):
      raise ValueError("Some samples from the posterior shape parameter are larger than 0.5; the variance is infinite.")
    vals = gpdist.std(c=self.shape_posterior, scale=self.scale_posterior, loc=self.threshold)
    if return_samples:
      return vals
    else:
      return np.mean(vals)

  def mean(self, return_samples=False) -> t.Union[float, np.array]:
    if np.any(self.shape_posterior >= 1):
      raise ValueError("Some samples from the posterior shape parameter are larger than 1; the mean is infinite.")
    vals = gpdist.mean(c=self.shape_posterior, scale=self.scale_posterior, loc=self.threshold)
    if return_samples:
      return vals
    else:
      return np.mean(vals)

  def moment(self, n: int, return_samples=False) -> t.Union[float, np.array]:
    if np.any(self.shape_posterior >= 1/n):
      raise ValueError(f"Some samples from the posterior shape parameter are larger than 1/{n}; the {n}-moment is infinite.")
    vals = np.array([gpdist.moment(c=shape, scale=scale, loc=self.threshold) for scale, shape in zip(self.scale_posterior, shape_posterior)])
    if return_samples:
      return vals
    else:
      return np.mean(vals)



class DiscreteWithBayesianGPTail(DiscreteWithGPTail):

  """Semiparametric Bayesian model with an empirical data distribution below a specified threshold and a Generalised Pareto exceedance model above it, fitted through Bayesian inference.
  """
  
  @classmethod
  def from_data(
    cls, 
    data: np.ndarray, 
    threshold: float, 
    **kwargs) -> DiscreteWithBayesianGPTail:
    """Fits a Generalied Pareto tail model from a given data array and threshold value, using Jeffrey's priors 
    
    Args:
        data (np.ndarray): data array 
        threshold (float): Threshold value to use for the tail model
        n_posterior_samples (int): Number of samples from posterior distribution
        **kwargs: Additional arguments to be passed to BayesianGPTail.fit
    
    Returns:
        DiscreteWithBayesianGPTail: Fitted model
    """
    exs_prob = 1 - Discrete.from_data(data).cdf(threshold)

    exceedances = data[data > threshold]

    discrete = Discrete.from_data(data[data <= threshold])

    tail = BayesianGPTail.fit(data = exceedances, threshold = threshold, **kwargs)

    return cls(
      weights = np.array([1 -exs_prob, exs_prob]),
      distributions = [discrete, tail])

    
# class GPTailMixture(BaseDistribution):

#   weights: np.ndarray
#   offsets: np.ndarray
#   base_distribution: GPTail

#   # @abstractmethod
#   # def simulate(self, size: int) -> np.ndarray:
#   #   """Produces simulated values from model
    
#   #   Args:
#   #       n (int): Number of samples
#   #   """
#   #   pass

#   def moment(self, n: int) -> float:
#     vals = np.array([gpdist.moment(
#       n,
#       loc=self.base_distribution.threshold + x, 
#       c=self.base_distribution.shape, 
#       scale=self.base_distribution.scale) for x in self,offsets])

#     return np.dot(self.weights, vals)

#   @abstractmethod
#   def ppf(self, q: float) -> float:
#     x0 = gpdist.ppf(
#       q,
#       loc=self.base_distribution.threshold + self.offsets, 
#       c=self.base_distribution.shape, 
#       scale=self.base_distribution.scale)

#     return np.dot(self.weights, vals)

#   def mean(self) -> float:
#     vals = gpdist.mean(
#       loc=self.base_distribution.threshold + self.offsets, 
#       c=self.base_distribution.shape, 
#       scale=self.base_distribution.scale)

#     return np.dot(self.weights, vals)


#   def cdf(self, x:float) -> float:
#     vals = gpdist.cdf(
#       x,
#       loc=self.base_distribution.threshold + self.offsets, 
#       c=self.base_distribution.shape, 
#       scale=self.base_distribution.scale)

#     return np.dot(self.weights, vals)

#   def pdf(self, x:float) -> float:
    
#     vals = gpdist.pdf(
#       x,
#       loc=self.base_distribution.threshold + self.offsets, 
#       c=self.base_distribution.shape, 
#       scale=self.base_distribution.scale)

#     return np.dot(self.weights, vals)