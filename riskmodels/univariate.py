"""
This module contains univariate models for risk analysis, namely, empirical (discrete) distributions and semiparametric distributions with Generalised Pareto tail models. Both MLE-based and Bayesian estimation are supported for the GP models; useful diagnostic plots are available for fitted models, and exceedance conditional distributions of the type X | X > u for a fitted model X are implemented through the >= and > operators for all model classes, as well as scalar addition and (positive) rescaling, Finally, Binned distributions with integer support are available and can be convolved to get the distribution of a sum of independent integer random variables. This is useful for risk models in energy procurement.
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
import matplotlib

import numpy as np
import emcee

from scipy.stats import genpareto as gpdist
from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.stats import gaussian_kde as kde
from scipy.signal import fftconvolve


from pydantic import BaseModel, ValidationError, validator, PositiveFloat
from functools import reduce

import statsmodels.distributions.empirical_distribution as ed

# class BaseWrapper(BaseModel):

#   class Config:
#     arbitrary_types_allowed = True


class BaseDistribution(BaseModel):

    """Base interface for available data model types"""

    _allowed_scalar_types = (int, float, np.int64, np.int32, np.float32, np.float64)
    _figure_color_palette = ["tab:cyan", "deeppink"]
    _error_tol = 1e-6

    def __repr__(self):
        return "Base distribution object"

    def __str__(self):
        return self.__repr__()

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
        """Calculates the corresponding quantile for a given probability value

        Args:
            q (float): probability level
        """
        pass

    def mean(self, **kwargs) -> float:
        """Calculates the expected value

        Args:
            **kwargs: Additional arguments passed to `moments`. This is needed for Bayesian model instances in which a `return_all` parameter can be passed.

        Returns:
            float: mean value

        """
        return self.moment(1, **kwargs)

    def std(self, **kwargs) -> float:
        """Calculates the standard deviation

        Args:
            **kwargs: Additional arguments passed to `moments`. This is needed for Bayesian model instances in which a `return_all` parameter can be passed.

        Returns:
            float: standard deviation value

        """
        return np.sqrt(self.moment(2, **kwargs) - self.mean(**kwargs) ** 2)

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Evaluates the cumulative probability function

        Args:
            x (float): a point in the support
        """
        pass

    @abstractmethod
    def pdf(self, x: float) -> float:
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

        # show histogram from 1k samples
        samples = self.simulate(size=size)
        plt.hist(
            samples, bins=25, edgecolor="white", color=self._figure_color_palette[0]
        )
        plt.title(f"Histogram from {np.round(size/1000,1)}K simulated samples")
        plt.show()

    def plot(self, size: int = 1000) -> None:
        """Plots a histogram of a simulated sample

        Args:
            size (int, optional): sample size

        """
        self.histogram(size)

    def cvar(self, p: float, **kwargs) -> float:
        """Calculates conditional value at risk for a probability level p, defined as the mean conditioned to an exceedance above the p-quantile.

        Args:
            p (float): Description
            **kwargs: Additional arguments passed to `moments`. This is needed for Bayesian model instances in which a `return_all` parameter can be passed.

        Returns:
            float: conditional value at risk

        Raises:
            ValueError: Description

        """
        if not isinstance(p, float) or p < 0 or p >= 1:
            raise ValueError("p must be in the open interval (0,1)")

        return (self >= self.ppf(p)).mean(**kwargs)

    @abstractmethod
    def __gt__(self, other: float) -> BaseDistribution:
        pass

    @abstractmethod
    def __ge__(self, other: float) -> BaseDistribution:
        pass

    @abstractmethod
    def __add__(self, other: self._allowed_scalar_types):
        pass

    @abstractmethod
    def __mul__(self, other: self._allowed_scalar_types):
        pass

    def __sub__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f"- is implemented for instances of types: {self._allowed_scalar_types}"
            )

        return self.__add__(-other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    # __radd__ = __add__

    # __rmul__ = __mul__

    # __rsub__ = __sub__


class Mixture(BaseDistribution):

    """This class represents a probability distribution given by a mixture of weighted continuous and empirical densities; as the base continuous densities can only be of class GPTail, this class is intended to represent either a semiparametric model with a Generalised Pareto tail, or the convolution of such a model with an integer distribution, as is the case for the power surplus distribution in power system reliability modeling.

    Args:
        distributions (t.List[BaseDistribution]): list of distributions that make up the mixture
        weights (np.ndarray): weights for each of the distribution. The weights must be a distribution themselves
    """

    distributions: t.List[BaseDistribution]
    weights: np.ndarray

    def __repr__(self):
        return f"Mixture with {len(self.weights)} components"

    @validator("weights", allow_reuse=True)
    def check_weigths(cls, weights):
        if not np.isclose(np.sum(weights), 1, atol=cls._error_tol):
            raise ValidationError(f"Weights don't sum 1 (sum = {np.sum(weights)})")
        elif np.any(weights <= 0):
            raise ValidationError("Negative or null weights are present")
        else:
            return weights

    def __mul__(self, factor: float):

        return Mixture(
            weights=self.weights,
            distributions=[factor * dist for dist in self.distributions],
        )

    def __add__(self, factor: float):

        return Mixture(
            weights=self.weights,
            distributions=[factor + dist for dist in self.distributions],
        )

    def __ge__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f">= is implemented for instances of types : {self._allowed_scalar_types}"
            )

        if self.cdf(other) == 1:
            raise ValueError("There is no probability mass above provided threshold")

        cond_weights = np.array(
            [
                1 - dist.cdf(other) + (isinstance(dist, Empirical)) * dist.pdf(other)
                for dist in self.distributions
            ]
        )
        new_weights = cond_weights * self.weights

        indices = (new_weights > 0).nonzero()[0]

        nz_weights = new_weights[indices]
        nz_dists = [self.distributions[i] for i in indices]

        return Mixture(
            weights=nz_weights / np.sum(nz_weights),
            distributions=[dist >= other for dist in nz_dists],
        )

    def __gt__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f"> is implemented for instances of types : {self._allowed_scalar_types}"
            )

        if self.cdf(other) == 1:
            raise ValueError("There is no probability mass above provided threshold")

        cond_weights = np.array(
            [
                1 - dist.cdf(other) + (isinstance(dist, Empirical)) * dist.pdf(other)
                for dist in self.distributions
            ]
        )

        new_weights = cond_weights * self.weights

        indices = (new_weights > 0).nonzero()[0]

        nz_weights = new_weights[indices]
        nz_dists = [self.distributions[i] for i in indices]

        return Mixture(
            weights=nz_weights / np.sum(nz_weights),
            distributions=[dist > other for dist in nz_dists],
        )

        # index = self.support > other

        # return type(self)(
        #   self.support[index],
        #   self.pdf_values[index]/np.sum(self.pdf_values[index]),
        #   self.data[self.data > other])

    def simulate(self, size: int) -> np.ndarray:
        """Simulate values from mixture distribution

        Args:
            size (int): Sample size

        Returns:
            np.ndarray: simulated sample
        """
        n_samples = np.random.multinomial(n=size, pvals=self.weights, size=1)[0]
        indices = (n_samples > 0).nonzero()[0]
        samples = [
            dist.simulate(size=k)
            for dist, k in zip(
                [self.distributions[k] for k in indices], n_samples[indices]
            )
        ]
        samples = np.concatenate(samples, axis=0)
        np.random.shuffle(samples)
        return samples

    def cdf(
        self, x: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        """Evaluate Mixture's CDF function

        Args:
            x (t.Union[float, np.ndarray]): point at which to evaluate CDF
            **kwargs: Additional arguments passed to individual mixture component's CDF function

        Returns:
            t.Union[float, np.ndarray]: CDF value
        """

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * dist.cdf(x, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)

    def pdf(
        self, x: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        """Evaluate Mixture's pdf function

        Args:
            x (t.Union[float, np.ndarray]): point at which to evaluate CDF
            **kwargs: Additional arguments passed to individual mixture component's pdf function

        Returns:
            t.Union[float, np.ndarray]: pdf value
        """

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * dist.pdf(x, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)

    def ppf(
        self, q: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        """Evaluate Mixture's quantile function function

        Args:
            x (t.Union[float, np.ndarray]): point at which to evaluate quantile function
            **kwargs: Additional arguments passed to individual mixture component's quantile function

        Returns:
            t.Union[float, np.ndarray]: quantile value
        """

        if isinstance(q, np.ndarray):
            return np.array([self.ppf(elem, **kwargs) for elem in q])

        def target_function(x):
            return self.cdf(x) - q

        vals = [
            w * dist.ppf(q, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        x0 = reduce(lambda x, y: x + y, vals)

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * dist.ppf(0.5 + q / 2, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        x1 = reduce(lambda x, y: x + y, vals)

        return root_scalar(target_function, x0=x0, x1=x1, method="secant").root

    def moment(self, n: int, **kwargs) -> float:
        """Evaluate Mixture's n-th moment

        Args:
            x (t.Union[float, np.ndarray]): Moment order
            **kwargs: Additional arguments passed to individual mixture components' moment function

        Returns:
            t.Union[float, np.ndarray]: moment value
        """

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * dist.moment(n, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)

    def mean(self, **kwargs) -> float:
        """Evaluate Mixture's mean

        Args:
            **kwargs: Additional arguments passed to individual mixture components' mean function

        Returns:
            float: mean value
        """

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * dist.mean(**kwargs) for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)

    def std(self, **kwargs) -> float:
        """Evaluate Mixture's standard deviation

        Args:
            **kwargs: Additional arguments passed to individual mixture components' standard deviation function

        Returns:
            float: standard deviation value
        """

        # the use of a list and a reduce is needed because mixture components might return scalars or vectors depending on their class and on the passed kwargs.
        vals = [
            w * (dist.std(**kwargs) ** 2 + dist.mean(**kwargs) ** 2)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return np.sqrt(reduce(lambda x, y: x + y, vals) - self.mean() ** 2)


class GPTail(BaseDistribution):
    """Representation of a fitted Generalized Pareto distribution as an exceedance model. It's density is given by

    $$ f(x) = \\frac{1}{\\sigma} \\left( 1 + \\xi \\left( \\frac{x - \\mu}{\\sigma} \\right) \\right)_{+}^{-(1 + 1/\\xi)} $$

    where \\( \\mu, \\sigma, \\xi \\) are the location, scale and shape parameters, and \\( (\\cdot)_{+} = \\max(\\cdot, 0)\\). The location parameter is also the lower endpoint (or threshold) of the distribution.

    Args:
        threshold (float): modeling threshold
        shape (float): fitted shape parameter
        scale (float): fitted scale parameter
        data (np.array, optional): exceedance data
    """

    threshold: float
    shape: float
    scale: PositiveFloat
    data: t.Optional[np.ndarray]

    def __repr__(self):
        return f"Generalised Pareto tail model with (mu, scale, shape) = ({self.threshold},{self.scale},{self.shape}) components"

    @property
    def endpoint(self):
        return self.threshold - self.scale / self.shape if self.shape < 0 else np.Inf

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
            raise TypeError(
                f"+ is implemented for instances of types: {self._allowed_scalar_types}"
            )

        return GPTail(
            threshold=self.threshold + other,
            shape=self.shape,
            scale=self.scale,
            data=self.data + other if self.data is not None else None,
        )

    def __ge__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f">= and > are implemented for instances of types: {self._allowed_scalar_types}"
            )

        if other >= self.endpoint:
            raise ValueError(
                f"No probability mass above endpoint ({self.endpoint}); conditional distribution X | X >= {other} does not exist"
            )

        if other < self.threshold:
            return self >= self.threshold
        else:
            # condition on empirical data if applicable
            if self.data is None or max(self.data) < other:
                new_data = None
                warnings.warn(
                    f"No observed data above {other}; setting data to None in conditional model.",
                    stacklevel=2,
                )
            else:
                new_data = self.data[self.data >= other]

            return GPTail(
                threshold=other,
                shape=self.shape,
                scale=self.scale + self.shape * (other - self.threshold),
                data=new_data,
            )

    def __gt__(self, other: float) -> GPTail:

        return self.__ge__(other)

    def __mul__(self, other: float) -> GPTail:

        if not isinstance(other, self._allowed_scalar_types) or other <= 0:
            raise TypeError(
                f"* is implemented for positive instances of: {self._allowed_scalar_types}"
            )

        new_data = other * self.data if self.data is not None else None

        return GPTail(
            threshold=other * self.threshold,
            shape=self.shape,
            scale=other * self.scale,
            data=new_data,
        )

    def simulate(self, size: int) -> np.ndarray:
        return self.model.rvs(size=size)

    def moment(self, n: int, **kwargs) -> float:
        return self.model.moment(n)

    def ppf(
        self, q: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        return self.model.ppf(q)

    def cdf(
        self, x: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        return self.model.cdf(x)

    def pdf(
        self, x: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        return self.model.pdf(x)

    def std(self, **kwargs):
        return self.model.std()

    def mle_cov(self) -> np.ndarray:
        """Returns the estimated parameter covariance matrix evaluated at the fitted parameters

        Returns:
            np.ndarray: Covariance matrix
        """

        if self.data is None:
            raise ValueError(
                "exceedance data not provided for this instance of GPTail; covariance matrix can't be estimated"
            )
        else:
            hess = self.loglik_hessian(
                [self.scale, self.shape], threshold=self.threshold, data=self.data
            )
            return np.linalg.inv(-hess)

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
    def loglik_grad(
        cls, params: t.List[float], threshold: float, data: np.ndarray
    ) -> np.ndarray:
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
            grad_scale = np.sum((y - scale) / (scale * (y * shape + scale)))
            grad_shape = np.sum(
                -((y * (1 + shape)) / (shape * (y * shape + scale)))
                + np.log(1 + (y * shape) / scale) / shape**2
            )
        else:
            grad_scale = np.sum((y - scale) / scale**2)
            grad_shape = np.sum(y * (y - 2 * scale) / (2 * scale**2))

        return np.array([grad_scale, grad_shape])

    @classmethod
    def loglik_hessian(
        cls, params: t.List[float], threshold: float, data: np.ndarray
    ) -> np.ndarray:
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
            d2scale = np.sum(
                -(1 / (shape * scale**2))
                + (1 + shape) / (shape * (y * shape + scale) ** 2)
            )

            # d2shape = (y (3 y ξ + y ξ^2 + 2 σ))/(ξ^2 (y ξ + σ)^2) - (2 Log[1 + (y ξ)/σ])/ξ^3
            d2shape = np.sum(
                (y * (3 * y * shape + y * shape**2 + 2 * scale))
                / (shape**2 * (y * shape + scale) ** 2)
                - (2 * np.log(1 + (y * shape) / scale)) / shape**3
            )

            # dscale_dshape = (y (-y + σ))/(σ (y ξ + σ)^2)
            dscale_dshape = np.sum(
                (y * (-y + scale)) / (scale * (y * shape + scale) ** 2)
            )
        else:
            d2scale = np.sum((scale - 2 * y) / scale**3)
            dscale_dshape = np.sum(-y * (y - scale) / scale**3)
            d2shape = np.sum(y**2 * (3 * scale - 2 * y) / (3 * scale**3))

        hessian = np.array([[d2scale, dscale_dshape], [dscale_dshape, d2shape]])

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

        return 0.5 * (np.log(scale) + shape**2)

    @classmethod
    def logreg_grad(cls, params: t.List[float]) -> float:
        """Returns the gradient of the regularisation term

        Args:
            params (t.List[float]): scale and shape parameters in that order

        """
        scale, shape = params
        return 0.5 * np.array([1.0 / scale, 2 * shape])

    @classmethod
    def logreg_hessian(cls, params: t.List[float]) -> float:
        """Returns the Hessian of the regularisation term

        Args:
            params (t.List[float]): scale and shape parameters in that order

        """
        scale, shape = params
        return 0.5 * np.array([-1.0 / scale**2, 0, 0, 2]).reshape((2, 2))

    @classmethod
    def loss(
        cls, params: t.List[float], threshold: float, data: np.ndarray
    ) -> np.ndarray:
        """Calculate the loss function on the provided parameters; this is the sum of the (negative) data log-likelihood and (negative) regularisation terms for the scale and shape. Everything is divided by the number of data points.

        Args:
            params (t.List[float]): Description
            threshold (float): Model threshold; eequivalently, location parameter
            data (np.ndarray): exceedance data above threshold

        """
        scale, shape = params
        n = len(data)
        unnorm = cls.loglik(params, threshold, data) + cls.logreg(params)
        return -unnorm / n

    @classmethod
    def loss_grad(
        cls, params: t.List[float], threshold: float, data: np.ndarray
    ) -> np.ndarray:
        """Calculate the loss function's gradient on the provided parameters

        Args:
            params (t.List[float]): Description
            threshold (float): Model threshold; eequivalently, location parameter
            data (np.ndarray): exceedance data above threshold

        """
        n = len(data)
        return -(cls.loglik_grad(params, threshold, data) + cls.logreg_grad(params)) / n

    @classmethod
    def loss_hessian(
        cls, params: t.List[float], threshold: float, data: np.ndarray
    ) -> np.ndarray:
        """Calculate the loss function's Hessian on the provided parameters

        Args:
            params (t.List[float]): Description
            threshold (float): Model threshold; eequivalently, location parameter
            data (np.ndarray): exceedance data above threshold

        """
        n = len(data)
        return (
            -(cls.loglik_hessian(params, threshold, data) + cls.logreg_hessian(params))
            / n
        )

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        threshold: float,
        x0: np.ndarray = None,
        return_opt_results=False,
    ) -> t.Union[GPTail, sp.optimize.OptimizeResult]:
        """Fits a eneralised Pareto tail model using a constrained trust region method

        Args:
            data (np.ndarray): exceedance data above threshold
            threshold (float): Model threshold
            x0 (np.ndarray, optional): Initial guess for optimization; if None, the result of scipy.stats.genpareto.fit is used as a starting point.
            return_opt_results (bool, optional): If True, return the OptimizeResult object; otherwise return fitted instance of GPTail

        Returns:
            t.Union[GPTail, sp.optimize.OptimizeResult]: Description
        """
        exceedances = data[data > threshold]

        # rescale exceedances and threshold so that both parameters are in roughly the same scale, improving numerical conditioning
        sdev = np.std(exceedances)

        # rescaling the data rescales the location and scale parameter, and leaves the shape parameter unchanged
        norm_exceedances = exceedances / sdev
        norm_threshold = threshold / sdev

        norm_max = max(norm_exceedances)

        constraints = LinearConstraint(
            A=np.array([[1 / (norm_max - norm_threshold), 1], [1, 0]]),
            lb=np.zeros((2,)),
            ub=np.Inf,
        )

        if x0 is None:
            # use default scipy fitter to get initial estimate
            # this is almost always good enough
            shape, _, scale = gpdist.fit(norm_exceedances, floc=norm_threshold)
            x0 = np.array([scale, shape])

        loss_func = lambda params: GPTail.loss(params, norm_threshold, norm_exceedances)
        loss_grad = lambda params: GPTail.loss_grad(
            params, norm_threshold, norm_exceedances
        )
        loss_hessian = lambda params: GPTail.loss_hessian(
            params, norm_threshold, norm_exceedances
        )

        res = minimize(
            fun=loss_func,
            x0=x0,
            method="trust-constr",
            jac=loss_grad,
            hess=loss_hessian,
            constraints=[constraints],
        )

        # print(res)
        if return_opt_results:
            warnings.warn(
                "Returning raw results for rescaled exceedance data (sdev ~ 1)."
            )
            return res
        else:
            scale, shape = list(res.x)

            return cls(
                threshold=sdev * norm_threshold,
                scale=sdev * scale,
                shape=shape,
                data=sdev * norm_exceedances,
            )

    def plot_diagnostics(self) -> None:
        """Returns a figure with fit diagnostic for the GP model"""
        if self.data is None:
            raise ValueError("Exceedance data was not provided for this model.")

        fig, axs = plt.subplots(3, 2)

        #################### Profile log-likelihood

        # set profile intervals based on MLE variance
        mle_cov = self.mle_cov()
        scale_bounds, shape_bounds = (
            self.scale - np.sqrt(mle_cov[0, 0]),
            self.scale + np.sqrt(mle_cov[0, 0]),
        ), (self.shape - np.sqrt(mle_cov[1, 1]), self.shape + np.sqrt(mle_cov[1, 1]))

        # set profile grids
        scale_grid = np.linspace(scale_bounds[0], scale_bounds[1], 50)
        shape_grid = np.linspace(shape_bounds[0], shape_bounds[1], 50)

        # declare profile functions
        scale_profile_func = lambda x: self.loglik(
            [x, self.shape], self.threshold, self.data
        )
        shape_profile_func = lambda x: self.loglik(
            [self.scale, x], self.threshold, self.data
        )

        loss_value = scale_profile_func(self.scale)

        scale_profile = np.array([scale_profile_func(x) for x in scale_grid])
        shape_profile = np.array([shape_profile_func(x) for x in shape_grid])

        alpha = 2 if loss_value > 0 else 0.5

        # filter to almost-optimal values

        def filter_grid(grid, optimum):
            radius = 2 * np.abs(optimum)
            return np.logical_and(
                np.logical_not(np.isnan(grid)),
                np.isfinite(grid),
                np.abs(grid - optimum) < radius,
            )

        scale_filter = filter_grid(scale_profile, loss_value)
        shape_filter = filter_grid(shape_profile, loss_value)

        valid_scales = scale_grid[scale_filter]
        valid_scale_profile = scale_profile[scale_filter]

        axs[0, 0].plot(
            valid_scales, valid_scale_profile, color=self._figure_color_palette[0]
        )
        axs[0, 0].vlines(
            x=self.scale,
            ymin=min(valid_scale_profile),
            ymax=max(valid_scale_profile),
            linestyle="dashed",
            colors=self._figure_color_palette[1],
        )
        axs[0, 0].title.set_text("Profile scale log-likelihood")
        axs[0, 0].set_xlabel("Scale")
        axs[0, 0].set_ylabel("log-likelihood")
        axs[0, 0].grid()

        valid_shapes = shape_grid[shape_filter]
        valid_shape_profile = shape_profile[shape_filter]

        axs[0, 1].plot(
            valid_shapes, valid_shape_profile, color=self._figure_color_palette[0]
        )
        axs[0, 1].vlines(
            x=self.shape,
            ymin=min(valid_shape_profile),
            ymax=max(valid_shape_profile),
            linestyle="dashed",
            colors=self._figure_color_palette[1],
        )
        axs[0, 1].title.set_text("Profile shape log-likelihood")
        axs[0, 1].set_xlabel("Shape")
        axs[0, 1].set_ylabel("log-likelihood")
        axs[0, 1].grid()

        ######################## Log-likelihood surface ###############
        scale_grid, shape_grid = np.mgrid[
            min(valid_scales) : max(valid_scales) : 2 * np.sqrt(mle_cov[0, 0]) / 50,
            shape_bounds[0] : shape_bounds[1] : 2 * np.sqrt(mle_cov[0, 0]) / 50,
        ]

        scale_mesh, shape_mesh = np.meshgrid(
            np.linspace(min(valid_scales), max(valid_scales), 50),
            np.linspace(min(valid_shapes), max(valid_shapes), 50),
        )

        max_x = max(self.data)

        z = np.empty(scale_mesh.shape)
        for i in range(scale_mesh.shape[0]):
            for j in range(scale_mesh.shape[1]):
                shape = shape_mesh[i, j]
                scale = scale_mesh[i, j]
                if shape < 0 and self.threshold - scale / shape < max_x:
                    z[i, j] = np.nan
                else:
                    z[i, j] = self.loglik([scale, shape], self.threshold, self.data)

        # negate z to recover true loglikelihood
        axs[1, 0].contourf(scale_mesh, shape_mesh, z, levels=15)
        axs[1, 0].scatter([self.scale], [self.shape], color="darkorange", s=2)
        axs[1, 0].annotate(text="MLE", xy=(self.scale, self.shape), color="darkorange")
        axs[1, 0].title.set_text("Log-likelihood surface")
        axs[1, 0].set_xlabel("Scale")
        axs[1, 0].set_ylabel("Shape")

        ############## histogram vs density ################
        hist_data = axs[1, 1].hist(
            self.data, bins=25, edgecolor="white", color=self._figure_color_palette[0]
        )

        range_min, range_max = min(self.data), max(self.data)
        x_axis = np.linspace(range_min, range_max, 100)
        pdf_vals = self.pdf(x_axis)
        y_axis = hist_data[0][0] / pdf_vals[0] * pdf_vals

        axs[1, 1].plot(x_axis, y_axis, color=self._figure_color_palette[1])
        axs[1, 1].title.set_text("Data vs rescaled fitted density")
        axs[1, 1].set_xlabel("Exceedance data")
        axs[1, 1].yaxis.set_visible(False)  # Hide only x axis
        # axs[0, 0].set_aspect('equal', 'box')

        ############# Q-Q plot ################
        probability_range = np.linspace(0.01, 0.99, 99)
        empirical_quantiles = np.quantile(self.data, probability_range)
        tail_quantiles = self.ppf(probability_range)

        axs[2, 0].scatter(
            tail_quantiles, empirical_quantiles, color=self._figure_color_palette[0]
        )
        min_x, max_x = min(tail_quantiles), max(tail_quantiles)
        # axs[0,1].set_aspect('equal', 'box')
        axs[2, 0].title.set_text("Q-Q plot")
        axs[2, 0].set_xlabel("model quantiles")
        axs[2, 0].set_ylabel("Data quantiles")
        axs[2, 0].grid()
        axs[2, 0].plot([min_x, max_x], [min_x, max_x], linestyle="--", color="black")

        ############ Mean return plot ###############
        # scale, shape = self.scale, self.shape

        # n_obs = len(self.data)
        # exceedance_frequency = 1/np.logspace(1,4,20)
        # return_levels = self.ppf(1 - exceedance_frequency)

        # axs[2,1].plot(1.0/exceedance_frequency,return_levels,color=self._figure_color_palette[0])
        # axs[2,1].set_xscale("log")
        # axs[2,1].title.set_text("Exceedance model's return levels")
        # axs[2,1].set_xlabel('1/frequency')
        # axs[2,1].set_ylabel('Return level')
        # axs[2,1].grid()

        # exs_prob = 1 #carried over from older code

        # m = 10**np.linspace(np.log(1/exs_prob + 1)/np.log(10), 3,20)
        # return_levels = self.ppf(1 - 1/(exs_prob*m))

        # axs[2,1].plot(m,return_levels,color=self._figure_color_palette[0])
        # axs[2,1].set_xscale("log")
        # axs[2,1].title.set_text('Exceedance return levels')
        # axs[2,1].set_xlabel('1/frequency')
        # axs[2,1].set_ylabel('Return level')
        # axs[2,1].grid()

        # try:
        #   #for this bit, look at An Introduction to Statistical selfing of Extreme Values, p.82
        #   mle_cov = self.mle_cov()
        #   eigenvals, eigenvecs = np.linalg.eig(mle_cov)
        #   if np.all(eigenvals > 0):
        #     covariance = np.eye(3)
        #     covariance[1::,1::] = mle_cov
        #     covariance[0,0] = exs_prob*(1-exs_prob)/n_obs
        #     #
        #     return_stdevs = []
        #     for m_ in m:
        #       quantile_grad = np.array([
        #         scale*m_**(shape)*exs_prob**(shape-1),
        #         shape**(-1)*((exs_prob*m_)**shape-1),
        #         -scale*shape**(-2)*((exs_prob*m_)**shape-1)+scale*shape**(-1)*(exs_prob*m_)**shape*np.log(exs_prob*m_)
        #         ])
        #       #
        #       sdev = np.sqrt(quantile_grad.T.dot(covariance).dot(quantile_grad))
        #       return_stdevs.append(sdev)
        #     #
        #     axs[2,1].fill_between(m, return_levels - return_stdevs, return_levels + return_stdevs, alpha=0.2, color=self._figure_color_palette[1])
        #   else:
        #     warnings.warn("Covariance MLE matrix is not positive definite; it might be ill-conditioned", stacklevel=2)
        # except Exception as e:
        #   warnings.warn(f"Confidence bands for return level could not be calculated; covariance matrix might be ill-conditioned; full trace: {traceback.format_exc()}", stacklevel=2)

        plt.tight_layout()
        return fig


class GPTailMixture(BaseDistribution):

    """Mixture distribution with generalised Pareto components. This is a base class for  Bayesian generalised Pareto tail models, which can be seen as an uniformly weighted mixture of the posterior samples; the convolution of a discrete (or empirical) distribution with a generalised Pareto distribution also results in a mixture of this kind. Most methods inherited from `BaseDistribution` have an extra argument `return_all`. When it is True, the full posterior sample of the method evaluation is returned.

    Args:
        data(np.ndarray, optional): Data that induced the model, if applicable
        weights (np.ndarray): component weights
        thresholds (np.ndarray): vector of threshold parameters, one for each component
        scales (np.ndarray): vector of scale parameters, one for each component
        shapes (np.ndarray): vector of shape parameters, one for each component
    """

    data: t.Optional[np.ndarray]
    weights: np.ndarray
    thresholds: np.ndarray
    scales: np.ndarray
    shapes: np.ndarray

    def __repr__(self):
        return f"Mixture of generalised Pareto distributions with {len(self.weights)} components"

    @validator("weights", allow_reuse=True)
    def check_weigths(cls, weights):
        if not np.isclose(np.sum(weights), 1, atol=cls._error_tol):
            raise ValueError(f"Weights don't sum 1 (sum = {np.sum(weights)})")
        elif np.any(weights <= 0):
            raise ValueError("Negative or null weights are present")
        else:
            return weights

    def moment(self, n: int, return_all: bool = False) -> float:
        """Returns the n-th order moment

        Args:
            n (int): Moment's order
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; otherwise  the moment value of the posterior predictive distribution is returned. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            float: moment value
        """
        vals = np.array(
            [
                gpdist.moment(n, loc=threshold, c=shape, scale=scale)
                for threshold, shape, scale in zip(
                    self.thresholds, self.shapes, self.scales
                )
            ]
        )

        if return_all:
            return vals
        else:
            return np.dot(self.weights, vals)

    def mean(self, return_all: bool = False) -> float:
        """Returns the mean value

        Args:
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; the mean of the posterior predictive distribution is evaluated. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            float: mean value
        """
        vals = gpdist.mean(loc=self.thresholds, c=self.shapes, scale=self.scales)

        if return_all:
            return vals
        else:
            return np.dot(self.weights, vals)

    def std(self, return_all: bool = False) -> float:
        """Standard deviation value

        Args:
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; otherwise the standard deviation of the posterior predictive distribution is returned. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            float: standard deviation value
        """
        var = gpdist.var(loc=self.thresholds, c=self.shapes, scale=self.scales)

        mean = gpdist.mean(loc=self.thresholds, c=self.shapes, scale=self.scales)

        if return_all:
            return np.sqrt(var)
        else:
            return np.sqrt(np.dot(self.weights, var + mean**2) - self.mean() ** 2)

    def cdf(
        self, x: t.Union[float, np.ndarray], return_all: bool = False
    ) -> t.Union[float, np.ndarray]:
        """Evaluates the CDF function

        Args:
            x (t.Union[float, np.ndarray]): Point at which to evaluate it
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; the CDF of the posterior predictive distribution is evaluated. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            t.Union[float, np.ndarray]: CDF value
        """

        if isinstance(x, np.ndarray):
            return np.array([self.cdf(elem, return_all) for elem in x])

        vals = gpdist.cdf(x, loc=self.thresholds, c=self.shapes, scale=self.scales)

        if return_all:
            return vals
        else:
            return np.dot(self.weights, vals)

    def pdf(
        self, x: t.Union[float, np.ndarray], return_all: bool = False
    ) -> t.Union[float, np.ndarray]:
        """Evaluates the pdf function

        Args:
            x (t.Union[float, np.ndarray]): Point at which to evaluate it
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; otherwise the pdf of the posterior predictive distribution is evaluated. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            t.Union[float, np.ndarray]: pdf value
        """

        if isinstance(x, np.ndarray):
            return np.array([self.pdf(elem, return_all) for elem in x])

        vals = gpdist.pdf(x, loc=self.thresholds, c=self.shapes, scale=self.scales)

        if return_all:
            return vals
        else:
            return np.dot(self.weights, vals)

    def ppf(
        self, q: t.Union[float, np.ndarray], return_all: bool = False
    ) -> t.Union[float, np.ndarray]:
        """Evaluates the quantile function

        Args:
            x (t.Union[float, np.ndarray]): probability level
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; otherwise  the quantile function of the posterior predictive distribution is evaluated. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            t.Union[float, np.ndarray]: quantile function value value
        """
        if isinstance(q, np.ndarray):
            return np.array([self.ppf(elem, return_all) for elem in q])

        vals = gpdist.ppf(q, loc=self.thresholds, c=self.shapes, scale=self.scales)

        if return_all:
            return vals
        else:

            def target_function(x):
                return self.cdf(x) - q

            x0 = np.dot(self.weights, vals)
            return root_scalar(target_function, x0=x0, x1=x0 + 1, method="secant").root

    def cvar(self, p: float, return_all: bool = False) -> float:
        """Returns the conditional value at risk for a given probability level

        Args:
            x (t.Union[float, np.ndarray]): probability level
            return_all (bool, optional): If `True`, all posterior samples of the value are returned; otherwise  the cvar of the posterior predictive distribution is evaluated. The posterior distribution is taken to be the empirical distribution of posterior samples.

        Returns:
            t.Union[float, np.ndarray]: conditional value at risk
        """
        if p < 0 or p >= 1:
            raise ValueError("p must be in the open interval (0,1)")

        return (self >= self.ppf(p)).mean(return_all)

    def simulate(self, size: int) -> np.ndarray:
        n_samples = np.random.multinomial(n=size, pvals=self.weights, size=1)[0]
        indices = (n_samples > 0).nonzero()[0]
        samples = [
            gpdist.rvs(
                size=n_samples[i],
                c=self.shapes[i],
                scale=self.scales[i],
                loc=self.thresholds[i],
            )
            for i in indices
        ]
        return np.concatenate(samples, axis=0)

    def __gt__(self, other: float) -> GPTailMixture:
        exceedance_prob = 1 - self.cdf(other)
        # prob_exceedance = np.dot(self.weights, prob_cond_exceedance)
        if exceedance_prob == 0:
            raise ValueError(
                f"There is no probability mass above {other}; conditional distribution does not exist."
            )

        conditional_weights = (
            self.weights
            * gpdist.sf(other, c=self.shapes, scale=self.scales, loc=self.thresholds)
            / exceedance_prob
        )

        indices = (conditional_weights > 0).nonzero()[
            0
        ]  # indices of mixture components with nonzero exceedance probability

        new_weights = conditional_weights[indices]

        # disable warnings temporarily
        # with warnings.catch_warnings():
        #   warnings.simplefilter("ignore")
        #   new_thresholds = np.array([(GPTail(threshold = mu, shape = xi, scale = sigma) >= other).threshold for mu, sigma, xi in zip(self.thresholds[indices], self.scales[indices], self.shapes[indices])])

        new_thresholds = np.array(
            [max(other, threshold) for threshold in self.thresholds[indices]]
        )
        new_shapes = self.shapes[indices]
        new_scales = self.scales[indices] + new_shapes * (
            new_thresholds - self.thresholds[indices]
        )
        # new_thresholds = np.clip(self.thresholds[indices], a_min=other, a_max = np.Inf)
        # new_shapes = self.shapes[indices]
        # new_scales = self.scales[indices] + new_shapes*np.clip(other - new_thresholds, a_min = 0.0, a_max=np.Inf)

        if self.data is not None and np.all(self.data < other):
            warnings.warn(
                f"No observed data above {other}; setting data to None in conditioned model",
                stacklevel=2,
            )
            new_data = None
        elif self.data is None:
            new_data = None
        else:
            new_data = self.data[self.data > other]

        return type(self)(
            weights=new_weights,
            thresholds=new_thresholds,
            shapes=new_shapes,
            scales=new_scales,
            data=new_data,
        )

    def __ge__(self, other: float) -> BaseDistribution:
        return self.__gt__(other)

    def __add__(self, other: self._allowed_scalar_types):

        if type(other) not in self._allowed_scalar_types:
            raise TypeError(f"+ is implemented for types {self._allowed_scalar_types}")

        new_data = None if self.data is None else self.data + other

        return type(self)(
            weights=self.weights,
            thresholds=self.thresholds + other,
            shapes=self.shapes,
            scales=self.scales,
            data=new_data,
        )

    def __mul__(self, other: self._allowed_scalar_types):

        if type(other) not in self._allowed_scalar_types:
            raise TypeError(f"* is implemented for types {self._allowed_scalar_types}")

        if other <= 0:
            raise ValueError(f"product supported for positive scalars only")

        new_data = None if self.data is None else other * self.data

        return type(self)(
            weights=self.weights,
            thresholds=other * self.thresholds,
            shapes=self.shapes,
            scales=other * self.scales,
            data=new_data,
        )


class Empirical(BaseDistribution):

    """Model for an empirical probability distribution, induced by a sample of data.

    Args:
        support (np.ndarray): distribution support
        pdf_values (np.ndarray): pdf array
        data (np.array, optional): data
    """

    _sum_compatible = (GPTail, Mixture, GPTailMixture)

    support: np.ndarray
    pdf_values: np.ndarray
    data: t.Optional[np.ndarray]

    def __repr__(self):
        if self.data is not None:
            return f"Empirical distribution with {len(self.data)} points"
        else:
            return f"Discrete distribution with support of size {len(self.pdf_values)}"

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
    def is_valid(self):
        n = len(self.support)
        m = len(self.pdf_values)

        if n != m:
            raise ValueError("Lengths of support and pdf arrays don't match")

        if not np.all(np.diff(self.support) > 0):
            raise ValueError("Support array must be in increasing order")

        return True

    @property
    def pdf_lookup(self):
        """Mapping from values in the support to their probability mass"""
        return {key: val for key, val in zip(self.support, self.pdf_values)}

    @property
    def min(self):
        """Minimum value in the support"""
        return self.support[0]

    @property
    def max(self):
        """Maximum value in the support"""
        return self.support[-1]

    @property
    def cdf_values(self):
        """Mapping from values in the support to their cumulative probability"""
        x = np.cumsum(self.pdf_values)
        # make sure cdf reaches 1
        x[-1] = 1.0
        return x

    @property
    def ecdf(self):
        """Mapping from values in the support to their cumulative probability"""
        cdf_vals = np.cumsum(self.pdf_values)
        # make sure cdf reaches 1
        cdf_vals[-1] = 1.0

        return ed.StepFunction(x=self.support, y=cdf_vals, side="right")

    @property
    def ecdf_inv(self):
        """Linearly interpolated mapping from probability values to their quantiles"""
        return ed.monotone_fn_inverter(self.ecdf, self.support)

    def __mul__(self, factor: float):

        if not isinstance(factor, self._allowed_scalar_types) or factor == 0:
            raise TypeError(
                f"multiplication is supported only for nonzero instances of type:{self._allowed_scalar_types}"
            )

        new_data = None if self.data is None else self.data * factor
        return Empirical(
            support=factor * self.support,
            pdf_values=np.copy(self.pdf_values, order="C"),
            data=new_data,
        )

    def __add__(self, other: t.Union[int, float, GPTail, Mixture, GPTailMixture]):

        if isinstance(other, self._allowed_scalar_types):
            new_data = None if self.data is None else self.data + other
            return Empirical(
                support=self.support + other,
                pdf_values=np.copy(self.pdf_values, order="C"),
                data=new_data,
            )

        elif isinstance(other, GPTail):

            indices = (self.pdf_values > 0).nonzero()[0]
            nz_pdf = self.pdf_values[indices]
            nz_support = self.support[indices]
            # mixed GPTails don't carry over exceedance data for efficient memory use
            # dists = [GPTail(threshold=other.threshold + x, scale=other.scale, shape=other.shape) for x in nz_support]

            return GPTailMixture(
                data=other.data,
                weights=nz_pdf,
                thresholds=other.threshold + nz_support,
                scales=np.array([other.scale for w in nz_support]),
                shapes=np.array([other.shape for w in nz_support]),
            )

        elif isinstance(other, Mixture):
            return Mixture(
                weights=other.weights,
                distributions=[self + dist for dist in other.distributions],
            )

        elif isinstance(other, GPTailMixture):
            # Return a new mixture where the old mixture is replicated for each point in the discrete support
            new_weights = np.concatenate(
                [w * other.weights for w in self.pdf_values], axis=0
            )
            new_th = np.concatenate(
                [other.thresholds + t for t in self.support], axis=0
            )
            new_scales = np.tile(other.scales, len(self.support))
            new_shapes = np.tile(other.shapes, len(self.support))
            return GPTailMixture(
                data=other.data,
                weights=new_weights[new_weights > 0],
                thresholds=new_th[new_weights > 0],
                scales=new_scales[new_weights > 0],
                shapes=new_shapes[new_weights > 0],
            )

        else:
            raise TypeError(f"+ is supported only for types: {self._sum_compatible}")

    def __neg__(self):

        return self.map(lambda x: -x)

    def __sub__(self, other: Empirical):

        if isinstance(other, (Empirical,) + self._allowed_scalar_types):

            return self + (-other)

        else:
            raise TypeError(
                "Subtraction is only defined for instances of Empirical or float "
            )

    def __ge__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f">= is implemented for instances of types : {self._allowed_scalar_types}"
            )

        if 1 - self.cdf(other) + self.pdf(other) == 0.0:
            raise ValueError(
                f"No probability mass above conditional threshold ({other})."
            )

        index = self.support >= other

        new_data = None if self.data is None else self.data[self.data >= other]

        pdf_vals = self.pdf_values[index] / np.sum(self.pdf_values[index])

        return type(self)(
            support=self.support[index], pdf_values=pdf_vals, data=new_data
        )

    def __gt__(self, other: float):

        if not isinstance(other, self._allowed_scalar_types):
            raise TypeError(
                f"> is implemented for instances of types: {self._allowed_scalar_types}"
            )

        if 1 - self.cdf(other) == 0:
            raise ValueError(
                f"No probability mass above conditional threshold ({other})."
            )

        index = self.support > other

        new_data = None if self.data is None else self.data[self.data > other]

        return type(self)(
            support=self.support[index],
            pdf_values=self.pdf_values[index] / np.sum(self.pdf_values[index]),
            data=new_data,
        )

    def simulate(self, size: int) -> np.ndarray:
        """Draws simulated values from the distribution

        Args:
            size (int): Sample size

        Returns:
            np.ndarray: simulated sample
        """
        return np.random.choice(self.support, size=size, p=self.pdf_values)

    def moment(self, n: int, **kwargs) -> float:
        """Evaluates the n-th moment of the distribution

        Args:
            n (int): moment order
            **kwargs: dummy additional arguments (not used)

        Returns:
            float: n-th moment value
        """
        return np.sum(self.pdf_values * self.support**n)

    def ppf(
        self, q: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        """Inverse CDF function; it uses linear interpolation.

        Args:
            q (t.Union[float, np.ndarray]): probability level

        Returns:
            t.Union[float, np.ndarray]: Linearly interpolated quantile function

        """
        is_scalar = isinstance(q, self._allowed_scalar_types)

        if is_scalar:
            q = np.array([q])

        if np.any(q < 0) or np.any(q > 1):
            raise ValueError(f"q needs to be in the interval [0,1]")

        ppf_values = np.empty((len(q),))

        left_vals_idx = q <= self.ecdf_inv.x[0]
        right_vals_idx = q >= self.ecdf_inv.x[-1]
        inside_vals_idx = np.logical_and(
            np.logical_not(left_vals_idx), np.logical_not(right_vals_idx)
        )

        ppf_values[left_vals_idx] = self.ecdf_inv.y[0]
        ppf_values[right_vals_idx] = self.ecdf_inv.y[-1]
        ppf_values[inside_vals_idx] = self.ecdf_inv(q[inside_vals_idx])

        if is_scalar:
            return ppf_values[0]
        else:
            return ppf_values

    def cdf(
        self, x: t.Union[float, np.ndarray], **kwargs
    ) -> t.Union[float, np.ndarray]:
        return self.ecdf(x)

    def pdf(self, x: t.Union[float, np.ndarray], **kwargs):

        if isinstance(x, np.ndarray):
            return np.array([self.pdf(elem) for elem in x])

        try:
            pdf_val = self.pdf_lookup[x]
            return pdf_val
        except KeyError as e:
            return 0.0

    def std(self, **kwargs):
        return np.sqrt(self.map(lambda x: x - self.mean()).moment(2))

    @classmethod
    def from_data(cls, data: np.array):

        support, unnorm_pdf = np.unique(data, return_counts=True)
        n = np.sum(unnorm_pdf)
        return cls(support=support, pdf_values=unnorm_pdf / n, data=data)

    def plot_mean_residual_life(self, threshold: float) -> matplotlib.figure.Figure:
        """Produces a mean residual life plot for the tail of the distribution above a given threshold

        Args:
            threshold (float): threshold value

        Returns:
            matplotlib.figure.Figure: figure


        """
        fig = plt.figure()
        fitted = self.fit_tail_model(threshold)
        scale, shape = fitted.tail.scale, fitted.tail.shape

        x_vals = np.linspace(threshold, max(self.data))
        if shape >= 1:
            raise ValueError(
                f"Expectation is not finite: fitted shape parameter is {shape}"
            )
        # y_vals = (scale + shape*x_vals)/(1-shape)
        y_vals = np.array(
            [np.mean(fitted.tail.data[fitted.tail.data >= x]) for x in x_vals]
        )
        plt.plot(x_vals, y_vals, color=self._figure_color_palette[0])
        plt.scatter(x_vals, y_vals, color=self._figure_color_palette[0])
        plt.title("Mean residual life plot")
        plt.xlabel("Threshold")
        plt.ylabel("Mean exceedance")
        return fig

    def fit_tail_model(
        self, threshold: float, bayesian=False, **kwargs
    ) -> t.Union[EmpiricalWithGPTail, EmpiricalWithBayesianGPTail]:
        """Fits a tail GP model above a specified threshold and return the fitted semiparametric model

        Args:
            threshold (float): Threshold above which a Generalised Pareto distribution will be fitted
            bayesian (bool, optional): If True, fit model through Bayesian inference
            **kwargs: Additional parameters passed to BayesianGPTail.fit or to GPTail.fit

        Returns:
            t.Union[EmpiricalWithGPTail, EmpiricalWithBayesianGPTail]

        """

        if threshold >= self.max:
            raise ValueError(
                "Empirical pdf is 0 above the provided threshold. Select a lower threshold for estimation."
            )

        if self.data is None:
            raise ValueError(
                "Data is not set for this distribution, so a tail model cannot be fitted. You can simulate from it and use the sampled data instead"
            )
        else:
            data = self.data

        if bayesian:
            return EmpiricalWithBayesianGPTail.from_data(
                data, threshold, bin_empirical=isinstance(self, Binned), **kwargs
            )
        else:
            return EmpiricalWithGPTail.from_data(
                data, threshold, bin_empirical=isinstance(self, Binned), **kwargs
            )

    def map(self, f: t.Callable) -> Empirical:
        """Returns the distribution resulting from an arbitrary transformation

        Args:
            f (t.Callable): Target transformation; it should take a numpy array as input

        """
        dist_df = pd.DataFrame({"pdf": self.pdf_values, "support": f(self.support)})
        mapped_dist_df = (
            dist_df.groupby("support").sum().reset_index().sort_values("support")
        )

        return Empirical(
            support=np.array(mapped_dist_df["support"]),
            pdf_values=np.array(mapped_dist_df["pdf"]),
            data=f(self.data) if self.data is not None else None,
        )

    def to_integer(self):
        """Convert to Binned distribution"""
        return Binned.from_empirical(self)


class Binned(Empirical):

    """Empirical distribution with an integer support. This allows it to be convolved with other integer distribution to obtain the distribution of a sum of random variables, assuming independence between the summands."""

    _supported_types = [np.int64, int]

    def __repr__(self):
        return f"Integer distribution with support of size {len(self.pdf_values)} ({np.sum(self.pdf_values> 0)} non-zero)"

    @validator("support", allow_reuse=True)
    def integer_support(cls, support):
        n = len(support)
        if not np.all(np.diff(support) == 1):
            raise ValueError(
                "The support vector must contain every integer between its minimum and maximum value"
            )
        elif support.dtype not in cls._supported_types:
            raise ValueError(
                f"Support entry types must be one of {self._supported_types}"
            )
        else:
            return support

    def __mul__(self, factor: self._allowed_scalar_types):

        if not isinstance(factor, self._allowed_scalar_types) or factor == 0:
            raise TypeError(
                f"multiplication is supported only for nonzero instances of type:{self._allowed_scalar_types}"
            )

        if isinstance(factor, int):
            return self.from_empirical(float(factor) * self)
            # new_data = None if self.data is None else self.data*factor
            # return Binned(support = factor*self.support, pdf_values = self.pdf_values, data = new_data)

        else:
            return super().__mul__(factor)

    def __add__(self, other: t.Union[float, int, Binned, GPTail, Mixture]):

        if isinstance(other, int):
            new_support = np.arange(
                min(self.support) + other, max(self.support) + other + 1
            )
            new_data = None if self.data is None else self.data + other
            return Binned(
                support=new_support,
                pdf_values=np.copy(self.pdf_values, order="C"),
                data=new_data,
            )

        if isinstance(other, Binned):
            new_support = np.arange(
                min(self.support) + min(other.support),
                max(self.support) + max(other.support) + 1,
            )

            pdf_vals = fftconvolve(self.pdf_values, other.pdf_values)
            pdf_vals = np.abs(
                pdf_vals
            )  # some values are negative due to numerical rounding error, set to positive (they are infinitesimal in any case)
            pdf_vals = pdf_vals / np.sum(pdf_vals)

            # this block removes trailing support elements with zero probability
            while pdf_vals[-1] == 0.0:
                new_support = new_support[0 : len(pdf_vals) - 1]
                pdf_vals = pdf_vals[0 : len(pdf_vals) - 1]

            # add any missing probability mass
            error = 1.0 - np.sum(pdf_vals)
            pdf_vals[0] += np.sign(error) * np.abs(error)

            return Binned(
                support=new_support,
                pdf_values=np.abs(pdf_vals),  # some negative values persist
                data=None,
            )

        else:
            return super().__add__(other)

    def __sub__(self, other: float):

        if isinstance(other, (int, Binned)):
            return self + (-other)
        else:
            super().__sub__(other)

    def __neg__(self):

        return super().__neg__().to_integer()

    @classmethod
    def from_data(cls, data: np.ndarray) -> Binned:
        """Instantiates a Binned object from observed data

        Args:
            data (np.ndarray): Observed data

        Returns:
            Binned: Integer distribution object
        """
        data = np.array(data)
        if data.dtype not in self._supported_types:
            warnings.warn(
                "Casting input data to integer values by rounding", stacklevel=2
            )
            data = data.astype(np.int64)

        return super().from_data(data).to_integer()

    @classmethod
    def from_empirical(cls, dist: Empirical) -> Binned:
        """Takes an Empirical instance with discrete support and creates a Binned instance by casting the support to integer values and filling the gaps in the support


        Args:
            empirical_dist (Empirical): Empirical instance

        Returns:
            Binned: Binned distribution


        """
        empirical_dist = dist.map(lambda x: np.round(x))
        base_support = empirical_dist.support.astype(Binned._supported_types[0])
        full_support = np.arange(min(base_support), max(base_support) + 1)

        full_pdf = np.zeros(
            (
                len(
                    full_support,
                )
            ),
            dtype=np.float64,
        )
        indices = base_support - min(base_support)
        full_pdf[indices] = empirical_dist.pdf_values
        # try:
        #   full_pdf[indices] = empirical_dist.pdf_values
        # except Exception as e:
        #   print(f"base support: {base_support}")
        #   print(f"full support: {full_support}")

        data = (
            None
            if empirical_dist.data is None
            else empirical_dist.data.astype(Binned._supported_types[0])
        )

        return Binned(
            support=full_support.astype(cls._supported_types[0]),
            pdf_values=full_pdf,
            data=data,
        )

    def pdf(self, x: t.Union[float, np.ndarray], **kwargs):

        if isinstance(x, np.ndarray):
            return np.array([self.pdf(elem) for elem in x])

        if not isinstance(x, (int, np.int32, np.int64)) or x > self.max or x < self.min:
            return 0.0
        else:
            return self.pdf_values[x - self.min]


class EmpiricalWithGPTail(Mixture):

    """Represents a semiparametric extreme value model with a fitted Generalized Pareto distribution above a certain threshold, and an empirical distribution below it"""

    threshold: float

    def __repr__(self):
        return f"Semiparametric model with generalised Pareto tail. Modeling threshold: {self.threshold}, exceedance probability: {self.exs_prob}"

    @property
    def empirical(self) -> Empirical:
        """Empirical distribution below the modeling threshold

        Returns:
            Empirical: Distribution object
        """
        return self.distributions[0]

    @property
    def tail(self) -> GPTail:
        """Generalised Pareto tail model above modeling threshold

        Returns:
            GPTail: Distribution
        """
        return self.distributions[1]

    # @property
    # def threshold(self) -> float:
    #   """Modeling threshold

    #   Returns:
    #       float: threshold value
    #   """
    #   return self.distributions[1].thresholds

    @property
    def exs_prob(self) -> float:
        """Probability mass above threshold; probability weight of tail model.

        Returns:
            float: weight
        """
        return self.weights[1]

    def ppf(self, q: t.Union[float, np.ndarray]) -> t.Union[float, np.ndarray]:

        is_scalar = isinstance(q, self._allowed_scalar_types)

        if is_scalar:
            q = np.array([q])

        lower_idx = q <= 1 - self.exs_prob
        higher_idx = np.logical_not(lower_idx)

        ppf_values = np.empty((len(q),))
        ppf_values[lower_idx] = self.empirical.ppf(q[lower_idx] / (1 - self.exs_prob))
        ppf_values[higher_idx] = self.tail.ppf(
            (q[higher_idx] - (1 - self.exs_prob)) / self.exs_prob
        )

        if is_scalar:
            return ppf_values[0]
        else:
            return ppf_values

    @classmethod
    def from_data(
        cls, data: np.ndarray, threshold: float, bin_empirical: bool = False, **kwargs
    ) -> EmpiricalWithGPTail:
        """Fits a model from a given data array and threshold value

        Args:
            data (np.ndarray): Data
            threshold (float): Threshold value to use for the tail model
            bin_empirical (bool, optional): Whether to cast empirical mixture component to an integer distribution by rounding
            **kwargs: Additional arguments passed to GPTail.fit


        Returns:
            EmpiricalWithGPTail: Fitted model
        """
        exs_prob = 1 - Empirical.from_data(data).cdf(threshold)

        exceedances = data[data > threshold]

        empirical = Empirical.from_data(data[data <= threshold])
        if bin_empirical:
            empirical = empirical.to_integer()

        tail = GPTail.fit(data=exceedances, threshold=threshold, **kwargs)

        return cls(
            distributions=[empirical, tail],
            weights=np.array([1 - exs_prob, exs_prob]),
            threshold=threshold,
        )

    def plot_diagnostics(self) -> matplotlib.figure.Figure:
        return self.tail.plot_diagnostics()

    def plot_return_levels(self) -> matplotlib.figure.Figure:
        """Returns a figure with a return level plot using the fitted tail model

        Returns:
            matplotlib.figure.Figure: figure

        """
        fig = plt.figure()
        scale, shape = self.tail.scale, self.tail.shape
        exs_prob = self.exs_prob

        if self.tail.data is None:
            n_obs = (
                np.Inf
            )  # this only means that threshold estimation variance is ignored in figure confidence bounds
        else:
            n_obs = len(self.tail.data)

        # exceedance_frequency = 1/np.logspace(1,4,20)
        # exceedance_frequency = exceedance_frequency[exceedance_frequency < exs_prob] #plot only levels inside fitted tail model
        # shown return levels go from largest power of 10th below exceedance prob, to 1/10000-th of that.

        x_min = np.floor(np.log(exs_prob) / np.log(10))
        x_max = x_min - 4
        exceedance_frequency = 10 ** (np.linspace(x_min, x_max, 50))
        return_levels = self.ppf(1 - exceedance_frequency)

        plt.plot(
            1.0 / exceedance_frequency,
            return_levels,
            color=self._figure_color_palette[0],
        )
        plt.xscale("log")
        plt.title(" Return levels")
        plt.xlabel("Return period")
        plt.ylabel("Return level")
        plt.grid()

        try:
            # for this bit, look at An Introduction to Statistical selfing of Extreme Values, p.82
            mle_cov = self.tail.mle_cov()
            eigenvals, eigenvecs = np.linalg.eig(mle_cov)
            if np.all(eigenvals > 0):
                covariance = np.eye(3)
                covariance[1::, 1::] = mle_cov
                covariance[0, 0] = exs_prob * (1 - exs_prob) / n_obs
                #
                return_stdevs = []
                for m in 1.0 / exceedance_frequency:
                    quantile_grad = np.array(
                        [
                            scale * m * (m * exs_prob) ** (shape - 1),
                            1 / shape * ((exs_prob * m) ** shape - 1),
                            -scale / shape**2 * ((exs_prob * m) ** shape - 1)
                            + scale
                            / shape
                            * (exs_prob * m) ** shape
                            * np.log(exs_prob * m),
                        ]
                    )
                    #
                    sdev = np.sqrt(quantile_grad.T.dot(covariance).dot(quantile_grad))
                    return_stdevs.append(sdev)
                #
                return_stdevs = np.array(return_stdevs)
                plt.fill_between(
                    1.0 / exceedance_frequency,
                    return_levels - 1.96 * return_stdevs,
                    return_levels + 1.96 * return_stdevs,
                    alpha=0.2,
                    color=self._figure_color_palette[1],
                    linestyle="dashed",
                )
            else:
                warnings.warn(
                    "Covariance MLE matrix is not positive definite; it might be ill-conditioned",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"Confidence bands for return level could not be calculated; covariance matrix might be ill-conditioned; full trace: {traceback.format_exc()}",
                stacklevel=2,
            )

        return fig


class BayesianGPTail(GPTailMixture):

    """Generalised Pareto tail model which is fitted through Bayesian inference, using uninformative (uniform) priors for the shape and scale parameters

    Args:
        threshold (float): modeling threshold
        data (np.array, optional): exceedance data
        shape (np.ndarray): sample from posterior shape distribution
        scale (np.ndarray): sample from posterior scale distribution
    """

    # _self.posterior_trace = None

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        threshold: float,
        max_posterior_samples: int = 1000,
        chain_length: int = 2000,
        x0: np.ndarray = None,
        plot_diagnostics: bool = False,
        n_walkers: int = 32,
        n_cores: int = 4,
        burn_in: int = 100,
        thinning: int = None,
        log_prior: t.Callable = None,
    ) -> BayesianGPTail:
        """Fits a Generalised Pareto model through Bayesian inference using exceedance data, starting with flat, uninformative priors for both and sampling from posterior shape and scale parameter distributions.

        Args:
            data (np.ndarray): observational data
            threshold (float): modeling threshold; location parameter for Generalised Pareto model
            max_posterior_samples (int, optional): Maximum number of posterior samples to keep
            chain_length (int, optional): timesteps in each chain
            x0 (np.ndarray, optional): Starting point for the chains. If None, MLE estimates are used.
            plot_diagnostics (bool, optional): If True, plots MCMC diagnostics in the background.
            n_walkers (int, optional): Number of concurrent paths to use
            n_cores (int, optional): Number of cores to use
            burn_in (int, optional): Number of initial samples to drop
            thinning (int, optional): Thinning factor to reduce autocorrelation; if None, an automatic estimate from emcee's `get_autocorr_time` is used.
            log_prior (t.Callable, optional): Function that takes as input a single length-2 iterable with scale and shape parameters and outputs the prior log-likelihood. If None, a constant prior on the valid parameter support is used.

        Returns:
            BayesianGPTail: fitted model

        """
        exceedances = data[data > threshold]
        x_max = max(data - threshold)

        # def log_likelihood(theta, data):
        #   scale, shape = theta
        #   return np.sum(gpdist.logpdf(data, c=shape, scale=scale, loc=threshold))

        if log_prior is None:

            def log_prior(theta):
                scale, shape = theta
                if scale > 0 and shape > -scale / (x_max):
                    return 0.0
                else:
                    return -np.Inf

        def log_probability(theta, data):
            prior = log_prior(theta)
            if np.isfinite(prior):
                ll = prior + GPTail.loglik(
                    theta, threshold, data
                )  # log_likelihood(theta, data)
                # print(theta, ll)
                return ll
            else:
                return -np.Inf

        exceedances = data[data > threshold]
        ndim = 2

        if x0 is None:
            # make initial guess
            mle_model = GPTail.fit(data=exceedances, threshold=threshold)
            shape, scale = mle_model.shape, mle_model.scale
            x0 = np.array([scale, shape])

        # create random walkers
        pos = x0 + 1e-4 * np.random.randn(n_walkers, ndim)

        with Pool(n_cores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=ndim,
                log_prob_fn=log_probability,
                args=(exceedances,),
            )
            sampler.run_mcmc(pos, chain_length, progress=True)

        samples = sampler.get_chain()

        tau = sampler.get_autocorr_time()
        thinning = int(np.round(np.mean(tau)))
        print(
            f"Using a thinning factor of {thinning} (from emcee.EnsembleSampler.get_autocorr_time)"
        )
        flat_samples = sampler.get_chain(discard=burn_in, thin=thinning, flat=True)

        if flat_samples.shape[0] > max_posterior_samples:
            np.random.shuffle(flat_samples)
            flat_samples = flat_samples[0:max_posterior_samples, :]

        n_samples = flat_samples.shape[0]
        print(f"Got {n_samples} posterior samples.")

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
            print("Use pyplot.show() to view chain diagnostics.")

        scale_posterior = flat_samples[:, 0]
        shape_posterior = flat_samples[:, 1]

        return cls(
            weights=1 / n_samples * np.ones((n_samples,), dtype=np.float64),
            thresholds=threshold * np.ones((n_samples,)),
            data=exceedances,
            shapes=shape_posterior,
            scales=scale_posterior,
        )

    def plot_diagnostics(self) -> matplotlib.figure.Figure:
        """Returns a figure with fit diagnostic plots for the GP model

        Returns:
            matplotlib.figure.Figure: figure

        """
        if self.data is None:
            raise ValueError("Exceedance data was not provided for this model.")

        def map_to_colors(vals):
            colours = np.zeros((len(vals), 3))
            norm = Normalize(vmin=vals.min(), vmax=vals.max())
            # Can put any colormap you like here.
            colours = [
                cm.ScalarMappable(norm=norm, cmap="cool").to_rgba(val) for val in vals
            ]
            return colours

        fig, axs = plt.subplots(3, 2)

        #################### Bayesian inference diagnostics: posterior histograms and scatterplot

        axs[0, 0].hist(
            self.scales, bins=25, edgecolor="white", color=self._figure_color_palette[0]
        )
        axs[0, 0].title.set_text("Posterior scale histogram")

        axs[0, 1].hist(
            self.shapes, bins=25, edgecolor="white", color=self._figure_color_palette[0]
        )
        axs[0, 1].title.set_text("Posterior shape histogram")

        posterior = np.concatenate(
            [self.scales.reshape((-1, 1)), self.shapes.reshape((-1, 1))], axis=1
        )
        kernel = kde(posterior.T)

        colours = map_to_colors(kernel.evaluate(posterior.T))

        axs[1, 0].scatter(self.scales, self.shapes, color=colours)
        axs[1, 0].title.set_text("Posterior sample")
        axs[1, 0].set_xlabel("Scale")
        axs[1, 0].set_ylabel("Shape")

        ############## histogram vs density ################
        hist_data = axs[1, 1].hist(
            self.data, bins=25, edgecolor="white", color=self._figure_color_palette[0]
        )

        range_min, range_max = min(self.data), max(self.data)
        x_axis = np.linspace(range_min, range_max, 100)

        mean_pdf_vals = np.array([self.pdf(x) for x in x_axis])
        # q025_pdf_vals = np.array( [np.quantile(self.pdf(x, return_all=True),0.025) for x in x_axis])
        # q975_pdf_vals = np.array( [np.quantile(self.pdf(x, return_all=True),0.975) for x in x_axis])

        y_axis = hist_data[0][0] / mean_pdf_vals[0] * mean_pdf_vals

        axs[1, 1].plot(x_axis, y_axis, color=self._figure_color_palette[1])
        # axs[1,1].fill_between(x_axis, q025_pdf_vals, q975_pdf_vals, alpha=0.2, color=self._figure_color_palette[1])
        axs[1, 1].title.set_text("Data vs fitted density")
        axs[1, 1].set_xlabel("Exceedance data")
        axs[1, 1].yaxis.set_visible(False)  # Hide only x axis
        # axs[0, 0].set_aspect('equal', 'box')

        ############# Q-Q plot ################
        probability_range = np.linspace(0.01, 0.99, 99)
        empirical_quantiles = np.quantile(self.data, probability_range)

        posterior_quantiles = [self.ppf(p, return_all=True) for p in probability_range]

        # hat_return_levels are not the mean of posterior return level samples, as the mean is not an unbiased estimator
        hat_tail_quantiles = np.array([self.ppf(p) for p in probability_range])

        # q025_tail_quantiles = np.array([np.quantile(q, 0.025) for q in posterior_quantiles])
        # q975_tail_quantiles = np.array([np.quantile(q, 0.975) for q in posterior_quantiles])

        axs[2, 0].scatter(
            hat_tail_quantiles, empirical_quantiles, color=self._figure_color_palette[0]
        )
        min_x, max_x = min(hat_tail_quantiles), max(hat_tail_quantiles)
        # axs[0,1].set_aspect('equal', 'box')
        axs[2, 0].title.set_text("Q-Q plot")
        axs[2, 0].set_xlabel("model quantiles")
        axs[2, 0].set_ylabel("Exceedance quantiles")
        axs[2, 0].grid()
        axs[2, 0].plot([min_x, max_x], [min_x, max_x], linestyle="--", color="black")
        # axs[2,0].fill_between(hat_tail_quantiles, q025_tail_quantiles, q975_tail_quantiles, alpha=0.2, color=self._figure_color_palette[1])

        plt.tight_layout()
        return fig


class EmpiricalWithBayesianGPTail(EmpiricalWithGPTail):

    """Semiparametric Bayesian model with an empirical data distribution below a specified threshold and a Generalised Pareto exceedance model above it, fitted through Bayesian inference."""

    @classmethod
    def from_data(
        cls, data: np.ndarray, threshold: float, bin_empirical: bool = False, **kwargs
    ) -> EmpiricalWithBayesianGPTail:
        """Fits a Generalied Pareto tail model from a given data array and threshold value, using Jeffrey's priors

        Args:
            data (np.ndarray): data array
            threshold (float): Threshold value to use for the tail model
            bin_empirical (bool, optional): Whether to cast empirical mixture component to an integer distribution by rounding
            **kwargs: Additional arguments to be passed to BayesianGPTail.fit

        Returns:
            EmpiricalWithBayesianGPTail: Fitted model

        Deleted Parameters:
            n_posterior_samples (int): Number of samples from posterior distribution

        """
        exs_prob = 1 - Empirical.from_data(data).cdf(threshold)

        exceedances = data[data > threshold]

        empirical = Empirical.from_data(data[data <= threshold])
        if bin_empirical:
            empirical = empirical.to_integer()

        tail = BayesianGPTail.fit(data=exceedances, threshold=threshold, **kwargs)

        return cls(
            weights=np.array([1 - exs_prob, exs_prob]),
            distributions=[empirical, tail],
            threshold=threshold,
        )

    def ppf(
        self, q: t.Union[float, np.ndarray], return_all=False
    ) -> t.Union[float, np.ndarray]:
        """Returns the quantile function evaluated at some probability level

        Args:
            q (t.Union[float, np.ndarray]): probability level
            return_all (bool, optional): If True, returns posterior ppf sample; otherwise return pointwise estimator

        Returns:
            t.Union[float, np.ndarray]: ppf value(s).
        """
        if isinstance(q, np.ndarray):
            return np.array([self.ppf(elem) for elem in q])

        if q <= 1 - self.exs_prob:
            val = self.empirical.ppf(q / (1 - self.exs_prob), return_all=return_all)
            # if a vector is expected as output, vectorize scalar
            if return_all:
                return val * np.ones((len(self.tail.shapes),))
            else:
                return val
        else:
            return self.tail.ppf(
                (q - (1 - self.exs_prob)) / self.exs_prob, return_all=return_all
            )

    def plot_return_levels(self) -> matplotlib.figure.Figure:
        """Returns a figure with a return levels using the fitted tail model

        Returns:
            matplotlib.figure.Figure: figure

        """
        fig = plt.figure()
        exs_prob = self.exs_prob

        exceedance_frequency = 1 / np.logspace(1, 4, 20)
        exceedance_frequency = exceedance_frequency[
            exceedance_frequency < exs_prob
        ]  # plot only levels inside fitted tail model

        return_levels = [self.ppf(1 - x, return_all=True) for x in exceedance_frequency]

        # hat_return_levels are not the mean of posterior return level samples, as the mean is not an unbiased estimator
        hat_return_levels = np.array(
            [self.ppf(1 - x, return_all=False) for x in exceedance_frequency]
        )
        q025_return_levels = np.array([np.quantile(r, 0.025) for r in return_levels])
        q975_return_levels = np.array([np.quantile(r, 0.975) for r in return_levels])

        plt.plot(
            1.0 / exceedance_frequency,
            hat_return_levels,
            color=self._figure_color_palette[0],
        )
        plt.fill_between(
            1.0 / exceedance_frequency,
            q025_return_levels,
            q975_return_levels,
            alpha=0.2,
            color=self._figure_color_palette[1],
            linestyle="dashed",
        )
        plt.xscale("log")
        plt.title("Exceedance return levels")
        plt.xlabel("1/frequency")
        plt.ylabel("Return level")
        plt.grid()

        return fig
