"""This module contains bivariate risk models to analyse exceedance dependence between components. Available exceedance models are inspired in bivariate generalised Pareto models, whose support is an inverted L-shaped subset of Euclidean space, where at least one component takes an extreme value above a specified threshold. Available parametric models in this module include the logistic model, equivalent to a Gumbel-Hougaard copula between exceedances, and a Gaussian model, equivalent to a Gaussian copula. The former exhibits asymptotic dependence and the latter asymtptotic independence, which characterises the dependence of extremes across components.
Finally, `Empirical` instances have methods to assess asymptotic dependence vs independence through hypothesis tests and visual inspection of the Pickands dependence function.
"""

from __future__ import annotations

import logging
import time
import typing as t
import traceback
import warnings
from abc import ABC, abstractmethod
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

from scipy.optimize import LinearConstraint, minimize, root_scalar, approx_fprime
from scipy.special import lambertw
from scipy.stats import (
    genpareto as gpdist,
    expon as exponential,
    gumbel_r as gumbel,
    norm as gaussian,
    multivariate_normal as mv_gaussian,
    gaussian_kde,
    rv_continuous as continuous_dist,
)

from pydantic import BaseModel, ValidationError, validator, PositiveFloat, Field
from functools import reduce

import riskmodels.univariate as univar

from riskmodels.utils.tmvn import TruncatedMVN as tmvn


class BaseDistribution(BaseModel, ABC):

    """Base interface for bivariate distributions"""

    _allowed_scalar_types = (int, float, np.int64, np.int32, np.float32, np.float64)
    _figure_color_palette = ["tab:cyan", "deeppink"]
    _error_tol = 1e-6

    data: t.Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return "Base distribution object"

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def pdf(self, x: np.ndarray) -> float:
        """Evaluate probability density function"""
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray):
        """Evaluate cumulative distribution function"""
        pass

    @abstractmethod
    def simulate(self, size: int):
        """Simulate from bivariate distribution"""
        pass

    def plot(self, size: int = 1000) -> matplotlib.figure.Figure:
        """Sample distribution and produce scatterplots and histograms

        Args:
            size (int, optional): Sample size

        Returns:
            matplotlib.figure.Figure: figure
        """
        sample = self.simulate(size)

        x = sample[:, 0]
        y = sample[:, 1]

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
        ax_scatter.tick_params(direction="in", top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction="in", labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction="in", labelleft=False)

        # the scatter plot:
        ax_scatter.scatter(x, y, color=self._figure_color_palette[0], alpha=0.35)

        # now determine nice limits by hand:
        # binwidth = 0.25
        # lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
        # ax_scatter.set_xlim((-lim, lim))
        # ax_scatter.set_ylim((-lim, lim))

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(
            x, bins=25, color=self._figure_color_palette[0], edgecolor="white"
        )
        # plt.title(f"Scatter plot from {np.round(size/1000,1)}K simulated samples")
        ax_histy.hist(
            y,
            bins=25,
            orientation="horizontal",
            color=self._figure_color_palette[0],
            edgecolor="white",
        )

        # ax_histx.set_xlim(ax_scatter.get_xlim())
        # ax_histy.set_ylim(ax_scatter.get_ylim())
        plt.tight_layout()
        return fig


class Frechet(object):
    """Minimal implementation of a unit Frechet distribution"""

    @classmethod
    def cdf(cls, x: np.ndarray):
        return np.exp(-1 / x)

    @classmethod
    def ppf(cls, p: np.ndarray):
        return -1.0 / np.log(p)


class Mixture(BaseDistribution):

    """Base interface for a bivariate mixture distribution"""

    distributions: t.List[BaseDistribution]
    weights: np.ndarray

    def __repr__(self):
        return f"Mixture with {len(self.weights)} components"

    def simulate(self, size: int) -> np.ndarray:

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

    def cdf(self, x: np.ndarray, **kwargs) -> float:
        vals = [
            w * dist.cdf(x, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)

    def pdf(self, x: np.ndarray, **kwargs) -> float:

        vals = [
            w * dist.pdf(x, **kwargs)
            for w, dist in zip(self.weights, self.distributions)
        ]
        return reduce(lambda x, y: x + y, vals)


class ExceedanceModel(Mixture):

    """Interface for exceedance models. This is a mixture of an empirical distribution with support below the exceedance threshold and an exceedance distribution (see `ExceedanceDistribution`) above it."""

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

    """Bivariate distribution with independent components"""

    x: univar.BaseDistribution
    y: univar.BaseDistribution

    def __repr__(self):
        return f"Independent bivariate distribution with marginals:\nx:{self.x.__repr__()}\ny:{self.y.__repr__()}"

    def pdf(self, x: np.ndarray):
        x1, x2 = x
        return self.x.pdf(x1) * self.y.pdf(x2)

    def cdf(self, x: np.ndarray):
        x1, x2 = x
        return self.x.cdf(x1) * self.y.cdf(x2)

    def simulate(self, size: int):
        return np.concatenate(
            [
                self.x.simulate(size).reshape((-1, 1)),
                self.y.simulate(size).reshape((-1, 1)),
            ],
            axis=1,
        )


class ExceedanceDistribution(BaseDistribution):

    """Main interface for exceedance distributions, which are defined on a region of the form \\( U \\nleq u \\), or equivalently \\( \\max\\{U_1,U_2\\} > u \\)."""

    quantile_threshold: float

    margin1: t.Union[univar.BaseDistribution, continuous_dist]
    margin2: t.Union[univar.BaseDistribution, continuous_dist]

    # default method variables below are the same as for the logistic model
    # this does not matter as this class should not be instantiated directly
    params: t.Optional[np.ndarray] = Field(
        default_factory=lambda: np.zeros((1,), dtype=np.float32)
    )
    _param_names = {
        0: "alpha"
    }  # mapping from params array indices to names for diagnostic plots
    _model_marginal_dist = gumbel
    _plotting_dist_name = "Gumbel"
    _default_x0 = np.array([0.0])

    @validator("data", allow_reuse=True)
    def validate_data(cls, data):
        if data is not None and (
            len(data.shape) != 2 or data.shape[1] != 2 or np.any(np.isnan(data))
        ):
            raise ValueError("Data is not an n x 2 numpy array")
        else:
            return data

    @classmethod
    @abstractmethod
    def unconditioned_cdf(cls, params: np.ndarray, data: np.ndarray):
        """Computes the raw model cdf, without conditioning to the exceedance region.

        Args:
            params (np.ndarray): array with dependence parameter as only element
            data (t.Union[np.ndarray, t.Iterable]): Observed data in model scale
        """
        pass

    @classmethod
    @abstractmethod
    def logpdf(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates logpdf function for exceedance distribution


        Args:
            params (np.ndarray): array with model parameters
            threshold (float): Exceedance threshold in model scale
            data (t.Union[np.ndarray, t.Iterable]): Observed data in model scale
        """
        pass

    @property
    def model_scale_threshold(self):
        return self._model_marginal_dist.ppf(self.quantile_threshold)

    @property
    def data_scale_threshold(self):
        return self.model_to_data_dist(
            self.bundle(self.model_scale_threshold, self.model_scale_threshold)
        )

    @property
    def mle_cov(self):
        return -np.linalg.inv(
            self.hessian(
                self.params,
                self.model_scale_threshold,
                self.data_to_model_dist(self.data),
            )
        )

    @classmethod
    def unbundle(
        cls, data: t.Union[np.ndarray, t.Iterable]
    ) -> t.Tuple[t.Union[np.ndarray, float], t.Union[np.ndarray, float]]:
        """Unbundles matrix or iterables into separate components

        Args:
            data (t.Union[np.ndarray, t.Iterable]): dara

        """
        if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        elif isinstance(data, Iterable):
            # if iterable, unroll
            x, y = data
        else:
            raise TypeError(
                "data must be an n x 2 numpy array or an iterable of length 2."
            )
        return x, y

    @classmethod
    def bundle(
        cls, x: t.Union[np.ndarray, float, int], y: t.Union[np.ndarray, float, int]
    ) -> t.Tuple[t.Union[np.ndarray, float], t.Union[np.ndarray, float]]:
        """bundle a pair of arrays or primitives into n x 2 matrix

        Args:
            data (t.Union[np.ndarray, t.Iterable])

        """
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and len(x) == len(y):
            z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
        elif issubclass(type(x), (float, int)) and issubclass(type(y), (float, int)):
            z = np.array([x, y]).reshape((1, 2))
        else:
            raise TypeError(
                "x, y must be 1-dimensional arrays or inherit from float or int."
            )
        return z

    def data_to_model_dist(self, data: t.Union[np.ndarray, t.Iterable]) -> np.ndarray:
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

        return self.bundle(x, y)

    def model_to_data_dist(self, data: t.Union[np.ndarray, t.Iterable]) -> np.ndarray:
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

        return self.bundle(u, w)

    def simulate(self, size: int):

        return self.model_to_data_dist(
            self.simulate_model(size, self.params, self.quantile_threshold)
        )

    @classmethod
    def loglik(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Computes log-likelihood for threshold exceedances"""
        return np.sum(cls.logpdf(params, threshold, data))

    def cdf(self, x: np.ndarray):
        mapped_data = self.data_to_model_dist(x)
        model_threshold = self.model_scale_threshold
        u = np.minimum(mapped_data, model_threshold)
        norm_factor = float(
            1
            - self.unconditioned_cdf(
                self.params, self.bundle(model_threshold, model_threshold)
            )
        )

        return (
            self.unconditioned_cdf(self.params, mapped_data)
            - self.unconditioned_cdf(self.params, u)
        ) / norm_factor

    def pdf(self, x: np.ndarray, eps=1e-5) -> np.ndarray:
        """Numerical approximation to the model's pdf in the original data scale. This is only non-zero when both marginal distributions are continuous on x

        Args:
            x (np.ndarray): Points to evaluate
            eps (float, optional): Numeric delta

        Returns:
            np.ndarray: pdf approximation
        """
        x1, x2 = self.unbundle(x)
        model_scale_data = self.data_to_model_dist(x)
        n = len(model_scale_data)

        e1, e2 = (
            np.stack(
                [np.ones((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)],
                axis=1,
            ),
            np.stack(
                [np.zeros((n,), dtype=np.float32), np.ones((n,), dtype=np.float32)],
                axis=1,
            ),
        )

        dz1_dx1 = (
            self.data_to_model_dist(x + eps * e1)[:, 0]
            - self.data_to_model_dist(x - eps * e1)[:, 0]
        ) / (2 * eps)
        dz2_dx2 = (
            self.data_to_model_dist(x + eps * e2)[:, 1]
            - self.data_to_model_dist(x - eps * e2)[:, 1]
        ) / (2 * eps)

        return (
            np.exp(self.logpdf(self.params, self.quantile_threshold, model_scale_data))
            * dz1_dx1
            * dz2_dx2
        )

    @classmethod
    def fit(
        cls,
        data: t.Union[np.ndarray, t.Iterable],
        quantile_threshold: float,
        margin1: univar.BaseDistribution = None,
        margin2: univar.BaseDistribution = None,
        return_opt_results=False,
        x0: float = None,
    ) -> Logistic:
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
            margin1 = univar.empirical.from_data(data[:, 0])
            warnings.warn(
                "margin1 is None; using an empirical distribution", stacklevel=2
            )

        if margin2 is None:
            margin1 = univar.empirical.from_data(data[:, 1])
            warnings.warn(
                "margin1 is None; using an empirical distribution", stacklevel=2
            )

        if (
            not isinstance(quantile_threshold, float)
            or quantile_threshold <= 0
            or quantile_threshold >= 1
        ):
            raise ValueError("quantile_threshold must be in the open interval (0,1)")

        mapped_data = cls(
            margin1=margin1,
            margin2=margin2,
            quantile_threshold=quantile_threshold,
        ).data_to_model_dist(data)

        x, y = cls.unbundle(mapped_data)

        # get threshold exceedances
        model_scale_threshold = cls._model_marginal_dist.ppf(quantile_threshold)

        exs_idx = np.logical_or(x > model_scale_threshold, y > model_scale_threshold)
        x = x[exs_idx]
        y = y[exs_idx]

        x0 = cls._default_x0 if x0 is None else x0

        mapped_exceedances = cls.bundle(x, y)
        n = len(mapped_exceedances)

        def logistic(x):
            return 1.0 / (1 + np.exp(-x))

        def loss(x, data):
            params = logistic(x)
            # optimise normalised log-likelihood
            return -cls.loglik(params, model_scale_threshold, data) / n

        res = minimize(fun=loss, x0=x0, method="BFGS", args=(mapped_exceedances,))

        if return_opt_results:
            return res

        return cls(
            quantile_threshold=quantile_threshold,
            params=logistic(res.x),
            data=data[exs_idx, :],
            margin1=margin1,
            margin2=margin2,
        )

    @classmethod
    def hessian(
        cls,
        params: np.ndarray,
        threshold: float,
        data: t.Union[np.ndarray, t.Iterable],
        eps=1e-6,
    ) -> np.ndarray:
        """Numerical approximation for the loglikelihood function's Hessian

        Args:
            params (np.ndarray): parameter array
            threshold (float): Threshold in standard scale (i.e. Gumbel or Gaussian)
            data (t.Union[np.ndarray, t.Iterable]): Data in standard scale (i.e. Gumbel or Gaussian)
            eps (float, optional): Numerical delta in each component

        Returns:
            np.ndarray: Hessian matrix

        """
        x0 = params
        grad0 = approx_fprime(x0, cls.loglik, eps, threshold, data)
        n = len(x0)
        hessian = np.zeros((n, n), dtype=np.float32)
        # The next loop fill in the matrix
        xx = x0
        for j in range(n):
            xx0 = xx[j]  # Store old value
            xx[j] = xx0 + eps  # Perturb with finite difference
            # Recalculate the partial derivatives for this new point
            current_grad = approx_fprime(xx, cls.loglik, eps, threshold, data)
            hessian[:, j] = (current_grad - grad0) / eps  # scale...
            xx[j] = xx0  # Restore initial value of x0
        return hessian

    def plot_diagnostics(self, eps=1e-6):
        """Returns a figure with the fitted exceedance model's profile log-likelihoods and fitted density in model scale.

        Args:
            eps (float, optional): numeric delta when calculating the Hessian matrix numerically; this is used to plot profile loglikelihoods.

        """
        # unbundle data
        if self.data is None:
            raise ValueError(
                "Diagnostics cannot be shown for models with no underlying data."
            )

        x, y = self.unbundle(self.data)
        z1, z2 = self.unbundle(self.data_to_model_dist(self.data))
        n = len(z1)
        params = self.params

        # create plot mosaic
        n_params = len(params)
        r, c = int(np.ceil(n_params / 2 + 1 / 2)), 2
        # this function is needed to handle mosaics with a varying number of rows
        def get_subplot(k):
            if r == 1:
                return axs[k]
            else:
                return axs[k // 2, k % 2]

        fig, axs = plt.subplots(r, c)

        # compute mle sdev
        model_scale_threshold = self.model_scale_threshold
        try:
            hessian = self.hessian(
                params, model_scale_threshold, self.bundle(z1, z2), eps
            )
            sdevs = np.sqrt(-np.linalg.inv(hessian))
        except Exception as e:
            raise Exception(
                f"Error when estimating fitted parameter variances; if fitted parameters are at the edge of their support, passing a lower eps value might help. Full trace: {traceback.format_exc()}"
            )
        # create profile log-likelihood plots
        for k in range(n_params):
            param = params[k]
            sdev = sdevs[k, k]
            grid = np.linspace(param - 1.96 * sdev, param + 1.96 * sdev, 100)
            feasible_grid = grid[np.logical_and(grid > 0, grid < 1)]
            perturbed_params = np.copy(params)
            profile_ll = []
            for x in feasible_grid:
                perturbed_params[k] = x
                profile_ll.append(
                    self.loglik(
                        perturbed_params, model_scale_threshold, self.bundle(z1, z2)
                    )
                )
            # filter to almost optimal values
            profile_ll = np.array(profile_ll)
            max_ll = max(profile_ll)
            almost_optimal = np.abs(profile_ll - max_ll) < np.abs(2 * max_ll)
            profile_ll = profile_ll[almost_optimal]
            feasible_grid = feasible_grid[almost_optimal]
            get_subplot(k).plot(
                feasible_grid, profile_ll, color=self._figure_color_palette[0]
            )
            get_subplot(k).vlines(
                x=param,
                ymin=min(profile_ll),
                ymax=max(profile_ll),
                linestyle="dashed",
                colors=self._figure_color_palette[1],
            )
            get_subplot(k).title.set_text("Profile log-likelihood")
            get_subplot(k).set_xlabel(self._param_names.get(k, f"params[{k}]"))
            get_subplot(k).set_ylabel("")
            get_subplot(k).grid()

        ### last subplot contains the fitted density in model scale
        # avoid plotting in Frechet scale as it makes it difficult to see anything
        if isinstance(self._model_marginal_dist, Frechet):
            # convert Frechet data to Gumbel
            z1 = np.log(z1)
            z2 = np.log(z2)
            dist_name = "Gumbel"
            logpdf = Logistic.logpdf
            contour_dist_threshold = np.log(model_scale_threshold)
        else:
            dist_name = self._plotting_dist_name
            logpdf = self.logpdf
            contour_dist_threshold = model_scale_threshold

        x_range = np.linspace(np.min(z1), np.max(z1), 50)
        y_range = np.linspace(np.min(z2), np.max(z2), 50)

        X, Y = np.meshgrid(x_range, y_range)
        bundled_grid = self.bundle(X.reshape((-1, 1)), Y.reshape((-1, 1)))
        Z = logpdf(
            data=bundled_grid, threshold=contour_dist_threshold, params=params
        ).reshape(X.shape)
        get_subplot(-1).contourf(X, Y, Z)
        get_subplot(-1).scatter(z1, z2, color=self._figure_color_palette[1], s=0.9)
        get_subplot(-1).title.set_text(f"Model density ({dist_name} scale)")
        get_subplot(-1).set_xlabel("x")
        get_subplot(-1).set_ylabel("y")

        plt.tight_layout()
        return fig


class Logistic(ExceedanceDistribution):

    """This model assumes association between exceedances at different components follow a Gumbel-Hougaard copula; in Gumbel scale, this is given by

    $$ \\mathbb{P}(\\textbf{X} \\leq \\mathbf{x}) = \\exp \\left(- \\left(  \\exp \\left( - \\frac{\\mathbf{x}_1}{\\alpha} \\right)+ \\left(  - \\frac{\\mathbf{x}_2}{\\alpha}  \\right) \\right)^\\alpha \\right), \\, 0 \\leq \\alpha\\leq 1, \\, \\mathbf{X} \\in \\mathbb{R}^2$$

    Exceedances in each component are defined as observations above a fixed quantile threshold \\( \\textbf{q}\\) for a high probability level \\(p \\approx 1\\), and so bivariate exceedances \\(\\textbf{Z}\\) are defined in an inverted-L-shaped region of space, \\( \\textbf{Z} \\nleq \\mathbf{q} \\): that in which there is an exceedance in at least one component. Consequently the model implemented here is only defined in the corresponding inverted-L-shaped region; the functional form of the dependence is the same as a Gumbel-Hougaard copula, but the normalisation constant is different because of this constraint.

    If the underlying marginal distributions follow a generalised Pareto above the quantile thresholds, this model can be seen as a pre-limit version of a bivariate generalised Pareto distribution with a logistic dependence model (see Rootzen and Tajvidi, 2006), to which this model converges as the quantile thresholds grow.
    """

    params: t.Optional[np.ndarray] = Field(
        default_factory=lambda: np.zeros((1,), dtype=np.float32)
    )

    _param_names = {
        0: "alpha"
    }  # mapping from params array indices to names for diagnostic plots

    _model_marginal_dist = gumbel
    # _plotting_dist = gumbel
    _plotting_dist_name = "Gumbel"
    _default_x0 = np.array([0.0])

    def __repr__(self):
        return f"{self.__class__.__name__} exceedance dependence model with alpha = {self.alpha} and quantile threshold {self.quantile_threshold}"

    @validator("params")
    def validate_params(cls, params):
        alpha = params[0]
        if alpha <= 0 or alpha > 1:
            raise TypeError(f"alpha must be in the interval (0,1]")
        else:
            return params

    @property
    def alpha(self):
        return self.params[0]

    @classmethod
    def logpdf(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates logpdf function for Gumbel exceedances


        Args:
            params (np.ndarray): array with dependence parameter as only element
            threshold (float): Exceedance threshold in Gumbel scale
            data (t.Union[np.ndarray, t.Iterable]): Observed data in Gumbel scale

        """
        alpha = params[0]

        x, y = cls.unbundle(data)

        nlogp = (np.exp(-x / alpha) + np.exp(-y / alpha)) ** alpha
        lognlogp = alpha * np.log(np.exp(-x / alpha) + np.exp(-y / alpha))
        rescaler = 1 - cls.unconditioned_cdf(params, cls.bundle(threshold, threshold))

        # a = np.exp((x + y - nlogp*alpha)/alpha)
        log_a = (x + y) / alpha - nlogp

        # b = nlogp
        log_b = lognlogp

        # c = 1 + alpha*(nlogp - 1)
        log_c = np.log(1 + alpha * (nlogp - 1))

        # d = 1.0/(alpha*(np.exp(x/alpha) + np.exp(y/alpha))**2)
        log_d = -(np.log(alpha) + 2 * np.log(np.exp(x / alpha) + np.exp(y / alpha)))

        log_density = log_a + log_b + log_c + log_d - np.log(rescaler)

        # density is 0 when both coordinates are below the threshold
        nil_density_idx = np.logical_and(x <= threshold, y <= threshold)
        log_density[nil_density_idx] = -np.Inf

        return log_density

    @classmethod
    def unconditioned_cdf(
        cls, params: np.ndarray, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates unconstrained standard Gumbel CDF"""
        alpha = params[0]
        x, y = cls.unbundle(data)
        return np.exp(-((np.exp(-x / alpha) + np.exp(-y / alpha)) ** (alpha)))

    @classmethod
    def simulate_model(
        cls, size: int, params: np.ndarray, quantile_threshold: float
    ) -> np.ndarray:
        """Simulates logistic exceedance model in Gumbel scale

        Args:
            size (int): Simulated sample size
            params (np.ndarray): Array with dependence parameter
            threshold (float): Exceedance threshold in both components

        Returns:
            np.ndarray: Simulated sample
        """
        ### simulate in Gumbel scale maximum component: z = max(x1, x2) ~ Gumbel(loc=alpha*np.log(2)) using inverse function method
        threshold = gumbel.ppf(quantile_threshold)
        alpha = params[0]

        q0 = gumbel.cdf(
            threshold, loc=alpha * np.log(2)
        )  # quantile of model's threshold in the maximum's distribution
        u = np.random.uniform(size=size, low=q0)
        maxima = gumbel.ppf(q=u, loc=alpha * np.log(2))

        if alpha == 0:
            r = 0
        elif alpha == 1:
            r = np.log(exponential.rvs(size=size, loc=1, scale=np.exp(maxima)))
        else:
            ###simulate difference between maxima and minima r = max(x,y) - min(x,y) using inverse function method
            u = np.random.uniform(size=size)
            r = (
                alpha
                * np.log(
                    (
                        -(
                            (alpha - 1)
                            * np.exp(maxima)
                            * lambertw(
                                -(
                                    np.exp(
                                        -maxima
                                        - (2**alpha * np.exp(-maxima) * alpha)
                                        / (alpha - 1)
                                    )
                                    * (-(2 ** (alpha - 1)) * (u - 1))
                                    ** (alpha / (alpha - 1))
                                    * alpha
                                )
                                / (alpha - 1)
                            )
                        )
                        / alpha
                    )
                    ** (1 / alpha)
                    - 1
                )
            ).real

        minima = maxima - r

        # allocate maxima randomly between components
        max_indices = np.random.binomial(1, 0.5, size)

        x = np.concatenate(
            [
                maxima[max_indices == 0].reshape((-1, 1)),
                minima[max_indices == 1].reshape((-1, 1)),
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                minima[max_indices == 0].reshape((-1, 1)),
                maxima[max_indices == 1].reshape((-1, 1)),
            ],
            axis=0,
        )

        return cls.bundle(x, y)


class Gaussian(ExceedanceDistribution):

    """This model assumes association between exceedances at different components follow a Gaussian copula. Exceedances in each component are defined as observations above a fixed quantile threshold \\( \\textbf{q}\\) for a high probability level \\(p \\approx 1\\), and so bivariate exceedances \\(\\textbf{Z}\\) are defined in an inverted-L-shaped region of space, \\( \\textbf{Z} \\nleq \\mathbf{q} \\): that in which there is an exceedance in at least one component. Consequently this copula model is only defined in the corresponding inverted-L-shaped region in \\( [\\textbf{0}, \\textbf{1}]\\); the functional form is the same as a Gaussian copula, but the normalisation constant is different.

    If the underlying marginal distributions follow a generalised Pareto above the quantile thresholds, this model can be seen as a pre-limit version of a bivariate generalised Pareto distribution (see Rootzen and Tajvidi, 2006) with a Gaussian dependence model. Because Gaussian copulas are asymptotically independent (this is, dependence  weakens at progressively more extreme levels regardless of the correlation parameter, and disappears in the limit), said limiting model is degenerate, with probability mass at \\(-\\infty\\). This pre-limit model on the other hand is non-degenerate and can be used to model asymptotically independent data.
    """

    _model_marginal_dist = gaussian
    # _plotting_dist = gaussian
    _plotting_dist_name = "Gaussian"
    _param_names = {
        0: "rho"
    }  # mapping from params array indices to names for diagnostic plots

    def __repr__(self):
        return f"{self.__class__.__name__} exceedance dependence model with rho = {self.params} and quantile threshold {self.quantile_threshold}"

    @validator("params")
    def validate_params(cls, params):
        rho = params[0]
        if rho < 0 or rho >= 1:
            raise TypeError(f"rho must be in the interval [0,1)")
        else:
            return params

    @property
    def cov(self):
        rho = self.params[0]
        return np.array([[1, rho], [rho, 1]])

    @property
    def rho(self):
        return self.params[0]

    @classmethod
    def unconditioned_cdf(cls, params: np.ndarray, data: np.ndarray):
        rho = params[0]
        """Calculates unconstrained standard Gaussian CDF"""
        return mv_gaussian.cdf(data, cov=np.array([[1, rho], [rho, 1]]))

    @classmethod
    def logpdf(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates logpdf for Gaussian exceedances"""
        rho = params[0]
        x, y = cls.unbundle(data)
        if isinstance(rho, (list, np.ndarray)):
            rho = rho[0]
        norm_factor = 1 - mv_gaussian.cdf(
            cls.bundle(threshold, threshold), cov=np.array([[1, rho], [rho, 1]])
        )
        density = mv_gaussian.logpdf(data, cov=np.array([[1, rho], [rho, 1]])) - np.log(
            norm_factor
        )

        # density is 0 when both coordinates are below the threshold
        nil_density_idx = np.logical_and(x <= threshold, y <= threshold)
        density[nil_density_idx] = -np.Inf

        return density

    @classmethod
    def simulate_model(
        cls, size: int, params: np.ndarray, quantile_threshold: float
    ) -> np.ndarray:
        """Simulate Gaussian exceedance model in Gaussian scale

        Args:
            size (int): Simulated sample size
            params (np.ndarray): Array with dependence parameter
            threshold (float): Exceedance threshold in both components

        Returns:
            np.ndarray: Simulated sample
        """

        # exceedance subregions:
        # r1 => exceedance in second component only, r2 => exceedance in both components, r3 => exceedance in first component only
        rho = params[0]
        cov = np.array([[1, rho], [rho, 1]])

        if quantile_threshold == 0:
            samples = mv_gaussian.rvs(size=size, cov=cov)
        else:
            threshold = gaussian.ppf(quantile_threshold)

            th = cls.bundle(threshold, threshold)
            th_cdf = mv_gaussian.cdf(th, cov=cov)

            p1 = quantile_threshold - th_cdf
            p2 = 1 - 2 * quantile_threshold + th_cdf
            p3 = 1 - th_cdf - (p1 + p2)

            p = np.array([p1, p2, p3])
            p = p / np.sum(p)

            # compute number of samples per subregion
            n1, n2, n3 = np.random.multinomial(n=size, pvals=p, size=1)[0].astype(
                np.int32
            )
            n1, n2, n3 = int(n1), int(n2), int(n3)

            r1_samples = (
                tmvn(
                    mu=np.zeros((2,)),
                    cov=cov,
                    lb=np.array([-np.Inf, threshold]),
                    ub=np.array([threshold, np.Inf]),
                )
                .sample(n1)
                .T
            )

            r2_samples = (
                tmvn(
                    mu=np.zeros((2,)),
                    cov=cov,
                    lb=np.array([threshold, threshold]),
                    ub=np.array([np.Inf, np.Inf]),
                )
                .sample(n2)
                .T
            )

            r3_samples = (
                tmvn(
                    mu=np.zeros((2,)),
                    cov=cov,
                    lb=np.array([threshold, -np.Inf]),
                    ub=np.array([np.Inf, threshold]),
                )
                .sample(n3)
                .T
            )

            samples = np.concatenate([r1_samples, r2_samples, r3_samples], axis=0)

        return samples


class AsymmetricLogistic(ExceedanceDistribution):

    """This model assumes association between exceedances at different components follow a copula induced by an asymmetric logistic model of extremal dependence; this model in unit Frechet scale is given by

    $$ \\mathbb{P}(\\textbf{X} \\leq \\mathbf{x}) = \\exp \\left( - \\frac{1 - \\beta}{\\mathbf{x}_1} - \\frac{1 - \\gamma}{\\mathbf{x}_2} - \\left(  \\left( \\frac{\\beta}{\\mathbf{x}_1} \\right)^{1/\\alpha} + \\left( \\frac{\\gamma}{\\mathbf{x}_2} \\right)^{1/\\alpha}\\right)^\\alpha \\right), \\, 0 \\leq \\alpha, \\beta, \\gamma \\leq 1, \\, \\mathbf{X} > 0$$

    Exceedances in each component are defined as observations above a fixed quantile threshold \\( \\textbf{q}\\) for a high probability level \\(p \\approx 1\\), and so bivariate exceedances \\(\\textbf{Z}\\) are defined in an inverted-L-shaped region of space, \\( \\textbf{Z} \\nleq \\mathbf{q} \\): that in which there is an exceedance in at least one component. Consequently the model implemented here is only defined in the corresponding inverted-L-shaped region.

    If the underlying marginal distributions follow a generalised Pareto above the quantile thresholds, this model can be seen as a pre-limit version of a bivariate generalised Pareto distribution (see Rootzen and Tajvidi, 2006) with an asymmetric logistic dependence model, to which this model converges as the quantile threshold grows.
    """

    params: t.Optional[np.ndarray] = Field(
        default_factory=lambda: np.zeros((3,), dtype=np.float32)
    )

    _param_names = {
        0: "alpha",
        1: "beta",
        2: "gamma",
    }  # mapping from params array indices to names for diagnostic plots
    _default_x0 = np.array([0, 0, 0])

    _model_marginal_dist = Frechet()

    # _plotting_dist = Frechet
    _plotting_dist_name = "Frechet"

    def __repr__(self):
        return f"{self.__class__.__name__} exceedance dependence model with (alpha, beta, gamma) = {self.params} and quantile threshold {self.quantile_threshold}"

    @validator("params")
    def validate_params(cls, params):
        alpha, beta, gamma = params
        if alpha <= 0 or alpha > 1:
            raise TypeError(f"alpha must be in the interval (0,1]")
        if (beta < 0 or beta > 1) or (gamma < 0 or gamma > 1):
            raise TypeError(f"beta, gamma must be in the interval [0,1]")
        else:
            return params

    @classmethod
    def logpdf(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates logpdf function for asymmetric logistic Frechet exceedances


        Args:
            params (np.ndarray): array with dependence and asymmetry parameters
            threshold (float): Exceedance threshold in Gumbel scale
            data (t.Union[np.ndarray, t.Iterable]): Observed data in Frechet scale

        """
        alpha, beta, gamma = params

        x, y = cls.unbundle(data)
        rescaler = 1 - cls.unconditioned_cdf(params, cls.bundle(threshold, threshold))

        alpha_prenorm = (beta / x) ** (1 / alpha) + (gamma / y) ** (1 / alpha)
        alpha_norm = (alpha_prenorm) ** alpha

        log_a = (-1 + beta) / x + (-1 + gamma - y * alpha_norm) / y
        log_b = -(2 * np.log(x) + 2 * np.log(y) + np.log(alpha))

        c_1 = (
            -x
            * y
            * (-1 + alpha)
            * (beta * gamma / (x * y)) ** (1 / alpha)
            * alpha_prenorm ** (-2 + alpha)
        )
        c_2 = (
            alpha
            * (1 - beta + x * (beta / x) ** (1 / alpha) * alpha_prenorm ** (-1 + alpha))
            * (
                1
                - gamma
                + y * (gamma / y) ** (1 / alpha) * alpha_prenorm ** (-1 + alpha)
            )
        )
        log_c = np.log(c_1 + c_2)

        log_density = log_a + log_b + log_c - np.log(rescaler)

        # density is 0 when both coordinates are below the threshold
        nil_density_idx = np.logical_and(x <= threshold, y <= threshold)
        log_density[nil_density_idx] = -np.Inf

        return log_density

    @classmethod
    def unconditioned_cdf(
        cls, params: np.ndarray, data: t.Union[np.ndarray, t.Iterable]
    ) -> np.ndarray:
        """Calculates unconstrained standard Frechet cdf with asymmetric logistic dependence

        Args:
            params (np.ndarray): array with dependence and asymmetry parameters
            data (t.Union[np.ndarray, t.Iterable]): Observed data in Frechet scale

        Returns:
            np.ndarray: cdf values
        """
        x, y = cls.unbundle(data)
        alpha, beta, gamma = params
        return np.exp(
            -(1 - beta) / x
            - (1 - gamma) / y
            - ((beta / x) ** (1 / alpha) + (gamma / y) ** (1 / alpha)) ** alpha
        )

    def simulate_model(
        self, size: int, params: np.ndarray, quantile_threshold: float
    ) -> np.ndarray:
        """Simulate asymmetric logistic exceedance model variates in Frechet scale using a method inspired by the one outlined in 'Simulating Multivariate Extreme Value Distributions of Logistic Type' by Stephenson (2003).

        Args:
            size (int): Sample size
            params (np.ndarray): array with dependence and asymmetry parameters
            quantile_threshold (float): Quantile threshold for simulated exceedances

        Returns:
            np.ndarray: Simulated sample
        """

        # Gumbel scales are used instead of Frechet in this method, and a the existing logistic model is used as a baseline sampler
        # in Gumbel scales asymmetry constants are offsets instead of rescaling factors
        gumbel_logistic = Logistic.simulate_model
        threshold = gumbel.ppf(quantile_threshold)
        # threshold from which we want to sample
        target_th = np.array([threshold, threshold]).reshape((1, 2))
        # unpack params
        alpha, beta, gamma = params

        def logistic_exs(size: int, alpha: float, a: float, b: float) -> np.ndarray:
            """
            Samples a non-deterministic number of exceedances from a logistic exceedance model. A smaller , symmetric proxy threshold is used to simulate variates that are later offset to match the target threshold. Because the proxy threshold is symmetric but the target threshold may not be, some samples may be discarded. The expected sample size is the passed size parameter. The target threshold value comes from outer scope.

            Args:
                size (int): Sample size
                alpha (float): Dependence parameter
                a (float): asymmetry parameter for first component
                b (float): Asymmetry parameter for second component

            Returns:
                np.ndarray: Simulated sample

            """
            alpha_param = np.array([alpha])
            ## offset from asymmetry parameters
            offset = np.array([np.log(a), np.log(b)]).reshape((1, 2))
            offset_th = target_th - offset
            # compute lower symmetric threshold to use as a proxy when sampling (some samples may be dropped)
            min_offset = np.min(offset_th)
            effective_th = min_offset * np.ones((2,)).reshape((1, 2))
            effective_q = gumbel.cdf(min_offset)
            # compute average sample drop rate
            s_effective = 1 - Logistic.unconditioned_cdf(alpha_param, effective_th)
            s_offset = 1 - Logistic.unconditioned_cdf(alpha_param, offset_th)
            drop_rate = (s_effective - s_offset) / s_effective
            if drop_rate >= 1 or drop_rate < 0:
                raise Exception(f"Invalid drop rate ({drop_rate})")
            effective_size = int(size / (1 - drop_rate))
            # sample from logistic model
            b2 = gumbel_logistic(effective_size, alpha_param, effective_q)
            # skew to symmetric logistic parameters
            b2[:, 0] += np.log(a)
            b2[:, 1] += np.log(b)
            # drop samples outside target region
            target_th1, target_th2 = target_th.reshape(-1)
            exs_idx = np.logical_or(b2[:, 0] > target_th1, b2[:, 1] > target_th2)
            b2 = b2[exs_idx, :]
            return b2

        def logistic_non_exs(size: int, alpha: float, a: float, b: float) -> np.ndarray:
            """
            same as above for non-exceedances

            Args:
                size (int): Sample size
                alpha (float): Dependence parameter
                a (float): asymmetry parameter for first component
                b (float): Asymmetry parameter for second component

            Returns:
                np.ndarray: Simulated sample

            """
            alpha_param = np.array([alpha])
            ## offset from asymmetry parameters
            offset = np.array([np.log(a), np.log(b)]).reshape((1, 2))
            offset_th = target_th - offset
            # compute lower symmetric threshold to use as a proxy when sampling (some samples may be dropped)
            min_offset = np.min(offset_th)
            effective_th = min_offset * np.ones((2,)).reshape((1, 2))
            # compute average sample drop rate
            drop_rate = 1 - Logistic.unconditioned_cdf(alpha_param, effective_th)
            if drop_rate >= 1 or drop_rate < 0:
                raise Exception(f"Invalid drop rate ({drop_rate})")
            effective_size = int(size / (1 - drop_rate))
            # sample from logistic model
            b2 = gumbel_logistic(effective_size, alpha_param, 0)
            # skew to symmetric logistic parameters
            b2[:, 0] += np.log(a)
            b2[:, 1] += np.log(b)
            # drop samples outside target region
            target_th1, target_th2 = target_th.reshape(-1)
            exs_idx = np.logical_and(b2[:, 0] <= target_th1, b2[:, 1] <= target_th2)
            b2 = b2[exs_idx, :]
            return b2

        def sample(
            sampler: t.Callable, size: int, alpha: float, a: float, b: float
        ) -> np.ndarray:
            """Wrapper for the functions above that makes the sample size deterministic

            Args:
                size (int): Sample size
                alpha (float): dependence parameter
                a (float): asymmetry parameter for first component
                b (float): asymmetry parameter for second component

            Returns:
                np.ndarray: Sample
            """
            # get bivariate samples ($B_2$ in the referenced paper)
            #
            b2 = sampler(size, alpha, a, b)
            while len(b2) < size:
                k = size - len(b2)
                updated_size = max(100, 2 * k)
                b2 = np.concatenate([b2, sampler(updated_size, alpha, a, b)], axis=0)
            b2 = b2[0:size, :]
            return b2

        # The paper referenced above takes the maximum from logistic model samples and rescales them (or offsets them, in Gumbel scale) in order to sample from an asymmetric logistic.
        # B_1 = bivariate independent offset Gumbel samples
        # B_2 = bivariate logistic offset Gumbel
        # B_1 is independent from B_2. Let M = max(B_1, B_2), then M ~ asym. logistic Gumbel

        alpha_param = np.array([alpha])

        b2_offset = np.log(np.array([beta, gamma]))
        b1_offset = np.log(np.array([1 - beta, 1 - gamma]))
        if quantile_threshold == 0:
            # if simulated region is unconditioned (i.e. the entire plane) use method from cited paper directly
            # independent gumbel variates (alpha = 1)
            b1 = (
                gumbel_logistic(size=size, params=np.array([1]), quantile_threshold=0)
                + b1_offset
            )
            # logistic gumbel variates
            b2 = (
                gumbel_logistic(size=size, params=alpha_param, quantile_threshold=0)
                + b2_offset
            )
            z = np.maximum(b1, b2)
        else:
            # Sampling exceedances requires some additional care:
            # exceedance in M <=> exceedance in B_1 or exceedance in B_2
            # exceedances can then be split in three cases: exceedance in B_1 alone, in B_2 alone or in both.
            # below the 3 cases are simulated separately and then concatenated

            # work out sample size proportions for the three cases
            p_exs_b1 = float(
                1 - Logistic.unconditioned_cdf(np.array([1]), target_th - b1_offset)
            )
            p_exs_b2 = float(
                1 - Logistic.unconditioned_cdf(alpha_param, target_th - b2_offset)
            )
            p_exs_any = p_exs_b1 + p_exs_b2 - p_exs_b1 * p_exs_b2
            p_exs = (
                np.array(
                    [
                        p_exs_b1 * (1 - p_exs_b2),
                        p_exs_b2 * (1 - p_exs_b1),
                        p_exs_b1 * p_exs_b2,
                    ]
                )
                / p_exs_any
            )  # relative sample sizes for all three cases

            b1_exs_size, b2_exs_size, both_exs_size = np.random.multinomial(
                n=size, pvals=p_exs, size=1
            )[0].astype(np.int32)

            # sample 6 cases: (case 1, 2 or 3) * (exceedance or non-exceedance)
            b1_exs = sample(
                sampler=logistic_exs, size=b1_exs_size, alpha=1, a=1 - beta, b=1 - gamma
            )

            b2_not_exs = sample(
                sampler=logistic_non_exs, size=b1_exs_size, alpha=alpha, a=beta, b=gamma
            )

            b2_exs = sample(
                sampler=logistic_exs, size=b2_exs_size, alpha=alpha, a=beta, b=gamma
            )

            b1_not_exs = sample(
                sampler=logistic_non_exs,
                size=b2_exs_size,
                alpha=1,
                a=1 - beta,
                b=1 - gamma,
            )

            b1_both_exs = sample(
                sampler=logistic_exs,
                size=both_exs_size,
                alpha=1,
                a=1 - beta,
                b=1 - gamma,
            )

            b2_both_exs = sample(
                sampler=logistic_exs, size=both_exs_size, alpha=alpha, a=beta, b=gamma
            )

            # take elementwise maximum and concatenate
            z = np.concatenate(
                [
                    np.maximum(b1_exs, b2_not_exs),
                    np.maximum(b2_exs, b1_not_exs),
                    np.maximum(b1_both_exs, b2_both_exs),
                ],
                axis=0,
            )

        # return in canonical model scale, i.e. unit Frechet
        return np.exp(z)


class LogisticGP(Logistic):

    """This model is an implementation of a bivariate generalised Pareto distribution with logistic dependence. In Gumbel scale this is given by

    $$ \\mathbb{P}(\\textbf{X} \\leq \\mathbf{x}) = \\frac{\\Psi(\\mathbf{x}) - \\Psi(\\min \\{\\mathbf{x},\\mathbf{0}\\})}{-\\Psi(\\mathbf{0})}, \\mathbf{X} \\nleq \\mathbf{0}$$

    where

    $$ \\Psi(\\mathbf{x}) = - \\left(  \\exp \\left( - \\frac{\\mathbf{x}_1}{\\alpha} \\right) + \\left(  - \\frac{\\mathbf{x}_2}{\\alpha}  \\right) \\right)^{\\alpha}, \\, 0 \\leq \\alpha \\leq 1, $$

    """

    @validator("quantile_threshold")
    def val_qt(cls, quantile_threshold):
        q = Logistic._model_marginal_dist.cdf(0)
        if quantile_threshold <= q:
            raise ValueError(
                f"This model does not support quantile thresholds lower than {np.round(q,2)}"
            )
        else:
            return quantile_threshold

    @classmethod
    def logpdf(
        cls, params: np.ndarray, threshold: float, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates the logpdf function


        Args:
            params (np.ndarray): array with dependence parameter as only element
            threshold (float): Exceedance threshold in Gumbel scale
            data (t.Union[np.ndarray, t.Iterable]): Observed data in Gumbel scale

        """
        alpha = params[0]

        x, y = cls.unbundle(data)

        rescaler = 1 - Logistic.unconditioned_cdf(
            params, cls.bundle(threshold, threshold)
        )

        log_a = -x / alpha - y / alpha
        log_b = (-2 + alpha) * np.log(np.exp(-x / alpha) + np.exp(-y / alpha))
        log_c = np.log(1 - alpha)
        log_d = -np.log(alpha)

        log_density = log_a + log_b + log_c + log_d - np.log(rescaler)

        # density is 0 when both coordinates are below the threshold
        nil_density_idx = np.logical_and(x <= threshold, y <= threshold)
        log_density[nil_density_idx] = -np.Inf

        return log_density

    @classmethod
    def unconditioned_cdf(
        cls, params: np.ndarray, data: t.Union[np.ndarray, t.Iterable]
    ):
        """Calculates cdf function with an exceedance threshold of zero"""
        alpha = params[0]
        x, y = cls.unbundle(data)
        G0 = Logistic.unconditioned_cdf(params, cls.bundle(0, 0))
        Ga = Logistic.unconditioned_cdf(params, data)
        Gb = Logistic.unconditioned_cdf(params, np.minimum(data, 0))
        return -1 / np.log(G0) * (np.log(Ga) - np.log(Gb))

    @classmethod
    def simulate_model(
        cls, size: int, params: np.ndarray, quantile_threshold: float
    ) -> np.ndarray:
        """Simulates exceedances from bivariate GP model

        Args:
            size (int): Simulated sample size
            params (np.ndarray): Array with dependence parameter
            threshold (float): Exceedance threshold in both components

        Returns:
            np.ndarray: Simulated sample
        """
        threshold = gumbel.ppf(quantile_threshold)
        alpha = params[0]

        maxima = exponential.rvs(size=size, loc=threshold)

        if alpha == 0:
            r = 0
        elif alpha == 1:
            r = np.Inf
        else:
            ###simulate difference between maxima and minima r = max(x,y) - min(x,y) using inverse function method
            u = np.random.uniform(size=size)
            r = alpha * np.log(((1 - u) / 2 ** (1 - alpha)) ** (1 / (alpha - 1)) - 1)

        minima = maxima - r

        # allocate maxima randomly between components
        max_indices = np.random.binomial(1, 0.5, size)

        x = np.concatenate(
            [
                maxima[max_indices == 0].reshape((-1, 1)),
                minima[max_indices == 1].reshape((-1, 1)),
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                minima[max_indices == 0].reshape((-1, 1)),
                maxima[max_indices == 1].reshape((-1, 1)),
            ],
            axis=0,
        )

        return cls.bundle(x, y)


class Empirical(BaseDistribution):

    """Bivariate empirical distribution induced by a sample of observed data"""

    data: np.ndarray
    pdf_values: np.ndarray

    _exceedance_models = {
        "logistic": Logistic,
        "gaussian": Gaussian,
        "asymmetric logistic": AsymmetricLogistic,
        "logistic gp": LogisticGP,
    }

    def __repr__(self):
        return f"Bivariate empirical distribution with {len(self.data)} points"

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
        if (
            not isinstance(data, np.ndarray)
            or len(data.shape) != 2
            or data.shape[1] != 2
        ):
            raise ValueError("data must be an n x 2 numpy array")

        n = len(data)
        return Empirical(
            data=data, pdf_values=1.0 / n * np.ones((n,), dtype=np.float64)
        )

    def pdf(self, x: np.ndarray):

        return np.mean(self.data == x.reshape((1, 2)))

    def cdf(self, x: np.ndarray):
        if len(x.shape) > 1:
            return np.array([self.cdf(elem) for elem in x])

        u = self.data <= x.reshape((1, 2))  # componentwise comparison
        v = (
            u.dot(np.ones((2, 1))) >= 2
        )  # equals 1 if and only if both components are below x
        return np.mean(v)

    def simulate(self, size: int):
        n = len(self.data)
        idx = np.random.choice(n, size=size)
        return self.data[idx]

    def get_marginals(self):

        return univar.Empirical.from_data(self.data[:, 0]), univar.Empirical.from_data(
            self.data[:, 1]
        )

    def fit_tail_model(
        self,
        model: str,
        quantile_threshold: float,
        margin1: univar.BaseDistribution = None,
        margin2: univar.BaseDistribution = None,
    ):
        """Fits a parametric model for threshold exceedances in the data. For a given threshold \\( u \\), exceedances are defined as vectors \\(Z\\) such that \\( \\max\\{Z_1,Z_2\\} > u \\), this is, an exceedance in at least one component, and encompasses an inverted L-shaped subset of Euclidean space.
        Currently, logistic and Gaussian models are available, with the former exhibiting asymptotic dependence, a strong type of dependence between extreme occurrences across components, and the latter exhibiting asymptotic independence, in which extremes occur relatively independently across components.

        Args:
            model (str): name of selected model, currently one of 'gaussian' or 'logistic' or 'asymmetric logistic'. For more information, see `Gaussian`,  `Logistic` and `AsymmetricLogistic` classes.
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
            warnings.warn(
                f"First marginal not provided. Fitting tail model using provided quantile threshold ({quantile_threshold} => {margin1.ppf(quantile_threshold)})",
                stacklevel=2,
            )

        if margin2 is None:

            _, margin2 = self.get_marginals()
            margin2 = margin2.fit_tail_model(threshold=margin2.ppf(quantile_threshold))
            warnings.warn(
                f"Second marginal not provided. Fitting tail model using provided quantile threshold ({quantile_threshold} => {margin2.ppf(quantile_threshold)})",
                stacklevel=2,
            )

        data = self.data

        x = data[:, 0]
        y = data[:, 1]

        exceedance_idx = np.logical_or(
            x > margin1.ppf(quantile_threshold), y > margin2.ppf(quantile_threshold)
        )

        exceedances = data[exceedance_idx]

        exceedance_model = self._exceedance_models[model].fit(
            data=exceedances,
            quantile_threshold=quantile_threshold,
            margin1=margin1,
            margin2=margin2,
        )

        empirical_model = Empirical.from_data(self.data[np.logical_not(exceedance_idx)])

        p = np.mean(np.logical_not(exceedance_idx))

        return ExceedanceModel(
            distributions=[empirical_model, exceedance_model],
            weights=np.array([p, 1 - p]),
        )

    def test_asymptotic_dependence(
        self, quantile_threshold: float = 0.95, prior: t.Optional[str] = "jeffreys"
    ) -> float:
        """Computes a Savage-Dickey ratio for the coefficient of tail dependence \\(\\eta\\) to test the hypothesis of asymptotic dependence (See 'Statistics of Extremes' by Beirlant, page 345-346). The hypothesis space is \\(\\eta \\in [0,1]\\) with \\(\\eta = 1\\) corresponding to asymptotic dependence. The posterior density is approximated through Gaussian Kernel density estimation.

        Args:
            quantile_threshold (float, optional): Quantile threshold over which the coefficient of tail dependence is to be estimated.
            log_prior (str, optional): Name of prior distribution to use for Bayesian inference. must be one of 'flat', which uses a flat prior on the support, or 'jeffreys' which uses an uninformative Jeffreys prior; defaults to 'jeffreys'.

        Returns:
            float: Savage-Dickey ratio. If larger than 1, this favors the asymptotic dependence hypothesis and vice versa.
        """

        def flat_prior(theta):
            scale, shape = theta
            if scale > 0 and shape >= 0 and shape <= 1:
                return 0.0
            else:
                return -np.Inf

        def jeffreys_prior(theta):
            scale, shape = theta
            if scale > 0 and shape >= 0 and shape <= 1:
                return -np.log(scale) - np.log(1 + shape) - 0.5 * np.log(1 + 2 * shape)
            else:
                return -np.Inf

        log_priors = {"flat": flat_prior, "jeffreys": jeffreys_prior}
        # Savage-Dickey ratios use the prior marginal density for eta = 1. Compute it for both priors
        prior_density = {
            "flat": 1,  # prior is uniform in [0,1] for eta
            "jeffreys": np.exp(log_priors["jeffreys"]([1, 1]))
            / (np.pi / 6),  # constant factor is pi/6 (integral in [0,1])
        }

        try:
            log_prior = log_priors[prior]
        except KeyError as e:
            raise ValueError(
                f"Prior name not recognised. Must be one of {log_priors.keys()}"
            )

        ### compute savage-dickey density ratio
        # Use generalised pareto to fit tails
        x, y = self.data.T
        x_dist, y_dist = univar.Empirical.from_data(x), univar.Empirical.from_data(y)
        x_dist, y_dist = x_dist.fit_tail_model(
            x_dist.ppf(quantile_threshold)
        ), y_dist.fit_tail_model(y_dist.ppf(quantile_threshold))

        # transform to approximate standard Frechet margins and map to test data t
        u1, u2 = x_dist.cdf(x), y_dist.cdf(y)
        z1, z2 = -1 / np.log(u1), -1 / np.log(u2)
        t = np.minimum(z1, z2)

        t_dist = univar.Empirical.from_data(t)
        # Pass initial point that enforces theoretical constraints of 0 <= eta <= 1. Approximation inaccuracies from the transformation to Frechet margins and MLE estimation can violate the bounds.
        mle_tail = t_dist.fit_tail_model(t_dist.ppf(quantile_threshold))
        x0 = np.array([mle_tail.tail.scale, max(0, min(0.99, mle_tail.tail.shape))])
        # sample posterior
        t_dist = t_dist.fit_tail_model(
            t_dist.ppf(quantile_threshold), bayesian=True, log_prior=log_prior, x0=x0
        )

        # approximate posterior distribution through Kernel density estimation. Evaluating it on 1 gives us the savage-dickey ratio
        # return gaussian_kde(t_dist.tail.shapes).evaluate(1)[0]

        return gaussian_kde(t_dist.tail.shapes).evaluate(1)[0] / prior_density[prior]

    @classmethod
    def pickands(
        self, p: np.ndarray, data: np.ndarray, quantile_threshold: float = 0.95
    ) -> np.ndarray:
        """Non-parametric Pickands dependence function approximation based on Hall and Tajvidi (2000).

        Args:
            p (np.ndarray): Pickands dependence function arguments
            data: (np.ndarray): data matrix of n x 2
            quantile_threshold (float, optional): Quantile threshold over which Pickands dependence will be approximated.

        Returns:
            np.ndarray: Nonparametric estimate of the Pickands dependence function
        """
        # find joint extremes
        if np.any(p > 1) or np.any(p < 0):
            raise ValueError("All argument values must be in [0,1]")

        x, y = data.T
        joint_extremes_idx = np.logical_and(
            x > np.quantile(x, quantile_threshold),
            y > np.quantile(y, quantile_threshold),
        )
        n = np.sum(joint_extremes_idx)

        # compute nonparametric scores
        x_exs, y_exs = x[joint_extremes_idx], y[joint_extremes_idx]
        x_exs_model, y_exs_model = (
            univar.Empirical.from_data(x_exs),
            univar.Empirical.from_data(y_exs),
        )

        # normalise to avoid copula values on the border of the unit square
        u1, u2 = n / (n + 1) * x_exs_model.cdf(x_exs), n / (n + 1) * y_exs_model.cdf(
            y_exs
        )
        # map to exponential
        s, t = -np.log(u1), -np.log(u2)

        pk = []
        for p_ in p:
            a = (s / np.mean(s)) / (1 - p_)
            b = (t / np.mean(t)) / p_
            pk.append(1.0 / np.mean(np.minimum(a, b)))

        return np.array(pk)

    def plot_pickands(
        self, quantile_threshold: float = 0.95
    ) -> matplotlib.figure.Figure:
        """Returns a plot of the empirical Pickands dependence function induced by joint exceedances above the specified quantile threshold.
        The Pickands dependence function \\(A: [0,1] \\to [1/2,1]\\) is convex and bounded by \\(\\max\\{t,1-t\\} \\leq A(t) \\leq 1\\); it can be used to assess extremal dependence, as there is a one-to-one correspondence between \\(A(t)\\) and extremal copulas; the closer it is to its lower bound, the stronger the extremal dependence. Conversely, for asymptotically independent data \\(A(t) = 1\\).
        The non-parametric approximation used here is based on Hall and Tajvidi (2000).

        Args:
            quantile_threshold (float, optional): Quantile threshold over which Pickands dependence will be approximated.

        Returns:
            matplotlib.figure.Figure: figure
        """
        fig = plt.figure(figsize=(5, 5))

        x = np.linspace(0, 1, 101)
        pk = self.pickands(x, self.data)

        plt.plot(x, pk, color="darkorange", label="Empirical")
        plt.plot(x, np.maximum(1 - x, x), linestyle="dashed", color="black")
        plt.plot(x, np.ones((len(x),)), linestyle="dashed", color="black")
        plt.xlabel("t")
        plt.ylabel("A(t)")
        plt.title("Empirical Pickands dependence of joint exceedances")
        plt.grid()
        return fig

    def plot_chi(self, direct_estimate: bool = True) -> matplotlib.figure.Figure:
        """Returns a plot of the empirical estimate of the coefficient of asymptotic dependence , defined as \\( \\chi = \\ \\lim_{p \\to 1} \\mathbb{P}(Y > q_y(p) \\,|\\, X > q_x (p)) \\), where \\( q_x, q_y\\) are the corresponding quantiles for a given probability level \\(p\\). There is asymptotic dependence when \\(\\chi > 0\\) and vice versa.

        Args:
            direct_estimate (optional, bool): if True, use the empirical copula's probability estimates directly. Otherwise use the approximation given by \\(\\chi = \\lim_{u \\to 1} 2 - \\log C(u,u) / \\log(u) \\) where \\(C(u,u)\\) is the empirical copula.

        Returns:
            matplotlib.figure.Figure: figure
        """

        def chi(p: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
            """Returns chi estimates and the corresponding estimate's standard deviation

            Args:
                t (np.ndarray): input probability levels

            Returns:
                t.Tuple[np.ndarray, np.ndarray]: estimate and standard deviation arrays
            """
            x, y = self.get_marginals()
            n = len(x.data)
            # map to rescaled copula values in the open set (0,1)
            u1, u2 = x.cdf(x.data) * n / (n + 1), y.cdf(y.data) * n / (n + 1)
            empirical_copula = Empirical.from_data(np.stack([u1, u2], axis=1))
            if direct_estimate:
                joint_p = 1 - 2 * p + empirical_copula.cdf(np.stack([p, p], axis=1))
                estimate = joint_p / (1 - p)
                n_obs = len(x.data) * (1 - p)
                std = np.sqrt(estimate * (1 - estimate) / n_obs)
            else:
                vals = empirical_copula.cdf(np.stack([p, p], axis=1))
                estimate = 2 - np.log(vals) / np.log(p)
                # standard error approximation using delta method
                std = np.sqrt(
                    vals * (1 - vals) / n * 1 / np.log(p) ** 2 * 1 / vals**2
                )
            return estimate, std

        fig = plt.figure(figsize=(5, 5))
        p = np.linspace(0.01, 0.99, 99)
        chi_central, chi_std = chi(p)
        ci_l, ci_u = chi_central - 1.96 * chi_std, chi_central + 1.96 * chi_std

        plt.plot(p, chi_central, color=self._figure_color_palette[0])
        plt.scatter(p, chi_central, color=self._figure_color_palette[0])
        plt.fill_between(
            p,
            ci_l,
            ci_u,
            linestyle="dashed",
            color=self._figure_color_palette[1],
            alpha=0.2,
        )
        plt.xlabel("quantiles")
        plt.ylabel("Chi(u)")
        plt.grid()
        return fig
